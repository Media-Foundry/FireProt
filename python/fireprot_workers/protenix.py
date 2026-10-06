"""Pinned Protenix model execution; Rust owns assets, lifecycle and acceptance."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import sys
import time

RESOURCE_NAMES = (
    'ccd_components_file', 'ccd_components_rdkit_mol_file',
    'pdb_cluster_file', 'obsolete_release_data_csv',
)


def check_request(request):
    workload = request['workload']
    if request['source']['commit'] != 'bb3be993b52713795796c72cf0a46ef8a7323442':
        raise ValueError('Protenix source is outside the integrated compatibility pin')
    if workload['model_name'] not in ('protenix_base_default_v0.5.0', 'protenix_mini_default_v0.5.0', 'protenix_mini_esm_v0.5.0', 'protenix_tiny_default_v0.5.0') or workload['precision'] != 'float32':
        raise ValueError('This worker integrates Base-default, Mini-default, Mini-ESM and Tiny-default v0.5.0 FP32')
    if set(request.get('resources', {})) != set(RESOURCE_NAMES):
        raise ValueError('Protenix requires exactly the four pinned CCD/data resources')
    checkpoint = Path(request['checkpoints']['model']['path'])
    if checkpoint.name != workload['model_name'] + '.pt':
        raise ValueError('Checkpoint filename must match the requested Protenix model')
    from .protenix_esm import check_checkpoints
    check_checkpoints(request, checkpoint)
    records = json.loads(Path(request['input']['path']).read_text())
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError('Protenix worker requires one input record')
    record = records[0]
    name = record.get('name', '')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', name) or name in ('.', '..'):
        raise ValueError('Input name must be a safe single path component')
    if set(record) != {'name', 'sequences'} or len(record['sequences']) != 1:
        raise ValueError('Only a sequence-only monomer input is integrated')
    entity = record['sequences'][0]
    if set(entity) != {'proteinChain'}:
        raise ValueError('Only a protein monomer is integrated')
    chain = entity['proteinChain']
    sequence = chain.get('sequence', '')
    if set(chain) - {'sequence', 'count'} or chain.get('count', 1) != 1:
        raise ValueError('Modified proteins, external features and multiple copies are not integrated')
    if not sequence or any(residue not in 'ACDEFGHIKLMNPQRSTVWYX' for residue in sequence):
        raise ValueError('Invalid protein sequence')
    if workload['model_name'] == 'protenix_mini_esm_v0.5.0' and len(sequence) > 4094:
        raise ValueError('Mini-ESM sequences above 4094 residues would be truncated upstream')
    if request['seed'] >= 2**32:
        raise ValueError('Protenix NumPy seeding requires a seed below 2**32')
    for key, expected in (('RANK', 0), ('LOCAL_RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_WORLD_SIZE', 1)):
        if int(os.environ.get(key, expected)) != expected:
            raise ValueError('Distributed model execution is not integrated')
    return name, checkpoint


def _install_native_construction(backend):
    """Honor the explicit initialization opt-in without installing GPU operators."""
    if backend != 'native' or os.environ.get('FIREPROT_ENABLE_FAST_MODEL_INIT') != '1':
        return None
    from fireprot.adapters.protenix import construction
    construction.install()
    return construction.manifest


def predict(request, channel):
    started = time.perf_counter()
    name, checkpoint = check_request(request)
    workload = request['workload']
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    esm_enabled = workload['model_name'] == 'protenix_mini_esm_v0.5.0'
    if esm_enabled and (Path.cwd() != output or (output / 'esm_embeddings').exists()):
        raise RuntimeError('Mini-ESM requires a fresh run-local embedding directory')
    os.environ['LAYERNORM_TYPE'] = 'torch'
    import torch
    import protenix
    if Path(protenix.__file__).resolve().parents[1] != Path(request['source']['root']):
        raise RuntimeError('Imported Protenix does not match the verified source root')
    # CCD module constants are initialized from these upstream configuration
    # values. Set explicit verified paths before any model/data module import.
    if 'protenix.data.core.ccd' in sys.modules:
        raise RuntimeError('Protenix CCD was imported before verified resource configuration')
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    for key in RESOURCE_NAMES:
        data_configs[key] = request['resources'][key]['path']
    inference_configs['load_checkpoint_dir'] = str(checkpoint.parent)
    inference_configs['sorted_by_ranking_score'] = False
    base_enabled = workload['model_name'] == 'protenix_base_default_v0.5.0'
    if base_enabled:
        # Base C10/D200 native repeats exceed the numerical gate otherwise.
        # This is mandatory for both backends of the Base worker contract.
        inference_configs['deterministic'] = True
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True

    runtime_manifest = None
    construction_manifest = _install_native_construction(request['backend'])
    kernel = 'torch'
    if request['backend'] == 'fireprot':
        from fireprot.adapters.protenix.runtime import install, manifest
        install('hipblas')
        runtime_manifest = manifest
        from fireprot.adapters.protenix.construction import manifest as construction_manifest
        kernel = 'fireprot'

    from runner import batch_inference, inference
    from protenix.utils.torch_utils import map_values_to_list
    from .transport import save_arrays
    torch.set_num_threads(1)
    torch.backends.cuda.preferred_blas_library('hipblas')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original_predict = inference.InferenceRunner.predict
    original_download = inference.download_from_url
    predictions = []
    calls, model_seconds = 0, 0.0

    def reject_download(*args, **kwargs):
        raise RuntimeError('Model downloads are disabled; provide all assets in the Rust request')

    def capture(runner, data):
        nonlocal calls, model_seconds
        cfg = runner.model.configs
        if base_enabled and (not cfg.deterministic
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.utils.deterministic.fill_uninitialized_memory):
            raise RuntimeError('Base requires strict deterministic execution with memory filling')
        if (cfg.model.N_cycle, cfg.sample_diffusion.N_step, cfg.sample_diffusion.N_sample, cfg.dtype) != (
                workload['cycles'], workload['diffusion_steps'], workload['samples'], 'fp32'):
            raise RuntimeError('Protenix workload changed at the model boundary')
        if data['sample_name'] != name or calls != 0:
            raise RuntimeError('Unexpected Protenix query or repeated model call')
        if bool(cfg.esm.enable) != esm_enabled:
            raise RuntimeError('Protenix ESM configuration changed at the model boundary')
        esm_features = None
        if esm_enabled:
            from .protenix_esm import validate_features
            esm_features = validate_features(request, data, cfg.esm, output)
        calls += 1
        channel.emit(dict(event='progress', stage='model'))
        torch.cuda.synchronize()
        begin = time.perf_counter()
        from fireprot.adapters.protenix.base_n1000 import install as base_optimizations
        with base_optimizations(runner.model, output, request, data):
            result = original_predict(runner, data)
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - begin
        if result['coordinate'].ndim != 3 or result['coordinate'].shape[0] != workload['samples']:
            raise RuntimeError('Unexpected Protenix coordinate sample axis')
        if len(result['summary_confidence']) != workload['samples'] or len(result['full_data']) != workload['samples']:
            raise RuntimeError('Unexpected Protenix confidence sample count')
        ranks = runner.dumper._get_ranker_indices(result)
        for sample in range(workload['samples']):
            arrays = {'coordinates': result['coordinate'][sample]}
            if esm_features is not None:
                arrays['input.esm_token_embedding'] = esm_features
            arrays.update({'confidence.' + key: value for key, value in result['full_data'][sample].items()
                           if key != 'atom_coordinate'})
            archive = f'sample-{sample}.npz'
            descriptors = save_arrays(output / archive, arrays)
            summary = map_values_to_list(copy.deepcopy(result['summary_confidence'][sample]))
            predictions.append(dict(sample=sample, archive=archive, arrays=descriptors, confidence=summary,
                structure=f"native/{name}/seed_{request['seed']}/predictions/{name}_sample_{int(ranks[sample])}.cif"))
        return result

    inference.InferenceRunner.predict = capture
    inference.download_from_url = reject_download
    try:
        batch_inference.inference_jsons(
            json_file=request['input']['path'], out_dir=str(output / 'native'),
            seeds=[request['seed']], n_cycle=workload['cycles'], n_step=workload['diffusion_steps'],
            n_sample=workload['samples'], dtype='fp32', model_name=workload['model_name'],
            trimul_kernel=kernel, triatt_kernel=kernel, enable_cache=True, enable_fusion=True,
            enable_tf32=False, use_msa=False, use_template=False, use_rna_msa=False,
            use_seeds_in_json=False, need_atom_confidence=False,
        )
    finally:
        inference.InferenceRunner.predict = original_predict
        inference.download_from_url = original_download
    # Upstream catches inference/data/serialization exceptions. Do not accept
    # an empty result or an output captured before a later dumper failure.
    errors = list((output / 'native' / 'ERR').glob('*'))
    structures = list((output / 'native').rglob('*.cif'))
    if errors or calls != 1 or len(predictions) != workload['samples'] or len(structures) != workload['samples']:
        raise RuntimeError('Incomplete Protenix inference; inspect native/ERR and worker logs')
    if construction_manifest is not None:
        (output / 'construction-runtime.json').write_text(
            json.dumps(construction_manifest(), indent=2, allow_nan=False) + '\n')
    if runtime_manifest is not None:
        (output / 'operator-runtime.json').write_text(json.dumps(runtime_manifest(), indent=2, allow_nan=False) + '\n')
    return dict(schema_version=2, run_id=request['run_id'], seed=request['seed'], workload=workload,
                predictions=predictions, model_seconds=model_seconds, request_seconds=time.perf_counter()-started)
