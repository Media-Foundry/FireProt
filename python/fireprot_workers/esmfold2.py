"""Pinned Biohub ESMFold2 monomer invocation, including its bundled ESMC LM."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

SOURCE_COMMIT = '43b4548b86762edfa747b07d5f440aad3c33acee'
WEIGHT_REVISION = '69869f737beffec5294845ede23db5fc0b4f509e'
ASSET_HASHES = {
    'model': '5e1f94eb5cd4b0ff0bfdeae3dbf64be564ba34352fe6ef3a6ee9830f584134b2',
    'model_config': '72054c1af92f432ac2d8e0628ba08b377d04c6a8679b5128b798dbc7f644ddc8',
    'ccd': '9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5',
    'model_shard_00001': 'a24f050e2a2f9746b43b435c53719afc575a374fa302731b6911356a3d8b5cac',
    'model_shard_00002': '1af1615f6f6d1b3c0a24a63d9052a0ecad5b3831d1513c1a5fea73870c1bec73',
    'model_shard_00003': '357b55ec56192fd01a1cfc39a5260caf2242503ac7c554f614e43c7a469f0ced',
    'model_shard_00004': 'f065aca1a13308bce41b78f728c28a0a29e3a9db6af531a74a258dbb06f8721b',
    'model_shard_00005': 'e0c507dce5b2ddb22f77305607b4332506137850c5378e98e9729345264668b7',
    'model_shard_00006': '90015eb13bbeab54237b18d6748dbd47f4bfd108fdf45704d66e5c3a93f8df7c',
}


def model_directory(request):
    """Bind every file consumed by the local sharded loader to a Rust-checked pin."""
    index = Path(request['checkpoints']['model']['path']).resolve()
    if index.name != 'model.safetensors.index.json':
        raise ValueError('ESMFold2 model checkpoint must be the sharded index')
    root = index.parent
    resources = request['resources']
    pins = dict(resources, model=request['checkpoints']['model'])
    if any(pins.get(name, {}).get('sha256') != digest for name, digest in ASSET_HASHES.items()):
        raise ValueError('ESMFold2 assets differ from the qualified release revision')
    required = {'model_config', 'ccd'} | {f'model_shard_{i:05}' for i in range(1, 7)}
    if set(resources) != required:
        raise ValueError('ESMFold2 needs config, CCD and all six bundled model/ESMC shards')
    if Path(resources['model_config']['path']).resolve() != root / 'config.json':
        raise ValueError('ESMFold2 config is outside the pinned checkpoint directory')
    shards = set(json.loads(index.read_text())['weight_map'].values())
    expected = {f'model-{i:05}-of-00006.safetensors' for i in range(1, 7)}
    if shards != expected:
        raise ValueError('Unexpected ESMFold2 shard set')
    for i in range(1, 7):
        if Path(resources[f'model_shard_{i:05}']['path']).resolve() != root / f'model-{i:05}-of-00006.safetensors':
            raise ValueError('ESMFold2 shard path differs from its Rust-verified resource')
    config = json.loads((root / 'config.json').read_text())
    if config.get('model_type') != 'esmfold2' or config.get('type') != 'release' or not config.get('esmc_config'):
        raise ValueError('Expected the standard ESMFold2 release with bundled ESMC')
    return root


def predict(request, channel):
    import torch
    ccd = Path(request['resources']['ccd']['path']).resolve()
    if Path(os.environ.get('ESMCFOLD_CCD_PATH', '')).resolve() != ccd:
        raise RuntimeError('The ESMFold2 CCD environment must match the verified resource')
    import esm
    from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    from .atlasfold import read_sequence, snapshot_input
    from .transport import save_arrays

    if request['source']['commit'] != SOURCE_COMMIT or Path(esm.__file__).resolve().parent.parent != Path(request['source']['root']).resolve():
        raise RuntimeError('Imported ESMFold2 does not match the pinned source')
    workload = request['workload']
    if workload['model_name'] != 'biohub/ESMFold2' or workload['precision'] != 'native':
        raise ValueError('Only biohub/ESMFold2 with native precision boundaries is integrated')
    root = model_directory(request)
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    retained = snapshot_input(request['input']['path'], output)
    _, sequence = read_sequence(retained)
    if any(aa not in 'ACDEFGHIKLMNPQRSTVWY' for aa in sequence):
        raise ValueError('ESMFold2 currently supports standard protein monomers only')
    torch.set_num_threads(8)
    torch.manual_seed(request['seed'])
    torch.use_deterministic_algorithms(workload['deterministic'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model = EsmFold2Model.from_pretrained(str(root), device='cuda', esmc_precision='bf16',
                                         local_files_only=True).eval()
    model.set_kernel_backend(None)
    model.set_chunk_size(workload['chunk_size'])
    if model.esmc is None:
        raise RuntimeError('The bundled ESMC language model is missing')
    from esm.models.esmfold2.conformers import CCD_PICKLE_PATH
    if CCD_PICKLE_PATH is None or CCD_PICKLE_PATH.resolve() != ccd:
        raise RuntimeError('Imported ESMFold2 CCD configuration differs from the resource pin')
    builder = ESMFold2InputBuilder(ccd_cache=ccd.parent)
    spi = StructurePredictionInput(sequences=[ProteinInput(id='A', sequence=sequence)])
    counts = dict(model=0, language_model=0, trunk=0, lm_encoder=0, diffusion=0)
    raw = {}
    model_seconds = 0.0
    hooks = []
    stages = []
    profiling = os.environ.get('FIREPROT_ESMFOLD2_PROFILE', '0') == '1'

    def count(name):
        def hook(module, args):
            counts[name] += 1
        return hook

    for name, module in [('language_model', model.esmc), ('trunk', model.folding_trunk),
                         ('lm_encoder', model.lm_encoder), ('diffusion', model.structure_head.diffusion_module)]:
        if module is not None:
            hooks.append(module.register_forward_pre_hook(count(name)))
            if profiling:
                stack = []
                def begin(module, args, _stack=stack):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    _stack.append(event)
                def end(module, args, result, _stack=stack, _name=name):
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    stages.append((_name, _stack.pop(), event))
                hooks += [module.register_forward_pre_hook(begin), module.register_forward_hook(end)]
    original = model.forward

    def forward(*args, **kwargs):
        nonlocal model_seconds
        if (kwargs.get('num_loops') != workload['loops'] or
                kwargs.get('num_sampling_steps') != workload['sampling_steps'] or
                kwargs.get('num_diffusion_samples') != workload['samples']):
            raise RuntimeError('ESMFold2 workload changed at the model boundary')
        counts['model'] += 1
        channel.emit(dict(event='progress', stage='model'))
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - begin
        for key in ('plddt_per_atom', 'pde', 'atom_pad_mask'):
            raw[key] = result[key].detach().cpu()
        return result

    model.forward = forward
    runtime = None
    if request['backend'] == 'fireprot':
        from fireprot.adapters.esmfold2.triangle import triangle_scope
        scope = triangle_scope(model)
    elif request['backend'] == 'native':
        scope = nullcontext()
    else:
        raise ValueError('Unsupported ESMFold2 backend')
    torch.cuda.reset_peak_memory_stats()
    try:
        with scope as runtime:
            result = builder.fold(model, spi, num_loops=workload['loops'],
                num_sampling_steps=workload['sampling_steps'], num_diffusion_samples=workload['samples'],
                seed=request['seed'], lm_dropout=0.3, complex_id='fireprot_prediction')
        torch.cuda.synchronize()
    finally:
        model.forward = original
        for hook in hooks:
            hook.remove()
    if counts['model'] != 1 or counts['language_model'] != 1 or counts['trunk'] != workload['loops'] + 1:
        raise RuntimeError(f'Unexpected ESMFold2 execution coverage: {counts}')
    if counts['lm_encoder'] != workload['loops'] + 1 or counts['diffusion'] == 0:
        raise RuntimeError(f'Incomplete ESMFold2 LM refinement or diffusion: {counts}')
    results = result if isinstance(result, list) else [result]
    if len(results) != workload['samples']:
        raise RuntimeError('ESMFold2 sample count changed')
    native = output / 'native'
    native.mkdir()
    predictions = []
    for i, sample in enumerate(results):
        arrays = dict(coordinates=sample.complex.atom_positions, plddt=sample.plddt,
            pae=sample.pae, ptm=sample.ptm, iptm=sample.iptm,
            pair_chains_iptm=sample.pair_chains_iptm,
            residue_index=sample.residue_index, entity_id=sample.entity_id,
            token_to_atoms=sample.complex.token_to_atoms,
            plddt_per_atom=raw['plddt_per_atom'][i], pde=raw['pde'][i],
            atom_mask=raw['atom_pad_mask'][0].bool())
        archive = f'sample-{i}.npz'
        descriptors = save_arrays(output / archive, arrays)
        structure = f'native/sample-{i}.cif'
        (output / structure).write_text(sample.complex.to_mmcif())
        predictions.append(dict(sample=i, archive=archive, structure=structure,
                                arrays=descriptors, confidence=None))
    stage_ms = {}
    for name, begin, end in stages:
        stage_ms.setdefault(name, []).append(begin.elapsed_time(end))
    manifest = dict(schema_version=1, source_commit=SOURCE_COMMIT, weight_revision=WEIGHT_REVISION,
        kernel_backend=None, fireprot=runtime.manifest() if runtime is not None else None,
        bundled_esmc=True, esmc_precision='bfloat16', other_precision='pinned native boundaries',
        requested_sampling_steps=workload['sampling_steps'], actual_denoiser_calls=counts['diffusion'],
        counts=counts, lm_dropout=0.3, chunk_size=workload['chunk_size'],
        stage_cuda_ms=stage_ms, profiling=profiling,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (output / 'operator-runtime.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return dict(schema_version=4, input_snapshot=retained.name, run_id=request['run_id'],
        seed=request['seed'], workload=workload, predictions=predictions,
        model_seconds=model_seconds, request_seconds=time.perf_counter() - started)
