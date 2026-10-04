"""AtlasFold public monomer API with the required AtlasLM-3B checkpoint."""
from __future__ import annotations
import dataclasses
import json
import os
from pathlib import Path
import time


def read_sequence(path):
    path = Path(path)
    if path.suffix == '.json':
        records = json.loads(path.read_text())
        if len(records) != 1 or len(records[0]['sequences']) != 1:
            raise ValueError('AtlasFold worker requires one monomer query')
        chain = records[0]['sequences'][0]['proteinChain']
        if chain.get('count', 1) != 1:
            raise ValueError('AtlasFold worker requires one chain')
        name, sequence = records[0]['name'], chain['sequence']
    else:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not lines or not lines[0].startswith('>') or sum(line.startswith('>') for line in lines) != 1:
            raise ValueError('AtlasFold worker requires one FASTA record')
        name, sequence = lines[0][1:], ''.join(lines[1:])
    if not name or not sequence or any(residue not in 'ACDEFGHIKLMNPQRSTVWYX' for residue in sequence):
        raise ValueError('Invalid AtlasFold monomer name or sequence')
    return name, sequence


def predict(request, channel):
    import numpy as np
    import torch
    import atlasfold
    from atlasfold.pretrained import load_model, get_runner
    from atlasfold.runner import get_sampling_config
    from .transport import save_arrays

    workload = request['workload']
    if request['source']['commit'] != '8ab3aca0e18c8b814d5ca6756b2617a07d72c68d':
        raise ValueError('AtlasFold source is outside the integrated compatibility pin')
    if Path(atlasfold.__file__).resolve().parents[2] != Path(request['source']['root']):
        raise RuntimeError('Imported AtlasFold does not match the verified source root')
    if workload['model_name'] != 'atlasfold-260703' or workload['precision'] != 'native':
        raise ValueError('Only AtlasFold 260703 with native mixed-precision boundaries is integrated')
    name, sequence = read_sequence(request['input']['path'])
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    torch.set_num_threads(8)
    torch.manual_seed(request['seed'])
    np.random.seed(request['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model = load_model('atlasfold', device='cuda', kernel='torch',
        model_path=Path(request['checkpoints']['model']['path']),
        lm_path=Path(request['checkpoints']['language_model']['path'])).eval()
    runner = get_runner(model)
    bucket = runner._get_length_bucket(len(sequence))
    sampling = dataclasses.replace(get_sampling_config(bucket), num_steps=workload['diffusion_steps'])
    original = model.inference
    calls, model_seconds, raw_dtypes = 0, 0.0, {}

    def inference(*args, **kwargs):
        nonlocal calls, model_seconds, raw_dtypes
        if kwargs['num_recycles'] != workload['recycles'] or kwargs['num_samples'] != workload['samples']:
            raise RuntimeError('AtlasFold workload changed at the model boundary')
        if kwargs['sampling_config'].num_steps != workload['diffusion_steps']:
            raise RuntimeError('AtlasFold diffusion step count changed')
        calls += 1
        channel.emit(dict(event='progress', stage='model'))
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - begin
        raw_dtypes = {key: str(value.dtype) for key, value in result.items()
                      if isinstance(value, torch.Tensor)}
        return result

    runtime = None
    if request['backend'] != 'native':
        from fireprot.adapters.atlasfold.runtime import AtlasFoldRuntime
        runtime = AtlasFoldRuntime(model, len(sequence), request['backend'].replace('_', '-'),
                                  allow_long1536=len(sequence) == 1536)
    model.inference = inference
    try:
        with torch.inference_mode():
            result = runner.fold(name, sequence, num_samples=workload['samples'], seeds=request['seed'],
                num_recycles=workload['recycles'], mlm_prob=0.15, sampling_config=sampling)
    finally:
        model.inference = original
        if runtime is not None:
            runtime.uninstall()
    if calls != 1:
        raise RuntimeError('Expected exactly one AtlasFold model call')
    predictions = []
    native = output / 'native'
    native.mkdir()
    for (seed, sample_index), sample in sorted(result.outputs.items()):
        if seed != request['seed']:
            raise RuntimeError('AtlasFold output seed changed')
        archive = f'sample-{sample_index}.npz'
        # Preserve the upstream public API (including its FP32 export and Python
        # float pTM). Internal model tensor dtypes are recorded separately below.
        arrays = dict(coordinates=sample.coordinates, plddt=sample.plddt, pae=sample.pae, ptm=sample.ptm)
        descriptors = save_arrays(output / archive, arrays)
        (native / f'sample-{sample_index}.cif').write_text(sample.to_mmcif())
        predictions.append(dict(sample=sample_index, archive=archive, arrays=descriptors, confidence=None,
                                structure=f'native/sample-{sample_index}.cif'))
    (output / 'atlasfold-model.json').write_text(json.dumps(dict(config=dataclasses.asdict(model.cfg),
        bucket=bucket, sampling=dataclasses.asdict(sampling), mlm_prob=0.15,
        trunk_passes=workload['recycles'] + 1, internal_tensor_dtypes=raw_dtypes), indent=2) + '\n')
    if runtime is not None:
        (output / 'operator-runtime.json').write_text(json.dumps(runtime.manifest(), indent=2, allow_nan=False) + '\n')
    return dict(schema_version=2, run_id=request['run_id'], seed=request['seed'], workload=workload,
        predictions=predictions, model_seconds=model_seconds, request_seconds=time.perf_counter()-started)
