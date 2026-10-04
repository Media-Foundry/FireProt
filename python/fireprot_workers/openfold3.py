"""OpenFold3/OpenBind-0 model invocation; no framework policy or monitoring."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time


def predict(request, channel):
    import torch
    import openfold3
    if request["source"]["commit"] != "7de748b7bc93adb5af0a4032d5e9f208b6e1325f":
        raise ValueError("OpenFold3 source version is outside the integrated compatibility pin")
    if Path(openfold3.__file__).resolve().parent.parent != Path(request["source"]["root"]):
        raise RuntimeError("Imported OpenFold3 does not match the verified source root")
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
    from openfold3.run_openfold import cli
    from .transport import save_arrays

    workload = request['workload']
    if workload['model_name'] != 'openbind-0' or workload['precision'] != 'float32':
        raise ValueError('Only OpenBind-0 with native FP32 policy is integrated')
    output = Path(os.environ['FIREPROT_OUTPUT_DIR'])
    config = {
        'experiment_settings': {'seeds': [request['seed']], 'use_msa_server': False, 'use_templates': False},
        'pl_trainer_args': {'devices': 1, 'num_nodes': 1, 'precision': '32-true'},
        'model_update': {'presets': ['predict'], 'custom': {
            'settings': {'memory': {'eval': {'use_triton_triangle_kernels': False,
                'use_deepspeed_evo_attention': False, 'use_cueq_triangle_kernels': False}}},
            'architecture': {'shared': {'num_recycles': workload['recycles'], 'diffusion': {
                'no_full_rollout_steps': workload['diffusion_steps'],
                'no_full_rollout_samples': workload['samples']}}}}}}
    config_path = output / 'runner.json'
    # JSON is a YAML subset accepted by the upstream runner-yaml loader.
    config_path.write_text(json.dumps(config, indent=2))
    original_forward, original_predict = OpenFold3.forward, OpenFold3AllAtom.predict_step
    predictions, calls, model_seconds = [], 0, 0.0

    def forward(model, *args, **kwargs):
        nonlocal calls, model_seconds
        shared = model.config.architecture.shared
        if shared.num_recycles != workload['recycles'] or shared.diffusion.no_full_rollout_steps != workload['diffusion_steps']:
            raise RuntimeError('Upstream model did not apply the requested inference workload')
        calls += 1
        channel.emit(dict(event='progress', stage='model'))
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_forward(model, *args, **kwargs)
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - started
        return result

    def predict_step(model, *args, **kwargs):
        result = original_predict(model, *args, **kwargs)
        if result is None:
            raise RuntimeError('Upstream returned no prediction')
        _, outputs = result
        tensors = {}
        def collect(prefix, obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    collect(prefix + '.' + str(key), value)
            elif isinstance(obj, torch.Tensor) or isinstance(obj, (float, int)):
                tensors[prefix] = obj
        collect('coordinates', outputs['atom_positions_predicted'])
        collect('confidence', outputs['confidence_scores'])
        coordinates = tensors['coordinates']
        if coordinates.ndim != 4 or tuple(coordinates.shape[:2]) != (1, workload['samples']):
            raise RuntimeError(f'Unexpected prediction batch/sample shape: {coordinates.shape}')
        for sample in range(workload['samples']):
            arrays = {}
            for name, value in tensors.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim < 2 or tuple(value.shape[:2]) != (1, workload['samples']):
                        raise RuntimeError(f'Unexpected batch/sample axes for {name}: {value.shape}')
                    value = value[0, sample]
                arrays[name] = value
            archive = f'sample-{sample}.npz'
            descriptors = save_arrays(output / archive, arrays)
            predictions.append(dict(sample=sample, archive=archive, arrays=descriptors, confidence=None))
        return result

    OpenFold3.forward, OpenFold3AllAtom.predict_step = forward, predict_step
    started = time.perf_counter()
    try:
        cli.main(args=['predict', '--query-json', request['input']['path'], '--use-msa-server=False',
            '--runner-yaml', str(config_path), '--inference-ckpt-path', request['checkpoints']['model']['path'],
            '--num-diffusion-samples', str(workload['samples']), '--output-dir', str(output / 'native')],
            standalone_mode=False)
        torch.cuda.synchronize()
    finally:
        OpenFold3.forward, OpenFold3AllAtom.predict_step = original_forward, original_predict
    if calls != 1 or len(predictions) != workload['samples']:
        raise RuntimeError('Expected exactly one query with the requested samples')
    return dict(schema_version=1, run_id=request['run_id'], seed=request['seed'], workload=workload,
                predictions=predictions, model_seconds=model_seconds,
                request_seconds=time.perf_counter() - started)
