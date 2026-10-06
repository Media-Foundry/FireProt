"""Explicit opt-in Base N1000 execution context; Rust remains the inference entry."""
from contextlib import ExitStack, contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path


FLAG = 'FIREPROT_ENABLE_BASE_N1000_OPTIMIZATIONS'


def validate_contract(request, data):
    expected = dict(model='protenix', model_name='protenix_base_default_v0.5.0',
                    precision='float32', cycles=10, diffusion_steps=200, samples=1)
    workload = request.get('workload', {})
    if request.get('backend') != 'fireprot' or any(workload.get(k) != v for k, v in expected.items()):
        raise ValueError('Base optimizations require FireProt Base FP32 C10/D200/S1')
    features = data.get('input_feature_dict', {})
    tokens = tuple(getattr(features.get('token_index'), 'shape', ()))
    atoms = tuple(getattr(features.get('atom_to_token_idx'), 'shape', ()))
    if tokens != (1000,) or len(atoms) != 1 or atoms[0] <= 0:
        raise ValueError('Base optimizations require 1000 tokens and a nonempty atom mapping')
    return int(atoms[0])


def _supports(signature):
    return (signature.device_type == 'cuda' and signature.architecture == 'gfx90a'
            and signature.dtype == 'float32' and signature.mode == 'inference'
            and signature.layout == 'model_features' and signature.mask == 'model_defined'
            and signature.inplace and len(signature.shape) == 2
            and signature.shape[0] == 1000 and signature.shape[1] > 0
            and dict(signature.attributes) == {'cycles': '10', 'steps': '200', 'samples': '1'})


@contextmanager
def _components(model, output, atom_count):
    from . import triangle_lt, dual_storage, global_bias, local_bias, transition
    with ExitStack() as stack:
        stack.enter_context(triangle_lt.install(model, output))
        stack.enter_context(dual_storage.install(model, output))
        stack.enter_context(global_bias.install(model, output))
        stack.enter_context(local_bias.install(model, output))
        stack.enter_context(transition.install(model, output, atom_count))
        yield


@lru_cache(maxsize=1)
def _registered_registry():
    from fireprot.ops.registry import registry
    registry.register(operation='protenix_base_n1000_context', name='amd-gfx90a-base-n1000-cached-v1',
                      function=_components, supports=_supports, priority=100, version='1')
    return registry


def _select(model, atom_count):
    import torch
    from fireprot.ops.registry import OperationSignature
    registry = _registered_registry()
    parameter = next(model.parameters())
    architecture = (torch.cuda.get_device_properties(parameter.device).gcnArchName.split(':')[0]
                    if parameter.is_cuda else 'unknown')
    signature = OperationSignature.create(operation='protenix_base_n1000_context',
        device_type=parameter.device.type, architecture=architecture,
        dtype=str(parameter.dtype).removeprefix('torch.'), shape=(1000, atom_count),
        layout='model_features', mode='training' if model.training else 'inference',
        mask='model_defined', inplace=True, attributes={'cycles': 10, 'steps': 200, 'samples': 1})
    selection = registry.resolve(signature)
    if selection.implementation is None:
        raise RuntimeError('No qualified Base context: ' + str(selection.rejected))
    return selection.implementation


@contextmanager
def install(model, output, request, data):
    mode = os.environ.get(FLAG, '0')
    if mode not in ('0', '1'):
        raise ValueError(FLAG + ' must be 0 or 1')
    if mode == '0':
        yield
        return
    atom_count = validate_contract(request, data)
    implementation = _select(model, atom_count)
    record = dict(schema_version=1, contract='base-n1000-fp32-c10-d200-s1-v1',
                  token_count=1000, atom_count=atom_count,
                  implementation=implementation.name,
                  components=['scoped_lt', 'dual_storage', 'global_bias_cache',
                              'local_bias_cache', 'transition_fusion'], error=None)
    try:
        with implementation.function(model, output, atom_count):
            yield
    except BaseException as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        Path(output, 'base-n1000-optimizations.json').write_text(json.dumps(record, indent=2)+'\n')
