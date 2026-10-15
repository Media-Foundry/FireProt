"""Experimental PairFormer-only autocast with explicit FP32 rollout boundaries.

This scope is tensor glue for a frozen diagnostic worker. The environment's
precision override must be declared separately from its base FP32 request;
finite outputs or an accepted archive do not establish mixed-precision parity.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


@contextmanager
def pairformer_precision_scope(model, output, precision, diagnostic=True,
                               triangle_precision='inherit'):
    from openfold3.core.model.layers.triangular_multiplicative_update import TriangleMultiplicativeUpdate

    dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
             'float16': torch.float16}[precision]
    if triangle_precision not in ('inherit', 'float32'):
        raise ValueError('Unsupported triangle precision policy')
    pins = {}
    for cls, expected in (
        (type(model), '1cd02947e24bcea5d30dba98122e6a38e707f88ebe1d4c4f2aa16f4b177f7e0c'),
        (type(model.pairformer_stack), 'cdb66c9553ef42b93835dec92f3e6dcf711dbf139a5a59aa1f2561216435af9f'),
        (TriangleMultiplicativeUpdate, '612caf7abd07a86a7fdf4eda7e0a2170c04b407f9121fc286ab864a648db1ba7'),
    ):
        path = Path(inspect.getfile(cls))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f'Unqualified precision boundary: {path}')
        pins[str(path)] = digest
    if model.training or not torch.is_inference_mode_enabled() or torch.is_grad_enabled():
        raise RuntimeError('Precision experiment requires inference mode')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('Master parameters must remain FP32')
    stack = model.pairformer_stack
    native_triangle = TriangleMultiplicativeUpdate._inference_forward
    # Enter before the experimental gate scope, whose raw ABI is FP32-only.
    if inspect.getfile(native_triangle) != inspect.getfile(TriangleMultiplicativeUpdate):
        raise RuntimeError('Enter precision scope before triangle replacements')
    had_local = 'forward' in stack.__dict__
    old_local = stack.__dict__.get('forward')
    original = stack.forward
    report = dict(completed=False, diagnostic=diagnostic, source_pins=pins,
                  base_request_precision='float32',
                  effective_precision=dict(pairformer_autocast=precision,
                                           triangle_multiplicative_update=triangle_precision,
                                           input_and_recycle_embeddings='float32',
                                           template_and_msa='float32',
                                           diffusion='float32', confidence='float32',
                                           master_parameters='float32'),
                  cycles=[], ranges={}, output_dtypes={}, boundary_calls={},
                  matmul_flags={name: getattr(torch.backends.cuda.matmul, name, None)
                      for name in ('allow_tf32', 'allow_fp16_reduced_precision_reduction',
                                   'allow_bf16_reduced_precision_reduction',
                                   'allow_fp16_accumulation')},
                  nonfinite=None, triangle_fp32_fusion_bypassed_cycles=0,
                  triangle_fp32_calls=0,
                  limits=['Diagnostic activation scans invalidate performance timing.',
                          'Mixed-precision outputs require independent aligned-coordinate and confidence evaluation.',
                          'No output clamping, changed recycle count or shortened diffusion.'])
    active = False
    handles = []
    triangle_modules = {id(m): name for name, m in stack.named_modules()
                        if isinstance(m, TriangleMultiplicativeUpdate)}

    def protected_triangle(module, *args, **kwargs):
        if not active or id(module) not in triangle_modules:
            raise RuntimeError('FP32 triangle protection escaped PairFormer')
        if any(t.is_floating_point() and t.dtype != torch.float32
               for t in tensors((args, kwargs))):
            raise RuntimeError('Protected triangle requires FP32 input/residual storage')
        with torch.amp.autocast('cuda', enabled=False):
            result = native_triangle(module, *args, **kwargs)
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in tensors(result)):
            raise RuntimeError('Protected triangle returned a non-FP32 tensor')
        report['triangle_fp32_calls'] += 1
        calls = report['cycles'][-1].setdefault('triangle_fp32_calls', {})
        name = triangle_modules[id(module)]
        calls[name] = calls.get(name, 0) + 1
        return result

    def observe(label, value, scan):
        for i, tensor in enumerate(tensors(value)):
            if not tensor.is_floating_point():
                continue
            key = f'{label}:{i}:{tensor.dtype}'
            report['output_dtypes'][str(tensor.dtype)] = report['output_dtypes'].get(str(tensor.dtype), 0) + 1
            if scan and tensor.numel():
                low, high = torch.aminmax(tensor.detach())
                low, high = float(low), float(high)
                if not math.isfinite(low) or not math.isfinite(high):
                    report['nonfinite'] = dict(cycle=len(report['cycles']),
                        module=label, dtype=str(tensor.dtype), shape=list(tensor.shape),
                        minimum=str(low), maximum=str(high))
                    raise FloatingPointError(f'Nonfinite activation: {label}, {tensor.dtype}')
                item = report['ranges'].setdefault(key, dict(calls=0, minimum=low,
                    maximum=high, max_abs=0.0, shape=list(tensor.shape)))
                item['calls'] += 1
                item['minimum'] = min(item['minimum'], low)
                item['maximum'] = max(item['maximum'], high)
                item['max_abs'] = max(item['max_abs'], abs(low), abs(high))
                if active:
                    cycle_range = report['cycles'][-1].setdefault('activation_ranges_by_dtype', {})
                    value = cycle_range.setdefault(str(tensor.dtype),
                        dict(calls=0, minimum=low, maximum=high, max_abs=0.0))
                    value['calls'] += 1
                    value['minimum'] = min(value['minimum'], low)
                    value['maximum'] = max(value['maximum'], high)
                    value['max_abs'] = max(value['max_abs'], abs(low), abs(high))

    def activation_hook(name):
        def hook(module, args, result):
            if active:
                observe(name, result, diagnostic)
        return hook

    def fp32_guard(name):
        def hook(module, args, kwargs):
            if active or (torch.is_autocast_enabled('cuda')
                          and torch.get_autocast_dtype('cuda') != torch.float32):
                raise RuntimeError(f'Lower precision escaped PairFormer into {name}')
            for tensor in tensors((args, kwargs)):
                if tensor.is_cuda and tensor.is_floating_point() and tensor.dtype != torch.float32:
                    raise RuntimeError(f'{name} received {tensor.dtype}')
            report['boundary_calls'][name] = report['boundary_calls'].get(name, 0) + 1
        return hook

    def diffusion_output(module, args, result):
        for tensor in tensors(result):
            if tensor.is_cuda and tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise RuntimeError('Diffusion produced a non-FP32 activation')

    def forward(*args, **kwargs):
        nonlocal active
        if active:
            raise RuntimeError('Nested PairFormer precision scope')
        if torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') != torch.float32:
            raise RuntimeError('Unexpected outer mixed-precision context')
        observe('pairformer.input', (args, kwargs), True)
        record = dict(index=len(report['cycles']), completed=False,
                      autocast=precision, returned_dtypes=[])
        report['cycles'].append(record)
        installed_triangle = TriangleMultiplicativeUpdate._inference_forward
        if dtype != torch.float32:
            TriangleMultiplicativeUpdate._inference_forward = (
                protected_triangle if triangle_precision == 'float32' else native_triangle)
            report['triangle_fp32_fusion_bypassed_cycles'] += 1
        active = True
        try:
            with torch.amp.autocast('cuda', dtype=dtype):
                result = original(*args, **kwargs)
            observe('pairformer.output', result, True)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError('Unexpected PairFormer return contract')
            result = tuple(t.float() for t in result)
            record['returned_dtypes'] = [str(t.dtype) for t in result]
            record['completed'] = True
            return result
        finally:
            active = False
            TriangleMultiplicativeUpdate._inference_forward = installed_triangle

    try:
        stack.forward = forward
        for name, module in stack.named_modules():
            if type(module).__name__ in ('Linear', 'LayerNorm', 'Attention',
                                        'TriangleMultiplicativeUpdate', 'PairFormerBlock'):
                handles.append(module.register_forward_hook(activation_hook(name)))
        for name, module in (('sample_diffusion', model.sample_diffusion),
                             ('diffusion_module', model.diffusion_module),
                             ('confidence', model.aux_heads)):
            handles.append(module.register_forward_pre_hook(fp32_guard(name), with_kwargs=True))
        for module in model.diffusion_module.modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(diffusion_output))
        yield report
        if len(report['cycles']) != model.shared.num_recycles + 1:
            raise RuntimeError('PairFormer recycle count changed')
        if any(not r['completed'] for r in report['cycles']):
            raise RuntimeError('Incomplete PairFormer cycle')
        if triangle_precision == 'float32' and dtype != torch.float32:
            # Upstream tunes chunk size by executing block zero before the
            # real stack. Require every module in every cycle, retaining all
            # additional tuning calls instead of assuming a fixed total.
            names = set(triangle_modules.values())
            for cycle in report['cycles']:
                calls = cycle.get('triangle_fp32_calls', {})
                if not names or set(calls) != names or any(v < 1 for v in calls.values()):
                    raise RuntimeError('Incomplete FP32 triangle coverage')
        if report['boundary_calls'].get('diffusion_module') != model.shared.diffusion.no_full_rollout_steps:
            raise RuntimeError('Diffusion step coverage changed')
        if report['boundary_calls'].get('sample_diffusion') != 1 or report['boundary_calls'].get('confidence') != 1:
            raise RuntimeError('Missing FP32 rollout/confidence boundary')
        if not report['output_dtypes'].get(str(dtype)):
            raise RuntimeError('Requested PairFormer dtype was never observed')
        report['completed'] = True
    finally:
        for handle in reversed(handles):
            handle.remove()
        if had_local:
            stack.forward = old_local
        else:
            delattr(stack, 'forward')
        Path(output, 'pairformer-precision.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
