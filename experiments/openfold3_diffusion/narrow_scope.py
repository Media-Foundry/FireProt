"""Opt-in, source-pinned128-to16 global diffusion projection experiment."""
from contextlib import contextmanager
import ctypes
import hashlib
import inspect
import json
from pathlib import Path

import torch

if __package__:
    from .bias_cache import validate_sources
else:
    from bias_cache import validate_sources


@contextmanager
def narrow_projection_scope(sample, output, receipt_path, verify_first=True):
    transformer = validate_sources(sample)
    receipt = json.loads(Path(receipt_path).read_text())
    library_path = Path(receipt['library'])
    if hashlib.sha256(library_path.read_bytes()).hexdigest() != receipt['library_sha256']:
        raise RuntimeError('Projection library digest mismatch')
    library = ctypes.CDLL(str(library_path))
    entry = library.fireprot_project_128_16
    entry.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    entry.restype = ctypes.c_int
    report = dict(completed=False, calls=0, fallbacks=0, native_checks=0,
                  verify_first=verify_first, library=str(library_path),
                  library_sha256=receipt['library_sha256'], calls_by_block={},
                  first_call=None,
                  scope='Only24 global diffusion linear_z projections; strict FP32 N1536, head-major output, no caching.')
    saved = []

    def wrap(index, module, original):
        checked = False

        def forward(value, *args, **kwargs):
            nonlocal checked
            weight = module.weight
            autocast_enabled = torch.is_autocast_enabled('cuda')
            autocast_dtype = torch.get_autocast_dtype('cuda')
            if report['first_call'] is None and type(value) is torch.Tensor:
                report['first_call'] = dict(
                    shape=list(value.shape), strides=list(value.stride()),
                    dtype=str(value.dtype), weight_dtype=str(weight.dtype),
                    weight_shape=list(weight.shape), weight_strides=list(weight.stride()),
                    device=str(value.device), precision=str(getattr(module,'precision',None)),
                    has_bias=module.bias is not None, training=module.training,
                    requires_grad=value.requires_grad,
                    inference_mode=torch.is_inference_mode_enabled(),
                    grad_enabled=torch.is_grad_enabled(),
                    autocast_enabled=autocast_enabled, autocast_dtype=str(autocast_dtype))
            eligible = (
                not args and not kwargs and type(value) is torch.Tensor
                and tuple(value.shape) == (1,1,1536,1536,128)
                and value.dtype == weight.dtype == torch.float32
                and value.is_cuda and value.device == weight.device
                and value.is_contiguous() and weight.is_contiguous()
                and tuple(weight.shape) == (16,128) and module.bias is None
                and getattr(module,'precision',None) in (None,torch.float32)
                and not value.requires_grad and not value.is_conj() and not value.is_neg()
                and not weight.is_conj() and not weight.is_neg()
                and not module.training and torch.is_inference_mode_enabled()
                and not torch.is_grad_enabled()
                # The pinned diffusion runner enables autocast with an FP32 target.
                # It preserves this FP32 linear's arithmetic; BF16/FP16 still fall back.
                and (not autocast_enabled or autocast_dtype == torch.float32)
                and torch.are_deterministic_algorithms_enabled()
                and not torch.is_deterministic_algorithms_warn_only_enabled())
            if not eligible:
                report['fallbacks'] += 1
                return original(value,*args,**kwargs)
            with torch.cuda.device(value.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Experimental projection excludes graph capture')
                packed = torch.empty((16,1536,1536),device=value.device,dtype=value.dtype)
                rc = entry(value.data_ptr(),weight.data_ptr(),packed.data_ptr(),1536*1536,1,
                           torch.cuda.current_stream().cuda_stream)
                if rc:
                    raise RuntimeError(f'Projection launch failed: {rc}')
                result = packed.permute(1,2,0).unsqueeze(0).unsqueeze(0)
                report['calls'] += 1
                report['calls_by_block'][str(index)] = report['calls_by_block'].get(str(index),0)+1
                if verify_first and not checked:
                    native = original(value)
                    if not torch.equal(result.contiguous().view(torch.int32),native.contiguous().view(torch.int32)):
                        raise RuntimeError(f'Trained projection differs in block{index}')
                    checked = True
                    report['native_checks'] += 1
                return result
        return forward

    try:
        for index, block in enumerate(transformer.blocks):
            module = block.attention_pair_bias.linear_z
            source = Path(inspect.getfile(type(module)))
            if hashlib.sha256(source.read_bytes()).hexdigest() != 'dc2da47e0ab77f9f4c73c8528e2d6d5069d267117f0327c2c0589411b97d3050':
                raise RuntimeError('Unqualified Linear source')
            saved.append((module,'forward' in module.__dict__,module.__dict__.get('forward')))
            module.forward = wrap(index,module,module.forward)
        yield report
        report['completed'] = True
    finally:
        for module, existed, previous in reversed(saved):
            if existed:
                module.forward = previous
            else:
                delattr(module,'forward')
        Path(output,'narrow-projection.json').write_text(json.dumps(report,indent=2)+'\n')
