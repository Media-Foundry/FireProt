"""Bitwise-exact ESM2 FP32 GEMMs for the qualified gfx1100 B4 bucket."""
from __future__ import annotations

import ctypes
import os
import pathlib
import threading
from typing import Any

from fireprot._paths import KERNEL_ROOT


_lock = threading.RLock()
_library: Any = None
_entry: Any = None
_error: str | None = None
_build: dict[str, Any] = {}


def load_extension() -> Any:
    """Build and load the source/compiler-fingerprinted gfx1100 C ABI once."""
    global _library, _entry, _error, _build
    with _lock:
        if _entry is not None or _error is not None:
            return _entry
        try:
            import torch
            if str(torch.__version__) != '2.13.0+rocm10.0.0' or torch.version.hip != '7.15.26333':
                raise RuntimeError('ESM2 gfx1100 GEMM requires its qualified Torch and HIP build')
            from ._published import select_c
            published = select_c(
                'fireprot_esm2_gemm', 'FIREPROT_ESM2_GEMM',
                'esm2-periodic-stagger-gemm-v1')
            if published is not None:
                _library, _build = published
            else:
                root = pathlib.Path(os.environ.get('FIREPROT_ROCM_HOME', '/opt/rocm'))
                from ._native_hip import load as build_native
                source = KERNEL_ROOT / 'fireprot_esm2_gemm.hip'
                _library, _build = build_native(
                    'fireprot_esm2_gemm', source, root, target='gfx1100',
                    compile_flags=['-ffp-contract=off'], extra_files=[__file__],
                    versions={'c_abi': 'esm2-periodic-stagger-gemm-v1'})
            _entry = _library.fireprot_esm2_gemm
            _entry.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
            _entry.restype = ctypes.c_int
        except Exception as error:
            _entry = None
            _error = f'{type(error).__name__}: {error}'
            return None
        return _entry


def extension_status() -> dict[str, Any]:
    with _lock:
        return dict(loaded=_entry is not None, error=_error, build=dict(_build))


def supported(x: Any, weight: Any, bias: Any) -> bool:
    """Fail closed outside the three measured contiguous ESM2 B4 projections."""
    import torch
    if bias is None or torch.is_grad_enabled():
        return False
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda
            and x.device == weight.device == bias.device
            and x.device.index == 0):
        return False
    if any(t.dtype != torch.float32 or not t.is_contiguous() or t.requires_grad
           or t.is_conj() or t.is_neg() for t in (x, weight, bias)):
        return False
    projection_shapes = {
        (2560, 2560): (1002, 4, 2560),
        (10240, 2560): (1002, 4, 2560),
        (2560, 10240): (1002, 4, 10240),
    }
    weight_shape = tuple(weight.shape)
    if tuple(x.shape) != projection_shapes.get(weight_shape):
        return False
    output_features = weight_shape[0]
    if tuple(bias.shape) != (output_features,):
        return False
    return torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0] == 'gfx1100'


def linear(x: Any, weight: Any, bias: Any, native: Any) -> tuple[Any, bool]:
    """Run the exact periodic-stagger kernel, or preserve the native operation."""
    if not supported(x, weight, bias):
        return native(x, weight, bias), False
    import torch
    entry = load_extension()
    if entry is None:
        raise RuntimeError(f'ESM2 gfx1100 GEMM extension unavailable: {_error}')
    output_features, input_features = weight.shape
    output = torch.empty((*x.shape[:-1], output_features), device=x.device, dtype=x.dtype)
    code = entry(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), output.data_ptr(),
                 4008, output_features, input_features,
                 torch.cuda.current_stream(x.device).cuda_stream)
    if code:
        raise RuntimeError(f'ESM2 gfx1100 GEMM failed: {code}')
    return output, True
