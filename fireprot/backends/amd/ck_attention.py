"""Validated exact-layout FP32 CK Tile triangle attention on gfx90a."""
from __future__ import annotations

import ctypes
import os
import pathlib
import threading
from typing import Any

from fireprot.backends.amd import triangle_attention as ta
from fireprot.ops.registry import OperationSignature, registry

IMPLEMENTATION = "amd-gfx90a-ck-tile-attention-f32-n1000"
LONG1536_IMPLEMENTATION = "amd-gfx90a-ck-tile-attention-f32-n1536"
STRIDES = (128000, 32, 128, 1)
_enabled = (
    os.environ.get("FIREPROT_ENABLE_CK_ATTENTION", "1") == "1"
    and os.environ.get("FIREPROT_DISABLE_CK_ATTENTION") != "1"
)
_exact1536_enabled = os.environ.get(
    "FIREPROT_ENABLE_CK_EXACT1536", "1" if ta._long1536_enabled else "0"
) == "1"
_long1536_enabled = _exact1536_enabled or os.environ.get("FIREPROT_ENABLE_CK_ATTENTION_1536") == "1"
_vector1000_enabled = os.environ.get("FIREPROT_ENABLE_CK_VECTOR_BIAS1000", "1") == "1"
_lock = threading.RLock()
_library: Any = None
_entry: Any = None
_entry_exact: Any = None
_entry_vector: Any = None
_error: str | None = None
_build: dict[str, Any] = {}


def load_extension() -> Any:
    """Build a source/header/compiler fingerprinted standalone HIP library once."""
    global _library, _entry, _entry_exact, _entry_vector, _error, _build
    with _lock:
        if _entry is not None or _error is not None:
            return _entry
        if os.environ.get("FIREPROT_DISABLE_CK_EXTENSION") == "1":
            _error = "disabled by FIREPROT_DISABLE_CK_EXTENSION"
            return None
        try:
            import torch

            if not torch.version.hip:
                raise RuntimeError("PyTorch HIP is required")
            root = pathlib.Path(os.environ.get("FIREPROT_ROCM_HOME", "/opt/rocm"))
            version = (root / "include/ck/version.h").read_text()
            if "8d1ae90eff7d022f26019ec55b2ec6a7674b3112" not in version:
                raise RuntimeError("CK installation is outside the measured version")
            from ._native_hip import load as build_native
            source = pathlib.Path(__file__).parent / "csrc/fireprot_ck_attention.cpp"
            if not (root / "include/ck_tile").is_dir():
                raise RuntimeError("CK Tile headers are unavailable")
            _library, _build = build_native('fireprot_ck_attention', source, root,
                extra_files=[__file__], versions={'ck_version': version})
            _entry = _library.ck_attention
            _entry.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
            _entry.restype = ctypes.c_int
            _entry_exact = _library.ck_attention_exact1536
            _entry_exact.argtypes = _entry.argtypes
            _entry_exact.restype = ctypes.c_int
            _entry_vector = _library.ck_attention_vector_bias1000
            _entry_vector.argtypes = _entry.argtypes
            _entry_vector.restype = ctypes.c_int
        except Exception as error:
            _entry = _entry_exact = _entry_vector = None
            _error = f"{type(error).__name__}: {error}"
            return None
        return _entry


def extension_status() -> dict[str, Any]:
    with _lock:
        return dict(enabled=_enabled, long1536_enabled=_long1536_enabled, exact1536_enabled=_exact1536_enabled,
                    vector1000_enabled=_vector1000_enabled,
                    loaded=_entry is not None, error=_error, build=dict(_build))


def _supports(signature: OperationSignature, *, shape=ta.LONG_QKV_SHAPE) -> bool | tuple[bool, str]:
    if not _enabled:
        return False, "CK Tile attention disabled by environment"
    if signature.shape != shape:
        return False, "CK Tile attention requires an exact measured bucket"
    n = shape[2]
    attributes = dict(signature.attributes)
    if attributes.get("bias_shapes") != repr(((1, 4, n, n),)):
        return False, "CK Tile attention requires implicit pair mask"
    strides = (n * 128, 32, 128, 1)
    if attributes.get("qkv_strides") != repr((strides,) * 3) or attributes.get("ck_tile_compatible") != "True":
        return False, "CK Tile attention requires exact QKV shape, strides, dtype, device and alignment"
    supported = (ta._supports_long1536_direct(signature) if shape == ta.LONG1536_QKV_SHAPE
                 else ta._supports_long_direct(signature))
    if supported is not True:
        return supported
    if load_extension() is None:
        return False, "CK Tile attention extension unavailable"
    return True


def _supports_long1536(signature: OperationSignature) -> bool | tuple[bool, str]:
    if not _long1536_enabled:
        return False, "CK N1536 requires the exact path or an explicit legacy opt-in"
    return _supports(signature, shape=ta.LONG1536_QKV_SHAPE)


def attention(*, module: Any, q: Any, k: Any, v: Any, biases: Any, native: Any) -> Any:
    import torch

    entry = load_extension()
    if entry is None:
        return ta._direct_sdpa(module=module, q=q, k=k, v=v, biases=biases, native=native)
    if q.shape[2] == 1536 and _exact1536_enabled:
        entry = _entry_exact
    elif q.shape[2] == 1000 and _vector1000_enabled:
        entry = _entry_vector
    with torch.cuda.device(q.device):
        if q.shape[2] == 1536:
            # Break the measured N1536 bias row-stride performance cliff.
            # The C ABI uses the same physical row stride, 1536 + 16.
            bias = torch.empty((1, 4, 1536, 1552), device=q.device, dtype=q.dtype)
            bias[..., :1536].copy_(biases[0])
        else:
            bias = biases[0].contiguous()
            # A contiguous slice may still have a misaligned storage offset.
            if _vector1000_enabled and bias.data_ptr() % 16:
                bias = bias.clone()
        output = torch.empty_like(q)
        status = entry(q.data_ptr(), k.data_ptr(), v.data_ptr(), bias.data_ptr(),
                       output.data_ptr(), q.shape[0], q.shape[2], torch.cuda.current_stream(q.device).cuda_stream)
        if status:
            raise RuntimeError(f"CK Tile attention launch failed with HIP status {status}")
    return output.transpose(-2, -3)


registry.register(operation=ta.OPERATION, name=IMPLEMENTATION, function=attention,
                  supports=_supports, priority=30, version="1")
registry.register(operation=ta.OPERATION, name=LONG1536_IMPLEMENTATION, function=attention,
                  supports=_supports_long1536, priority=30, version="exact-shape-1")
