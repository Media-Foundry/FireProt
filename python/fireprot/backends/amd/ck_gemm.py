"""Capability-gated direct CK GEMM paths observed in Protenix."""

from __future__ import annotations

import threading
import weakref
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from fireprot.backends.amd.ck_extension import extension_status, load_ck_extension
from fireprot.ops.registry import OperationSignature, registry


OPERATION = "linear"
IMPLEMENTATION = "amd-ck-gemm-f32-m124-k64-n32"
ROWS = 124
IN_FEATURES = 64
OUT_FEATURES = 32
_fast_path_counts: Counter[tuple[str, str]] = Counter()
_fast_path_lock = threading.RLock()
_packed_lock = threading.RLock()
_packed_weights: weakref.WeakKeyDictionary[Any, tuple[tuple[Any, ...], Any]] = (
    weakref.WeakKeyDictionary()
)


@lru_cache(maxsize=None)
def _device_architecture(device_index: int) -> str:
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]


def _architecture(tensor: Any) -> str:
    try:
        return _device_architecture(int(tensor.device.index or 0))
    except Exception:
        return "unknown"


def _row_count(shape: tuple[int, ...]) -> int:
    rows = 1
    for value in shape[:-1]:
        rows *= value
    return rows


def _supports_linear(signature: OperationSignature) -> bool | tuple[bool, str]:
    attributes = dict(signature.attributes)
    if signature.device_type != "cuda":
        return False, "device is not CUDA/HIP"
    if signature.architecture != "gfx90a":
        return False, "architecture is not gfx90a"
    if signature.dtype != "float32":
        return False, "dtype is not float32"
    if signature.mode != "inference":
        return False, "training or autograd is enabled"
    if signature.layout != "contiguous":
        return False, "input is not contiguous"
    if signature.shape[-1:] != (IN_FEATURES,) or _row_count(signature.shape) != ROWS:
        return False, "matrix shape is not the validated 124x64 bucket"
    if attributes.get("out_features") != str(OUT_FEATURES):
        return False, "output width is not 32"
    if attributes.get("bias") != "False":
        return False, "only bias-free linear is supported"
    if attributes.get("weight_contiguous") != "True":
        return False, "weight is not contiguous"
    if load_ck_extension() is None:
        return False, "direct CK extension is unavailable"
    return True


def _weight_key(weight: Any) -> tuple[Any, ...]:
    return (
        weight.data_ptr(),
        getattr(weight, "_version", None),
        str(weight.device),
        str(weight.dtype),
        tuple(weight.shape),
    )


def _packed_weight(module: Any) -> Any:
    weight = module.weight
    key = _weight_key(weight)
    with _packed_lock:
        cached = _packed_weights.get(module)
        if cached is not None and cached[0] == key:
            return cached[1]
        packed = weight.transpose(0, 1).contiguous()
        _packed_weights[module] = (key, packed)
        return packed


def _ck_linear(
    *,
    module: Any,
    x: Any,
    native: Callable[[], Any],
    extension: Any | None = None,
) -> Any:
    extension = extension or load_ck_extension()
    if extension is None:
        return native()
    original_shape = x.shape
    matrix = x.reshape(ROWS, IN_FEATURES)
    output = extension.mm_f32(matrix, _packed_weight(module))
    return output.reshape(*original_shape[:-1], OUT_FEATURES)


def _ck_outer_product_mean(
    *,
    module: Any,
    m: Any,
    mask: Any,
    chunk_size: int | None,
    inplace_safe: bool,
    native: Callable[[], Any],
    extension: Any | None = None,
) -> Any:
    """Preserve Protenix OPM semantics while replacing its two projections."""
    import torch

    extension = extension or load_ck_extension()
    if extension is None:
        return native()
    if mask is None:
        mask = m.new_ones(m.shape[:-1])

    normalized = module.layer_norm(m)
    expanded_mask = mask.unsqueeze(-1)
    left = _ck_linear(
        module=module.linear_1,
        x=normalized,
        native=lambda: module.linear_1(normalized),
        extension=extension,
    )
    left = left * expanded_mask
    right = _ck_linear(
        module=module.linear_2,
        x=normalized,
        native=lambda: module.linear_2(normalized),
        extension=extension,
    )
    right = right * expanded_mask
    del normalized

    left = left.transpose(-2, -3)
    right = right.transpose(-2, -3)
    if chunk_size is not None:
        outer = module._chunk(left, right, chunk_size)
    else:
        outer = module._opm(left, right)

    norm = torch.einsum("...abc,...adc->...bdc", expanded_mask, expanded_mask)
    norm = norm + module.eps
    if inplace_safe:
        outer /= norm
        return outer
    return outer / norm


registry.register(
    operation=OPERATION,
    name=IMPLEMENTATION,
    function=_ck_linear,
    supports=_supports_linear,
    priority=30,
    version="1",
)


def dispatch_ck_outer_product_mean(
    *,
    module: Any,
    m: Any,
    mask: Any,
    chunk_size: int | None,
    inplace_safe: bool,
    native: Callable[[], Any],
) -> Any:
    import torch

    supported = (
        m.device.type == "cuda"
        and _architecture(m) == "gfx90a"
        and m.dtype is torch.float32
        and not module.training
        and not torch.is_grad_enabled()
        and not m.requires_grad
        and m.is_contiguous()
        and m.shape[-1:] == (IN_FEATURES,)
        and _row_count(tuple(m.shape)) == ROWS
        and module.c_m == IN_FEATURES
        and module.c_hidden == OUT_FEATURES
        and module.linear_1.bias is None
        and module.linear_2.bias is None
        and module.linear_1.weight.is_contiguous()
        and module.linear_2.weight.is_contiguous()
    )
    if supported:
        extension = load_ck_extension()
        if extension is not None:
            with _fast_path_lock:
                _fast_path_counts[(OPERATION, IMPLEMENTATION)] += 2
            return _ck_outer_product_mean(
                module=module,
                m=m,
                mask=mask,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
                native=native,
                extension=extension,
            )
    with _fast_path_lock:
        _fast_path_counts[(OPERATION, "native")] += 2
    return native()


def fast_path_manifest() -> dict[str, Any]:
    with _fast_path_lock, _packed_lock:
        return {
            "selection_counts": [
                {
                    "operation": operation,
                    "implementation": implementation,
                    "count": count,
                }
                for (operation, implementation), count in sorted(
                    _fast_path_counts.items()
                )
            ],
            "extension": extension_status(),
            "packed_weight_count": len(_packed_weights),
        }


def _reset_for_testing() -> None:
    with _fast_path_lock, _packed_lock:
        _fast_path_counts.clear()
        _packed_weights.clear()
