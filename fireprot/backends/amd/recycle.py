"""Capability-gated CK-style HIP operations for the Protenix recycle trunk."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from fireprot.backends.amd.extension import load_amd_extension
from fireprot.ops.registry import OperationSignature, registry


PAIR_ROWS = 124 * 124
PAIR_CHANNELS = 128
LAYER_NORM_OPERATION = "layer_norm"
TRANSITION_OPERATION = "gated_transition"
LAYER_NORM_IMPLEMENTATION = "amd-gfx90a-hip-layernorm-f32-c128"
TRANSITION_IMPLEMENTATION = "amd-gfx90a-hip-transition-f32-c128"
_fast_path_counts: Counter[tuple[str, str]] = Counter()


def _architecture(tensor: Any) -> str:
    try:
        return _device_architecture(int(tensor.device.index or 0))
    except Exception:
        return "unknown"


@lru_cache(maxsize=None)
def _device_architecture(device_index: int) -> str:
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]


def _base_support(signature: OperationSignature) -> bool | tuple[bool, str]:
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
    if signature.shape[-1:] != (PAIR_CHANNELS,):
        return False, "last dimension is not 128"
    if not signature.shape or _row_count(signature.shape) != PAIR_ROWS:
        return False, "row count is not the validated 124x124 bucket"
    if load_amd_extension() is None:
        return False, "FireProt AMD extension is unavailable"
    return True


def _row_count(shape: tuple[int, ...]) -> int:
    rows = 1
    for value in shape[:-1]:
        rows *= value
    return rows


def _supports_layer_norm(
    signature: OperationSignature,
) -> bool | tuple[bool, str]:
    supported = _base_support(signature)
    if supported is not True:
        return supported
    attributes = dict(signature.attributes)
    if attributes.get("offset") == "True" and attributes.get("scale") != "True":
        return False, "offset-only LayerNorm is not implemented"
    if attributes.get("epsilon") != "1e-05":
        return False, "epsilon is not the validated 1e-5"
    return True


def _supports_transition(
    signature: OperationSignature,
) -> bool | tuple[bool, str]:
    supported = _base_support(signature)
    if supported is not True:
        return supported
    attributes = dict(signature.attributes)
    if attributes.get("expansion") != "4":
        return False, "transition expansion is not four"
    return True


def _hip_layer_norm(
    *, module: Any, x: Any, native: Callable[[], Any]
) -> Any:
    extension = load_amd_extension()
    if extension is None:
        return native()
    if module.weight is not None and module.bias is not None:
        return extension.layer_norm_f32_128(
            x, module.weight, module.bias, module.eps
        )
    if module.weight is not None:
        return extension.layer_norm_f32_128_scale(x, module.weight, module.eps)
    if module.bias is None:
        return extension.layer_norm_f32_128_plain(x, module.eps)
    return native()


def _hip_transition(
    *, module: Any, x: Any, native: Callable[[], Any]
) -> Any:
    extension = load_amd_extension()
    if extension is None:
        return native()
    original_shape = x.shape
    matrix = x.reshape(-1, PAIR_CHANNELS)
    normalized = module.layernorm1(matrix)
    activation = module.linear_no_bias_a(normalized)
    gate = module.linear_no_bias_b(normalized)
    del normalized
    gate = extension.silu_mul_f32(activation, gate)
    del activation
    output = module.linear_no_bias(gate)
    return output.reshape(original_shape)


registry.register(
    operation=LAYER_NORM_OPERATION,
    name=LAYER_NORM_IMPLEMENTATION,
    function=_hip_layer_norm,
    supports=_supports_layer_norm,
    priority=20,
    version="1",
)
registry.register(
    operation=TRANSITION_OPERATION,
    name=TRANSITION_IMPLEMENTATION,
    function=_hip_transition,
    supports=_supports_transition,
    priority=20,
    version="1",
)


def _mode(module: Any, tensor: Any) -> str:
    import torch

    return (
        "inference"
        if not module.training and not torch.is_grad_enabled() and not tensor.requires_grad
        else "training"
    )


def dispatch_layer_norm(
    *, module: Any, x: Any, native: Callable[[], Any]
) -> Any:
    import torch

    supported = (
        x.device.type == "cuda"
        and _architecture(x) == "gfx90a"
        and x.dtype is torch.float32
        and _mode(module, x) == "inference"
        and x.is_contiguous()
        and x.shape[-1:] == (PAIR_CHANNELS,)
        and _row_count(tuple(x.shape)) == PAIR_ROWS
        and not (module.weight is None and module.bias is not None)
        and module.eps == 1e-5
    )
    if supported:
        extension = load_amd_extension()
        if extension is not None:
            _fast_path_counts[(LAYER_NORM_OPERATION, LAYER_NORM_IMPLEMENTATION)] += 1
            return _hip_layer_norm(module=module, x=x, native=native)
    _fast_path_counts[(LAYER_NORM_OPERATION, "native")] += 1
    return native()


def dispatch_transition(
    *, module: Any, x: Any, native: Callable[[], Any]
) -> Any:
    import torch

    supported = (
        x.device.type == "cuda"
        and _architecture(x) == "gfx90a"
        and x.dtype is torch.float32
        and _mode(module, x) == "inference"
        and x.is_contiguous()
        and tuple(x.shape) == (124, 124, PAIR_CHANNELS)
        and module.n == 4
        and module.c_in == PAIR_CHANNELS
    )
    if supported:
        extension = load_amd_extension()
        if extension is not None:
            _fast_path_counts[(TRANSITION_OPERATION, TRANSITION_IMPLEMENTATION)] += 1
            return _hip_transition(module=module, x=x, native=native)
    _fast_path_counts[(TRANSITION_OPERATION, "native")] += 1
    return native()


def fast_path_manifest() -> dict[str, Any]:
    return {
        "selection_counts": [
            {"operation": operation, "implementation": implementation, "count": count}
            for (operation, implementation), count in sorted(_fast_path_counts.items())
        ]
    }
