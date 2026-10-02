"""Capability-gated CK-style HIP operations for the Protenix recycle trunk."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from fireprot.backends.amd.extension import load_amd_extension
from fireprot.ops.registry import OperationSignature, registry


PAIR_ROWS = 124 * 124
PAIR_CHANNELS = 128
LONG_LAYER_NORM_ROWS = frozenset({1000 * 244, 1000 * 256, 1000 * 1000})
LONG_TRANSITION_SHAPE = (1000, 1000, PAIR_CHANNELS)
LONG1536_ROWS = 1536 * 1536
LONG1536_TRANSITION_SHAPE = (1536, 1536, PAIR_CHANNELS)
_long1536_epilogues_enabled = (
    os.environ.get("FIREPROT_ENABLE_LONG1536") == "1"
    and os.environ.get("FIREPROT_ENABLE_LONG1536_EPILOGUES", "1") == "1"
)
LAYER_NORM_OPERATION = "layer_norm"
TRANSITION_OPERATION = "gated_transition"
LAYER_NORM_IMPLEMENTATION = "amd-gfx90a-hip-layernorm-f32-c128"
LONG_LAYER_NORM_IMPLEMENTATION = "amd-gfx90a-hip-layernorm-f32-c128-long"
CHANNEL_FIRST_LAYER_NORM_IMPLEMENTATION = (
    "amd-gfx90a-hip-layernorm-f32-c128-channel-first"
)
TRANSITION_IMPLEMENTATION = "amd-gfx90a-hip-transition-f32-c128"
LONG_TRANSITION_IMPLEMENTATION = "amd-gfx90a-hip-transition-f32-c128-long"
_fast_path_counts: Counter[tuple[str, str]] = Counter()
_channel_first_layer_norm_enabled = (
    os.environ.get("FIREPROT_DISABLE_CHANNEL_FIRST_LAYERNORM") != "1"
)
_long_layer_norm_enabled = (
    os.environ.get("FIREPROT_DISABLE_LONG_LAYERNORM") != "1"
)
_long_transition_enabled = (
    os.environ.get("FIREPROT_DISABLE_LONG_TRANSITION") != "1"
)


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


def _base_support(
    signature: OperationSignature,
    *,
    validated_rows: frozenset[int] = frozenset({PAIR_ROWS}),
) -> bool | tuple[bool, str]:
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
    if not signature.shape or _row_count(signature.shape) not in validated_rows:
        return False, "row count is not in a validated bucket"
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


def _supports_long_layer_norm(
    signature: OperationSignature,
) -> bool | tuple[bool, str]:
    if not _long_layer_norm_enabled:
        return False, "disabled by FIREPROT_DISABLE_LONG_LAYERNORM"
    rows = (LONG_LAYER_NORM_ROWS | {LONG1536_ROWS}
            if _long1536_epilogues_enabled else LONG_LAYER_NORM_ROWS)
    supported = _base_support(signature, validated_rows=rows)
    if supported is not True:
        return supported
    attributes = dict(signature.attributes)
    if attributes.get("offset") == "True" and attributes.get("scale") != "True":
        return False, "offset-only LayerNorm is not implemented"
    if attributes.get("epsilon") != "1e-05":
        return False, "epsilon is not the validated 1e-5"
    return True


def _supports_channel_first_layer_norm(
    signature: OperationSignature,
) -> bool | tuple[bool, str]:
    attributes = dict(signature.attributes)
    if not _channel_first_layer_norm_enabled:
        return False, "disabled by FIREPROT_DISABLE_CHANNEL_FIRST_LAYERNORM"
    if signature.device_type != "cuda":
        return False, "device is not CUDA/HIP"
    if signature.architecture != "gfx90a":
        return False, "architecture is not gfx90a"
    if signature.dtype != "float32":
        return False, "dtype is not float32"
    if signature.mode != "inference":
        return False, "training or autograd is enabled"
    if signature.shape != (124, 124, PAIR_CHANNELS):
        return False, "shape is not the validated 124x124x128 bucket"
    if signature.layout != "strided":
        return False, "input is not strided"
    if attributes.get("stride") != "(124, 1, 15376)":
        return False, "stride is not the validated channel-first layout"
    if attributes.get("scale") != "True" or attributes.get("offset") != "True":
        return False, "full affine LayerNorm is required"
    if attributes.get("epsilon") != "1e-05":
        return False, "epsilon is not the validated 1e-5"
    if load_amd_extension() is None:
        return False, "FireProt AMD extension is unavailable"
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


def _supports_long_transition(
    signature: OperationSignature,
) -> bool | tuple[bool, str]:
    if not _long_transition_enabled:
        return False, "disabled by FIREPROT_DISABLE_LONG_TRANSITION"
    if signature.shape != LONG_TRANSITION_SHAPE and not (
        _long1536_epilogues_enabled and signature.shape == LONG1536_TRANSITION_SHAPE
    ):
        return False, "shape is not an enabled exact long bucket"
    supported = _base_support(
        signature,
        validated_rows=frozenset({_row_count(signature.shape)}),
    )
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


def _hip_channel_first_layer_norm(
    *, module: Any, x: Any, native: Callable[[], Any]
) -> Any:
    extension = load_amd_extension()
    if extension is None:
        return native()
    return extension.layer_norm_f32_128_channel_first(
        x, module.weight, module.bias, module.eps
    )


registry.register(
    operation=LAYER_NORM_OPERATION,
    name=LAYER_NORM_IMPLEMENTATION,
    function=_hip_layer_norm,
    supports=_supports_layer_norm,
    priority=20,
    version="1",
)
registry.register(
    operation=LAYER_NORM_OPERATION,
    name=LONG_LAYER_NORM_IMPLEMENTATION,
    function=_hip_layer_norm,
    supports=_supports_long_layer_norm,
    priority=30,
    version="1",
)
registry.register(
    operation=LAYER_NORM_OPERATION,
    name=CHANNEL_FIRST_LAYER_NORM_IMPLEMENTATION,
    function=_hip_channel_first_layer_norm,
    supports=_supports_channel_first_layer_norm,
    priority=25,
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
registry.register(
    operation=TRANSITION_OPERATION,
    name=LONG_TRANSITION_IMPLEMENTATION,
    function=_hip_transition,
    supports=_supports_long_transition,
    priority=30,
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

    rows = _row_count(tuple(x.shape)) if x.shape else 0
    long_bucket = _long_layer_norm_enabled and (
        rows in LONG_LAYER_NORM_ROWS
        or (_long1536_epilogues_enabled and rows == LONG1536_ROWS)
    )
    common = (
        x.device.type == "cuda"
        and _architecture(x) == "gfx90a"
        and x.dtype is torch.float32
        and _mode(module, x) == "inference"
        and x.shape[-1:] == (PAIR_CHANNELS,)
        and (rows == PAIR_ROWS or long_bucket)
        and not (module.weight is None and module.bias is not None)
        and module.eps == 1e-5
    )
    channel_first = (
        _channel_first_layer_norm_enabled
        and common
        and tuple(x.shape) == (124, 124, PAIR_CHANNELS)
        and tuple(x.stride()) == (124, 1, PAIR_ROWS)
        and module.weight is not None
        and module.bias is not None
    )
    if channel_first:
        extension = load_amd_extension()
        if extension is not None:
            _fast_path_counts[
                (LAYER_NORM_OPERATION, CHANNEL_FIRST_LAYER_NORM_IMPLEMENTATION)
            ] += 1
            return _hip_channel_first_layer_norm(module=module, x=x, native=native)
    if common and x.is_contiguous():
        extension = load_amd_extension()
        if extension is not None:
            implementation = (
                LONG_LAYER_NORM_IMPLEMENTATION
                if long_bucket
                else LAYER_NORM_IMPLEMENTATION
            )
            _fast_path_counts[(LAYER_NORM_OPERATION, implementation)] += 1
            return _hip_layer_norm(module=module, x=x, native=native)
    _fast_path_counts[(LAYER_NORM_OPERATION, "native")] += 1
    return native()


def dispatch_transition(
    *, module: Any, x: Any, native: Callable[[], Any]
) -> Any:
    import torch

    shape = tuple(x.shape)
    long_bucket = _long_transition_enabled and (
        shape == LONG_TRANSITION_SHAPE
        or (_long1536_epilogues_enabled and shape == LONG1536_TRANSITION_SHAPE)
    )
    supported = (
        x.device.type == "cuda"
        and _architecture(x) == "gfx90a"
        and x.dtype is torch.float32
        and _mode(module, x) == "inference"
        and x.is_contiguous()
        and (shape == (124, 124, PAIR_CHANNELS) or long_bucket)
        and module.n == 4
        and module.c_in == PAIR_CHANNELS
    )
    if supported:
        extension = load_amd_extension()
        if extension is not None:
            implementation = (
                LONG_TRANSITION_IMPLEMENTATION
                if long_bucket
                else TRANSITION_IMPLEMENTATION
            )
            _fast_path_counts[(TRANSITION_OPERATION, implementation)] += 1
            return _hip_transition(module=module, x=x, native=native)
    _fast_path_counts[(TRANSITION_OPERATION, "native")] += 1
    return native()


def fast_path_manifest() -> dict[str, Any]:
    return {
        "channel_first_layer_norm_enabled": _channel_first_layer_norm_enabled,
        "long_layer_norm_enabled": _long_layer_norm_enabled,
        "long_transition_enabled": _long_transition_enabled,
        "long1536_epilogues_enabled": _long1536_epilogues_enabled,
        "selection_counts": [
            {"operation": operation, "implementation": implementation, "count": count}
            for (operation, implementation), count in sorted(_fast_path_counts.items())
        ]
    }
