"""AMD inference path for Protenix triangle multiplication."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from fireprot.backends.amd.extension import load_amd_extension
from fireprot.ops.registry import (
    OperationSignature,
    registry,
    signature_from_tensor,
)


OPERATION = "triangle_multiplication"
IMPLEMENTATION = "amd-short-sequence-torch"
FUSED_GATE_IMPLEMENTATION = "amd-gfx90a-hip-triangle-gates-f32-n124-c128"
LONG_FUSED_GATE_IMPLEMENTATION = "amd-gfx90a-hip-triangle-gates-f32-n1000-c128"
LONG_PACKED_IMPLEMENTATION = "amd-gfx90a-hip-triangle-packed-f32-n1000-c128"
LONG1536_PACKED_IMPLEMENTATION = "amd-gfx90a-hip-triangle-packed-f32-n1536-c128"
MAX_TOKENS = 256
VALIDATED_SHAPE = (124, 124, 128)
LONG_VALIDATED_SHAPE = (1000, 1000, 128)
LONG1536_SHAPE = (1536, 1536, 128)
_long1536_enabled = os.environ.get("FIREPROT_ENABLE_LONG1536") == "1"
_fused_gate_enabled = os.environ.get("FIREPROT_DISABLE_TRIANGLE_GATE_FUSION") != "1"
_long_fused_gate_enabled = (
    os.environ.get("FIREPROT_DISABLE_LONG_TRIANGLE_GATES") != "1"
)
_long_packed_enabled = os.environ.get("FIREPROT_DISABLE_LONG_TRIANGLE_PACKED") != "1"
_vector_packed_enabled = os.environ.get("FIREPROT_ENABLE_TRIANGLE_VECTOR", "1") == "1"
_vector_counts = {"n1000": 0, "n1536": 0, "fallback": 0}
_vector_norm_enabled = os.environ.get("FIREPROT_ENABLE_VECTOR_PACKED_NORM", "1") == "1"
_norm_counts = {"n1000": 0, "n1536": 0}
_dual_projection_enabled = os.environ.get("FIREPROT_ENABLE_DUAL_PROJECTION", "0") == "1"
_dual_counts = {"n1000": 0, "n1536": 0, "fallback": 0}


def vector_manifest() -> dict[str, Any]:
    return {"enabled": _vector_packed_enabled, "counts": dict(_vector_counts),
            "packed_norm_enabled": _vector_norm_enabled, "packed_norm_counts": dict(_norm_counts),
            "dual_projection_enabled": _dual_projection_enabled, "dual_projection_counts": dict(_dual_counts)}



def _architecture(tensor: Any) -> str:
    try:
        import torch

        properties = torch.cuda.get_device_properties(tensor.device)
        return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    except Exception:
        return "unknown"


def _supports_short_sequence(
    signature: OperationSignature,
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
    if not signature.inplace:
        return False, "native non-inplace path is already direct"
    if len(signature.shape) != 3:
        return False, "only unbatched pair tensors are supported"
    if signature.shape[-2] > MAX_TOKENS:
        return False, f"token count exceeds {MAX_TOKENS}"
    return True


def _short_sequence_torch(
    *,
    module: Any,
    z: Any,
    mask: Any,
    inplace_safe: bool,
    add_with_inplace: bool,
    inplace_chunk_size: int | None,
    native: Callable[[], Any],
) -> Any:
    del inplace_chunk_size, native
    update = module.forward(
        z=z,
        mask=mask,
        inplace_safe=False,
        _add_with_inplace=False,
        triangle_multiplicative="torch",
    )
    if add_with_inplace:
        z.add_(update)
    else:
        z.copy_(update)
    return z


def _supports_fused_gates(
    signature: OperationSignature,
) -> bool | tuple[bool, str]:
    if not _fused_gate_enabled:
        return False, "disabled by FIREPROT_DISABLE_TRIANGLE_GATE_FUSION"
    supported = _supports_short_sequence(signature)
    if supported is not True:
        return supported
    if signature.shape != VALIDATED_SHAPE:
        return False, "shape is not the validated 124x124x128 bucket"
    if signature.mask not in {"none", "dense"}:
        return False, "only absent or dense pair masks are supported"
    attributes = dict(signature.attributes)
    if attributes.get("c_z") != "128" or attributes.get("c_hidden") != "128":
        return False, "c_z and c_hidden must both be 128"
    if load_amd_extension() is None:
        return False, "FireProt AMD extension is unavailable"
    return True


def _short_sequence_hip_fused(
    *,
    module: Any,
    z: Any,
    mask: Any,
    inplace_safe: bool,
    add_with_inplace: bool,
    inplace_chunk_size: int | None,
    native: Callable[[], Any],
) -> Any:
    del inplace_safe, inplace_chunk_size
    extension = load_amd_extension()
    if (
        extension is None
        or (
            mask is not None
            and (
                tuple(mask.shape) != VALIDATED_SHAPE[:2]
                or not mask.is_contiguous()
                or mask.dtype != z.dtype
                or mask.device != z.device
            )
        )
    ):
        return _short_sequence_torch(
            module=module,
            z=z,
            mask=mask,
            inplace_safe=True,
            add_with_inplace=add_with_inplace,
            inplace_chunk_size=None,
            native=native,
        )

    normalized = module.layer_norm_in(z)

    a_gate = module.linear_a_g(normalized)
    a = module.linear_a_p(normalized)
    if mask is None:
        a = extension.sigmoid_mul_f32(a_gate, a)
    else:
        a = extension.sigmoid_mul_mask_f32(a_gate, a, mask)
    del a_gate

    b_gate = module.linear_b_g(normalized)
    b = module.linear_b_p(normalized)
    if mask is None:
        b = extension.sigmoid_mul_f32(b_gate, b)
    else:
        b = extension.sigmoid_mul_mask_f32(b_gate, b, mask)
    del b_gate

    x = module._combine_projections(a, b)
    del a, b
    x = module.layer_norm_out(x)
    x = module.linear_z(x)

    output_gate = module.linear_g(normalized)
    x = extension.sigmoid_mul_f32(output_gate, x)
    del output_gate, normalized

    if add_with_inplace:
        z.add_(x)
    else:
        z.copy_(x)
    return z


def _supports_long_fused_gates(
    signature: OperationSignature,
    *, shape: tuple[int, ...] = LONG_VALIDATED_SHAPE,
) -> bool | tuple[bool, str]:
    if not _long_fused_gate_enabled:
        return False, "disabled by FIREPROT_DISABLE_LONG_TRIANGLE_GATES"
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
    if not signature.inplace:
        return False, "long path requires inplace inference"
    if signature.shape != shape:
        return False, "shape is not the exact supported long bucket"
    if signature.mask != "none":
        return False, "the validated long-sequence bucket has no explicit mask"
    attributes = dict(signature.attributes)
    if attributes.get("c_z") != "128" or attributes.get("c_hidden") != "128":
        return False, "c_z and c_hidden must both be 128"
    if load_amd_extension() is None:
        return False, "FireProt AMD extension is unavailable"
    return True


def _long_sequence_hip_fused(
    *,
    module: Any,
    z: Any,
    mask: Any,
    inplace_safe: bool,
    add_with_inplace: bool,
    inplace_chunk_size: int | None,
    native: Callable[[], Any],
) -> Any:
    del inplace_safe
    extension = load_amd_extension()
    if extension is None or mask is not None:
        return native()

    def projection_fusion(gate: Any, projection: Any, projection_mask: Any) -> Any:
        if projection_mask is None:
            return extension.sigmoid_mul_f32(gate, projection)
        gate.sigmoid_()
        projection.mul_(gate)
        projection.mul_(projection_mask)
        return projection

    def gate_fusion(gate: Any, output: Any) -> Any:
        return extension.sigmoid_mul_f32(gate, output)

    return module._inference_forward(
        z,
        mask,
        inplace_chunk_size=inplace_chunk_size,
        with_add=add_with_inplace,
        projection_fusion=projection_fusion,
        gate_fusion=gate_fusion,
    )


def _supports_long_packed(signature: OperationSignature) -> bool | tuple[bool, str]:
    if not _long_packed_enabled:
        return False, "disabled by FIREPROT_DISABLE_LONG_TRIANGLE_PACKED"
    return _supports_long_fused_gates(signature)


def _supports_long1536_packed(signature: OperationSignature) -> bool | tuple[bool, str]:
    if not _long1536_enabled or not _long_packed_enabled:
        return False, "N1536 packed triangle requires its length opt-in and packed path"
    return _supports_long_fused_gates(signature, shape=LONG1536_SHAPE)


def _supports_dual_projection(module: Any, tensor: Any) -> bool:
    import torch

    if not _dual_projection_enabled or torch.is_grad_enabled() or module.training:
        return False
    if (tuple(tensor.shape) not in (LONG_VALIDATED_SHAPE, LONG1536_SHAPE)
            or tensor.dtype != torch.float32 or not tensor.is_cuda
            or not tensor.is_contiguous() or tensor.data_ptr() % 16
            or _architecture(tensor) != "gfx90a"):
        return False
    for name in ("linear_a_g", "linear_a_p", "linear_b_g", "linear_b_p"):
        layer = getattr(module, name, None)
        if not isinstance(layer, torch.nn.Linear) or layer.bias is not None:
            return False
        weight = layer.weight
        if (tuple(weight.shape) != (128, 128) or weight.dtype != tensor.dtype
                or weight.device != tensor.device or not weight.is_contiguous()
                or weight.data_ptr() % 16):
            return False
    return True


def _long_sequence_packed(
    *, module: Any, z: Any, mask: Any, inplace_safe: bool,
    add_with_inplace: bool, inplace_chunk_size: int | None,
    native: Callable[[], Any],
) -> Any:
    import torch

    extension = load_amd_extension()
    if extension is None or mask is not None:
        return native()
    vector = (_vector_packed_enabled and tuple(z.shape) in (LONG_VALIDATED_SHAPE, LONG1536_SHAPE)
              and z.is_contiguous() and z.data_ptr() % 16 == 0)
    pack = extension.triangle_vector_pack if vector else extension.sigmoid_pack_f32_128
    normalized = module.layer_norm_in(z)
    if _supports_dual_projection(module, normalized):
        a = extension.dual_projection(normalized, module.linear_a_g.weight, module.linear_a_p.weight)
        b = extension.dual_projection(normalized, module.linear_b_g.weight, module.linear_b_p.weight)
        _dual_counts[f"n{z.shape[0]}"] += 1
    else:
        if _dual_projection_enabled:
            _dual_counts["fallback"] += 1
        a = pack(module.linear_a_g(normalized), module.linear_a_p(normalized))
        b = pack(module.linear_b_g(normalized), module.linear_b_p(normalized))
    from fireprot.backends.amd.lt_bmm import packed_bmm
    contracted = packed_bmm(a, b, module._outgoing)
    del a, b
    norm = module.layer_norm_out
    vector_norm = _vector_norm_enabled and tuple(z.shape) in (LONG_VALIDATED_SHAPE, LONG1536_SHAPE)
    norm_fn = extension.packed_norm_vector if vector_norm else extension.packed_layer_norm_f32_128
    output = norm_fn(contracted, norm.weight, norm.bias, norm.eps)
    if vector_norm:
        _norm_counts[f"n{z.shape[0]}"] += 1
    del contracted
    output = module.linear_z(output)
    if vector:
        result = extension.triangle_residual(module.linear_g(normalized), output, z, add_with_inplace)
        _vector_counts[f"n{z.shape[0]}"] += 1
        return result
    _vector_counts["fallback"] += 1
    output = extension.sigmoid_mul_f32(module.linear_g(normalized), output)
    del normalized
    if add_with_inplace:
        z.add_(output)
    else:
        z.copy_(output)
    return z


registry.register(
    operation=OPERATION,
    name=LONG_PACKED_IMPLEMENTATION,
    function=_long_sequence_packed,
    supports=_supports_long_packed,
    priority=40,
    version="1",
)
registry.register(
    operation=OPERATION,
    name=LONG1536_PACKED_IMPLEMENTATION,
    function=_long_sequence_packed,
    supports=_supports_long1536_packed,
    priority=40,
    version="experimental-1",
)
registry.register(
    operation=OPERATION,
    name=LONG_FUSED_GATE_IMPLEMENTATION,
    function=_long_sequence_hip_fused,
    supports=_supports_long_fused_gates,
    priority=30,
    version="1",
)
registry.register(
    operation=OPERATION,
    name=FUSED_GATE_IMPLEMENTATION,
    function=_short_sequence_hip_fused,
    supports=_supports_fused_gates,
    priority=20,
    version="1",
)
registry.register(
    operation=OPERATION,
    name=IMPLEMENTATION,
    function=_short_sequence_torch,
    supports=_supports_short_sequence,
    priority=10,
    version="1",
)


def dispatch_triangle_multiplication(
    *,
    module: Any,
    z: Any,
    mask: Any,
    inplace_safe: bool,
    add_with_inplace: bool,
    inplace_chunk_size: int | None,
    native: Callable[[], Any],
) -> Any:
    import torch

    mode = (
        "inference"
        if not module.training and not torch.is_grad_enabled() and not z.requires_grad
        else "training"
    )
    signature = signature_from_tensor(
        OPERATION,
        z,
        architecture=_architecture(z),
        mode=mode,
        mask="none" if mask is None else "dense",
        inplace=inplace_safe,
        attributes={
            "direction": "outgoing" if module._outgoing else "incoming",
            "add_with_inplace": add_with_inplace,
            "c_z": module.c_z,
            "c_hidden": module.c_hidden,
        },
    )
    return registry.dispatch(
        signature,
        native,
        module=module,
        z=z,
        mask=mask,
        inplace_safe=inplace_safe,
        add_with_inplace=add_with_inplace,
        inplace_chunk_size=inplace_chunk_size,
        native=native,
    )
