"""AMD efficient-attention path for Protenix triangle attention."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from fireprot.ops.registry import (
    OperationSignature,
    registry,
    signature_from_tensor,
)


OPERATION = "triangle_attention"
IMPLEMENTATION = "amd-gfx90a-efficient-sdpa"
LONG_DIRECT_IMPLEMENTATION = "amd-gfx90a-efficient-sdpa-direct-n1000"
SUPPORTED_QKV_SHAPE = (124, 4, 124, 32)
SUPPORTED_BIAS_SHAPES = ((124, 1, 1, 124), (1, 4, 124, 124))
SUPPORTED_UNMASKED_BIAS_SHAPES = ((1, 4, 124, 124),)
LONG_QKV_SHAPE = (1000, 4, 1000, 32)
LONG_BIAS_SHAPES = ((1000, 1, 1, 1000), (1, 4, 1000, 1000))
LONG_UNMASKED_BIAS_SHAPES = ((1, 4, 1000, 1000),)
OUTER_BATCH_CHUNK = 64
_long_sequence_enabled = (
    os.environ.get("FIREPROT_DISABLE_LONG_TRIANGLE_ATTENTION") != "1"
)
_long_direct_enabled = os.environ.get("FIREPROT_DISABLE_LONG_DIRECT_ATTENTION") != "1"


def _architecture(tensor: Any) -> str:
    try:
        import torch

        properties = torch.cuda.get_device_properties(tensor.device)
        return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    except Exception:
        return "unknown"


def _supports_efficient_sdpa(
    signature: OperationSignature,
) -> bool | tuple[bool, str]:
    attributes = dict(signature.attributes)
    if signature.device_type != "cuda":
        return False, "device is not CUDA/HIP"
    if signature.architecture != "gfx90a":
        return False, "architecture is not gfx90a"
    if signature.dtype != "float32":
        return False, "dtype is not float32"
    if signature.mode != "inference":
        return False, "training or autograd is enabled"
    if signature.layout != "strided":
        return False, "projected Q must use the observed head-major strided layout"
    if signature.shape == SUPPORTED_QKV_SHAPE:
        expected_bias_shapes = {
            repr(SUPPORTED_BIAS_SHAPES),
            repr(SUPPORTED_UNMASKED_BIAS_SHAPES),
        }
    elif signature.shape == LONG_QKV_SHAPE:
        if not _long_sequence_enabled:
            return False, "disabled by FIREPROT_DISABLE_LONG_TRIANGLE_ATTENTION"
        expected_bias_shapes = {
            repr(LONG_BIAS_SHAPES),
            repr(LONG_UNMASKED_BIAS_SHAPES),
        }
    else:
        return False, "QKV shape is not a validated short or long bucket"
    if attributes.get("bias_shapes") not in expected_bias_shapes:
        return False, "mask and triangle bias shapes are not validated"
    return True


def _efficient_sdpa(
    *,
    module: Any,
    q: Any,
    k: Any,
    v: Any,
    biases: Sequence[Any],
    native: Callable[..., Any],
) -> Any:
    """Return [*, Q, H, D] using ROCm's memory-efficient SDPA kernel."""
    del module, native
    import torch.nn.functional as functional

    # Protenix produces the triangle bias with heads as the fastest-moving
    # dimension. Materializing it in Q/K order before broadcasting avoids an
    # implicit SDPA input copy that is larger than the attention kernel itself.
    triangle_bias = biases[-1].contiguous()
    if q.shape[0] <= OUTER_BATCH_CHUNK:
        attention_bias = (
            triangle_bias if len(biases) == 1 else biases[0] + triangle_bias
        )
        output = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0,
            scale=1.0,
        )
    else:
        output = q.new_empty(q.shape)
        for start in range(0, q.shape[0], OUTER_BATCH_CHUNK):
            stop = min(start + OUTER_BATCH_CHUNK, q.shape[0])
            attention_bias = (
                triangle_bias
                if len(biases) == 1
                else biases[0][start:stop] + triangle_bias
            )
            output[start:stop] = functional.scaled_dot_product_attention(
                q[start:stop],
                k[start:stop],
                v[start:stop],
                attn_mask=attention_bias,
                dropout_p=0.0,
                scale=1.0,
            )
    return output.transpose(-2, -3)


def _supports_long_direct(signature: OperationSignature) -> bool | tuple[bool, str]:
    if not _long_direct_enabled:
        return False, "disabled by FIREPROT_DISABLE_LONG_DIRECT_ATTENTION"
    if signature.shape != LONG_QKV_SHAPE or dict(signature.attributes).get(
        "bias_shapes"
    ) != repr(LONG_UNMASKED_BIAS_SHAPES):
        return False, "direct attention requires the exact unmasked long bucket"
    return _supports_efficient_sdpa(signature)


def _direct_sdpa(
    *, module: Any, q: Any, k: Any, v: Any,
    biases: Sequence[Any], native: Callable[..., Any],
) -> Any:
    """Keep the head-last output layout without an outer copy/assembly loop."""
    import torch.nn.functional as functional

    if len(biases) != 1:
        return _efficient_sdpa(module=module, q=q, k=k, v=v, biases=biases, native=native)
    output = functional.scaled_dot_product_attention(
        q, k, v, attn_mask=biases[0].contiguous(), dropout_p=0.0, scale=1.0,
    )
    return output.transpose(-2, -3)


registry.register(
    operation=OPERATION,
    name=LONG_DIRECT_IMPLEMENTATION,
    function=_direct_sdpa,
    supports=_supports_long_direct,
    priority=20,
    version="1",
)


registry.register(
    operation=OPERATION,
    name=IMPLEMENTATION,
    function=_efficient_sdpa,
    supports=_supports_efficient_sdpa,
    priority=10,
    version="1",
)


def dispatch_triangle_attention(
    *,
    module: Any,
    q: Any,
    k: Any,
    v: Any,
    biases: Sequence[Any],
    native: Callable[..., Any],
) -> Any:
    import torch

    mode = (
        "inference"
        if (
            not module.training
            and not torch.is_grad_enabled()
            and not any(tensor.requires_grad for tensor in (q, k, v))
        )
        else "training"
    )
    bias_shapes = tuple(tuple(int(value) for value in bias.shape) for bias in biases)
    signature = signature_from_tensor(
        OPERATION,
        q,
        architecture=_architecture(q),
        mode=mode,
        mask=(
            "triangle-bias"
            if len(biases) == 1
            else "additive-mask-and-triangle-bias"
            if len(biases) == 2
            else "unsupported"
        ),
        inplace=False,
        attributes={
            "bias_count": len(biases),
            "bias_shapes": repr(bias_shapes),
            "heads": q.shape[-3] if q.ndim >= 3 else -1,
            "head_dim": q.shape[-1] if q.ndim else -1,
        },
    )
    return registry.dispatch(
        signature,
        native,
        module=module,
        q=q,
        k=k,
        v=v,
        biases=biases,
        native=native,
    )
