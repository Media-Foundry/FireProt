"""AMD efficient-attention path for Protenix triangle attention."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fireprot.ops.registry import (
    OperationSignature,
    registry,
    signature_from_tensor,
)


OPERATION = "triangle_attention"
IMPLEMENTATION = "amd-gfx90a-efficient-sdpa"
SUPPORTED_QKV_SHAPE = (124, 4, 124, 32)
SUPPORTED_BIAS_SHAPES = ((124, 1, 1, 124), (1, 4, 124, 124))
OUTER_BATCH_CHUNK = 64


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
    if signature.shape != SUPPORTED_QKV_SHAPE:
        return False, f"QKV shape is not the validated {SUPPORTED_QKV_SHAPE}"
    if attributes.get("bias_shapes") != repr(SUPPORTED_BIAS_SHAPES):
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
    triangle_bias = biases[1].contiguous()
    if q.shape[0] <= OUTER_BATCH_CHUNK:
        combined_bias = biases[0] + triangle_bias
        output = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_bias,
            dropout_p=0.0,
            scale=1.0,
        )
    else:
        output = q.new_empty(q.shape)
        for start in range(0, q.shape[0], OUTER_BATCH_CHUNK):
            stop = min(start + OUTER_BATCH_CHUNK, q.shape[0])
            combined_bias = biases[0][start:stop] + triangle_bias
            output[start:stop] = functional.scaled_dot_product_attention(
                q[start:stop],
                k[start:stop],
                v[start:stop],
                attn_mask=combined_bias,
                dropout_p=0.0,
                scale=1.0,
            )
    return output.transpose(-2, -3)


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
        mask="additive-mask-and-triangle-bias" if len(biases) == 2 else "unsupported",
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
