"""AMD inference path for Protenix triangle multiplication."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fireprot.ops.registry import (
    OperationSignature,
    registry,
    signature_from_tensor,
)


OPERATION = "triangle_multiplication"
IMPLEMENTATION = "amd-short-sequence-torch"
MAX_TOKENS = 256


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
