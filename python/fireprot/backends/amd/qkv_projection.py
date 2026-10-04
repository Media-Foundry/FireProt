"""Validated FP32 self-attention QKV preparation for exact long gfx90a buckets."""
from __future__ import annotations

import os
import threading
from typing import Any

from fireprot.backends.amd.extension import load_amd_extension

_lock = threading.RLock()
_counts = {"n1000": 0, "n1536": 0, "fallback": 0}


def supported(module: Any, q_x: Any, kv_x: Any, apply_scale: bool) -> bool:
    import torch

    if os.environ.get("FIREPROT_ENABLE_QKV_PROJECTION", "1") != "1":
        return False
    if module.training or torch.is_grad_enabled() or not apply_scale:
        return False
    if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.float32:
        return False
    if tuple(q_x.shape) not in {(1000, 1000, 128), (512, 1536, 128)}:
        return False
    if (q_x.shape != kv_x.shape or q_x.stride() != kv_x.stride()
            or q_x.device != kv_x.device or q_x.dtype != kv_x.dtype
            or q_x.data_ptr() != kv_x.data_ptr() or kv_x.requires_grad):
        return False
    if q_x.shape[1] == 1536 and os.environ.get("FIREPROT_ENABLE_LONG1536") != "1":
        return False
    if module.no_heads != 4 or module.c_hidden != 32:
        return False
    if not torch.version.hip or not q_x.is_cuda:
        return False
    if q_x.dtype != torch.float32 or not q_x.is_contiguous() or q_x.requires_grad or q_x.data_ptr() % 16:
        return False
    from protenix.model.triangular.layers import OpenfoldLinear

    for linear in (module.linear_q, module.linear_k, module.linear_v):
        if type(linear) not in (torch.nn.Linear, OpenfoldLinear) or linear.bias is not None:
            return False
        if getattr(linear, "precision", None) is not None:
            return False
        if linear._forward_hooks or linear._forward_pre_hooks or "forward" in linear.__dict__:
            return False
        weight = linear.weight
        if tuple(weight.shape) != (128, 128) or weight.device != q_x.device:
            return False
        if weight.dtype != torch.float32 or not weight.is_contiguous() or weight.data_ptr() % 16:
            return False
    return torch.cuda.get_device_properties(q_x.device).gcnArchName.split(":")[0] == "gfx90a"


def prepare(*, module: Any, q_x: Any, kv_x: Any, apply_scale: bool, native: Any) -> Any:
    extension = load_amd_extension() if supported(module, q_x, kv_x, apply_scale) else None
    if extension is None:
        with _lock:
            _counts["fallback"] += 1
        return native()
    output = extension.qkv_projection(q_x, module.linear_q.weight, module.linear_k.weight, module.linear_v.weight)
    batch, length, _ = q_x.shape
    result = tuple(t.view(batch, length, 4, 32).transpose(1, 2) for t in output.unbind(0))
    with _lock:
        _counts[f"n{length}"] += 1
    return result


def manifest() -> dict[str, Any]:
    with _lock:
        return {"enabled": os.environ.get("FIREPROT_ENABLE_QKV_PROJECTION", "1") == "1", "counts": dict(_counts)}
