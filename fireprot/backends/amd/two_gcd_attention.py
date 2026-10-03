"""Experimental exact N1536 attention using both explicitly visible gfx90a GCDs."""
from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Any

_state = threading.local()
_lock = threading.Lock()
_counts: Counter[str] = Counter()


def enabled() -> bool:
    return os.environ.get("FIREPROT_ENABLE_TWO_GCD_ATTENTION") == "1"


def rejection_reason(module: Any, q: Any, k: Any, v: Any, biases: Any) -> str | None:
    import torch

    if not enabled():
        return "disabled"
    if os.environ.get("HIP_VISIBLE_DEVICES") not in {"6,7", "7,6"} or os.environ.get("ROCR_VISIBLE_DEVICES"):
        return "requires explicit physical GPUs 6 and 7 without nested visibility"
    if not torch.version.hip or not q.is_cuda:
        return "requires HIP tensors"
    if (module is not None and module.training) or torch.is_grad_enabled():
        return "requires inference"
    if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != torch.float32:
        return "requires FP32 autocast or disabled autocast"
    if len(biases) != 1 or tuple(biases[0].shape) != (1, 4, 1536, 1536):
        return "requires exact triangle bias"
    tensors = (q, k, v, biases[0])
    if any(t.dtype != torch.float32 or t.device != q.device or t.requires_grad for t in tensors):
        return "requires matching FP32 inference tensors"
    if any(tuple(t.shape) != (512, 4, 1536, 32) or tuple(t.stride()) != (196608, 32, 128, 1) for t in (q, k, v)):
        return "requires exact QKV shape and strides"
    if torch.cuda.device_count() != 2:
        return "requires exactly two visible devices"
    primary = q.device.index
    secondary = 1 - primary
    for device in (primary, secondary):
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                return "graph capture is unsupported"
        if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx90a":
            return "requires two gfx90a devices"
    if not torch.cuda.can_device_access_peer(primary, secondary) or not torch.cuda.can_device_access_peer(secondary, primary):
        return "requires bidirectional peer access"
    return None


def try_attention(*, module: Any, q: Any, k: Any, v: Any, biases: Any) -> Any | None:
    """Return head-last output, or None to retain single-device SDPA.

    The caller must exclusively own both visible GCDs. Cross-device work
    rejoins the caller stream before output use or model end-event recording.
    """
    reason = rejection_reason(module, q, k, v, biases)
    if reason is not None:
        with _lock:
            _counts["fallback: " + reason] += 1
        return None
    import torch
    from torch.nn import functional as F

    primary, secondary = q.device.index, 1 - q.device.index
    streams = getattr(_state, "streams", None)
    if streams is None:
        streams = _state.streams = {}
    if primary not in streams:
        streams[primary] = (torch.cuda.Stream(device=primary), torch.cuda.Stream(device=secondary))
    transfer, remote = streams[primary]
    with torch.cuda.device(primary):
        local = torch.cuda.current_stream(primary)
        packed = biases[0].contiguous()
        transfer.wait_stream(local)
        for tensor in (q, k, v):
            tensor.record_stream(local)
            tensor.record_stream(transfer)
        with torch.cuda.stream(transfer), torch.cuda.stream(remote):
            rq, rk, rv = [t[320:].to(secondary, non_blocking=True) for t in (q, k, v)]
            rb = packed.to(secondary, non_blocking=True)
            ro = F.scaled_dot_product_attention(rq, rk, rv, attn_mask=rb, dropout_p=0.0, scale=1.0)
        lo = F.scaled_dot_product_attention(q[:320], k[:320], v[:320], attn_mask=packed, dropout_p=0.0, scale=1.0)
        with torch.cuda.stream(transfer), torch.cuda.stream(remote):
            returned = ro.to(primary, non_blocking=True)
        local.wait_stream(remote)
        returned.record_stream(local)
        result = torch.cat((lo.transpose(1, 2), returned.transpose(1, 2)), dim=0)
    with _lock:
        _counts["selected"] += 1
    return result


def manifest() -> dict[str, Any]:
    import torch

    with _lock:
        result = dict(enabled=enabled(), counts=dict(_counts), primary_batch=320, secondary_batch=192)
    if enabled() and torch.cuda.is_initialized() and torch.cuda.device_count() == 2:
        result["peak_allocated_by_logical_device_at_manifest"] = {str(i): torch.cuda.max_memory_allocated(i) for i in (0, 1)}
        result["visible_devices"] = os.environ.get("HIP_VISIBLE_DEVICES")
    return result
