"""Opt-in exact N1536 attention output-gate fusion; native fallback otherwise."""
from __future__ import annotations
import os
import threading
from typing import Any
from fireprot.backends.amd.extension import load_amd_extension

_lock=threading.RLock()
_counts={"selected":0,"fallback":0}
_fallback_signatures: dict[str,Any]={}


def supported(module: Any, o: Any, x: Any) -> bool:
    import torch
    if os.environ.get('FIREPROT_ENABLE_ATTENTION_GATE_1536')!='1': return False
    if module.training or torch.is_grad_enabled(): return False
    if torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda')!=torch.float32: return False
    if getattr(module,'linear_g',None) is None or module.no_heads!=4 or module.c_hidden!=32: return False
    if tuple(x.shape)!=(512,1536,128) or tuple(o.shape)!=(512,1536,4,32): return False
    if not torch.version.hip or not x.is_cuda or x.device!=o.device: return False
    if any(t.dtype!=torch.float32 or not t.is_contiguous() or t.requires_grad or t.data_ptr()%16 for t in (x,o)): return False
    for linear in (module.linear_g,module.linear_o):
        if linear.weight.dtype!=torch.float32 or linear.weight.device!=x.device: return False
        if linear.bias is not None and (linear.bias.dtype!=torch.float32 or linear.bias.device!=x.device): return False
    return torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0]=='gfx90a'


def wrap_up(*,module: Any,o: Any,x: Any,native: Any) -> Any:
    extension=load_amd_extension() if supported(module,o,x) else None
    if extension is None:
        import torch
        details = dict(x_shape=list(x.shape) if x is not None else None,
                       o_shape=list(o.shape) if o is not None else None,
                       x_stride=list(x.stride()) if x is not None else None,
                       o_stride=list(o.stride()) if o is not None else None,
                       x_dtype=str(x.dtype) if x is not None else None,
                       o_dtype=str(o.dtype) if o is not None else None,
                       training=getattr(module,'training',None),grad_enabled=torch.is_grad_enabled(),
                       autocast=torch.is_autocast_enabled('cuda'),
                       autocast_dtype=str(torch.get_autocast_dtype('cuda')),
                       heads=getattr(module,'no_heads',None),hidden=getattr(module,'c_hidden',None),
                       x_requires_grad=getattr(x,'requires_grad',None),
                       o_requires_grad=getattr(o,'requires_grad',None))
        key=repr(details)
        with _lock:
            _counts['fallback']+=1
            if key in _fallback_signatures:
                _fallback_signatures[key]['calls']+=1
            elif len(_fallback_signatures)<16:
                _fallback_signatures[key]=dict(details,calls=1)
        return native()
    gate=module.linear_g(x)
    output=extension.sigmoid_mul_out_f32(gate,o.view(512,1536,128))
    result=module.linear_o(output)
    with _lock: _counts['selected']+=1
    return result


def manifest() -> dict[str,Any]:
    with _lock:
        return dict(enabled=os.environ.get('FIREPROT_ENABLE_ATTENTION_GATE_1536')=='1',counts=dict(_counts),fallback_signatures=list(_fallback_signatures.values()))
