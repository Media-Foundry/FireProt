"""Opt-in pair transpose views for exact long inference layouts."""
from __future__ import annotations
import os
from typing import Any
from fireprot.backends.amd.extension import load_amd_extension


def transpose_pair(x: Any, *, training: bool) -> Any:
    import torch
    view=x.transpose(-2,-3)
    enabled=(os.environ.get('FIREPROT_ENABLE_PAIR_LAYOUT')=='1'
             and os.environ.get('FIREPROT_ENABLE_TRANSPOSED_LAYERNORM')=='1'
             and os.environ.get('FIREPROT_DISABLE_LONG_LAYERNORM')!='1')
    supported=(enabled and not training and not torch.is_grad_enabled()
               and torch.version.hip and x.is_cuda and x.dtype==torch.float32
               and not x.requires_grad and x.ndim==3
               and x.shape[0] in (1000,1536) and x.shape[1:]==(x.shape[0],128)
               and (x.shape[0]!=1536 or os.environ.get('FIREPROT_ENABLE_LONG1536')=='1')
               and tuple(x.stride()) in ((x.shape[0]*128,128,1),(128,x.shape[0]*128,1))
               and x.data_ptr()%16==0
               and torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0]=='gfx90a')
    if supported:
        extension=load_amd_extension()
        if extension is not None and hasattr(extension,'transposed_layer_norm'):
            return view
    return view.contiguous()
