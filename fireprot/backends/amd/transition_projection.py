"""Opt-in FP32 dual projection and SiLU for exact long pair transitions."""
from __future__ import annotations
import os
from typing import Any


def supported(module: Any, matrix: Any) -> bool:
    import torch
    if os.environ.get('FIREPROT_ENABLE_TRANSITION_PROJECTION') != '1':
        return False
    if module.training or torch.is_grad_enabled() or matrix.requires_grad:
        return False
    if torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') != torch.float32:
        return False
    if not torch.version.hip or not matrix.is_cuda or matrix.dtype != torch.float32:
        return False
    if tuple(matrix.shape) not in {(1000000,128),(2359296,128)}:
        return False
    if matrix.shape[0] == 2359296 and os.environ.get('FIREPROT_ENABLE_LONG1536') != '1':
        return False
    if not matrix.is_contiguous() or matrix.data_ptr() % 16 or module.n != 4 or module.c_in != 128:
        return False
    from protenix.model.modules.primitives import Linear
    global_hooks=torch.nn.modules.module
    if global_hooks._global_forward_hooks or global_hooks._global_forward_pre_hooks:
        return False
    for linear in (module.linear_no_bias_a,module.linear_no_bias_b):
        if type(linear) not in (torch.nn.Linear,Linear) or linear.bias is not None:
            return False
        if getattr(linear,'precision',None) is not None:
            return False
        if linear._forward_hooks or linear._forward_pre_hooks or 'forward' in linear.__dict__:
            return False
        weight=linear.weight
        if tuple(weight.shape)!=(512,128) or weight.device!=matrix.device or weight.dtype!=torch.float32:
            return False
        if not weight.is_contiguous() or weight.data_ptr()%16:
            return False
    return torch.cuda.get_device_properties(matrix.device).gcnArchName.split(':')[0]=='gfx90a'
