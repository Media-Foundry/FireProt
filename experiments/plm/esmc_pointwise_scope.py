"""Isolated ESMC pointwise experiments, preserving native BF16 intermediates."""
from contextlib import contextmanager
import ctypes
import math
import os

import torch
import torch.nn.functional as F


class Pointwise:
    def __init__(self, path, vector=8):
        self.lib = ctypes.CDLL(str(path))
        self.fn = self.lib.esmc_pointwise
        self.fn.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
        self.fn.restype = ctypes.c_int
        self.vector = vector

    def run(self, kind, x, r=None, factor=1.0):
        assert x.is_cuda and x.device.index == 0 and x.is_contiguous() and x.dtype == torch.bfloat16
        assert not torch.is_grad_enabled() and not x.requires_grad
        assert x.shape[-1] == (2560 if kind == 0 else 13824)
        assert math.isfinite(factor) and factor > 0
        if r is not None:
            assert r.shape == x.shape and r.dtype == x.dtype and r.device == x.device and r.is_contiguous()
        out = torch.empty((*x.shape[:-1], 2560 if kind == 0 else 6912), device=x.device, dtype=x.dtype)
        # Torch's scalar division computes the FP32 reciprocal before multiplication.
        reciprocal = ctypes.c_float(1.0 / ctypes.c_float(factor).value).value
        rc = self.fn(kind, self.vector, x.data_ptr(), None if r is None else r.data_ptr(),
                     out.data_ptr(), x.numel()//x.shape[-1], reciprocal, torch.cuda.current_stream().cuda_stream)
        if rc: raise RuntimeError(f'ESMC pointwise HIP status {rc}')
        return out

    @contextmanager
    def scope(self, model, mode, audit=False):
        assert mode in ('residual', 'swiglu', 'both')
        assert not model.training and len(model.transformer.blocks) == 80
        changes = []
        state = dict(residual_calls=0, swiglu_calls=0, audited=0, vector=self.vector, mode=mode)
        def check(actual, expected):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), 'intermediate byte mismatch'
            state['audited'] += 1
        try:
            for block in model.transformer.blocks:
                if mode in ('swiglu', 'both'):
                    module = block.ffn
                    native = module.forward
                    def ffn(x, _m=module):
                        z = F.layer_norm(x, (_m.hidden_size,), _m.layer_norm_weight, _m.layer_norm_bias, _m.eps)
                        z = F.linear(z, _m.fc1_weight)
                        out = self.run(1, z)
                        if audit:
                            a, b = z.chunk(2, dim=-1)
                            check(out, F.silu(a)*b)
                        state['swiglu_calls'] += 1
                        return F.linear(out, _m.fc2_weight)
                    changes.append((module, native)); module.forward = ffn
                if mode in ('residual', 'both'):
                    native = block.forward
                    def forward(x, sequence_id, output_attentions=False, _b=block):
                        r, weights = _b.attn(x, sequence_id, output_attentions=output_attentions)
                        y = self.run(0, x, r, _b.scaling_factor)
                        if audit: check(y, x+r/_b.scaling_factor)
                        r = _b.ffn(y)
                        z = self.run(0, y, r, _b.scaling_factor)
                        if audit: check(z, y+r/_b.scaling_factor)
                        state['residual_calls'] += 2
                        return z, weights
                    changes.append((block, native)); block.forward = forward
            yield state
            assert state['residual_calls'] == (160 if mode in ('residual', 'both') else 0)
            assert state['swiglu_calls'] == (80 if mode in ('swiglu', 'both') else 0)
        finally:
            for module, native in reversed(changes): module.forward = native


@contextmanager
def trial_scope(model, request, batch_size):
    """Opt-in composition in a frozen experimental deployment only."""
    flag = os.environ.get('FIREPROT_ESMC_POINTWISE', '0')
    if flag == '0':
        yield request, None
        return
    if flag != '1' or request['workload']['model_name'] != 'esmc-fold2-6b' or batch_size != 4:
        raise ValueError('Pointwise composition requires ESMC B4 and flag 0/1')
    resource = request['resources']['hip_esmc_pointwise']
    op = Pointwise(resource['path'], 4)
    selected = dict(request, resources=dict(request['resources'], hip_esm=resource))
    with op.scope(model, 'residual') as state:
        state['library'] = resource
        state['vector_swiglu'] = 8
        yield selected, state
