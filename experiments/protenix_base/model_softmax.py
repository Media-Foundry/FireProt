"""Experimental trained-call verification; native QK/PV, owned HIP score softmax.

Installed only in an isolated worker snapshot. Publication, selection and model
supervision remain Rust-owned. This module is not a maintained default adapter.
"""
import ctypes
import json
import os
from pathlib import Path

import torch
import torch.utils.deterministic
from fireprot.backends.amd._published import select_c
from fireprot.ops.registry import registry

NAME = 'experimental-base-native-bmm-hip-bias-softmax'


class Kernel:
    def __init__(self, mode):
        if mode not in ('verify', 'candidate'):
            raise ValueError('Unknown Base softmax experiment mode')
        self.mode = mode
        selected = select_c('fireprot_base_bias_softmax', 'FIREPROT_BASE_BIAS_SOFTMAX',
                            'base-bias-softmax-f32-v1')
        if selected is None:
            raise RuntimeError('Explicit published Base softmax bundle required')
        self.library, self.receipt = selected
        self.entry = self.library.fireprot_base_bias_softmax_f32
        self.entry.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                             + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
        self.entry.restype = ctypes.c_int
        self.calls = self.verified = self.fallbacks = 0
        self.shapes = {}

    def supports(self, signature):
        shape = signature.shape
        return (signature.device_type == 'cuda' and signature.architecture == 'gfx90a'
                and signature.dtype == 'float32' and signature.mode == 'inference'
                and len(shape) == 4 and shape[1] == 4 and shape[2] in (600, 1000)
                and shape[3] == 32 and 0 < shape[0] <= shape[2]
                and signature.mask in ('triangle-bias', 'additive-mask-and-triangle-bias'))

    def call(self, *, module, q, k, v, biases, native):
        if (torch.is_grad_enabled() or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.utils.deterministic.fill_uninitialized_memory):
            raise RuntimeError('Strict deterministic inference required')
        n, batch = q.shape[-2], q.shape[0]
        mask = biases[0] if len(biases) == 2 else None
        bias = biases[-1]
        tensors = (q, k, v, *biases)
        compatible = (len(biases) in (1, 2) and k.shape == q.shape and v.shape == q.shape
            and tuple(bias.shape) == (1, 4, n, n)
            and (mask is None or (mask.shape[0] in (1, batch) and tuple(mask.shape[1:]) == (1, 1, n)))
            and all(type(t) is torch.Tensor and t.device == q.device and t.dtype == torch.float32
                    and not t.requires_grad and not t.is_conj() and not t.is_neg()
                    and all(s >= 0 for s in t.stride()) for t in tensors))
        if not compatible:
            self.fallbacks += 1
            return native()
        reference = native() if self.mode == 'verify' else None
        # This newly allocated score tensor is private to this call. Preserve
        # native matmul shapes, input strides and arithmetic; only consume scores.
        scores = torch.matmul(q, k.transpose(-1, -2))
        if not scores.is_contiguous() or scores.storage_offset() != 0:
            raise RuntimeError('Unexpected native score storage')
        with torch.cuda.device(q.device):
            code = self.entry(scores.data_ptr(), None if mask is None else mask.data_ptr(),
                bias.data_ptr(), scores.data_ptr(), batch, n,
                0 if mask is None or mask.shape[0] == 1 else mask.stride(0),
                0 if mask is None else mask.stride(-1), *bias.stride()[-3:],
                2, 1, torch.cuda.current_stream(q.device).cuda_stream)
            if code:
                raise RuntimeError(f'HIP softmax status {code}')
        result = torch.matmul(scores, v).transpose(-2, -3)
        self.calls += 1
        shape = f'B{batch}_N{n}_biases{len(biases)}'
        self.shapes[shape] = self.shapes.get(shape, 0) + 1
        if reference is not None:
            if not torch.equal(reference.contiguous().view(torch.int32), result.contiguous().view(torch.int32)):
                raise RuntimeError(f'Trained Base attention mismatch at call {self.calls}: {shape}')
            self.verified += 1
            return reference
        return result


def run(original_predict, request, channel):
    mode = os.environ.get('FIREPROT_BASE_BIAS_SOFTMAX_MODE', 'off')
    if mode == 'off':
        return original_predict(request, channel)
    if request['backend'] != 'fireprot' or request['workload']['model_name'] != 'protenix_base_default_v0.5.0':
        raise RuntimeError('Experiment only accepts the Base FireProt route')
    from fireprot.adapters.protenix import runtime
    original_install = runtime.install
    kernel = None

    def install(*args, **kwargs):
        nonlocal kernel
        result = original_install(*args, **kwargs)
        kernel = Kernel(mode)
        registry.register(operation='triangle_attention', name=NAME, function=kernel.call,
                          supports=kernel.supports, priority=100, version='experimental-v1')
        return result

    runtime.install = install
    try:
        result = original_predict(request, channel)
        if kernel is None or kernel.calls == 0 or (mode == 'verify' and kernel.verified != kernel.calls):
            raise RuntimeError('Base softmax experiment did not verify selected calls')
        Path(os.environ['FIREPROT_OUTPUT_DIR'], 'base-bias-softmax.json').write_text(json.dumps(
            dict(mode=mode, calls=kernel.calls, verified=kernel.verified, fallbacks=kernel.fallbacks,
                 shapes=kernel.shapes, published=kernel.receipt), indent=2) + '\n')
        return result
    finally:
        runtime.install = original_install
        if kernel is not None:
            registry.unregister('triangle_attention', NAME)
