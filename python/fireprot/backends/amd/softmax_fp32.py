"""Prepared FP32 HIP mask/bias/softmax boundary, not yet enabled by an adapter.

Rust verifies the published C ABI artifact. This module only checks tensors and
launches the kernel; callers own capability selection and native fallback.
"""
from __future__ import annotations

import ctypes

import torch

TORCH_VERSION = "2.12.0a0+git78258b9"
HIP_VERSION = "7.14.60850"
PACKAGE = "fireprot_softmax_fp32"
ENVIRONMENT_PREFIX = "FIREPROT_SOFTMAX_FP32"
C_ABI = "softmax-fp32-v1"


def compatible_tensors(scores, biases):
    """Pure tensor contract; GPU/software compatibility is checked separately."""
    if not isinstance(biases, (list, tuple)) or len(biases) != 2:
        return False
    tensors = (scores, *biases)
    if any(type(x) is not torch.Tensor for x in tensors):
        return False
    if any(x.dtype != torch.float32 or x.ndim != 4 or x.device != scores.device
           or x.requires_grad or x.is_conj() or x.is_neg()
           or any(s < 0 for s in x.stride()) for x in tensors):
        return False
    if (not scores.is_contiguous() or not 0 < scores.shape[0] <= 1536
            or scores.shape[1] != 4 or scores.shape[2] not in (1000, 1536)
            or scores.shape[3] != scores.shape[2]):
        return False
    mask, bias = biases
    n = scores.shape[-1]
    return (mask.shape[0] in (1, scores.shape[0])
            and tuple(mask.shape[1:]) == (1, 1, n)
            and tuple(bias.shape) == (1, 4, n, n))


class SoftmaxFP32:
    """Out-of-place launch of an explicitly published gfx90a implementation."""

    def __init__(self):
        if str(torch.__version__) != TORCH_VERSION or torch.version.hip != HIP_VERSION:
            raise RuntimeError("Unvalidated Torch/HIP FP32 softmax implementation")
        if not torch.cuda.is_available():
            raise RuntimeError("HIP GPU unavailable")
        device = torch.cuda.current_device()
        if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx90a":
            raise RuntimeError("FP32 softmax requires gfx90a")
        from ._published import select_c
        selected = select_c(PACKAGE, ENVIRONMENT_PREFIX, C_ABI)
        if selected is None:
            raise RuntimeError("An explicitly pinned FP32 softmax bundle is required")
        self.library, self.receipt = selected
        self.device = device
        self.launches_submitted = 0
        self.entry = self.library.fireprot_softmax_f32
        self.entry.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                              + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2
                              + [ctypes.c_void_p])
        self.entry.restype = ctypes.c_int

    def supports(self, scores, biases):
        return (not torch.is_grad_enabled() and compatible_tensors(scores, biases)
                and scores.is_cuda and scores.device.index == self.device)

    def __call__(self, scores, biases):
        if not self.supports(scores, biases):
            raise ValueError("Unsupported FP32 softmax tensors or device")
        mask, bias = biases
        with torch.cuda.device(self.device):
            bias = bias.contiguous()
            output = torch.empty_like(scores)
            status = self.entry(
                scores.data_ptr(), mask.data_ptr(), bias.data_ptr(), output.data_ptr(),
                scores.shape[0], scores.shape[-1],
                mask.stride(0) if mask.shape[0] > 1 else 0, mask.stride(-1),
                *bias.stride()[-3:], 2, 1, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"FP32 softmax HIP launch failed: {status}")
        self.launches_submitted += 1
        return output
