"""Experimental exact copies for measured triangle layouts; no model dispatch."""
import ctypes
from pathlib import Path

import torch


class CopyRuntime:
    def __init__(self, library):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.fn = self.library.fireprot_triangle_copy
        self.fn.argtypes = ([ctypes.c_void_p]*2 + [ctypes.c_int]*3
                           + [ctypes.c_int64]*6 + [ctypes.c_void_p])
        self.fn.restype = ctypes.c_int
        self.calls = 0
        self.fallbacks = 0

    @staticmethod
    def layout(t):
        shape, strides = list(t.shape), list(t.stride())
        if len(shape) == 4 and shape[0] == 1:
            shape, strides = shape[1:], strides[1:]
        if len(shape) != 3 or sorted(shape) != [128, 256, 1536]:
            return None
        # Sufficient injectivity check, accepting contiguous tensors and slices.
        extent = 1
        for n, s in sorted(zip(shape, strides), key=lambda v: v[1]):
            if s < extent:
                return None
            extent = s*n
        return shape, strides

    def copy_(self, dst, src):
        eligible = (type(src) is torch.Tensor and type(dst) is torch.Tensor
            and src.dtype == dst.dtype == torch.float32
            and src.device == dst.device and src.is_cuda
            and torch.version.hip is not None
            and not torch.is_grad_enabled()
            and not src.requires_grad and not dst.requires_grad
            and not src.is_conj() and not dst.is_conj()
            and not src.is_neg() and not dst.is_neg()
            and src.shape == dst.shape
            and torch.are_deterministic_algorithms_enabled()
            and torch.utils.deterministic.fill_uninitialized_memory
            and src.untyped_storage().data_ptr() != dst.untyped_storage().data_ptr())
        a, b = (self.layout(src), self.layout(dst)) if eligible else (None, None)
        if a and b and 1 in a[1] and 1 in b[1]:
            x, y = a[1].index(1), b[1].index(1)
            if x != y:
                z = 3-x-y
                axes = (x,y,z)
                with torch.cuda.device(src.device):
                    stream = torch.cuda.current_stream(src.device)
                    code = self.fn(src.data_ptr(), dst.data_ptr(),
                        *(a[0][i] for i in axes), *(a[1][i] for i in axes),
                        *(b[1][i] for i in axes), stream.cuda_stream)
                    if code:
                        raise RuntimeError(f'triangle HIP copy launch failed: {code}')
                    src.record_stream(stream)
                    dst.record_stream(stream)
                self.calls += 1
                return dst
        self.fallbacks += 1
        return dst.copy_(src)

    def contiguous(self, src):
        if src.is_contiguous():
            return src
        return self.copy_(torch.empty_like(src, memory_format=torch.contiguous_format), src)
