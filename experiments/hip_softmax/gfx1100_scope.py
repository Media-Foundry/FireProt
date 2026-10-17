"""Experimental OpenFold3 gfx1100 dispatch; not a maintained production route.

Only synthetic operator gates have passed. Full trained-model parity and Rust
E2E qualification are required before promotion. Explicit library hashes are
mandatory; this module neither compiles nor discovers a library implicitly.
"""
import ctypes
import hashlib
from pathlib import Path

import torch

from fireprot.adapters.openfold3.attention import AttentionDispatcher
from fireprot.backends.amd.softmax_fp32 import compatible_tensors
from fireprot.ops.registry import signature_from_tensor


CASES = {(1, 1000), (1, 1536), (8, 1000), (8, 1536), (128, 1536)}


class Gfx1100Softmax:
    def __init__(self, path, sha256):
        if str(torch.__version__) != "2.13.0+rocm10.0.0" or torch.version.hip != "7.15.26333":
            raise RuntimeError("Unvalidated Torch/HIP for experimental gfx1100 softmax")
        self.device = torch.cuda.current_device()
        if torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0] != "gfx1100":
            raise RuntimeError("Experimental wave32 softmax requires gfx1100")
        path = Path(path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError("Experimental softmax library SHA-256 mismatch")
        self.library = ctypes.CDLL(str(path))
        self.entry = self.library.fireprot_softmax_gfx1100
        self.entry.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                              + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
        self.entry.restype = ctypes.c_int
        self.launches_submitted = 0
        self.receipt = dict(path=str(path), sha256=digest, architecture="gfx1100",
                            scope="experimental synthetic-operator qualification only",
                            cases=sorted(CASES), variant_calls={})

    def supports(self, scores, biases):
        return (not torch.is_grad_enabled() and compatible_tensors(scores, biases)
                and scores.is_cuda and scores.device.index == self.device
                and (scores.shape[0], scores.shape[-1]) in CASES)

    def __call__(self, scores, biases):
        if not self.supports(scores, biases):
            raise ValueError("Unsupported experimental gfx1100 softmax input")
        mask, bias = biases
        batch, n = scores.shape[0], scores.shape[-1]
        ending = bias.stride(-1) > bias.stride(-2)
        waves, reuse, packed = 8, 0, False
        if n == 1000:
            waves, reuse = (8, 1) if ending else (1, 0)
        elif batch == 128 and ending:
            waves, reuse, packed = 4, 1, True
        with torch.cuda.device(self.device):
            if packed:
                bias = bias.contiguous()
            output = torch.empty_like(scores)
            status = self.entry(scores.data_ptr(), mask.data_ptr(), bias.data_ptr(), output.data_ptr(),
                batch, n, mask.stride(0) if mask.shape[0] > 1 else 0, mask.stride(-1),
                *bias.stride()[-3:], waves, reuse, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"gfx1100 softmax launch failed: {status}")
        self.launches_submitted += 1
        key = f"n{n}-b{batch}-w{waves}-r{reuse}-packed{int(packed)}"
        calls = self.receipt["variant_calls"]
        calls[key] = calls.get(key, 0) + 1
        return output


class Gfx1100Dispatcher(AttentionDispatcher):
    def _eligible(self, query, key, value, biases):
        return (super()._eligible(query, key, value, biases)
                and (query.shape[0], query.shape[2]) in CASES)

    def __call__(self, native, query, key, value, biases, use_high_precision=False):
        if type(query) is not torch.Tensor:
            return native(query, key, value, biases, use_high_precision=use_high_precision)
        signature = signature_from_tensor("attention", query, architecture="gfx1100",
            mode="training" if torch.is_grad_enabled() else "inference", mask="pair_and_token",
            inplace=False, attributes={"eligible": self._eligible(query, key, value, biases),
                                       "high_precision": use_high_precision})
        return self.registry.dispatch(signature, native, query, key, value, biases,
                                      use_high_precision=use_high_precision)
