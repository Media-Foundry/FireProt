"""Published gfx1100 FP32 softmax; exact screened signatures only."""
import ctypes

import torch

from fireprot.backends.amd.softmax_fp32 import compatible_tensors


# N1000 model traces also contain outer chunks 256 and 232. Keep the 1000-row
# autotuner probe native, and retain native fallback for untested shapes.
CASES = {(1, 1000), (1, 1536), (8, 1000), (8, 1536), (128, 1536),
         (232, 1000), (256, 1000)}


class Gfx1100Softmax:
    def __init__(self):
        if str(torch.__version__) != "2.13.0+rocm10.0.0" or torch.version.hip != "7.15.26333":
            raise RuntimeError("Unvalidated Torch/HIP for experimental gfx1100 softmax")
        self.device = torch.cuda.current_device()
        if torch.cuda.get_device_properties(self.device).gcnArchName.split(":")[0] != "gfx1100":
            raise RuntimeError("Experimental wave32 softmax requires gfx1100")
        from ._published import select_c
        selected = select_c('fireprot_softmax_gfx1100', 'FIREPROT_SOFTMAX_GFX1100', 'softmax-gfx1100-v1')
        if selected is None:
            raise RuntimeError('An explicitly pinned gfx1100 softmax bundle is required')
        self.library, self.receipt = selected
        self.entry = self.library.fireprot_softmax_gfx1100
        self.entry.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
                              + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p])
        self.entry.restype = ctypes.c_int
        self.launches_submitted = 0
        self.receipt.update(cases=sorted(CASES), variant_calls={})

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
