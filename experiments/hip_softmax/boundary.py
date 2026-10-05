"""Experimental tensor boundary for the Rust-published FP32 HIP kernel."""
import ctypes

import torch


def compatible(shape, device, biases):
    if (len(shape) != 4 or shape[1] != 4 or shape[2] not in (1000,1536)
            or shape[3] != shape[2] or not 0 < shape[0] <= 1536
            or len(biases) != 2):
        return False
    mask, bias = biases
    return (all(x.dtype == torch.float32 and x.device == device and x.ndim == 4
                and all(s >= 0 for s in x.stride()) for x in biases)
            and tuple(mask.shape[1:]) == (1,1,shape[3])
            and mask.shape[0] in (1,shape[0])
            and tuple(bias.shape) == (1,4,shape[2],shape[3]))


class FusedSoftmax:
    def __init__(self, waves=2, reuse=1, pack=False):
        from fireprot.backends.amd._published import select_c
        selected = select_c('fireprot_experimental_softmax_f32',
            'FIREPROT_EXPERIMENTAL_SOFTMAX', 'experimental-softmax-f32-v1')
        if selected is None:
            raise RuntimeError('An explicitly pinned experimental bundle is required')
        self.library, self.receipt = selected
        self.entry = self.library.fireprot_softmax_f32
        self.entry.argtypes = ([ctypes.c_void_p]*4 + [ctypes.c_int]*2
            + [ctypes.c_int64]*5 + [ctypes.c_int]*2 + [ctypes.c_void_p])
        self.entry.restype = ctypes.c_int
        if waves not in (2,4) or reuse not in (0,1):
            raise ValueError('Unsupported launch configuration')
        self.waves, self.reuse = waves,reuse
        self.pack = pack

    def __call__(self, scores, biases):
        if (torch.is_grad_enabled() or not scores.is_cuda
                or scores.dtype != torch.float32 or not scores.is_contiguous()
                or not compatible(scores.shape,scores.device,biases)):
            raise ValueError('Unsupported experimental softmax input')
        mask,bias = biases
        if self.pack:
            bias = bias.contiguous()
        output = torch.empty_like(scores)
        with torch.cuda.device(scores.device):
            status = self.entry(scores.data_ptr(),mask.data_ptr(),bias.data_ptr(),
                output.data_ptr(),scores.shape[0],scores.shape[-1],
                mask.stride(0) if mask.shape[0] > 1 else 0, mask.stride(-1),
                *bias.stride()[-3:],self.waves,self.reuse,
                torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f'Experimental HIP launch failed: {status}')
        return output
