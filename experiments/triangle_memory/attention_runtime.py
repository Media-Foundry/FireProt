"""Narrow attention key layout experiment; arithmetic remains unchanged."""
import ctypes
import torch


class KeyPacker:
    def __init__(self, library):
        self.library=ctypes.CDLL(str(library))
        self.fn=self.library.fireprot_triangle_pack_key
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]+[ctypes.c_int64]*3+[ctypes.c_void_p]
        self.fn.restype=ctypes.c_int
        self.calls=0

    def __call__(self,key):
        if not (type(key) is torch.Tensor and key.is_cuda and key.dtype==torch.float32
            and tuple(key.shape)==(128,4,1536,32)
            and tuple(key.stride())==(196608,32,128,1)
            and not torch.is_grad_enabled() and not key.requires_grad
            and not key.is_neg() and not key.is_conj()):
            return key.transpose(-1,-2).reshape(key.shape[0]*4,32,1536)
        with torch.cuda.device(key.device):
            out=torch.empty((512,32,1536),device=key.device,dtype=key.dtype)
            stream=torch.cuda.current_stream(key.device)
            rc=self.fn(key.data_ptr(),out.data_ptr(),128,*key.stride()[:3],stream.cuda_stream)
            if rc:raise RuntimeError(f'Attention copy launch failed: {rc}')
            key.record_stream(stream);out.record_stream(stream)
        self.calls+=1
        return out
