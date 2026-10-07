import ctypes
import torch
class Gate:
 def __init__(self,path):
  self.library=ctypes.CDLL(path);self.entry=self.library.fireprot_gate_pack
  self.entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_int64]*2+[ctypes.c_void_p];self.entry.restype=ctypes.c_int;self.calls=0
 def __call__(self,g,p,m):
  assert g.shape==p.shape and g.ndim==4 and g.shape[0]==1 and g.shape[-1]==128
  assert g.is_cuda and p.device==g.device and m.device==g.device and g.is_contiguous() and p.is_contiguous()
  assert g.dtype==p.dtype==m.dtype==torch.float32 and m.shape==g.shape[:-1]+(1,)
  out=torch.empty((1,128,g.shape[1],g.shape[2]),device=g.device,dtype=g.dtype)
  rc=self.entry(g.data_ptr(),p.data_ptr(),m.data_ptr(),out.data_ptr(),g.shape[1],g.shape[2],m.stride(1),m.stride(2),torch.cuda.current_stream().cuda_stream)
  if rc:raise RuntimeError(rc)
  self.calls+=1;return out.permute(0,2,3,1)
