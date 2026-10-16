"""Experimental FFN plan creation delayed until the first matching projection."""
from contextlib import contextmanager
import ctypes
import os
import time
import torch
import torch.nn.functional as F


class FixedProjection:
    def __init__(self, path, batch_size):
        if batch_size not in (1,4): raise ValueError('Only B1/B4 are qualified for screening')
        self.batch_size=batch_size
        self.path=path
        self.plan=ctypes.c_void_p()

    def _initialize(self):
        begin=time.perf_counter()
        batch_size=self.batch_size
        self.lib=ctypes.CDLL(str(self.path))
        self.lib.esmc_lt_create.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_size_t)]
        self.lib.esmc_lt_create.restype=ctypes.c_int
        self.lib.esmc_lt_run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_size_t,ctypes.c_void_p]
        self.lib.esmc_lt_run.restype=ctypes.c_int
        self.lib.esmc_lt_destroy.argtypes=[ctypes.c_void_p];self.lib.esmc_lt_destroy.restype=None
        self.plan=ctypes.c_void_p();size=ctypes.c_size_t()
        rc=self.lib.esmc_lt_create(batch_size*1002,ctypes.byref(self.plan),ctypes.byref(size))
        if rc: raise RuntimeError(f'Fixed hipBLASLt plan rejected: {rc}')
        try:self.workspace=torch.empty(size.value,device='cuda:0',dtype=torch.uint8)
        except BaseException:self.close();raise
        self.initialization_host_seconds=time.perf_counter()-begin

    def close(self):
        if self.plan.value:
            self.lib.esmc_lt_destroy(self.plan);self.plan=ctypes.c_void_p()

    @contextmanager
    def scope(self,check_each=False):
        native=F.linear;state=dict(calls=0,workspace_bytes=self.workspace.numel() if self.plan.value else 0,
                                 plan_created=False,initialization_host_seconds=0.0)
        def linear(x,w,bias=None):
            if tuple(w.shape)!=(13824,2560):return native(x,w,bias)
            if (tuple(x.shape)!=(self.batch_size,1002,2560) or bias is not None
                    or x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16
                    or not x.is_contiguous() or not w.is_contiguous()
                    or x.device!=torch.device('cuda:0') or w.device!=x.device
                    or torch.is_grad_enabled()):
                raise ValueError('Unexpected ESMC FFN projection contract')
            if not self.plan.value:
                self._initialize()
                state['plan_created']=True
                state['initialization_host_seconds']=self.initialization_host_seconds
                state['workspace_bytes']=self.workspace.numel()
            out=torch.empty((*x.shape[:-1],13824),device=x.device,dtype=x.dtype)
            rc=self.lib.esmc_lt_run(self.plan,x.data_ptr(),w.data_ptr(),out.data_ptr(),
                self.workspace.data_ptr(),self.workspace.numel(),torch.cuda.current_stream().cuda_stream)
            if rc:raise RuntimeError(f'Fixed hipBLASLt projection failed: {rc}')
            if check_each:
                reference=native(x,w,bias)
                if not torch.equal(reference.view(torch.uint8),out.view(torch.uint8)):
                    raise RuntimeError(f'Projection {state["calls"]} changed output bits')
            state['calls']+=1
            return out
        F.linear=linear
        try:
            yield state
            if state['calls']!=80:raise RuntimeError(f'Expected 80 FFN projections: {state}')
        finally:F.linear=native


_plans = {}


@contextmanager
def fixed_gemm_scope(request, batch_size):
    flag=os.environ.get('FIREPROT_ESMC_FIXED_GEMM','0')
    if flag=='0':yield None;return
    if flag!='1':raise ValueError('FIREPROT_ESMC_FIXED_GEMM must be 0 or 1')
    if (request['backend']!='fireprot' or request['workload']['model_name']!='esmc-fold2-6b'
            or request['workload']['precision']!='native' or batch_size not in (1,4)):
        raise ValueError('Fixed ESMC GEMMs require native N1000 B1/B4')
    resource=request['resources']['hip_esmc_lt']
    # Rust verifies the shared library pin before this worker starts.
    key=(resource['path'],resource['sha256'],batch_size)
    if key not in _plans:_plans[key]=FixedProjection(resource['path'],batch_size)
    with _plans[key].scope() as state:
        state['library']=resource
        state['algorithm']=4335 if batch_size==1 else 4139
        state['online_tuning']=False
        yield state
