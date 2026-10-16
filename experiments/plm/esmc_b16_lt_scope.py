"""Experimental lazy direct FFN plan for the qualified B16 signature."""
from contextlib import contextmanager
import ctypes
import os
from esmc_lt_lazy_scope import FixedProjection


class B16Projection(FixedProjection):
    def __init__(self,path):
        self.batch_size=16
        self.path=path
        self.plan=ctypes.c_void_p()


_plans={}


@contextmanager
def b16_gemm_scope(request,batch_size):
    flag=os.environ.get('FIREPROT_ESMC_B16_FFN','0')
    if flag=='0':
        yield None
        return
    if (flag!='1' or batch_size!=16 or request['backend']!='fireprot'
            or request['workload']['model_name']!='esmc-fold2-6b'
            or request['workload']['precision']!='native'):
        raise ValueError('B16 fixed FFN requires ESMC native precision and explicit flag 0/1')
    resource=request['resources']['hip_esmc_b16_lt'];key=(resource['path'],resource['sha256'])
    if key not in _plans:_plans[key]=B16Projection(resource['path'])
    with _plans[key].scope() as state:
        state['library']=resource;state['online_tuning']=False
        yield state
