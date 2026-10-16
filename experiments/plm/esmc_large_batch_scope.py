"""Experimental N1000 ESMC B8/B16 scope; native GEMM and attention stay intact."""
from contextlib import contextmanager
import os
import torch


@contextmanager
def large_batch_scope(model, request, length, batch_size):
    if (request['workload']['model_name']!='esmc-fold2-6b' or not request['workload'].get('batch')
            or length!=1000 or batch_size not in (8,16) or model.training or torch.is_grad_enabled()
            or len(model.transformer.blocks)!=80):
        raise ValueError('Large-batch prototype requires ESMC N1000 B8/B16 inference')
    if str(torch.__version__)!='2.12.0a0+git78258b9' or torch.version.hip!='7.14.60850':
        raise ValueError('Unqualified large-batch Torch/HIP runtime')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':
        raise ValueError('Large-batch prototype requires gfx90a')
    if os.environ.get('FIREPROT_ESMC_FIXED_GEMM')!='0':
        raise ValueError('B4 fixed GEMM must be explicitly disabled for larger groups')
    from .esm import esm_scope
    from .pointwise import Pointwise
    resource=request['resources']['hip_esmc_large_batch']
    selected=dict(request,resources=dict(request['resources'],hip_esm=resource))
    pointwise=Pointwise(resource['path'],4)
    with pointwise.scope(model,'residual') as residual,esm_scope(model,selected,batch_size) as state:
        state['large_batch_prototype']=True
        state['fixed_gemm_used']=False
        state['pointwise']=residual
        residual['library']=resource;residual['vector_swiglu']=8
        yield state
