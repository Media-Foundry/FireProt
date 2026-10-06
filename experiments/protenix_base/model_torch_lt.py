"""Isolated per-contraction Torch hipBLASLt preference; all other BLAS unchanged."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import torch


@contextmanager
def install(model, output):
    mode=os.environ.get('FIREPROT_BASE_TORCH_LT_MODE','off')
    if mode not in ('off','on'):raise ValueError('Unknown Torch Lt mode')
    if mode=='off':
        yield
        return
    if (model.training or not torch.version.hip
            or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a'
            or not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError('Strict gfx90a inference required')
    from fireprot.backends.amd import lt_bmm
    original=lt_bmm.packed_bmm
    records=dict(mode=mode,incoming=0,outgoing=0,fallback=0,error=None,
                 preference_scope='Only synchronous BMM dispatch; restore before returning')
    def call(a,b,outgoing):
        supported=(not torch.is_grad_enabled() and torch.utils.deterministic.fill_uninitialized_memory
            and a.device==b.device and a.is_cuda
            and all(x.dtype==torch.float32 and tuple(x.shape)==(128,1000,1000)
                    and tuple(x.stride())==(1000128,1000,1) and not x.requires_grad
                    and not x.is_conj() and not x.is_neg() for x in (a,b))
            and not torch.backends.cuda.matmul.allow_tf32
            and not torch.cuda.is_current_stream_capturing())
        if not supported:
            records['fallback']+=1
            return original(a,b,outgoing)
        previous=torch.backends.cuda.preferred_blas_library()
        try:
            torch.backends.cuda.preferred_blas_library('hipblaslt')
            result=torch.bmm(a,b.transpose(-1,-2)) if outgoing else torch.bmm(a.transpose(-1,-2),b)
        finally:torch.backends.cuda.preferred_blas_library(previous)
        records['outgoing' if outgoing else 'incoming']+=1
        return result
    lt_bmm.packed_bmm=call
    try:
        yield
        if records['incoming']!=524 or records['outgoing']!=524 or records['fallback']!=0:
            raise RuntimeError('Torch Lt did not cover expected Base contractions')
    except BaseException as error:
        records['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        lt_bmm.packed_bmm=original
        Path(output,'base-torch-lt.json').write_text(json.dumps(records,indent=2)+'\n')
