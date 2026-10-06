"""Isolated Base N1000 Lt contraction adapter; Rust owns published acceptance."""
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import torch
from fireprot.backends.amd._published import select_c


@contextmanager
def install(model, output):
    mode=os.environ.get('FIREPROT_BASE_LT_PADDED_MODE','off')
    if mode not in ('on','off'):raise ValueError('Unknown Lt mode')
    if mode=='off':
        yield
        return
    if model.training or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Strict inference required')
    from fireprot.backends.amd import lt_bmm
    original=lt_bmm.packed_bmm
    selected=select_c('fireprot_base_lt_padded','FIREPROT_BASE_LT_PADDED','base-lt-padded-v1')
    if selected is None:raise RuntimeError('Published Lt bundle required')
    library,receipt=selected
    entry=library.lt_base_bmm
    entry.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]*2+[ctypes.c_size_t];entry.restype=ctypes.c_int
    workspace=None
    records=dict(mode=mode,algorithm=0,incoming=0,outgoing=0,fallback=0,error=None,published=receipt)
    def call(a,b,outgoing):
        nonlocal workspace
        supported=(not torch.is_grad_enabled() and torch.utils.deterministic.fill_uninitialized_memory
            and a.device==b.device and a.is_cuda
            and all(x.dtype==torch.float32 and tuple(x.shape)==(128,1000,1000)
                    and tuple(x.stride())==(1000128,1000,1) and not x.requires_grad
                    and x.data_ptr()%16==0 for x in (a,b))
            and not torch.backends.cuda.matmul.allow_tf32
            and not torch.cuda.is_current_stream_capturing())
        if not supported:
            records['fallback']+=1
            return original(a,b,outgoing)
        with torch.cuda.device(a.device):
            if workspace is None:workspace=torch.empty(32*1024*1024,device=a.device,dtype=torch.uint8)
            result=torch.empty((128,1000,1000),device=a.device,dtype=torch.float32)
            status=entry(a.data_ptr(),b.data_ptr(),result.data_ptr(),int(outgoing),0,torch.cuda.current_stream(a.device).cuda_stream,workspace.data_ptr(),workspace.numel())
            if status:raise RuntimeError(f'Lt contraction failed {status}')
        records['outgoing' if outgoing else 'incoming']+=1
        return result
    lt_bmm.packed_bmm=call
    try:
        yield
        if records['incoming']!=524 or records['outgoing']!=524 or records['fallback']!=0:
            raise RuntimeError('Lt selection did not cover expected Base contractions')
    except BaseException as error:
        records['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        lt_bmm.packed_bmm=original
        Path(output,'base-lt-padded.json').write_text(json.dumps(records,indent=2)+'\n')
