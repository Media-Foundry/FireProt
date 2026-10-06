"""Experimental graph-aware dual storage; derived from maintained dual_storage.py at 0881f0a."""
from contextlib import contextmanager
import ctypes
import json
from pathlib import Path
import torch
from fireprot.backends.amd._published import select_c


@contextmanager
def install(model, output, contract, mode="on"):
    if mode not in ('on','off'):raise ValueError('Unknown dual reuse mode')
    if mode=='off':
        yield
        return
    if model.training or not torch.version.hip or not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError('Strict HIP inference required')
    from fireprot.backends.amd.extension import load_amd_extension
    from fireprot.backends.amd import lt_bmm
    ext=load_amd_extension()
    selected=select_c('fireprot_base_dual_tiles','FIREPROT_BASE_DUAL_TILES','base-dual-tiles-v1')
    if ext is None or selected is None:raise RuntimeError('Published kernels required')
    library,receipt=selected
    entry=library.base_dual_tile
    entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_void_p]
    entry.restype=ctypes.c_int
    original_project,original_bmm=ext.dual_projection_tiled,lt_bmm.packed_bmm
    storage=None
    views=None
    phase=0
    owner=None
    input_key=None
    records=dict(mode=mode,projection_calls=0,consumed_pairs=0,allocations=0,error=None,published=receipt)

    def stream_key(tensor):
        return (tensor.device.index,torch.cuda.current_stream(tensor.device).cuda_stream)

    def project(x,wg,wp):
        nonlocal storage,views,phase,owner,input_key
        if (torch.is_grad_enabled() or not torch.utils.deterministic.fill_uninitialized_memory
                or not x.is_cuda or x.dtype!=torch.float32 or tuple(x.shape)!=(1000,1000,128)
                or not x.is_contiguous() or x.requires_grad or x.is_conj() or x.is_neg()
                or torch.backends.cuda.matmul.allow_tf32
                or torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0]!='gfx90a'):
            raise RuntimeError('Dual reuse input outside qualified contract')
        for weight in (wg,wp):
            if (weight.device!=x.device or weight.dtype!=x.dtype or tuple(weight.shape)!=(128,128)
                    or not weight.is_contiguous() or weight.is_conj() or weight.is_neg()):
                raise RuntimeError('Dual reuse weight outside qualified contract')
        if any(t.data_ptr()%16 for t in (x,wg,wp)):
            raise RuntimeError('Dual reuse alignment mismatch')
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing() != contract.capturing:
                raise RuntimeError('Capture outside declared trunk partition')
            current=stream_key(x)
            if storage is None:
                owner=current
                storage=torch.empty((2,128,1000128),device=x.device,dtype=x.dtype)
                views=tuple(storage[i].as_strided((128,1000,1000),(1000128,1000,1)) for i in range(2))
                records['allocations']+=1
            if current!=owner or phase>=2:raise RuntimeError('Unsafe stream change or unconsumed projections')
            key=(x.data_ptr(),x._version)
            if phase==0:input_key=key
            elif key!=input_key:raise RuntimeError('A/B normalization input mismatch')
            result=views[phase]
            status=entry(x.data_ptr(),wg.data_ptr(),wp.data_ptr(),result.data_ptr(),1000000,1000128,0,current[1])
            if status:raise RuntimeError(f'Dual reuse launch failed {status}')
            phase+=1
            records['projection_calls']+=1
            return result

    def bmm(a,b,outgoing):
        nonlocal phase,input_key
        if phase!=2 or a is not views[0] or b is not views[1] or stream_key(a)!=owner:
            raise RuntimeError('Contraction does not consume the retained A/B pair')
        result=original_bmm(a,b,outgoing)
        # Same stream: consumption is enqueued before the next projection overwrites storage.
        phase=0
        input_key=None
        records['consumed_pairs']+=1
        return result

    def replay_guard():
        if phase != 0 or storage is None or stream_key(storage) != owner:
            raise RuntimeError('Graph replay changed dual-storage ownership or consumption')
        if any(views[i].data_ptr() != storage[i].data_ptr() for i in range(2)):
            raise RuntimeError('Graph replay changed dual-storage addresses')

    contract.guards.append(replay_guard)
    ext.dual_projection_tiled=project
    lt_bmm.packed_bmm=bmm
    try:
        yield
        contract.validate_completed()
        expected = contract.python_contractions
        records['logical_consumed_pairs'] = contract.logical_contractions
        if phase!=0 or records['projection_calls']!=2*expected or records['consumed_pairs']!=expected:
            raise RuntimeError('Incomplete dual reuse coverage')
    except BaseException as error:
        records['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        ext.dual_projection_tiled=original_project
        lt_bmm.packed_bmm=original_bmm
        Path(output,'base-dual-reuse.json').write_text(json.dumps(records,indent=2)+'\n')
