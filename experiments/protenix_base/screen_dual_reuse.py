"""Screen retained projection storage with deterministic initialization still enabled."""
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import torch
import torch.utils.deterministic
from fireprot.backends.amd.extension import load_amd_extension
from fireprot.backends.amd._published import select_c


def main(destination):
    if os.environ.get('HIP_VISIBLE_DEVICES') not in ('6','7'):
        raise RuntimeError('HIP6/7 only')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(101)
    selected=select_c('fireprot_base_dual_tiles','FIREPROT_BASE_DUAL_TILES','base-dual-tiles-v1')
    if selected is None:raise RuntimeError('Published dual tile bundle required')
    lib,receipt=selected
    entry=lib.base_dual_tile
    entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_void_p]
    entry.restype=ctypes.c_int
    ext=load_amd_extension()
    if ext is None:raise RuntimeError('Reference extension unavailable')
    with torch.inference_mode():
        x=torch.randn((1000,1000,128),device='cuda')
        wg,wp=[torch.randn((128,128),device='cuda')/128**0.5 for _ in range(2)]
        storage=torch.empty((128,1000128),device='cuda')
        def retained():
            status=entry(x.data_ptr(),wg.data_ptr(),wp.data_ptr(),storage.data_ptr(),1000000,1000128,0,torch.cuda.current_stream().cuda_stream)
            if status:raise RuntimeError(f'Launch failure {status}')
            return storage.as_strided((128,1000,1000),(1000128,1000,1))
        def native():return ext.dual_projection_tiled(x,wg,wp)
        checks=[]
        for iteration in range(3):
            if iteration:x.add_(0.125)
            reference=native();result=retained();torch.cuda.synchronize()
            exact=torch.equal(reference.view(torch.int32),result.view(torch.int32))
            checks.append(dict(input_update=iteration,bitwise_equal=exact))
            if not exact:raise RuntimeError('Full overwrite check failed')
            del reference,result
        times={'maintained':[],'retained':[]}
        functions=[('maintained',native),('retained',retained)]
        for iteration in range(24):
            for name,fn in (functions if iteration%2==0 else list(reversed(functions))):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();result=fn();b.record();b.synchronize()
                if iteration>=4:times[name].append(a.elapsed_time(b))
                del result
    report=dict(scope='Synthetic steady-state retained output versus maintained allocation; not E2E; initial retained allocation excluded',
                dtype='float32',shape=[1000,1000,128],seed=101,hip_device=os.environ['HIP_VISIBLE_DEVICES'],
                deterministic_fill=True,checks=checks,published=receipt,
                results={name:dict(median_ms=statistics.median(v),values_ms=v) for name,v in times.items()})
    Path(destination).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main(sys.argv[1])
