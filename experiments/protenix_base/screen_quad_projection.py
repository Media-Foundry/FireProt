"""Synthetic N1000 fused A/B projection screen against the published maintained kernel."""
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import torch
import torch.utils.deterministic
from fireprot.backends.amd._native_hip import load
from fireprot.backends.amd.extension import load_amd_extension

TILES = [(128,128,8,8),(128,128,16,8),(64,128,8,4),(64,128,16,4),(128,64,16,8)]


def main(destination):
    if os.environ.get('HIP_VISIBLE_DEVICES') not in ('6','7'):
        raise RuntimeError('HIP6/7 only')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':
        raise RuntimeError('gfx90a required')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(101)
    lib,receipt=load('fireprot_base_quad_projection',Path(__file__).with_name('quad_projection.cpp'),Path('/opt/rocm'),
                     extra_files=[__file__],versions={'c_abi':'base-quad-v1'})
    dest=Path(destination)
    dest.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    entry=lib.base_quad
    entry.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*3+[ctypes.c_void_p]
    entry.restype=ctypes.c_int
    ext=load_amd_extension()
    if ext is None:raise RuntimeError('Maintained reference unavailable')
    checks=[]
    with torch.inference_mode():
        x=torch.randn((1000,1000,128),device='cuda',dtype=torch.float32)
        wg,wp,bg,bp=[torch.randn((128,128),device='cuda',dtype=torch.float32)/128**0.5 for _ in range(4)]
        def native():return (ext.dual_projection_tiled(x,wg,wp),ext.dual_projection_tiled(x,bg,bp))
        def candidate(variant):
            storage=torch.empty((2,128,1000128),device='cuda',dtype=torch.float32)
            status=entry(x.data_ptr(),wg.data_ptr(),wp.data_ptr(),bg.data_ptr(),bp.data_ptr(),storage[0].data_ptr(),storage[1].data_ptr(),1000000,1000128,variant,torch.cuda.current_stream().cuda_stream)
            if status:raise RuntimeError(f'Kernel error {status}')
            return tuple(storage[i].as_strided((128,1000,1000),(1000128,1000,1)) for i in range(2))
        for scale in (1.,8.):
            if scale!=1.:
                wg.mul_(scale);bg.mul_(scale)
            reference=native()
            for variant,tile in enumerate(TILES):
                result=candidate(variant);torch.cuda.synchronize()
                exact=all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(reference,result))
                checks.append(dict(tile=tile,gate_scale=scale,bitwise_equal=exact,max_abs_delta=max((a-b).abs().max().item() for a,b in zip(reference,result))))
                if not exact:raise RuntimeError(f'Exact rejection {tile} scale {scale}')
                del result
            del reference
        wg.div_(8.);bg.div_(8.)
        functions=[('maintained',native)]+[(str(i),lambda i=i:candidate(i)) for i in range(len(TILES))]
        timings={name:[] for name,_ in functions}
        for iteration in range(20):
            order=functions if iteration%2==0 else list(reversed(functions))
            for name,function in order:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();result=function();b.record();b.synchronize()
                if iteration>=4:timings[name].append(a.elapsed_time(b))
                del result
    report=dict(scope='Synthetic random FP32 N1000 input; exact maintained-reference gate; not E2E or trained qualification',
                seed=101,hip_device=os.environ['HIP_VISIBLE_DEVICES'],tiles=TILES,checks=checks,
                allocation_policy='Fresh output allocation and deterministic fill included; input and weights reused',
                results=[dict(name=name,median_ms=statistics.median(values),values_ms=values) for name,values in timings.items()],
                library=receipt['library'],library_sha256=receipt['library_sha256'])
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main(sys.argv[1])
