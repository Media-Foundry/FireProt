"""Screen Lt algorithms for exact padded Base triangle inputs; no model claim."""
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import torch
import torch.utils.deterministic
from fireprot.backends.amd._native_hip import load


def main(destination):
    if os.environ.get('HIP_VISIBLE_DEVICES') not in ('6','7'):
        raise RuntimeError('HIP6/7 only')
    if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':
        raise RuntimeError('gfx90a required')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.preferred_blas_library('hipblas')
    torch.manual_seed(101)
    root=Path('/opt/rocm')
    lib,receipt=load('fireprot_base_lt_padded',Path(__file__).with_name('lt_padded.cpp'),root,
        link_flags=[f'-L{root}/lib',f'-Wl,-rpath,{root}/lib','-lhipblaslt'],
        libraries=[root/'lib/libhipblaslt.so'],extra_files=[__file__],versions={'c_abi':'base-lt-padded-v1'})
    dest=Path(destination);dest.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    count=lib.lt_base_count;count.argtypes=[ctypes.c_int];count.restype=ctypes.c_int
    entry=lib.lt_base_bmm;entry.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]*2+[ctypes.c_size_t];entry.restype=ctypes.c_int
    records=[]
    with torch.inference_mode():
        a,b=[torch.empty_strided((128,1000,1000),(1000128,1000,1),device='cuda',dtype=torch.float32) for _ in range(2)]
        for x in (a,b):x.normal_().div_(1000**0.5)
        workspace=torch.empty(32*1024*1024,device='cuda',dtype=torch.uint8)
        for outgoing in (0,1):
            left,right=(a,b.transpose(-1,-2)) if outgoing else (a.transpose(-1,-2),b)
            def native():return torch.bmm(left,right)
            ref=native()
            n=count(outgoing)
            if n<0:raise RuntimeError('Lt initialization failed')
            functions=[('hipblas',native)]
            for algorithm in range(n):
                def call(algorithm=algorithm):
                    out=torch.empty((128,1000,1000),device='cuda',dtype=torch.float32)
                    status=entry(a.data_ptr(),b.data_ptr(),out.data_ptr(),outgoing,algorithm,torch.cuda.current_stream().cuda_stream,workspace.data_ptr(),workspace.numel())
                    if status:raise RuntimeError(f'Lt launch failed {status}')
                    return out
                out=call();torch.cuda.synchronize()
                delta=(out-ref).abs().max().item()
                try:torch.testing.assert_close(out,ref,atol=2e-5,rtol=2e-5)
                except AssertionError:
                    records.append(dict(outgoing=outgoing,algorithm=algorithm,status='numerical-rejection',max_abs_delta=delta));del out;continue
                records.append(dict(outgoing=outgoing,algorithm=algorithm,status='numerics-passed',max_abs_delta=delta,bitwise_equal=torch.equal(out.view(torch.int32),ref.view(torch.int32))))
                del out;functions.append((str(algorithm),call))
            del ref
            times={name:[] for name,_ in functions}
            for iteration in range(16):
                order=functions if iteration%2==0 else list(reversed(functions))
                for name,function in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();out=function();end.record();end.synchronize()
                    if iteration>=4:times[name].append(start.elapsed_time(end))
                    del out
            records.append(dict(outgoing=outgoing,status='timing',heuristic_count=n,results=[dict(name=name,median_ms=statistics.median(v),values_ms=v) for name,v in times.items()]))
    report=dict(scope='Synthetic FP32 N1000 B128 strided inputs; not trained model/E2E qualification',input_stride=[1000128,1000,1],seed=101,hip_device=os.environ['HIP_VISIBLE_DEVICES'],records=records,library=receipt['library'],library_sha256=receipt['library_sha256'],workspace_bytes=32*1024*1024,allocation_policy='Output allocation/fill included; reusable 32MiB workspace excluded')
    dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main(sys.argv[1])
