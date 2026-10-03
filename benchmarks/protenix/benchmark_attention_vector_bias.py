"""N1000 CK bias vectorization probe with unchanged padded-tail semantics."""
from __future__ import annotations
import argparse
import ctypes
import gc
import json
import os
import pathlib
import statistics
import subprocess
from benchmarks.protenix.profile_inference import REPO_ROOT, _rocm_preflight, _additional_device_preflight, _GpuProcessMonitor, _sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=pathlib.Path)
    args=parser.parse_args()
    visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select GPU 6 or 7')
    source=REPO_ROOT/'benchmarks/protenix/csrc/ck_attention_vector_bias.cpp'
    build=REPO_ROOT/'runtime/ck_vector_bias';build.mkdir(parents=True,exist_ok=True)
    builds=[]
    for n in (64,32):
        library=build/f'n{n}.so'
        command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a','-I/opt/rocm/include',f'-DBN={n}',str(source),'-o',str(library)]
        with (build/f'n{n}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        builds.append(dict(n=n,library=str(library),library_sha256=_sha256(library),command=command))
    import torch
    from fireprot.backends.amd import ck_attention as ck
    # Preserve the original padded control after production promotion.
    ck._vector1000_enabled = False
    if ck.load_extension() is None:raise RuntimeError(str(ck.extension_status()))
    policy=dict(device_policy='exclusive')
    preflight=_rocm_preflight(policy)
    if not preflight['exclusive_check']['passed']:raise RuntimeError('GPU unavailable')
    extra=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1.0) for d in (int(visible),7 if visible=='6' else 6)]
    for m in monitors:m.start()
    report=dict(scope=__doc__,shape=[1000,4,1000,32],dtype='float32',tf32=False,seed=907,warmup_groups=5,measured_groups=20,cache_state='warm; compilation excluded',physical_device=int(visible),allocator=os.environ.get('PYTORCH_ALLOC_CONF'),builds=builds,preflight=preflight,sibling_preflight=extra,baseline_build=ck.extension_status(),source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in (source,REPO_ROOT/'fireprot/backends/amd/csrc/fireprot_ck_attention.cpp',pathlib.Path(__file__))},records=[])
    handles=[]
    try:
        if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':raise RuntimeError('Requires gfx90a HIP')
        torch.manual_seed(907);torch.backends.cuda.matmul.allow_tf32=False
        report.update(torch=torch.__version__,hip=torch.version.hip)
        with torch.inference_mode():
            q,k,v=[torch.randn(1000,1000,4,32,device='cuda').transpose(1,2) for _ in range(3)];q.div_(32**0.5)
            bias=torch.randn(1000,1000,4,device='cuda').permute(2,0,1).unsqueeze(0)
            kwargs=dict(module=None,q=q,k=k,v=v,biases=[bias],native=lambda:None)
            from fireprot.backends.amd.triangle_attention import _direct_sdpa
            reference=_direct_sdpa(**kwargs)
            functions=[('ck_baseline_before',lambda:ck.attention(**kwargs))]
            for item in builds:
                handle=ctypes.CDLL(item['library']);handles.append(handle)
                entry=handle.ck_attention_vector_bias;entry.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p];entry.restype=ctypes.c_int
                assert entry(None,None,None,None,None,1000,1001,None)==2
                def call(entry=entry,guard=False):
                    packed=bias.contiguous()
                    storage=torch.full((q.numel()+8192,),123.5,device='cuda') if guard else None
                    output=storage[4096:-4096].view(1000,1000,4,32).transpose(1,2) if guard else torch.empty_like(q)
                    status=entry(q.data_ptr(),k.data_ptr(),v.data_ptr(),packed.data_ptr(),output.data_ptr(),1000,1000,torch.cuda.current_stream().cuda_stream)
                    if status:raise RuntimeError(f'CK launch status {status}')
                    if guard:
                        assert bool((storage[:4096]==123.5).all()) and bool((storage[-4096:]==123.5).all())
                    return output.transpose(1,2)
                actual=call(guard=True);torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5);del actual
                saved=bias.clone();bias[:,:,0,:]=-float('inf');bias[:,:,1,0]=80
                torch.testing.assert_close(call(guard=True),_direct_sdpa(**kwargs),atol=2e-5,rtol=2e-5)
                bias.copy_(saved);del saved
                functions.append((f'vector_bias_n{item["n"]}',call))
            functions.append(('ck_baseline_after',lambda:ck.attention(**kwargs)))
            for group in (1,8):
                for name,call in functions:
                    result=call();torch.testing.assert_close(result,reference,atol=2e-5,rtol=2e-5)
                    delta=(result-reference).abs().max().item();del result
                    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    values=[]
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];start.record()
                        for _ in range(group):result=call();del result
                        end.record();end.synchronize()
                        if i>=5:values.append(start.elapsed_time(end)/group)
                    row=dict(implementation=name,group_size=group,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
        report['status']='ok'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
