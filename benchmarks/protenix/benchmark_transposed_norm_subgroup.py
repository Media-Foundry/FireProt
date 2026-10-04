"""Tune 32-lane pair rows against the current production transposed LayerNorm."""
import argparse, ctypes, hashlib, json, os, pathlib, statistics, subprocess
from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _rocm_preflight, _additional_device_preflight, _GpuProcessMonitor, _sha256,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select physical GPU 6 or 7')
    import torch
    policy={'device_policy':'exclusive'}
    preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    sibling=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for m in monitors:m.start()
    source=REPO_ROOT/'benchmarks/protenix/csrc/transposed_norm_subgroup_probe.cpp'
    report=dict(status='running',seed=1043,dtype='float32',tf32=False,warmups=5,repetitions=20,
                scope='Standalone strided pair LayerNorm; no model predictions',
                cycle_step_samples='Not applicable',cache_state='warm; allocation included, compile excluded',
                preflight=preflight,sibling_preflight=sibling,torch=torch.__version__,hip=torch.version.hip,
                source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in (source,pathlib.Path(__file__))},
                records=[],validation=[])
    try:
        from fireprot.backends.amd.extension import load_amd_extension
        ext=load_amd_extension();assert ext is not None
        report['production_source_sha256']=_sha256(REPO_ROOT/'kernels/amd/fireprot_amd_ops.cu')
        compiler=subprocess.check_output(['/opt/rocm/bin/hipcc','--version'],text=True)
        directory=REPO_ROOT/'runtime/transposed_norm_subgroup'/hashlib.sha256(source.read_bytes()+compiler.encode()).hexdigest()[:16]
        directory.mkdir(parents=True,exist_ok=True);lib=directory/'probe.so'
        command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',str(source),'-o',str(lib)]
        if not lib.exists():subprocess.run(command,check=True)
        report.update(compiler=compiler,command=command,library_sha256=_sha256(lib))
        handle=ctypes.CDLL(str(lib));entry=handle.transposed_norm
        entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_void_p];entry.restype=ctypes.c_int
        assert entry(None,None,None,None,999,4,1e-5,None)==2
        assert entry(None,None,None,None,1000,4,1e-5,None)==3
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        torch.manual_seed(1043);torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
            for n in (1000,1536):
                storage=torch.randn(n,n,128,device='cuda');x=storage.transpose(0,1)
                weight=torch.randn(128,device='cuda');bias=torch.randn(128,device='cuda')
                def native():return torch.nn.functional.layer_norm(x,(128,),weight,bias,1e-5)
                def production():return ext.transposed_layer_norm(x,weight,bias,1e-5)
                def candidate(waves,guard=False):
                    buf=torch.full((x.numel()+64,),123.5,device='cuda') if guard else torch.empty(x.numel(),device='cuda')
                    output=(buf[32:-32] if guard else buf).view(n,n,128)
                    status=entry(x.data_ptr(),weight.data_ptr(),bias.data_ptr(),output.data_ptr(),n,waves,1e-5,torch.cuda.current_stream().cuda_stream)
                    if status:raise RuntimeError(f'HIP status {status}')
                    if guard:assert bool((buf[:32]==123.5).all()) and bool((buf[-32:]==123.5).all())
                    return output
                variants=(4,8,84,88,164,168,324,328)
                accepted={v:True for v in variants}
                for case in ('normal','edge'):
                    if case=='edge':
                        x[0,0]=1000.;x[0,1]=1.+torch.arange(128,device='cuda')*1e-6
                        x[0,2]=1000.+torch.arange(128,device='cuda')*.001
                        x[0,3]=0.;x[0,4]=-1000.
                    expected=native()
                    for waves in variants:
                        actual=candidate(waves,True)
                        row=dict(n=n,case=case,waves=waves,max_abs_delta=(actual-expected).abs().max().item())
                        try:torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5);row['passed']=True
                        except AssertionError as error:row.update(passed=False,error=str(error));accepted[waves]=False
                        report['validation'].append(row);print(json.dumps(row),flush=True);del actual
                    del expected
                functions=[('production_before',production),('native_before',native),('linear_before',lambda:candidate(8))]+[(f'variant{w}',lambda w=w:candidate(w)) for w in variants if accepted[w]]+[('linear_after',lambda:candidate(8)),('native_after',native),('production_after',production)]
                for name,fn in functions:
                    torch.cuda.reset_peak_memory_stats();values=[]
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();out=fn();end.record();end.synchronize();del out
                        if i>=5:values.append(start.elapsed_time(end))
                    row=dict(n=n,input_stride=list(x.stride()),implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                del storage,x,weight,bias
                torch.cuda.empty_cache()
        report['status']='ok' if all(r['passed'] for r in report['validation']) else 'completed_with_rejections'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status'] in ('ok','completed_with_rejections') else 1


if __name__=='__main__':raise SystemExit(main())
