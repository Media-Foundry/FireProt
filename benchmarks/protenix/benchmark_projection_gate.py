"""Compare current packing, transposed BLAS, and CK fused triangle projections."""
import argparse,ctypes,json,os,pathlib,statistics,subprocess
from benchmarks.protenix.profile_inference import REPO_ROOT,_rocm_preflight,_additional_device_preflight,_GpuProcessMonitor,_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select GPU 6 or 7')
    source=REPO_ROOT/'benchmarks/protenix/csrc/ck_projection_gate_probe.cpp'
    build=REPO_ROOT/'runtime/ck_projection_gate';build.mkdir(parents=True,exist_ok=True)
    builds=[]
    for m,n in [(64,64),(64,128),(128,64),(128,128)]:
        lib=build/f'm{m}n{n}.so'
        command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a','-I/opt/rocm/include',f'-DTILE_M={m}',f'-DTILE_N={n}',str(source),'-o',str(lib)]
        with (build/f'm{m}n{n}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        builds.append(dict(m=m,n=n,library=str(lib),library_sha256=_sha256(lib),command=command))
    import torch
    policy=dict(device_policy='exclusive');preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    extra=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for monitor in monitors:monitor.start()
    report=dict(seed=983,dtype='float32',tf32=False,warmups=5,repetitions=20,cache_state='warm; compilation excluded; value projection and allocations included',preflight=preflight,sibling_preflight=extra,builds=builds,source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in [source,pathlib.Path(__file__),REPO_ROOT/'fireprot/backends/amd/csrc/fireprot_amd_ops.cu']},records=[])
    handles=[]
    try:
        from fireprot.backends.amd.extension import load_amd_extension
        from fireprot.backends.amd.blas import configure_process_blas
        configure_process_blas('hipblas')
        ext=load_amd_extension();assert ext is not None
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        torch.manual_seed(983);torch.backends.cuda.matmul.allow_tf32=False
        entries=[]
        for b in builds:
            handle=ctypes.CDLL(b['library']);handles.append(handle);entry=handle.ck_projection_gate
            entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];entry.restype=ctypes.c_int
            assert entry(None,None,None,None,999,None)==2
            entries.append((f'ck_m{b["m"]}n{b["n"]}',entry))
        with torch.inference_mode():
            for length in (1000,1536):
                x=torch.randn(length,length,128,device='cuda')
                wg=torch.randn(128,128,device='cuda')*.1;wp=torch.randn_like(wg)*.1
                xt=x.view(-1,128).T
                def baseline():return ext.triangle_vector_pack(torch.nn.functional.linear(x,wg),torch.nn.functional.linear(x,wp))
                def transposed():
                    gate=torch.mm(wg,xt);value=torch.mm(wp,xt)
                    return ext.sigmoid_mul_f32(gate,value).view(128,length,length)
                reference=baseline()
                functions=[('baseline_before',baseline),('transposed_blas',transposed)]
                for name,entry in entries:
                    def fused(entry=entry):
                        value=torch.mm(wp,xt);output=torch.empty_like(value)
                        status=entry(wg.data_ptr(),x.data_ptr(),value.data_ptr(),output.data_ptr(),length,torch.cuda.current_stream().cuda_stream)
                        if status:raise RuntimeError(f'CK launch status {status}')
                        return output.view(128,length,length)
                    functions.append((name,fused))
                functions.append(('baseline_after',baseline))
                for name,fn in functions:
                    actual=fn();torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                    delta=(actual-reference).abs().max().item();del actual
                    values=[];torch.cuda.reset_peak_memory_stats()
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();out=fn();end.record();end.synchronize();del out
                        if i>=5:values.append(start.elapsed_time(end))
                    row=dict(length=length,implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                del x,wg,wp,xt,reference
                torch.cuda.empty_cache()
        report['status']='ok'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
