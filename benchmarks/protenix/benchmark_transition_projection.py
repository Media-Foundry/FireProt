"""Compare fused HIP transition projections against current hipBLAS plus HIP SiLU."""
from __future__ import annotations
import argparse,ctypes,gc,hashlib,json,os,pathlib,statistics,subprocess
from benchmarks.protenix.profile_inference import REPO_ROOT,_additional_device_preflight,_GpuProcessMonitor,_rocm_preflight,_sha256


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select physical GPU 6 or 7')
    os.environ['LAYERNORM_TYPE']='torch'
    import torch
    from fireprot.backends.amd.blas import configure_process_blas
    from fireprot.backends.amd.extension import load_amd_extension
    policy={'device_policy':'exclusive'}
    preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    sibling=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for monitor in monitors:monitor.start()
    source=REPO_ROOT/'benchmarks/protenix/csrc/transition_projection_probe.cpp'
    report=dict(status='running',seed=1029,dtype='fp32',tf32=False,model='Protenix pair Transition shape replay; random weights',cycle_step_sample_count='not applicable to operator replay',warmups=5,repetitions=20,cache_state='warm; compilation excluded; allocations included',memory_scope='whole probe including reference tensors, not model peak',parameters='normal inputs; normal(0,0.1) weights; zero/+100/-100 input rows',preflight=preflight,sibling_preflight=sibling,torch=torch.__version__,hip=torch.version.hip,source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in (source,pathlib.Path(__file__).resolve(),REPO_ROOT/'fireprot/backends/amd/csrc/fireprot_amd_ops.cu')},builds=[],records=[],rejections=[])
    try:
        report['blas']=configure_process_blas('hipblas')
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        extension=load_amd_extension();assert extension is not None
        compiler=subprocess.check_output(['/opt/rocm/bin/hipcc','--version'],text=True);report['compiler']=compiler
        digest=hashlib.sha256(source.read_bytes()+compiler.encode()).hexdigest()[:16]
        directory=REPO_ROOT/'runtime/transition_projection'/digest;directory.mkdir(parents=True,exist_ok=True)
        candidates=[]
        for rows,channels,waves in ((64,64,8),(128,64,8),(128,128,8),(128,64,16)):
            name=f'r{rows}c{channels}w{waves}';binary=directory/(name+'.so');pending=binary.with_suffix('.pending.so')
            command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',f'-DROW_TILE={rows}',f'-DCHANNEL_TILE={channels}',f'-DWAVES={waves}',str(source),'-o',str(pending)]
            if not binary.is_file():subprocess.run(command,check=True);pending.replace(binary)
            library=ctypes.CDLL(str(binary));project=library.transition_projection;project._library_handle=library
            project.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];project.restype=ctypes.c_int
            assert project(None,None,None,None,1,None)==2
            assert project(None,None,None,None,131,None)==3
            candidates.append((name,project));report['builds'].append(dict(name=name,command=command,library_sha256=_sha256(binary)))
        torch.manual_seed(1029);torch.backends.cuda.matmul.allow_tf32=False
        def compare(actual,expected):
            maximum=0.0
            for start in range(0,actual.shape[0],16384):
                a,b=actual[start:start+16384],expected[start:start+16384]
                maximum=max(maximum,(a-b).abs().max().item())
                torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
            return maximum
        with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
            for rows in (131,1000000,1536*1536):
                x=torch.randn(rows,128,device='cuda');x[0]=0;x[1]=100;x[2]=-100
                wa=torch.randn(512,128,device='cuda')*.1;wb=torch.randn_like(wa)*.1;wo=torch.randn(128,512,device='cuda')*.1
                scale=torch.randn(128,device='cuda')*.1;bias=torch.randn_like(scale)*.1
                normalized=extension.layer_norm_f32_128(x,scale,bias,1e-5)
                def native_prepare(y=normalized):
                    a=torch.nn.functional.linear(y,wa);b=torch.nn.functional.linear(y,wb)
                    return extension.silu_mul_f32(a,b)
                def native_complete():
                    y=extension.layer_norm_f32_128(x,scale,bias,1e-5)
                    h=native_prepare(y);del y
                    return torch.nn.functional.linear(h,wo)
                preparation=[('baseline_before',native_prepare)];complete=[('baseline_before',native_complete)]
                expected=native_prepare()
                for name,project in candidates:
                    def fused_prepare(y=normalized,project=project,guard=False):
                        count=rows*512
                        storage=torch.full((count+64,),123.5,device='cuda') if guard else torch.empty(count,device='cuda')
                        out=storage[32:-32] if guard else storage
                        status=project(y.data_ptr(),wa.data_ptr(),wb.data_ptr(),out.data_ptr(),rows,torch.cuda.current_stream().cuda_stream)
                        if status:raise RuntimeError(f'HIP status {status}')
                        if guard:
                            assert bool((storage[:32]==123.5).all()) and bool((storage[-32:]==123.5).all())
                        return out.view(rows,512)
                    actual=fused_prepare(guard=True)
                    try:delta=compare(actual,expected)
                    except AssertionError as error:
                        report['rejections'].append(dict(rows=rows,scope='preparation',implementation=name,error=str(error)));print(json.dumps(report['rejections'][-1]),flush=True);del actual;continue
                    del actual;preparation.append((name,fused_prepare))
                    def fused_complete(prepare=fused_prepare):
                        y=extension.layer_norm_f32_128(x,scale,bias,1e-5)
                        h=prepare(y);del y
                        return torch.nn.functional.linear(h,wo)
                    complete.append((name,fused_complete))
                del expected
                preparation.append(('baseline_after',native_prepare));complete.append(('baseline_after',native_complete))
                for scope,functions in (('preparation',preparation),('complete',complete)):
                    expected=functions[0][1]()
                    for name,function in functions:
                        gc.collect();actual=function()
                        try:delta=compare(actual,expected)
                        except AssertionError as error:
                            report['rejections'].append(dict(rows=rows,scope=scope,implementation=name,error=str(error)));print(json.dumps(report['rejections'][-1]),flush=True);del actual;continue
                        del actual;values=[];torch.cuda.reset_peak_memory_stats()
                        for i in range(25):
                            start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];start.record();output=function();end.record();end.synchronize();del output
                            if i>=5:values.append(start.elapsed_time(end))
                        row=dict(rows=rows,scope=scope,implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,peak_allocated_bytes=torch.cuda.max_memory_allocated());report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                    del expected
                del x,normalized,wa,wb,wo,scale,bias,preparation,complete,functions,function,fused_prepare,fused_complete,native_prepare,native_complete
                gc.collect();torch.cuda.empty_cache()
        report['status']='completed_with_rejections' if report['rejections'] else 'ok'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status'] in {'ok','completed_with_rejections'} else 1


if __name__=='__main__':raise SystemExit(main())
