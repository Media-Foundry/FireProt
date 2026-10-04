"""Vectorized long packed LayerNorm candidates with complete numerical checks."""
import argparse,ctypes,json,os,pathlib,statistics,subprocess
from benchmarks.protenix.profile_inference import REPO_ROOT,_rocm_preflight,_additional_device_preflight,_GpuProcessMonitor,_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select physical GPU 6 or 7')
    source=REPO_ROOT/'benchmarks/protenix/csrc/packed_norm_vector_probe.cpp'
    build=REPO_ROOT/'runtime/packed_norm_vector';build.mkdir(parents=True,exist_ok=True)
    lib=build/'probe.so';command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',str(source),'-o',str(lib)]
    subprocess.run(command,check=True)
    import torch
    handle=ctypes.CDLL(str(lib));entry=handle.packed_norm_vector
    entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_float,ctypes.c_void_p];entry.restype=ctypes.c_int
    assert entry(None,None,None,None,999,32,0,1e-5,None)==2
    assert entry(None,None,None,None,1000,32,0,1e-5,None)==3
    policy=dict(device_policy='exclusive');preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    extra=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for m in monitors:m.start()
    report=dict(seed=967,dtype='float32',tf32=False,warmup_groups=5,measured_groups=20,group_size=8,
                cache_state='warm; compile excluded; output allocation included',preflight=preflight,sibling_preflight=extra,
                command=command,library_sha256=_sha256(lib),source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in [source,pathlib.Path(__file__),REPO_ROOT/'kernels/amd/fireprot_amd_ops.cu']},records=[],edge_validation=[])
    try:
        from fireprot.backends.amd.extension import load_amd_extension
        ext=load_amd_extension();assert ext is not None
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        torch.manual_seed(967);torch.backends.cuda.matmul.allow_tf32=False
        report.update(torch=torch.__version__,hip=torch.version.hip)
        with torch.inference_mode():
            for n in (1000,1536):
                packed=torch.randn(128,n,n,device='cuda');weight=torch.randn(128,device='cuda');bias=torch.randn(128,device='cuda')
                reference=ext.packed_layer_norm_f32_128(packed,weight,bias,1e-5)
                def call(tile,vec,guard=False):
                    storage=torch.full((packed.numel()+8192,),123.5,device='cuda') if guard else None
                    output=storage[4096:-4096].view(n,n,128) if guard else torch.empty((n,n,128),device='cuda')
                    code=entry(packed.data_ptr(),weight.data_ptr(),bias.data_ptr(),output.data_ptr(),n,tile,vec,1e-5,torch.cuda.current_stream().cuda_stream)
                    if code:raise RuntimeError(f'HIP launch {code}')
                    if guard:
                        assert bool((storage[:4096]==123.5).all()) and bool((storage[-4096:]==123.5).all())
                    return output
                functions=[('baseline_before',lambda:ext.packed_layer_norm_f32_128(packed,weight,bias,1e-5))]
                for tile in (32,64):
                    for vec in (0,1,2):
                        output=call(tile,vec,True);torch.testing.assert_close(output,reference,atol=2e-5,rtol=2e-5);del output
                        functions.append((f'tile{tile}_vecout{vec}',lambda t=tile,v=vec:call(t,v)))
                functions.append(('baseline_after',lambda:ext.packed_layer_norm_f32_128(packed,weight,bias,1e-5)))
                for name,fn in functions:
                    actual=fn();delta=(actual-reference).abs().max().item();del actual
                    values=[]
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)];start.record()
                        for _ in range(8):out=fn();del out
                        end.record();end.synchronize()
                        if i>=5:values.append(start.elapsed_time(end)/8)
                    row=dict(n=n,implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta)
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                # Constant and low-variance tails stress variance and epsilon handling.
                packed[:,-1,-2]=1000.;packed[:,-1,-1]=1.+torch.arange(128,device='cuda')*1e-6
                reference=ext.packed_layer_norm_f32_128(packed,weight,bias,1e-5)
                for tile in (32,64):
                    for vec in (0,1,2):
                        actual=call(tile,vec,True)
                        result=dict(n=n,tile=tile,vector_output=vec,max_abs_delta=(actual-reference).abs().max().item())
                        try:
                            torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                            result['passed']=True
                        except AssertionError as error:
                            result.update(passed=False,error=str(error))
                        report['edge_validation'].append(result)
                        del actual
                del packed,weight,bias,reference
                torch.cuda.empty_cache()
        report['status']='ok'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
