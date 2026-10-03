"""Measure QKV preparation at the actual long triangle-attention batch sizes."""
import argparse
import ctypes
import hashlib
import json
import math
import os
import pathlib
import statistics
import subprocess
from benchmarks.protenix.profile_inference import (
    REPO_ROOT,_rocm_preflight,_additional_device_preflight,_GpuProcessMonitor,_sha256,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--full-attention',action='store_true')
    parser.add_argument('--wide-sweep',action='store_true')
    parser.add_argument('--hip-mfma',action='store_true')
    args=parser.parse_args();visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select GPU 6 or 7')
    if args.full_attention and not args.hip_mfma:parser.error('Full attention requires --hip-mfma')
    if args.full_attention:os.environ['FIREPROT_ENABLE_LONG1536']='1'
    source=REPO_ROOT/'benchmarks/protenix/csrc/qkv_split_probe.cpp'
    compiler=subprocess.check_output(['/opt/rocm/bin/hipcc','--version'],text=True)
    build=REPO_ROOT/'runtime/qkv_projection'/hashlib.sha256(source.read_bytes()+compiler.encode()).hexdigest()[:16]
    build.mkdir(parents=True,exist_ok=True);lib=build/'split.so'
    command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',str(source),'-o',str(lib.with_suffix('.pending.so'))]
    if not lib.is_file():subprocess.run(command,check=True);lib.with_suffix('.pending.so').replace(lib)
    import torch
    policy={'device_policy':'exclusive'}
    preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    extra=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for m in monitors:m.start()
    report=dict(full_attention=args.full_attention,wide_sweep=args.wide_sweep,hip_mfma=args.hip_mfma,seed=1019,dtype='float32',tf32=False,warmups=5,repetitions=20,
                parameters='random normal inputs and normal(0,0.1) weights',
                cache_state='warm; weight concatenation cached outside timing; allocations and Q scaling included',
                compiler=compiler,command=command,library_sha256=_sha256(lib),torch=torch.__version__,hip=torch.version.hip,
                source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in (source,pathlib.Path(__file__),REPO_ROOT/'Protenix/protenix/model/triangular/layers.py')},
                preflight=preflight,sibling_preflight=extra,records=[])
    try:
        from fireprot.backends.amd.blas import configure_process_blas
        report['blas']=configure_process_blas('hipblas')
        os.environ['LAYERNORM_TYPE']='torch'
        from protenix.model.triangular.layers import Attention
        handle=ctypes.CDLL(str(lib));entry=handle.qkv_split
        entry.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int,ctypes.c_void_p];entry.restype=ctypes.c_int
        assert entry(None,None,1,None)==2
        candidates=[]
        if args.hip_mfma:
            mfma_source=REPO_ROOT/'benchmarks/protenix/csrc/qkv_projection_probe.cpp'
            report['source_sha256'][str(mfma_source.relative_to(REPO_ROOT))]=_sha256(mfma_source)
            mfma_build=REPO_ROOT/'runtime/qkv_mfma'/hashlib.sha256(mfma_source.read_bytes()+compiler.encode()).hexdigest()[:16]
            mfma_build.mkdir(parents=True,exist_ok=True)
            variants=((128,64,16,8),(64,64,16,4),(64,128,16,8),(64,128,8,8),(32,64,16,4),(64,32,16,4))
            if args.wide_sweep:variants=((64,64,16,4),(64,64,8,4),(128,64,8,8),(128,128,8,16),(128,128,16,16),(128,64,16,16),(256,64,16,16))
            if args.full_attention:variants=((64,64,16,4),(128,64,16,16))
            for r,c,kk,w in variants:
                name=f'r{r}c{c}k{kk}w{w}';binary=mfma_build/(name+'.so')
                command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',f'-DROW_TILE={r}',f'-DCHANNEL_TILE={c}',f'-DK_TILE={kk}',f'-DWAVES={w}',str(mfma_source),'-o',str(binary.with_suffix('.pending.so'))]
                if not binary.is_file():subprocess.run(command,check=True);binary.with_suffix('.pending.so').replace(binary)
                library=ctypes.CDLL(str(binary));project=library.qkv_projection;project._library_handle=library
                project.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];project.restype=ctypes.c_int
                candidates.append((name,project));report.setdefault('mfma_builds',[]).append(dict(name=name,command=command,library_sha256=_sha256(binary)))
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a' 
        torch.manual_seed(1019);torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode():
            for batch,n in ((1000,1000),(512,1536)):
                module=Attention(128,128,128,32,4).cuda().eval()
                for p in module.parameters():p.normal_(0,.1)
                x=torch.randn(batch,n,128,device='cuda')
                weight=torch.cat([module.linear_q.weight,module.linear_k.weight,module.linear_v.weight])
                def layout(t):return t.view(batch,n,4,32).transpose(1,2)
                def production():return module._prep_qkv(x,x)
                def joint_strided():
                    packed=torch.nn.functional.linear(x,weight)
                    q,k,v=packed.split(128,dim=-1);q.div_(math.sqrt(32))
                    return tuple(layout(t) for t in (q,k,v))
                def joint_copies():
                    packed=torch.nn.functional.linear(x,weight)
                    q,k,v=[t.contiguous() for t in packed.split(128,dim=-1)];q.div_(math.sqrt(32))
                    return tuple(layout(t) for t in (q,k,v))
                def joint_hip_split(guard=False):
                    packed=torch.nn.functional.linear(x,weight);count=3*batch*n*128
                    storage=torch.full((count+64,),123.5,device='cuda') if guard else torch.empty(count,device='cuda')
                    output=storage[32:-32] if guard else storage
                    status=entry(packed.data_ptr(),output.data_ptr(),batch*n,torch.cuda.current_stream().cuda_stream)
                    if status:raise RuntimeError(f'Split HIP status {status}')
                    if guard:assert bool((storage[:32]==123.5).all()) and bool((storage[-32:]==123.5).all())
                    return tuple(layout(t) for t in output.view(3,batch,n,128).unbind(0))
                reference=production()
                for a,b in zip(joint_hip_split(True),reference):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
                functions=[('production_before',production),('joint_strided',joint_strided),('joint_torch_copies',joint_copies),('joint_hip_split',joint_hip_split)]
                for name,project in candidates:
                    def mfma(project=project,guard=False):
                        count=3*batch*n*128
                        storage=torch.full((count+64,),123.5,device='cuda') if guard else torch.empty(count,device='cuda')
                        output=storage[32:-32] if guard else storage
                        status=project(x.data_ptr(),module.linear_q.weight.data_ptr(),module.linear_k.weight.data_ptr(),module.linear_v.weight.data_ptr(),output.data_ptr(),batch*n,torch.cuda.current_stream().cuda_stream)
                        if status:raise RuntimeError(f'MFMA HIP status {status}')
                        if guard:assert bool((storage[:32]==123.5).all()) and bool((storage[-32:]==123.5).all())
                        return tuple(layout(t) for t in output.view(3,batch,n,128).unbind(0))
                    for a,b in zip(mfma(guard=True),reference):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
                    functions.append((name,mfma))
                functions.append(('production_after',production))
                if args.full_attention:
                    from fireprot.backends.amd import ck_attention as ck
                    assert ck.load_extension() is not None and ck._exact1536_enabled and ck._vector1000_enabled
                    bias=torch.randn(1,4,n,n,device='cuda')*.1
                    wrapped=[]
                    for name,prepare in functions:
                        if name.startswith('joint_'):continue
                        def complete(prepare=prepare):
                            q,k,v=prepare()
                            output=ck.attention(module=module,q=q,k=k,v=v,biases=[bias],native=lambda **_:None)
                            return (module._wrap_up(output,x),)
                        wrapped.append((name,complete))
                    functions=wrapped;reference=functions[0][1]()
                    report['ck_attention']=ck.extension_status()
                    report['scope']='QKV preparation + CK attention/bias packing + output gate/projection; one actual outer-batch chunk, excludes triangle input LayerNorm and bias projection'
                for name,fn in functions:
                    actual=fn()
                    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
                    delta=max((a-b).abs().max().item() for a,b in zip(actual,reference))
                    strides=[list(t.stride()) for t in actual];del actual,a,b
                    values=[];torch.cuda.reset_peak_memory_stats()
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();output=fn();end.record();end.synchronize();del output
                        if i>=5:values.append(start.elapsed_time(end))
                    row=dict(batch=batch,length=n,implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,strides=strides,ck_layout_compatible=None if args.full_attention else all(s==[n*128,32,128,1] for s in strides),peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                del x,weight,module,reference
                if args.full_attention:del bias
                torch.cuda.empty_cache()
        report['status']='ok'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
