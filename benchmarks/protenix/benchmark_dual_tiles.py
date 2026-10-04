"""Tune spatial/channel/K tiles against the promoted dual projection kernel."""
import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _rocm_preflight, _additional_device_preflight, _GpuProcessMonitor, _sha256,
)


def full_triangle(torch,ext,entry,n,padding,report):
    os.environ['LAYERNORM_TYPE']='torch'
    from fireprot.adapters.protenix.runtime import install
    from fireprot.backends.amd import triangle_multiplication as tm
    from fireprot.backends.amd.lt_bmm import packed_bmm
    from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
    install('hipblas');tm._dual_projection_enabled=True
    tm._dual_tiled_enabled=False  # Preserve the measured C64/R64 control.
    tm._vector_packed_enabled=True;tm._vector_norm_enabled=True
    report['blas']='hipblas'
    report['triangle_config']={'channels':128,'rows':128,'k':16,'waves':8,'padding':padding,
                               'control':'promoted C64/R64/K16/W4, contiguous output'}
    for outgoing in (True,False):
        module=TriangleMultiplicativeUpdate(128,128,outgoing).cuda().eval()
        for p in module.parameters():p.normal_(0,.1)
        source=torch.randn(n,n,128,device='cuda');z=torch.empty_like(source)
        def project(x,wg,wp):
            output=torch.empty((128,n*n+padding),device='cuda')[:,:n*n]
            status=entry(x.data_ptr(),wg.data_ptr(),wp.data_ptr(),output.data_ptr(),n,padding,torch.cuda.current_stream().cuda_stream)
            if status:raise RuntimeError(f'HIP triangle status {status}')
            return output.view(128,n,n)
        for add in (True,False):
            def production():
                return tm._long_sequence_packed(module=module,z=z,mask=None,inplace_safe=True,
                    add_with_inplace=add,inplace_chunk_size=256,native=lambda:None)
            def candidate():
                x=module.layer_norm_in(z)
                a=project(x,module.linear_a_g.weight,module.linear_a_p.weight)
                b=project(x,module.linear_b_g.weight,module.linear_b_p.weight)
                contracted=packed_bmm(a,b,outgoing);del a,b
                norm=module.layer_norm_out
                out=ext.packed_norm_vector(contracted,norm.weight,norm.bias,norm.eps);del contracted
                out=module.linear_z(out)
                return ext.triangle_residual(module.linear_g(x),out,z,add)
            z.copy_(source);reference=production().clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                z.copy_(source);actual=candidate()
                torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                delta=(actual-reference).abs().max().item()
            torch.cuda.synchronize()
            for name,fn in [('production_before',production),('tiled_projection',candidate),('production_after',production)]:
                samples=[];torch.cuda.reset_peak_memory_stats()
                for i in range(25):
                    z.copy_(source)
                    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record();fn();end.record();end.synchronize()
                    if i>=5:samples.append(start.elapsed_time(end))
                row=dict(length=n,outgoing=outgoing,add=add,implementation=name,median_ms=statistics.median(samples),
                         p95_ms=sorted(samples)[18],max_abs_delta=delta,values_ms=samples,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
            del reference,actual
        del module,source,z
        torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--padding',type=int,choices=(0,16,64,128),default=0)
    parser.add_argument('--full-triangle',action='store_true')
    parser.add_argument('--launch-bounds',action='store_true')
    parser.add_argument('--resource-tiles',action='store_true')
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args()
    visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select GPU 6 or 7')
    source=REPO_ROOT/'benchmarks/protenix/csrc/dual_projection_probe.cpp'
    compiler=subprocess.check_output(['/opt/rocm/bin/hipcc','--version'],text=True)
    build=REPO_ROOT/'runtime/dual_tiles'/hashlib.sha256(source.read_bytes()+compiler.encode()).hexdigest()[:16]
    build.mkdir(parents=True,exist_ok=True)
    variants=[(64,64,16,4),(64,128,16,4),(128,64,8,4),
              (64,128,16,8),(128,128,16,8),(128,128,8,8),
              (64,256,16,8),(64,256,8,8),(64,64,16,2),(32,64,16,2)]
    if args.compact:variants=[(64,64,16,4),(64,128,16,4),(128,128,16,8)]
    if args.resource_tiles:variants=[(128,128,16,4),(128,128,16,8),(64,256,16,4),(64,256,16,8)]
    if args.full_triangle:variants=[(128,128,16,8)]
    builds=[]
    for c,r,k,w in variants:
        name=f'c{c}r{r}k{k}w{w}b{int(args.launch_bounds)}';lib=build/(name+'.so');pending=lib.with_suffix('.pending.so')
        command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',
                 f'-DCHANNEL_TILE={c}',f'-DROW_TILE={r}',f'-DK_TILE={k}',f'-DWAVES={w}',f'-DBOUNDED_LAUNCH={int(args.launch_bounds)}',str(source),'-o',str(pending)]
        if not lib.is_file():
            subprocess.run(command,check=True);pending.replace(lib)
        builds.append(dict(name=name,channels=c,rows=r,k=k,waves=w,library=str(lib),library_sha256=_sha256(lib),command=command))
    import torch
    from fireprot.backends.amd.extension import load_amd_extension
    policy={'device_policy':'exclusive'}
    preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    extra=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for m in monitors:m.start()
    report=dict(launch_bounds=args.launch_bounds,full_triangle=args.full_triangle,padding=args.padding,padding_scope='output channel stride only; logical protein length unchanged',dtype='float32',tf32=False,seed=1009,parameters='random inputs and normal(0,0.1) weights',
                warmups=5,repetitions=20,cache_state='warm; allocations included; compilation excluded',
                compiler=compiler,torch=torch.__version__,hip=torch.version.hip,builds=builds,
                source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in (source,pathlib.Path(__file__),REPO_ROOT/'kernels/amd/fireprot_amd_ops.cu')},
                preflight=preflight,sibling_preflight=extra,records=[])
    try:
        ext=load_amd_extension();assert ext is not None
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        torch.manual_seed(1009);torch.backends.cuda.matmul.allow_tf32=False
        entries=[]
        for b in builds:
            lib=ctypes.CDLL(b['library']);entry=lib.run_dual_projection_strided;entry._library_handle=lib
            entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];entry.restype=ctypes.c_int
            assert entry(None,None,None,None,999,0,None)==2
            entries.append((b['name'],entry))
        with torch.inference_mode():
            for n in (1000,1536):
                if args.full_triangle:
                    full_triangle(torch,ext,entries[0][1],n,args.padding,report)
                    continue
                x=torch.randn(n,n,128,device='cuda');wg=torch.randn(128,128,device='cuda')*.1;wp=torch.randn_like(wg)*.1
                def production():return ext.dual_projection(x,wg,wp)
                reference=production()
                functions=[('production_before',production)]
                for name,entry in entries:
                    def candidate(entry=entry,guard=False):
                        stride=n*n+args.padding;count=128*stride
                        storage=torch.full((count+64,),123.5,device='cuda') if guard else torch.empty(count,device='cuda')
                        body=storage[32:-32] if guard else storage
                        output=body.view(128,stride)[:,:n*n]
                        status=entry(x.data_ptr(),wg.data_ptr(),wp.data_ptr(),output.data_ptr(),n,args.padding,torch.cuda.current_stream().cuda_stream)
                        if status:raise RuntimeError(f'HIP launch status {status}')
                        if guard:
                            assert bool((storage[:32]==123.5).all()) and bool((storage[-32:]==123.5).all())
                            assert bool((body.view(128,stride)[:,n*n:]==123.5).all())
                        return output.view(128,n,n)
                    torch.testing.assert_close(candidate(guard=True),reference,atol=2e-5,rtol=2e-5)
                    functions.append((name,candidate))
                functions.append(('production_after',production))
                for name,fn in functions:
                    actual=fn();torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                    delta=(actual-reference).abs().max().item();del actual
                    samples=[]
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();output=fn();end.record();end.synchronize();del output
                        if i>=5:samples.append(start.elapsed_time(end))
                    row=dict(length=n,implementation=name,median_ms=statistics.median(samples),p95_ms=sorted(samples)[18],max_abs_delta=delta,values_ms=samples)
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                del x,wg,wp,reference
                torch.cuda.empty_cache()
        report['status']='ok'
    except Exception as error:
        report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
