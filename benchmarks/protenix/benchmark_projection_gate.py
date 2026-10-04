"""Compare current packing, transposed BLAS, and CK fused triangle projections."""
import argparse,ctypes,json,os,pathlib,statistics,subprocess,hashlib
from benchmarks.protenix.profile_inference import REPO_ROOT,_rocm_preflight,_additional_device_preflight,_GpuProcessMonitor,_sha256


def run_triangle(torch,ext,handle,n,report,dual_entry=None):
    """Measure the complete current triangle operator against fused projections."""
    os.environ['LAYERNORM_TYPE']='torch'
    os.environ['FIREPROT_ENABLE_LONG1536']='1'
    from fireprot.adapters.protenix.runtime import install
    from fireprot.backends.amd import triangle_multiplication as tm
    from fireprot.backends.amd.lt_bmm import packed_bmm
    from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
    install('hipblas')
    tm._vector_packed_enabled=True;tm._vector_norm_enabled=True
    tm._dual_projection_enabled=False  # Preserve the measured pre-dual control.
    entry=handle.ck_projection_gate_rows
    entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_void_p]
    entry.restype=ctypes.c_int
    chunk=1000000
    report.setdefault('triangle_config',{})[str(n)]={'chunk':None if dual_entry else chunk,'tile':[64,16] if dual_entry else [128,64 if n==1000 else 128],'implementation':'dual_hip' if dual_entry else 'ck','vector_pack_control':True,'vector_norm':True}
    for outgoing in (True,False):
        module=TriangleMultiplicativeUpdate(128,128,outgoing).cuda().eval()
        for p in module.parameters():p.normal_(0,.1)
        source_z=torch.randn(n,n,128,device='cuda');z=torch.empty_like(source_z)
        def project(normalized,wg,wp):
            if dual_entry is not None:
                output=torch.empty((128,n,n),device='cuda')
                status=dual_entry(normalized.data_ptr(),wg.data_ptr(),wp.data_ptr(),output.data_ptr(),n,torch.cuda.current_stream().cuda_stream)
                if status:raise RuntimeError(f'Dual triangle launch status {status}')
                return output
            rows=n*n;flat=normalized.view(rows,128)
            output=torch.empty((128,rows),device='cuda')
            for first in range(0,rows,chunk):
                count=min(chunk,rows-first);features=flat[first:first+count]
                value=torch.mm(wp,features.T)
                status=entry(wg.data_ptr(),features.data_ptr(),value.data_ptr(),output.data_ptr()+4*first,count,count,rows,torch.cuda.current_stream().cuda_stream)
                if status:raise RuntimeError(f'CK triangle launch status {status}')
            return output.view(128,n,n)
        for add in (True,False):
            def baseline():
                return tm._long_sequence_packed(module=module,z=z,mask=None,inplace_safe=True,add_with_inplace=add,inplace_chunk_size=256,native=lambda:None)
            def candidate():
                normalized=module.layer_norm_in(z)
                a=project(normalized,module.linear_a_g.weight,module.linear_a_p.weight)
                b=project(normalized,module.linear_b_g.weight,module.linear_b_p.weight)
                contracted=packed_bmm(a,b,outgoing);del a,b
                norm=module.layer_norm_out
                output=ext.packed_norm_vector(contracted,norm.weight,norm.bias,norm.eps);del contracted
                output=module.linear_z(output)
                return ext.triangle_residual(module.linear_g(normalized),output,z,add)
            z.copy_(source_z);reference=baseline().clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                z.copy_(source_z);actual=candidate()
                torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                delta=(actual-reference).abs().max().item()
            torch.cuda.synchronize()
            for name,fn in [('baseline_before',baseline),('dual_hip_projection' if dual_entry else 'ck_fused_projection',candidate),('baseline_after',baseline)]:
                values=[];torch.cuda.reset_peak_memory_stats()
                for i in range(25):
                    z.copy_(source_z)
                    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record();fn();end.record();end.synchronize()
                    if i>=5:values.append(start.elapsed_time(end))
                row=dict(length=n,outgoing=outgoing,add=add,implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
            del reference,actual
        del module,source_z,z
        torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--padding',type=int,choices=(0,16,64),default=0)
    parser.add_argument('--dual-hip',action='store_true')
    parser.add_argument('--full-triangle',action='store_true')
    parser.add_argument('--chunks',type=int,nargs='*',default=[])
    args=parser.parse_args()
    if any(c<=0 or c%4 for c in args.chunks):parser.error('Chunks must be positive multiples of four')
    visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select GPU 6 or 7')
    source=REPO_ROOT/'benchmarks/protenix/csrc/ck_projection_gate_probe.cpp'
    compiler_version=subprocess.check_output(['/opt/rocm/bin/hipcc','--version'],text=True)
    digest=hashlib.sha256();headers=sorted(pathlib.Path('/opt/rocm/include/ck').rglob('*.hpp'))
    for h in headers:
        digest.update(str(h.relative_to('/opt/rocm/include')).encode()+b'\0')
        digest.update(hashlib.sha256(h.read_bytes()).digest())
    header_sha256=digest.hexdigest();digest.update(source.read_bytes()+compiler_version.encode())
    build=REPO_ROOT/'runtime/ck_projection_gate'/digest.hexdigest()[:16];build.mkdir(parents=True,exist_ok=True)
    builds=[]
    for m,n in [(64,64),(64,128),(128,64),(128,128)]:
        lib=build/f'm{m}n{n}.so'
        command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a','-I/opt/rocm/include',f'-DTILE_M={m}',f'-DTILE_N={n}',str(source),'-o',str(lib.with_suffix('.pending.so'))]
        if not lib.is_file():
            with (build/f'm{m}n{n}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            lib.with_suffix('.pending.so').replace(lib)
        builds.append(dict(m=m,n=n,library=str(lib),library_sha256=_sha256(lib),command=command))
    import torch
    policy=dict(device_policy='exclusive');preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    extra=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for monitor in monitors:monitor.start()
    report=dict(dual_hip=args.dual_hip,full_triangle=args.full_triangle,chunks=args.chunks,padding=args.padding,padding_scope='CK value/output channel stride only; logical protein length unchanged',compiler_version=compiler_version,ck_header_sha256=header_sha256,ck_header_count=len(headers),torch=torch.__version__,hip=torch.version.hip,weight_prepacking='gate/value weights concatenated once outside timed region for joint projection',parameters='random normal inputs; normal(0,0.1) weights, not checkpoint weights',seed=983,dtype='float32',tf32=False,warmups=5,repetitions=20,cache_state='warm; compilation excluded; value projection and allocations included',preflight=preflight,sibling_preflight=extra,builds=builds,source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in [source,pathlib.Path(__file__),REPO_ROOT/'kernels/amd/fireprot_amd_ops.cu']},records=[])
    handles=[]
    dual_entries=[]
    try:
        if args.dual_hip:
            dual_source=REPO_ROOT/'benchmarks/protenix/csrc/dual_projection_probe.cpp'
            report['source_sha256'][str(dual_source.relative_to(REPO_ROOT))]=_sha256(dual_source)
            dual_build=REPO_ROOT/'runtime/dual_projection'/hashlib.sha256(dual_source.read_bytes()+compiler_version.encode()).hexdigest()[:16]
            dual_build.mkdir(parents=True,exist_ok=True)
            for tile,k_tile in ((64,16),(64,32),(128,16),(128,32)):
                lib=dual_build/f'c{tile}k{k_tile}.so'
                command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',f'-DCHANNEL_TILE={tile}',f'-DK_TILE={k_tile}',str(dual_source),'-o',str(lib)]
                if not lib.is_file():subprocess.run(command,check=True)
                handle=ctypes.CDLL(str(lib));entry=handle.run_dual_projection
                entry._library_handle=handle
                entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];entry.restype=ctypes.c_int
                dual_entries.append((f'dual_hip_c{tile}k{k_tile}',entry))
                report.setdefault('dual_builds',[]).append(dict(tile=tile,k_tile=k_tile,command=command,library_sha256=_sha256(lib)))
        from fireprot.backends.amd.extension import load_amd_extension
        from fireprot.backends.amd.blas import configure_process_blas
        report['blas']=configure_process_blas('hipblas')
        ext=load_amd_extension();assert ext is not None
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        torch.manual_seed(983);torch.backends.cuda.matmul.allow_tf32=False
        entries=[]
        for b in builds:
            handle=ctypes.CDLL(b['library']);handles.append(handle);entry=handle.ck_projection_gate
            entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p];entry.restype=ctypes.c_int
            assert entry(None,None,None,None,999,0,None)==2
            entries.append((f'ck_m{b["m"]}n{b["n"]}',entry))
        with torch.inference_mode():
            for length in (1000,1536):
                if args.full_triangle:
                    handle=handles[2 if length==1000 else 3]
                    run_triangle(torch,ext,handle,length,report,dual_entries[0][1] if args.dual_hip else None)
                    continue
                x=torch.randn(length,length,128,device='cuda')
                wg=torch.randn(128,128,device='cuda')*.1;wp=torch.randn_like(wg)*.1
                xt=x.view(-1,128).T
                combined_weight=torch.cat((wg,wp),dim=0)
                def baseline():return ext.triangle_vector_pack(torch.nn.functional.linear(x,wg),torch.nn.functional.linear(x,wp))
                def transposed():
                    gate=torch.mm(wg,xt);value=torch.mm(wp,xt)
                    return ext.sigmoid_mul_f32(gate,value).view(128,length,length)
                def joint_projection():
                    both=torch.mm(combined_weight,xt)
                    return ext.sigmoid_mul_f32(both[:128],both[128:]).view(128,length,length)
                reference=baseline()
                functions=[('baseline_before',baseline),('transposed_blas',transposed),('joint_transposed_blas',joint_projection)]
                for name,entry in dual_entries:
                    def dual(entry=entry):
                        output=torch.empty((128,length,length),device='cuda')
                        status=entry(x.data_ptr(),wg.data_ptr(),wp.data_ptr(),output.data_ptr(),length,torch.cuda.current_stream().cuda_stream)
                        if status:raise RuntimeError(f'Dual HIP launch status {status}')
                        return output
                    functions.append((name,dual))
                for name,entry in entries:
                    def fused(entry=entry,guard=False):
                        rows=length*length;stride=rows+args.padding
                        if guard:
                            value_storage=torch.full((128,stride),123.5,device='cuda')
                            output_storage=torch.full((128,stride),123.5,device='cuda')
                        else:
                            value_storage=torch.empty((128,stride),device='cuda')
                            output_storage=torch.empty_like(value_storage)
                        value=value_storage[:,:rows];output=output_storage[:,:rows]
                        torch.mm(wp,xt,out=value)
                        status=entry(wg.data_ptr(),x.data_ptr(),value.data_ptr(),output.data_ptr(),length,args.padding,torch.cuda.current_stream().cuda_stream)
                        if status:raise RuntimeError(f'CK launch status {status}')
                        if guard and args.padding:
                            assert bool((value_storage[:,rows:]==123.5).all()) and bool((output_storage[:,rows:]==123.5).all())
                        return output.view(128,length,length)
                    torch.testing.assert_close(fused(guard=True),reference,atol=2e-5,rtol=2e-5)
                    functions.append((name,fused))
                for b,handle in zip(builds,handles):
                    chunk_entry=handle.ck_projection_gate_rows
                    chunk_entry.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_void_p]
                    chunk_entry.restype=ctypes.c_int
                    assert chunk_entry(None,None,None,None,3,4,4,None)==2
                    for chunk in args.chunks:
                        def chunked(entry=chunk_entry,chunk=chunk,guard=False):
                            rows=length*length;stride=rows+args.padding
                            storage=torch.full((128,stride),123.5,device='cuda') if guard else torch.empty((128,stride),device='cuda')
                            flat=x.view(rows,128)
                            for first in range(0,rows,chunk):
                                count=min(chunk,rows-first)
                                features=flat[first:first+count]
                                value=torch.mm(wp,features.T)
                                status=entry(wg.data_ptr(),features.data_ptr(),value.data_ptr(),storage.data_ptr()+4*first,count,count,stride,torch.cuda.current_stream().cuda_stream)
                                if status:raise RuntimeError(f'CK chunk launch status {status}')
                            if guard and args.padding:assert bool((storage[:,rows:]==123.5).all())
                            return storage[:,:rows].view(128,length,length)
                        torch.testing.assert_close(chunked(guard=True),reference,atol=2e-5,rtol=2e-5)
                        functions.append((f'ck_m{b["m"]}n{b["n"]}_chunk{chunk}',chunked))
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
                del x,wg,wp,xt,combined_weight,reference
                torch.cuda.empty_cache()
        report['status']='ok'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
