"""Diagnose packed triangle stages and test fused gated residual updates."""
import argparse
import ctypes
import json
import os
import pathlib
import statistics
import subprocess
from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _rocm_preflight, _additional_device_preflight, _GpuProcessMonitor, _sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--length', type=int, choices=(1000,1536), required=True)
    parser.add_argument('--vector-pack',action='store_true')
    args = parser.parse_args()
    visible = os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}: parser.error('Select physical GPU 6 or 7')
    n = args.length
    source = REPO_ROOT/'benchmarks/protenix/csrc/triangle_residual_probe.cpp'
    build = REPO_ROOT/'runtime/triangle_residual_probe'; build.mkdir(parents=True,exist_ok=True)
    lib = build/'probe.so'
    command = ['/opt/rocm/bin/hipcc','-O3','-ffp-contract=off','-std=c++17','-shared','-fPIC',
               '--offload-arch=gfx90a',str(source),'-o',str(lib)]
    subprocess.run(command,check=True)
    import torch  # Load the matched HIP runtime before the standalone library.
    library = ctypes.CDLL(str(lib)); entry = library.triangle_residual
    entry.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
    entry.restype = ctypes.c_int
    assert entry(None,None,None,999,1,None)==2
    assert entry(None,None,None,n,1,None)==3
    policy = dict(device_policy='exclusive')
    preflight = _rocm_preflight(policy)
    assert preflight['exclusive_check']['passed']
    extra = _additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors = [_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for m in monitors: m.start()
    report = dict(shape=[n,n,128],dtype='float32',tf32=False,seed=929,
                  warmups=5,repetitions=20,parameters='random normal(0,0.1), not checkpoint weights',
                  cache_state='warm; compilation excluded; source reset outside timed interval',
                  preflight=preflight,sibling_preflight=extra,records=[],command=command,
                  source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in
                                 (source,pathlib.Path(__file__),REPO_ROOT/'python/fireprot/backends/amd/triangle_multiplication.py')},
                  library_sha256=_sha256(lib))
    try:
        os.environ['LAYERNORM_TYPE']='torch'
        import torch
        from fireprot.adapters.protenix.runtime import install
        from fireprot.backends.amd.extension import load_amd_extension
        from fireprot.backends.amd import triangle_multiplication as tm
        from fireprot.backends.amd.lt_bmm import packed_bmm
        from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate
        install('hipblas'); ext=load_amd_extension(); assert ext is not None
        tm._vector_packed_enabled = False  # Keep the pre-promotion control explicit.
        tm._vector_norm_enabled = False
        tm._dual_projection_enabled = False
        pack = ext.sigmoid_pack_f32_128
        if args.vector_pack:
            pack_source=REPO_ROOT/'benchmarks/protenix/csrc/triangle_pack_vector_probe.cpp'
            pack_lib=build/'pack.so'
            pack_command=command.copy();pack_command[-3]=str(pack_source);pack_command[-1]=str(pack_lib)
            subprocess.run(pack_command,check=True)
            pack_library=ctypes.CDLL(str(pack_lib));pack_entry=pack_library.triangle_pack_vector
            pack_entry.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
            pack_entry.restype=ctypes.c_int
            report['vector_pack']=dict(command=pack_command,source_sha256=_sha256(pack_source),library_sha256=_sha256(pack_lib),tile_rows=64)
            def pack(gate,value):
                output=torch.empty((128,n,n),device=gate.device,dtype=gate.dtype)
                status=pack_entry(gate.data_ptr(),value.data_ptr(),output.data_ptr(),n,64,torch.cuda.current_stream().cuda_stream)
                if status:raise RuntimeError(f'Pack HIP status {status}')
                return output
        assert torch.version.hip and torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        torch.manual_seed(929); torch.backends.cuda.matmul.allow_tf32=False
        report.update(torch=torch.__version__,hip=torch.version.hip)
        with torch.inference_mode():
            for outgoing in (True,False):
                module=TriangleMultiplicativeUpdate(128,128,outgoing).cuda().eval()
                for p in module.parameters(): p.normal_(0,0.1)
                source_z=torch.randn(n,n,128,device='cuda');z=torch.empty_like(source_z)
                def production():
                    return tm._long_sequence_packed(module=module,z=z,mask=None,inplace_safe=True,
                        add_with_inplace=True,inplace_chunk_size=256,native=lambda:None)
                def candidate(add=True,timers=None):
                    def stage(name,fn):
                        if timers is None:return fn()
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();result=fn();end.record();timers.append((name,start,end));return result
                    normalized=stage('input_norm',lambda:module.layer_norm_in(z))
                    ag=stage('a_gate_projection',lambda:module.linear_a_g(normalized))
                    ap=stage('a_value_projection',lambda:module.linear_a_p(normalized))
                    a=stage('a_pack',lambda:pack(ag,ap));del ag,ap
                    bg=stage('b_gate_projection',lambda:module.linear_b_g(normalized))
                    bp=stage('b_value_projection',lambda:module.linear_b_p(normalized))
                    b=stage('b_pack',lambda:pack(bg,bp));del bg,bp
                    contracted=stage('contraction',lambda:packed_bmm(a,b,outgoing));del a,b
                    norm=module.layer_norm_out
                    out=stage('output_norm_unpack',lambda:ext.packed_layer_norm_f32_128(contracted,norm.weight,norm.bias,norm.eps));del contracted
                    out=stage('output_projection',lambda:module.linear_z(out))
                    gate=stage('output_gate_projection',lambda:module.linear_g(normalized))
                    def fused():
                        status=entry(gate.data_ptr(),out.data_ptr(),z.data_ptr(),n,int(add),torch.cuda.current_stream().cuda_stream)
                        if status:raise RuntimeError(f'HIP launch status {status}')
                    stage('gated_residual',fused)
                    return z
                z.copy_(source_z); reference=production().clone()
                stream=torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    # Explicit handoff before consuming data produced on the default stream.
                    stream.wait_stream(torch.cuda.default_stream())
                    z.copy_(source_z); actual=candidate()
                    torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                    delta=(actual-reference).abs().max().item()
                torch.cuda.synchronize()
                for name,fn in [('baseline_before',production),('vector_pack_and_residual' if args.vector_pack else 'fused_residual',candidate),('baseline_after',production)]:
                    values=[];torch.cuda.reset_peak_memory_stats()
                    for i in range(25):
                        z.copy_(source_z)
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();fn();end.record();end.synchronize()
                        if i>=5:values.append(start.elapsed_time(end))
                    row=dict(outgoing=outgoing,implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                stages={}
                for i in range(6):
                    z.copy_(source_z);timers=[];candidate(timers=timers);torch.cuda.synchronize()
                    if i:
                        for name,start,end in timers:stages.setdefault(name,[]).append(start.elapsed_time(end))
                report['records'].append(dict(outgoing=outgoing,diagnostic_stages_median_ms={k:statistics.median(v) for k,v in stages.items()}))
                del module,z,source_z,reference,actual
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
