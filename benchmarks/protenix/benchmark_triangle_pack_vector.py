"""Compare scalar and vector global accesses for long sigmoid packing."""
import argparse,ctypes,json,os,pathlib,statistics,subprocess
from benchmarks.protenix.profile_inference import REPO_ROOT,_rocm_preflight,_additional_device_preflight,_GpuProcessMonitor,_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args()
    visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select GPU 6 or 7')
    source=REPO_ROOT/'benchmarks/protenix/csrc/triangle_pack_vector_probe.cpp'
    build=REPO_ROOT/'runtime/triangle_pack_vector';build.mkdir(parents=True,exist_ok=True)
    lib=build/'probe.so'
    command=['/opt/rocm/bin/hipcc','-O3','-ffp-contract=off','-std=c++17','-shared','-fPIC','--offload-arch=gfx90a',str(source),'-o',str(lib)]
    subprocess.run(command,check=True)
    import torch
    library=ctypes.CDLL(str(lib));entry=library.triangle_pack_vector
    entry.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p];entry.restype=ctypes.c_int
    policy=dict(device_policy='exclusive')
    preflight=_rocm_preflight(policy);assert preflight['exclusive_check']['passed']
    extra=_additional_device_preflight([7 if visible=='6' else 6],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1) for d in (6,7)]
    for m in monitors:m.start()
    report=dict(preflight=preflight,sibling_preflight=extra,seed=937,dtype='float32',tf32=False,
                warmup_groups=5,measured_groups=20,group_size=8,cache_state='warm; compilation excluded',
                command=command,source_sha256={str(p.relative_to(REPO_ROOT)):_sha256(p) for p in
                    (source,pathlib.Path(__file__),REPO_ROOT/'fireprot/backends/amd/csrc/fireprot_amd_ops.cu')},
                library_sha256=_sha256(lib),records=[])
    try:
        from fireprot.backends.amd.extension import load_amd_extension
        ext=load_amd_extension();assert ext is not None
        torch.manual_seed(937);torch.backends.cuda.matmul.allow_tf32=False
        assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]=='gfx90a'
        with torch.inference_mode():
            for n in (1000,1536):
                gate=torch.randn(n,n,128,device='cuda');value=torch.randn_like(gate)
                gate.view(-1)[:4]=torch.tensor([-80.,80.,-float('inf'),float('inf')],device='cuda')
                reference=ext.sigmoid_pack_f32_128(gate,value)
                def call(tile):
                    output=torch.empty_like(reference)
                    status=entry(gate.data_ptr(),value.data_ptr(),output.data_ptr(),n,tile,torch.cuda.current_stream().cuda_stream)
                    if status:raise RuntimeError(f'HIP status {status}')
                    return output
                for name,fn in [('baseline_before',lambda:ext.sigmoid_pack_f32_128(gate,value)),
                                ('vector32',lambda:call(32)),('vector64',lambda:call(64)),
                                ('baseline_after',lambda:ext.sigmoid_pack_f32_128(gate,value))]:
                    actual=fn();torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-6)
                    delta=(actual-reference).abs().max().item();del actual
                    timings=[]
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        for _ in range(8):out=fn();del out
                        end.record();end.synchronize()
                        if i>=5:timings.append(start.elapsed_time(end)/8)
                    row=dict(n=n,implementation=name,median_ms=statistics.median(timings),p95_ms=sorted(timings)[18],values_ms=timings,max_abs_delta=delta)
                    report['records'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='values_ms'}),flush=True)
                del gate,value,reference
                torch.cuda.empty_cache()
        report['status']='ok'
    except Exception as error:report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
