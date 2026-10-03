"""Replay the integrated two-GCD route on caller streams and allocator policies."""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import statistics
import time
from types import SimpleNamespace
from benchmarks.protenix.profile_inference import _rocm_preflight, _additional_device_preflight, _GpuProcessMonitor, _sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=pathlib.Path)
    parser.add_argument('--queued-calls',type=int,default=1,choices=(1,8,32))
    parser.add_argument('--primary',type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'7,6','6,7'}:parser.error('Requires physical GPUs 6 and 7')
    physical=list(map(int,os.environ['HIP_VISIBLE_DEVICES'].split(',')))
    primary=args.primary;secondary=1-primary
    policy=dict(device_policy='exclusive')
    preflight=_rocm_preflight(policy)
    if not preflight['exclusive_check']['passed']:raise RuntimeError('Primary GPU unavailable')
    extra=_additional_device_preflight([physical[1]],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1.0) for d in (7,6)]
    for m in monitors:m.start()
    report=dict(scope='Integrated operator diagnostic, not model performance',visible_devices=physical,primary_logical=primary,preflight=preflight,extra_preflight=extra,
                shape=[512,4,1536,32],seed=419,dtype='float32',tf32=False,warmups=5,repetitions=20,
                allocator=os.environ.get('PYTORCH_ALLOC_CONF'),cache_state='warm',queued_calls=args.queued_calls,records=[],
                script_sha256=_sha256(pathlib.Path(__file__)))
    try:
        import torch
        from fireprot.backends.amd.two_gcd_attention import try_attention,manifest
        os.environ['FIREPROT_ENABLE_TWO_GCD_ATTENTION']='1'
        torch.cuda.set_device(primary);torch.manual_seed(419);torch.backends.cuda.matmul.allow_tf32=False
        report.update(torch=torch.__version__,hip=torch.version.hip)
        report['runtime_source_sha256']=_sha256(pathlib.Path(__file__).resolve().parents[2]/'fireprot/backends/amd/two_gcd_attention.py')
        def sync():
            torch.cuda.synchronize(0);torch.cuda.synchronize(1)
        with torch.inference_mode():
            q,k,v=[torch.randn(512,1536,4,32,device=f'cuda:{primary}').transpose(1,2) for _ in range(3)]
            q.mul_(32**-0.5)
            bias=torch.randn(1536,1536,4,device=f'cuda:{primary}').permute(2,0,1).unsqueeze(0)*0.1
            module=SimpleNamespace(training=False)
            def single():
                return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias.contiguous(),dropout_p=0.0,scale=1.0).transpose(1,2)
            def dual():
                return try_attention(module=module,q=q,k=k,v=v,biases=[bias])
            sync();reference=single();sync()
            side=torch.cuda.Stream(device=primary)
            probe_local=torch.cuda.Stream(device=primary)
            probe_copy=torch.cuda.Stream(device=primary)
            probe_remote=torch.cuda.Stream(device=secondary)
            def probe():
                probe_local.wait_stream(torch.cuda.current_stream(primary))
                with torch.cuda.stream(probe_local):
                    packed=bias.contiguous()
                    probe_copy.wait_stream(probe_local)
                    with torch.cuda.stream(probe_copy),torch.cuda.stream(probe_remote):
                        rq,rk,rv=[t[320:].to(f'cuda:{secondary}',non_blocking=True) for t in (q,k,v)]
                        rb=packed.to(f'cuda:{secondary}',non_blocking=True)
                        ro=torch.nn.functional.scaled_dot_product_attention(rq,rk,rv,attn_mask=rb,dropout_p=0.0,scale=1.0)
                    lo=torch.nn.functional.scaled_dot_product_attention(q[:320],k[:320],v[:320],attn_mask=packed,dropout_p=0.0,scale=1.0)
                    with torch.cuda.stream(probe_copy),torch.cuda.stream(probe_remote):
                        returned=ro.to(f'cuda:{primary}',non_blocking=True)
                    probe_local.wait_stream(probe_remote)
                    out=torch.cat((lo.transpose(1,2),returned.transpose(1,2)),dim=0)
                    probe_copy.wait_stream(probe_local)
                torch.cuda.current_stream(primary).wait_stream(probe_local)
                out.record_stream(torch.cuda.current_stream(primary))
                return out
            for name,call,stream in [('single_before',single,torch.cuda.default_stream(primary)),('probe_schedule',probe,torch.cuda.default_stream(primary)),('dual_default',dual,torch.cuda.default_stream(primary)),('dual_explicit',dual,side),('single_after',single,torch.cuda.default_stream(primary))]:
                with torch.cuda.stream(stream):
                    actual=call();sync();torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                    delta=(actual-reference).abs().max().item();del actual
                    values=[]
                    for device in (0,1):torch.cuda.reset_peak_memory_stats(device)
                    for i in range(25):
                        sync();start=time.perf_counter()
                        for _ in range(args.queued_calls):
                            result=call();del result
                        sync()
                        if i>=5:values.append((time.perf_counter()-start)*1000/args.queued_calls)
                record=dict(implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,peak_reserved={str(d):torch.cuda.max_memory_reserved(d) for d in (0,1)},peak_allocated={str(d):torch.cuda.max_memory_allocated(d) for d in (0,1)})
                report['records'].append(record);print(json.dumps({k:v for k,v in record.items() if k!='values_ms'}),flush=True)
        report.update(status='ok',runtime=manifest())
    except Exception as error:
        report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report.get('status')=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
