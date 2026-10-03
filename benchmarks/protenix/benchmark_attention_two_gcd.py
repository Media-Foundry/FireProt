"""Experimental two-GCD SDPA split, including input/output peer transfers.

Requires physical GPUs 6 and 7; compares two GCDs against one GCD, not a
single-kernel speedup. No model dispatch is changed.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import pathlib
import statistics
import time
from benchmarks.protenix.profile_inference import REPO_ROOT, _rocm_preflight, _GpuProcessMonitor, _sha256, _json_load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--trace',type=pathlib.Path,help='Optional untimed two-device Chrome trace')
    parser.add_argument('--overlap-inputs',action='store_true',help='Use a separate primary copy stream')
    parser.add_argument('--remote-batch',type=int,choices=(128,192,256,320,384),default=256)
    args=parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES')!='6,7':parser.error('Requires HIP_VISIBLE_DEVICES=6,7')
    measurement=_json_load(REPO_ROOT/'benchmarks/protenix/configs/mi250-long1536.json')['measurement']
    preflight=_rocm_preflight(measurement)
    if not preflight['exclusive_check']['passed']:raise RuntimeError('GPU 6 is unavailable')
    card=preflight['data']['card7']
    second_check=dict(physical_device=7,gpu_use=int(card['GPU use (%)']),vram_used_bytes=int(card['VRAM Total Used Memory (B)']))
    second_check['passed']=second_check['gpu_use']<=5 and second_check['vram_used_bytes']<=268435456
    if not second_check['passed']:raise RuntimeError('GPU 7 is unavailable')
    monitors=[_GpuProcessMonitor(d,os.getpid(),1.0) for d in (6,7)]
    for monitor in monitors:monitor.start()
    report=dict(scope='One-GCD versus two-GCD attention, includes peer transfers; not model speedup',
                devices={'primary':7,'secondary':6},seed=419,dtype='float32',tf32=False,
                shape=[512,4,1536,32],remote_batch=args.remote_batch,overlap_inputs=args.overlap_inputs,warmups=5,repetitions=20,cache_state='warm; remote tensors copied each call',
                timing='wall time through synchronization of both devices',preflight=preflight,
                second_preflight=second_check,script_sha256=_sha256(pathlib.Path(__file__)),records=[])
    try:
        import torch
        from torch.nn import functional as F
        if not torch.version.hip or torch.cuda.device_count()!=2:raise RuntimeError('Requires two visible HIP devices')
        if any(torch.cuda.get_device_properties(d).gcnArchName.split(':')[0]!='gfx90a' for d in (0,1)):raise RuntimeError('Requires gfx90a')
        report.update(torch=torch.__version__,hip=torch.version.hip,
                      peer_access=[torch.cuda.can_device_access_peer(0,1),torch.cuda.can_device_access_peer(1,0)])
        if not all(report['peer_access']):raise RuntimeError('Bidirectional peer access is required')
        torch.cuda.set_device(1);torch.manual_seed(419);torch.backends.cuda.matmul.allow_tf32=False
        local=torch.cuda.Stream(device=1);remote=torch.cuda.Stream(device=0)
        transfer=torch.cuda.Stream(device=1) if args.overlap_inputs else local
        cut=512-args.remote_batch
        def synchronize():
            torch.cuda.synchronize(0);torch.cuda.synchronize(1)
        with torch.inference_mode():
            q,k,v=[torch.randn(512,1536,4,32,device='cuda:1').transpose(1,2) for _ in range(3)]
            q.mul_(32**-0.5)
            bias=torch.randn(1536,1536,4,device='cuda:1').permute(2,0,1).unsqueeze(0)*0.1
            def single():
                return F.scaled_dot_product_attention(q,k,v,attn_mask=bias.contiguous(),dropout_p=0.0,scale=1.0)
            def split():
                local.wait_stream(torch.cuda.current_stream(1))
                with torch.cuda.stream(local):
                    packed_bias=bias.contiguous()
                    # Submit transfers before local compute so source-side copy
                    # dependencies cannot serialize the two attention kernels.
                    if args.overlap_inputs:transfer.wait_stream(local)
                    with torch.cuda.stream(transfer), torch.cuda.stream(remote):
                        rq=q[cut:].to('cuda:0',non_blocking=True)
                        rk=k[cut:].to('cuda:0',non_blocking=True)
                        rv=v[cut:].to('cuda:0',non_blocking=True)
                        rb=packed_bias.to('cuda:0',non_blocking=True)
                        ro=F.scaled_dot_product_attention(rq,rk,rv,attn_mask=rb,dropout_p=0.0,scale=1.0)
                    lo=F.scaled_dot_product_attention(q[:cut],k[:cut],v[:cut],attn_mask=packed_bias,dropout_p=0.0,scale=1.0)
                    with torch.cuda.stream(transfer), torch.cuda.stream(remote):
                        returned=ro.to('cuda:1',non_blocking=True)
                    local.wait_stream(remote)
                    output=torch.cat((lo.transpose(1,2),returned.transpose(1,2)),dim=0).transpose(1,2)
                torch.cuda.current_stream(1).wait_stream(local)
                return output
            synchronize();expected=single();synchronize();actual=split();synchronize()
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            report['max_abs_delta']=(actual-expected).abs().max().item()
            del expected,actual
            gc.collect()
            for d in (0,1):
                with torch.cuda.device(d):torch.cuda.empty_cache()
            for name,call in [('single_before',single),('dual',split),('single_after',single)]:
                for d in (0,1):torch.cuda.reset_peak_memory_stats(d)
                values=[]
                for i in range(25):
                    synchronize();start=time.perf_counter();result=call();synchronize()
                    elapsed=(time.perf_counter()-start)*1000
                    if i>=5:values.append(elapsed)
                    del result
                record=dict(implementation=name,median_ms=statistics.median(values),p95_ms=sorted(values)[18],
                            values_ms=values,peak_allocated_bytes={str(d):torch.cuda.max_memory_allocated(d) for d in (0,1)})
                report['records'].append(record);print(json.dumps({k:v for k,v in record.items() if k!='values_ms'}),flush=True)
            if args.trace is not None:
                synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                    result=split();synchronize()
                del result
                args.trace.parent.mkdir(parents=True,exist_ok=True)
                profile.export_chrome_trace(str(args.trace))
                report['trace']=dict(path=str(args.trace),sha256=_sha256(args.trace),scope='Untimed diagnostic call after measurements')
        report['status']='ok'
    except Exception as error:
        report.update(status='error',error=str(error));raise
    finally:
        report['gpu_process_monitors']=[m.stop() for m in monitors]
        if not all(m['passed'] for m in report['gpu_process_monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
