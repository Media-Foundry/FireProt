"""Measure complete N1536 attention across outer chunk sizes, including assembly."""
from __future__ import annotations
import argparse
import gc
import json
import os
import pathlib
import statistics
from benchmarks.protenix.profile_inference import REPO_ROOT, _rocm_preflight, _GpuProcessMonitor, _sha256, _json_load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6','7'}:parser.error('Select physical GPU 6 or 7')
    os.environ['LAYERNORM_TYPE']='torch'
    import torch
    from torch.nn import functional as F
    from protenix.model.triangular.layers import Attention
    from fireprot.backends.amd.blas import configure_process_blas
    measurement=_json_load(REPO_ROOT/'benchmarks/protenix/configs/mi250-long1536.json')['measurement']
    preflight=_rocm_preflight(measurement)
    if not preflight['exclusive_check']['passed']:raise RuntimeError('GPU isolation failed')
    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':raise RuntimeError('Requires gfx90a HIP')
    monitor=_GpuProcessMonitor(int(os.environ['HIP_VISIBLE_DEVICES']),os.getpid(),1.0);monitor.start()
    report=dict(scope='Standalone full outer-batch attention; not model performance',shape=[1536,1536,128],
                dtype='float32',tf32=False,seed=727,weight_std=0.02,warmups=5,repetitions=20,
                cache_state='warm; includes projections, per-chunk bias packing, gate and output assembly',
                preflight=preflight,torch=torch.__version__,hip=torch.version.hip,
                script_sha256=_sha256(pathlib.Path(__file__)),
                attention_source_sha256=_sha256(REPO_ROOT/'Protenix/protenix/model/triangular/layers.py'),records=[])
    try:
        report['blas']=configure_process_blas('hipblas')
        torch.manual_seed(727);torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode():
            module=Attention(128,128,128,32,4).cuda().eval()
            for parameter in module.parameters():parameter.normal_(std=0.02)
            x=torch.randn(1536,1536,128,device='cuda')
            bias=(torch.randn(1536,1536,4,device='cuda')*0.1).permute(2,0,1).unsqueeze(0)
            report['input_stride']=list(x.stride());report['bias_stride']=list(bias.stride())
            def attention(part):
                q,k,v=module._prep_qkv(part,part)
                value=F.scaled_dot_product_attention(q,k,v,attn_mask=bias.contiguous(),dropout_p=0.0,scale=1.0)
                return module._wrap_up(value.transpose(1,2),part)
            def call(chunk):
                if chunk==1536:return attention(x)
                result=torch.empty_like(x)
                for start in range(0,1536,chunk):
                    result[start:start+chunk].copy_(attention(x[start:start+chunk]))
                return result
            for label,chunk in [('baseline',512),('smaller',256),('larger',768),('full',1536),('baseline_repeat',512)]:
                expected=call(512);actual=call(chunk)
                torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
                delta=(actual-expected).abs().max().item()
                del expected,actual
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                values=[]
                for i in range(25):
                    start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record();result=call(chunk);end.record();end.synchronize()
                    if i>=5:values.append(start.elapsed_time(end))
                    del result
                record=dict(label=label,chunk=chunk,median_ms=statistics.median(values),p95_ms=sorted(values)[18],
                            values_ms=values,max_abs_delta=delta,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                report['records'].append(record);print(json.dumps({k:v for k,v in record.items() if k!='values_ms'}),flush=True)
        report['status']='ok'
    except Exception as error:
        report.update(status='error',error=str(error));raise
    finally:
        report['gpu_process_monitor']=monitor.stop()
        if not report['gpu_process_monitor']['passed']:report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
