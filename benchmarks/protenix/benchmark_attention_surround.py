"""Compare N1536 triangle-attention projections and gate fusion, including copies."""
from __future__ import annotations
import argparse
import ctypes
import gc
import json
import math
import os
import pathlib
import statistics
from benchmarks.protenix.profile_inference import REPO_ROOT, _rocm_preflight, _GpuProcessMonitor, _sha256, _json_load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    args=p.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6','7'}: p.error('Select GPU 6 or 7')
    import torch
    from torch.nn import functional as F
    from fireprot.backends.amd.blas import configure_process_blas
    os.environ["LAYERNORM_TYPE"]="torch"
    from protenix.model.triangular.layers import Attention
    config=_json_load(REPO_ROOT/'benchmarks/protenix/configs/mi250-long1536.json')
    preflight=_rocm_preflight(config['measurement'])
    if not preflight['exclusive_check']['passed']: raise RuntimeError('GPU isolation failed')
    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':
        raise RuntimeError('Requires gfx90a HIP')
    monitor=_GpuProcessMonitor(int(os.environ['HIP_VISIBLE_DEVICES']),os.getpid(),1.0);monitor.start()
    report=dict(scope='Standalone attention with real N1536 chunk shape; not model speedup',
                seed=719,dtype='float32',tf32=False,shape=[512,1536,128],warmups=5,repetitions=20,
                cache_state='warm; merged weights cached before timing; layout copies included',
                preflight=preflight,torch=torch.__version__,hip=torch.version.hip,
                script_sha256=_sha256(pathlib.Path(__file__)),library_sha256=_sha256(args.library),records=[])
    try:
        report['blas']=configure_process_blas('hipblas')
        torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(719)
        lib=ctypes.CDLL(str(args.library.resolve()));entry=lib.attention_gate
        entry.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int64,ctypes.c_void_p];entry.restype=ctypes.c_int
        split=lib.qkv_split_scale
        split.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int64,ctypes.c_void_p];split.restype=ctypes.c_int
        with torch.inference_mode():
            module=Attention(128,128,128,32,4).cuda().eval()
            for weight in module.parameters(): weight.normal_(std=0.02)
            x=torch.randn(512,1536,128,device='cuda')
            o=torch.randn(512,1536,4,32,device='cuda')
            bias=torch.randn(1,4,1536,1536,device='cuda')*0.1
            combined=torch.cat([module.linear_q.weight,module.linear_k.weight,module.linear_v.weight])
            def prep_native(): return module._prep_qkv(x,x)
            def prep_combined():
                packed=F.linear(x,combined)
                q,k,v=[part.contiguous().view(512,1536,4,32).transpose(1,2) for part in packed.split(128,-1)]
                q.div_(math.sqrt(32))
                return q,k,v
            def prep_combined_views():
                packed=F.linear(x,combined)
                q,k,v=[part.view(512,1536,4,32).transpose(1,2) for part in packed.split(128,-1)]
                q.div_(math.sqrt(32))
                return q,k,v
            def prep_combined_fused():
                packed=F.linear(x,combined)
                outputs=[torch.empty_like(x) for _ in range(3)]
                status=split(packed.data_ptr(),*[t.data_ptr() for t in outputs],512*1536,torch.cuda.current_stream().cuda_stream)
                if status: raise RuntimeError(f'Split kernel status {status}')
                return tuple(t.view(512,1536,4,32).transpose(1,2) for t in outputs)
            def wrap_native(value=o): return module._wrap_up(value,x)
            def wrap_fused(value=o):
                gate=module.linear_g(x)
                value=value.reshape(512,1536,128)
                out=torch.empty_like(value)
                status=entry(gate.data_ptr(),value.data_ptr(),out.data_ptr(),out.numel(),torch.cuda.current_stream().cuda_stream)
                if status: raise RuntimeError(f'Gate kernel status {status}')
                return module.linear_o(out)
            def full(prep,wrap):
                q,k,v=prep()
                value=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.0,scale=1.0).transpose(1,2)
                return wrap(value)
            groups=[('qkv',prep_native,prep_combined),('qkv_fused_split',prep_native,prep_combined_fused),
                    ('qkv_views',prep_native,prep_combined_views),('wrap',wrap_native,wrap_fused),
                    ('attention_gate_only',lambda:full(prep_native,wrap_native),lambda:full(prep_native,wrap_fused)),
                    ('attention_fused_split',lambda:full(prep_native,wrap_native),lambda:full(prep_combined_fused,wrap_fused)),
                    ('attention_qkv_views',lambda:full(prep_native,wrap_native),lambda:full(prep_combined_views,wrap_fused))]
            for label,native,candidate in groups:
                expected=native();actual=candidate()
                left=expected if isinstance(expected,tuple) else (expected,)
                right=actual if isinstance(actual,tuple) else (actual,)
                delta=0.0
                for a,b in zip(left,right):
                    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
                    delta=max(delta,(a-b).abs().max().item())
                del expected,actual,left,right,a,b
                gc.collect();torch.cuda.empty_cache()
                for name,call in [('native',native),('candidate',candidate),('native_repeat',native)]:
                    torch.cuda.reset_peak_memory_stats();values=[]
                    for i in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record();result=call();end.record();end.synchronize()
                        if i>=5: values.append(start.elapsed_time(end))
                        del result
                    record=dict(group=label,implementation=name,median_ms=statistics.median(values),
                                p95_ms=sorted(values)[18],values_ms=values,max_abs_delta=delta,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    report['records'].append(record);print(json.dumps({k:v for k,v in record.items() if k!='values_ms'}),flush=True)
        report['status']='ok'
    except Exception as error:
        report.update(status='error',error=str(error));raise
    finally:
        report['gpu_process_monitor']=monitor.stop()
        if not report['gpu_process_monitor']['passed']:report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__': raise SystemExit(main())
