"""Replay the best N1536 CK candidates and SDPA with contiguous launch groups.

Includes per-call bias packing and output allocation. Group averages describe
queued operator throughput, not full-model latency. Uses a hash-verified probe.
"""
from __future__ import annotations
import argparse
import ctypes
import gc
import json
import os
import pathlib
import statistics
from benchmarks.protenix.profile_inference import REPO_ROOT, _rocm_preflight, _additional_device_preflight, _GpuProcessMonitor, _sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=pathlib.Path,default=REPO_ROOT/'runtime/ck_bias_stride_benchmark/probe.so')
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--build-manifest',type=pathlib.Path,help='Optional key-tile build manifest')
    args=parser.parse_args()
    visible=os.environ.get('HIP_VISIBLE_DEVICES')
    if visible not in {'6','7'}:parser.error('Select physical GPU 6 or 7 only')
    # Pinned to provenance/protenix_attention_bias_stride_fb06beb.json.
    # Runtime deployments intentionally omit the top-level provenance tree.
    previous=dict(library_sha256='1cfdf44c81586cb9d37fae91d55b8bc503a31dcda5badffb539a28729df56b10',
                  source_sha256='c4f4b6e0f3322e6b4207922e4f2900042001d31a72996043a70a3d1ac6191d58')
    if _sha256(args.library)!=previous['library_sha256'] or _sha256(args.library.with_suffix('.cpp'))!=previous['source_sha256']:
        raise RuntimeError('Requires the recorded M128/N64/W4 batch-group-8 probe')
    policy=dict(device_policy='exclusive')
    preflight=_rocm_preflight(policy)
    if not preflight['exclusive_check']['passed']:raise RuntimeError('Primary GPU unavailable')
    sibling=7 if visible=='6' else 6
    extra=_additional_device_preflight([sibling],policy)
    monitors=[_GpuProcessMonitor(d,os.getpid(),1.0) for d in (int(visible),sibling)]
    for monitor in monitors:monitor.start()
    report=dict(scope=__doc__,shape=[512,4,1536,32],seed=419,dtype='float32',tf32=False,
                warmup_groups=5,measured_groups=20,cache_state='warm',physical_device=int(visible),
                allocator=os.environ.get('PYTORCH_ALLOC_CONF'),timing='GPU event time per call within each group',
                preflight=preflight,sibling_preflight=extra,library_sha256=_sha256(args.library),
                probe_source_sha256=previous['source_sha256'],script_sha256=_sha256(pathlib.Path(__file__)),records=[])
    try:
        import torch
        from torch.nn import functional as F
        if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx90a':raise RuntimeError('Requires gfx90a HIP')
        torch.manual_seed(419);torch.backends.cuda.matmul.allow_tf32=False
        report.update(torch=torch.__version__,hip=torch.version.hip)
        library=ctypes.CDLL(str(args.library.resolve()));entry=library.ck_attention
        entry.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p];entry.restype=ctypes.c_int
        entries=[('ck',entry)]
        handles=[library]
        if args.build_manifest is not None:
            build=json.loads(args.build_manifest.read_text())
            if build.get('base_source_sha256',build['source_sha256'])!=previous['source_sha256'] or _sha256(pathlib.Path(build['source']))!=build['source_sha256']:
                raise RuntimeError('Unverified key-tile source')
            report['key_tile_build']=build;entries=[]
            for item in build['builds']:
                if item['returncode']!=0:continue
                path=pathlib.Path(item['library'])
                if _sha256(path)!=item['library_sha256']:raise RuntimeError('Key-tile library hash mismatch')
                handle=ctypes.CDLL(str(path));handles.append(handle)
                function=handle.ck_attention
                function.argtypes=entry.argtypes;function.restype=ctypes.c_int
                entries.append((item['name'],function))
            if not entries:raise RuntimeError('No key-tile candidates compiled')
        with torch.inference_mode():
            q,k,v=[torch.randn(512,1536,4,32,device='cuda').transpose(1,2) for _ in range(3)]
            q.div_(32**0.5)
            bias=torch.randn(1536,1536,4,device='cuda').permute(2,0,1).unsqueeze(0)
            report.update(qkv_strides=[list(t.stride()) for t in (q,k,v)],bias_stride=list(bias.stride()))
            def sdpa():
                return F.scaled_dot_product_attention(q,k,v,attn_mask=bias.contiguous(),dropout_p=0.0,scale=1.0)
            def ck(padding,function):
                packed=torch.empty(1,4,1536,1536+padding,device='cuda');packed[...,:1536].copy_(bias)
                output=torch.empty_like(q)
                status=function(q.data_ptr(),k.data_ptr(),v.data_ptr(),packed.data_ptr(),output.data_ptr(),512,1536,padding,torch.cuda.current_stream().cuda_stream)
                if status:raise RuntimeError(f'CK HIP launch returned {status}')
                return output
            reference=sdpa();torch.cuda.synchronize()
            candidates=[('sdpa_before',sdpa)]
            candidates.extend((f'{name}_pad{padding}',lambda f=function,p=padding:ck(p,f)) for name,function in entries for padding in (8,16))
            candidates.append(('sdpa_after',sdpa))
            for group in (1,8):
                for name,call in candidates:
                    actual=call();torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
                    delta=(actual-reference).abs().max().item();del actual
                    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    times=[]
                    for iteration in range(25):
                        start,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        for _ in range(group):
                            output=call();del output
                        end.record();end.synchronize()
                        if iteration>=5:times.append(start.elapsed_time(end)/group)
                    item=dict(implementation=name,group_size=group,median_ms=statistics.median(times),p95_ms=sorted(times)[18],values_ms=times,max_abs_delta=delta,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    report['records'].append(item);print(json.dumps({k:v for k,v in item.items() if k!='values_ms'}),flush=True)
        report['status']='ok'
    except Exception as error:
        report.update(status='error',error=str(error));raise
    finally:
        report['monitors']=[monitor.stop() for monitor in monitors]
        if not all(m['passed'] for m in report['monitors']):report['status']='contaminated'
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='ok' else 1


if __name__=='__main__':raise SystemExit(main())
