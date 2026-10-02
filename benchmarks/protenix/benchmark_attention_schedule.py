"""Compare shared-bias launch scheduling using prebuilt, isolated CK probes.

No candidate is registered in model dispatch. Include bias packing and the
required head-last output layout in every measurement.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import pathlib
import statistics

from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _GpuProcessMonitor, _json_load, _rocm_preflight, _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-dir', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--contiguous-qkv', action='store_true',
                        help='Probe libraries must use contiguous BHND QKV strides')
    args = parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6', '7'}:
        parser.error('Select physical GPU 6 or 7')
    report = dict(scope='N1536 attention operator scheduling diagnosis', seed=419,
                  shape=[512, 4, 1536, 32], dtype='float32', tf32=False,
                  warmups=5, repetitions=20, cache_state='warm',
                  physical_device=os.environ['HIP_VISIBLE_DEVICES'], records=[],
                  probe_contiguous_qkv=args.contiguous_qkv,
                  script_sha256=_sha256(pathlib.Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    monitor = None
    handles = []
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from fireprot.backends.amd import ck_attention
        from fireprot.backends.amd.triangle_attention import _direct_sdpa

        if ck_attention.load_extension() is None:
            raise RuntimeError(str(ck_attention.extension_status()))
        report['ck_build'] = ck_attention.extension_status()
        report['probe_build'] = _json_load(args.library_dir / 'build.json')
        measurement = _json_load(REPO_ROOT / 'benchmarks/protenix/configs/mi250-long.json')['measurement']
        report['preflight'] = _rocm_preflight(measurement)
        if not report['preflight']['exclusive_check']['passed']:
            raise RuntimeError('GPU isolation preflight failed')
        monitor = _GpuProcessMonitor(int(os.environ['HIP_VISIBLE_DEVICES']), os.getpid(), 1.0)
        monitor.start()
        if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
            raise RuntimeError('Requires gfx90a HIP')
        report.update(torch=torch.__version__, hip=torch.version.hip)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(419)
        with torch.inference_mode():
            q, k, v = [torch.randn(512, 1536, 4, 32, device='cuda').transpose(1, 2) for _ in range(3)]
            q.div_(32**0.5)
            bias = torch.randn(1536, 1536, 4, device='cuda').permute(2, 0, 1).unsqueeze(0)
            report.update(q_stride=list(q.stride()), bias_stride=list(bias.stride()))
            kw = dict(module=None, q=q, k=k, v=v, biases=[bias], native=lambda: None)
            reference = _direct_sdpa(**kw)

            def swapped_sdpa():
                packed_bias = bias.contiguous().transpose(0, 1)
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                        attn_mask=packed_bias, dropout_p=0.0, scale=1.0)
                return out.permute(1, 2, 0, 3).contiguous()

            def contiguous_sdpa():
                inputs = [x.contiguous() for x in (q, k, v)]
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        *inputs, attn_mask=bias.contiguous(), dropout_p=0.0, scale=1.0)
                return out.transpose(1, 2).contiguous()

            functions = [('sdpa', lambda: _direct_sdpa(**kw)),
                         ('ck_padded', lambda: ck_attention.attention(**kw)),
                         ('sdpa_batch_head_swap', swapped_sdpa)]
            if args.contiguous_qkv:
                functions.append(('sdpa_contiguous_qkv', contiguous_sdpa))
            report['libraries'] = {}
            contiguous_build = report['probe_build'].get('qkv_layout') == 'BHND contiguous'
            if contiguous_build != args.contiguous_qkv:
                raise RuntimeError('Probe QKV layout does not match build manifest')
            for build in report['probe_build']['builds']:
                if build['returncode'] != 0:
                    raise RuntimeError('A probe failed to compile')
                path = args.library_dir / (build['name'] + '.so')
                handle = ctypes.CDLL(str(path.resolve()))
                handles.append(handle)
                entry = handle.ck_attention
                entry.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
                entry.restype = ctypes.c_int
                report['libraries'][path.name] = _sha256(path)

                def call(entry=entry):
                    inputs = [x.contiguous() for x in (q, k, v)] if args.contiguous_qkv else (q, k, v)
                    packed = torch.empty(1, 4, 1536, 1552, device='cuda')
                    packed[..., :1536].copy_(bias)
                    out = torch.empty_like(q)
                    status = entry(*(x.data_ptr() for x in inputs), packed.data_ptr(),
                                   out.data_ptr(), 512, 1536, torch.cuda.current_stream().cuda_stream)
                    if status:
                        raise RuntimeError(f'CK HIP launch status {status}')
                    return out.transpose(1, 2)

                functions.append((f"ck_{build['name']}", call))
            functions.append(('sdpa_repeat', lambda: _direct_sdpa(**kw)))
            for name, function in functions:
                # Assertion diagnostics can retain tensor-bearing traceback
                # cycles even after the candidate output has been released.
                gc.collect()
                try:
                    result = function()
                    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
                    delta = (result-reference).abs().max().item()
                    del result
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    timings = []
                    for iteration in range(25):
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        result = function()
                        end.record()
                        end.synchronize()
                        if iteration >= 5:
                            timings.append(start.elapsed_time(end))
                        del result
                    record = dict(name=name, status='ok', median_ms=statistics.median(timings),
                                  p95_ms=sorted(timings)[18], values_ms=timings,
                                  max_abs_delta=delta, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                except (RuntimeError, AssertionError) as error:
                    record = dict(name=name, status='rejected', error=str(error))
                finally:
                    # Failed numerical gates must not retain a large candidate
                    # output during the following candidate's memory measurement.
                    result = None
                report['records'].append(record)
                print(json.dumps({k:v for k,v in record.items() if k != 'values_ms'}), flush=True)
        report['status'] = 'ok'
    except Exception as error:
        report.update(status='error', error=str(error))
        raise
    finally:
        if monitor is not None:
            report['gpu_process_monitor'] = monitor.stop()
            if not report['gpu_process_monitor']['passed']:
                report['status'] = 'contaminated'
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
