"""Scan physical CK bias row strides with fixed M128/N64/W4 and batch group 8."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import statistics
import subprocess

from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _GpuProcessMonitor, _json_load, _rocm_preflight, _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule-dir', required=True, type=pathlib.Path,
                        help='Private overlay produced by build_attention_schedule')
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6', '7'}:
        parser.error('Select physical GPU 6 or 7')
    base = args.schedule_dir.resolve()
    source = (base / 'probe.cpp').read_text()
    manifest = _json_load(base / 'build.json')
    if manifest.get('qkv_layout') != 'BSHD' or _sha256(base / 'probe.cpp') != manifest['source_sha256']:
        raise RuntimeError('Requires a verified BSHD schedule probe')
    relative = pathlib.Path('ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp')
    if _sha256(base / relative) != manifest['patched_header_sha256']:
        raise RuntimeError('Schedule header differs from build manifest')
    for old, new in (
        ('int batch,int n,void* stream', 'int batch,int n,int padding,void* stream'),
        ('n + 16 : n', 'n + padding : n'),
    ):
        if source.count(old) != 1:
            raise RuntimeError(f'Unexpected source pattern: {old}')
        source = source.replace(old, new)
    build = REPO_ROOT / 'runtime/ck_bias_stride_benchmark'
    build.mkdir(parents=True, exist_ok=True)
    probe = build / 'probe.cpp'
    probe.write_text(source)
    library = build / 'probe.so'
    command = ['/opt/rocm/bin/hipcc', '-O3', '-std=c++17', '-fPIC', '-shared',
               '--offload-arch=gfx90a', f'-I{base}', '-I/opt/rocm/include',
               '-DBG=8', '-DBM=128', '-DWM=4', str(probe), '-o', str(library)]
    with (build / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = dict(scope='Complete N1536 operator; physical bias padding only',
                  shape=[512, 4, 1536, 32], seed=419, dtype='float32', tf32=False,
                  warmups=5, repetitions=20, cache_state='warm; compilation excluded',
                  physical_device=os.environ['HIP_VISIBLE_DEVICES'], records=[],
                  command=command, source_sha256=_sha256(probe),
                  script_sha256=_sha256(pathlib.Path(__file__)),
                  library_sha256=_sha256(library), schedule_manifest=manifest)
    monitor = None
    try:
        measurement = _json_load(REPO_ROOT / 'benchmarks/protenix/configs/mi250-long1536.json')['measurement']
        report['preflight'] = _rocm_preflight(measurement)
        if not report['preflight']['exclusive_check']['passed']:
            raise RuntimeError('GPU isolation preflight failed')
        monitor = _GpuProcessMonitor(int(os.environ['HIP_VISIBLE_DEVICES']), os.getpid(), 1.0)
        monitor.start()
        import torch
        from fireprot.backends.amd.triangle_attention import _direct_sdpa

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(419)
        report.update(torch=torch.__version__, hip=torch.version.hip)
        if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
            raise RuntimeError('Requires gfx90a HIP')
        handle = ctypes.CDLL(str(library))
        entry = handle.ck_attention
        entry.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        entry.restype = ctypes.c_int
        with torch.inference_mode():
            q, k, v = [torch.randn(512, 1536, 4, 32, device='cuda').transpose(1, 2) for _ in range(3)]
            q.div_(32**0.5)
            bias = torch.randn(1536, 1536, 4, device='cuda').permute(2, 0, 1).unsqueeze(0)
            kw = dict(module=None, q=q, k=k, v=v, biases=[bias], native=lambda: None)
            reference = _direct_sdpa(**kw)
            for padding in (None, 0, 4, 8, 16, 32, 48, 64, 96, 128, 256, None):
                def call():
                    if padding is None:
                        return _direct_sdpa(**kw)
                    packed = torch.empty(1, 4, 1536, 1536 + padding, device='cuda')
                    packed[..., :1536].copy_(bias)
                    output = torch.empty_like(q)
                    status = entry(q.data_ptr(), k.data_ptr(), v.data_ptr(), packed.data_ptr(),
                                   output.data_ptr(), 512, 1536, padding, torch.cuda.current_stream().cuda_stream)
                    if status:
                        raise RuntimeError(f'HIP launch status {status}')
                    return output.transpose(1, 2)

                result = call()
                torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
                delta = (result-reference).abs().max().item()
                del result
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                values = []
                for iteration in range(25):
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    result = call()
                    end.record()
                    end.synchronize()
                    if iteration >= 5:
                        values.append(start.elapsed_time(end))
                    del result
                record = dict(implementation='sdpa' if padding is None else 'ck', padding=padding,
                              median_ms=statistics.median(values), p95_ms=sorted(values)[18],
                              values_ms=values, max_abs_delta=delta,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
