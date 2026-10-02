"""Measure the actual transposed BMM layouts used by packed triangle contraction.

Run each backend in a separate process. First write full hipBLAS references,
then validate every output element before timing other backends.
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
    parser.add_argument('--backend', choices=('hipblas', 'hipblaslt', 'ck', 'direct_ck', 'direct_lt'), required=True)
    parser.add_argument('--library', type=pathlib.Path)
    parser.add_argument('--length', type=int, choices=(1000, 1536), default=1536)
    parser.add_argument('--reference-dir', type=pathlib.Path, required=True)
    parser.add_argument('--write-reference', action='store_true')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if os.environ.get('HIP_VISIBLE_DEVICES') not in {'6', '7'}:
        parser.error('Select physical GPU 6 or 7')
    if args.write_reference and args.backend != 'hipblas':
        parser.error('Only hipBLAS may write the native reference')
    if args.backend in {'direct_ck', 'direct_lt'} and (args.library is None or args.length != 1536):
        parser.error('Direct library requires --library and length 1536')
    args.reference_dir.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, backend=args.backend, shape=[128, args.length, args.length],
                  seed=443, dtype='float32', tf32=False, warmups=5, repetitions=20,
                  cache_state='warm; reference serialization and checking excluded',
                  scope='Exact packed triangle contraction layouts; not model performance',
                  script_sha256=_sha256(pathlib.Path(__file__)), records=[])
    monitor = None
    try:
        measurement = _json_load(REPO_ROOT / 'benchmarks/protenix/configs/mi250-long1536.json')['measurement']
        report['preflight'] = _rocm_preflight(measurement)
        if not report['preflight']['exclusive_check']['passed']:
            raise RuntimeError('GPU isolation preflight failed')
        monitor = _GpuProcessMonitor(int(os.environ['HIP_VISIBLE_DEVICES']), os.getpid(), 1.0)
        monitor.start()
        import torch
        from fireprot.backends.amd.blas import configure_process_blas

        if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx90a':
            raise RuntimeError('Requires gfx90a HIP')
        report['policy'] = configure_process_blas('hipblas' if args.backend in {'direct_ck', 'direct_lt'} else args.backend)
        report.update(torch=torch.__version__, hip=torch.version.hip)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(443)
        n = args.length
        direct = None
        if args.backend in {'direct_ck', 'direct_lt'}:
            library = ctypes.CDLL(str(args.library.resolve()))
            direct = getattr(library, 'ck_packed_bmm' if args.backend == 'direct_ck' else 'lt_packed_bmm')
            direct.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
            direct.restype = ctypes.c_int
            report['library_sha256'] = _sha256(args.library)
            report['library'] = str(args.library.resolve())
            if args.backend == 'direct_lt':
                report['external_workspace_bytes'] = 2 * 32 * 1024 * 1024
                report['memory_note'] = 'Two persistent Lt workspaces are outside PyTorch peak allocation; diagnostic only'
        with torch.inference_mode():
            a, b = [torch.randn(128, n, n, device='cuda').div_(n**0.5) for _ in range(2)]
            for direction in ('outgoing', 'incoming'):
                reference_path = args.reference_dir / f'n{n}-{direction}.pt'
                metadata_path = reference_path.with_suffix('.json')
                left, right = (a, b.transpose(-1, -2)) if direction == 'outgoing' else (a.transpose(-1, -2), b)
                record = dict(direction=direction, left_stride=list(left.stride()), right_stride=list(right.stride()))
                def contract():
                    if direct is None:
                        return torch.bmm(left, right)
                    output = torch.empty_like(a)
                    status = direct(a.data_ptr(), b.data_ptr(), output.data_ptr(), 128, n,
                                    int(direction == 'outgoing'), torch.cuda.current_stream().cuda_stream)
                    if status:
                        raise RuntimeError(f'Direct library returned status {status}')
                    return output

                result = reference = None
                try:
                    result = contract()
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(result).all().item()):
                        raise RuntimeError('Nonfinite output')
                    identity = dict(seed=443, shape=[128, n, n], dtype='float32', tf32=False,
                                    torch=torch.__version__, hip=torch.version.hip, direction=direction)
                    if args.write_reference:
                        if reference_path.exists() or metadata_path.exists():
                            raise RuntimeError(f'Reference already exists: {reference_path}')
                        torch.save(result.cpu(), reference_path)
                        metadata_path.write_text(json.dumps(identity | {'sha256': _sha256(reference_path)}, indent=2) + '\n')
                        delta = 0.0
                    else:
                        metadata = _json_load(metadata_path)
                        if any(metadata.get(k) != v for k, v in identity.items()):
                            raise RuntimeError('Reference identity mismatch')
                        if _sha256(reference_path) != metadata['sha256']:
                            raise RuntimeError('Reference file hash mismatch')
                        reference = torch.load(reference_path, map_location='cuda', weights_only=True)
                        torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
                        delta = (result-reference).abs().max().item()
                    result = reference = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    values = []
                    for iteration in range(25):
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        result = contract()
                        end.record()
                        end.synchronize()
                        if iteration >= 5:
                            values.append(start.elapsed_time(end))
                        result = None
                    record.update(status='ok', median_ms=statistics.median(values),
                                  p95_ms=sorted(values)[18], values_ms=values, max_abs_delta=delta,
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  reference_metadata=_json_load(metadata_path))
                except (RuntimeError, AssertionError) as error:
                    record.update(status='rejected', error=str(error))
                finally:
                    result = reference = None
                report['records'].append(record)
                print(json.dumps({k:v for k,v in record.items() if k not in {'values_ms', 'reference_metadata'}}), flush=True)
                gc.collect()
        report['status'] = 'ok' if all(r['status'] == 'ok' for r in report['records']) else 'rejected'
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
