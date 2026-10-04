#!/usr/bin/env python3
"""Compare chunked and packed long triangle updates with controlled random weights."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics

from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _GpuProcessMonitor, _git_commit, _json_load, _rocm_preflight,
    _selected_physical_device, _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--blas-backend", default="hipblas", choices=["hipblas", "hipblaslt", "ck"])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--length", type=int, choices=(1000, 1536), default=1000)
    parser.add_argument("--include-native", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be nonnegative and repetitions positive")
    measurement = _json_load(REPO_ROOT / "benchmarks/protenix/configs/mi250-long.json")["measurement"]
    preflight = _rocm_preflight(measurement)
    if not preflight["exclusive_check"]["passed"]:
        raise RuntimeError("Exclusive-device preflight failed")
    _, physical_index = _selected_physical_device(os.environ)
    monitor = _GpuProcessMonitor(physical_index, os.getpid(), 1.0)
    monitor.start()
    os.environ["LAYERNORM_TYPE"] = "torch"
    report = {"schema_version": 1, "status": "running", "preflight": preflight}
    try:
        import torch
        from fireprot.adapters.protenix.runtime import install
        from fireprot.backends.amd import triangle_multiplication as tm
        from fireprot.backends.amd.extension import extension_status, load_amd_extension
        from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate

        install(args.blas_backend)
        if load_amd_extension() is None:
            raise RuntimeError(str(extension_status()))
        torch.manual_seed(307)
        torch.backends.cuda.matmul.allow_tf32 = False
        records = []
        with torch.inference_mode():
            for outgoing in (True, False):
                module = TriangleMultiplicativeUpdate(128, 128, outgoing).cuda().eval()
                for parameter in module.parameters():
                    parameter.normal_(0, 0.1)
                source = torch.randn(args.length, args.length, 128, device="cuda")
                target = torch.empty_like(source)
                outputs = {}
                functions = [
                    ("chunked", tm._long_sequence_hip_fused),
                    ("packed", tm._long_sequence_packed),
                ]
                if args.include_native:
                    functions.insert(0, ("native", lambda **kw: module(
                        kw["z"], inplace_safe=True, _add_with_inplace=True)))
                for name, function in functions:
                    timings = []
                    torch.cuda.reset_peak_memory_stats()
                    for iteration in range(args.warmups + args.repetitions):
                        target.copy_(source)
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        result = function(
                            module=module, z=target, mask=None, inplace_safe=True,
                            add_with_inplace=True, inplace_chunk_size=256, native=lambda: None,
                        )
                        end.record()
                        end.synchronize()
                        assert result.data_ptr() == target.data_ptr()
                        if iteration >= args.warmups:
                            timings.append(start.elapsed_time(end))
                    outputs[name] = result.cpu()
                    ordered = sorted(timings)
                    records.append(dict(
                        outgoing=outgoing, implementation=name,
                        values_ms=timings, median_ms=statistics.median(timings),
                        p95_ms=ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)],
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    ))
                reference = outputs["native"] if args.include_native else outputs["chunked"]
                delta = (reference - outputs["packed"]).abs()
                torch.testing.assert_close(outputs["packed"], reference, rtol=2e-5, atol=1e-5)
                records.append(dict(outgoing=outgoing, max_abs_delta=delta.max().item(),
                                    rms_delta=delta.square().mean().sqrt().item()))
                del source, target, result, module, outputs
                torch.cuda.empty_cache()
        report.update(
            status="ok", seed=307, shape=[args.length, args.length, 128], dtype="float32", tf32=False,
            parameters="random nonzero normal(0,0.1); not checkpoint weights",
            warmups=args.warmups, repetitions=args.repetitions,
            cache_state="warm within process; source restored before each event",
            blas_backend=args.blas_backend, records=records,
            torch=torch.__version__, hip=torch.version.hip,
            source_commit=os.environ.get("FIREPROT_SOURCE_COMMIT") or _git_commit(REPO_ROOT),
            source_sha256={name: _sha256(REPO_ROOT / name) for name in (
                "benchmarks/protenix/benchmark_triangle.py",
                "python/fireprot/backends/amd/triangle_multiplication.py",
                "kernels/amd/fireprot_amd_ops.cu",
                "kernels/amd/fireprot_amd_ops.cpp",
            )},
        )
    except Exception as error:
        report.update(status="error", error=str(error))
        raise
    finally:
        report["gpu_process_monitor"] = monitor.stop()
        if not report["gpu_process_monitor"]["passed"]:
            report["status"] = "contaminated"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
