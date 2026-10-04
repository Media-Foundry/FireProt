#!/usr/bin/env python3
"""Measure complete N1536 pair LayerNorm and gated transition operations."""
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
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be nonnegative and repetitions positive")
    measurement = _json_load(REPO_ROOT / "benchmarks/protenix/configs/mi250-long1536.json")["measurement"]
    preflight = _rocm_preflight(measurement)
    if not preflight["exclusive_check"]["passed"]:
        raise RuntimeError("Exclusive-device preflight failed")
    _, physical = _selected_physical_device(os.environ)
    monitor = _GpuProcessMonitor(physical, os.getpid(), 1.0)
    monitor.start()
    report = dict(schema_version=1, status="running", preflight=preflight,
                  shape=[1536, 1536, 128], dtype="float32", seed=641, tf32=False,
                  warmups=args.warmups, repetitions=args.repetitions,
                  scope="complete random-weight operators; not model performance",
                  cache_state="warm within process; compilation excluded", records=[])
    try:
        os.environ["LAYERNORM_TYPE"] = "torch"
        os.environ["FIREPROT_ENABLE_LONG1536"] = "1"
        os.environ["FIREPROT_ENABLE_LONG1536_EPILOGUES"] = "1"
        import torch
        import torch.nn.functional as F
        from fireprot.adapters.protenix.runtime import install
        from fireprot.backends.amd import recycle
        from fireprot.backends.amd.extension import extension_status, load_amd_extension
        from protenix.model.modules.primitives import Transition

        install("hipblas")
        if load_amd_extension() is None:
            raise RuntimeError(str(extension_status()))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(641)
        with torch.inference_mode():
            module = Transition(c_in=128, n=4).cuda().eval()
            for parameter in module.parameters():
                parameter.normal_(0, 0.05)
            x = torch.randn(1536, 1536, 128, device="cuda")
            norm = module.layernorm1

            def native_norm():
                return F.layer_norm(x, (128,), norm.weight, norm.bias, norm.eps)

            def native_transition():
                normalized = native_norm()
                return F.linear(F.silu(F.linear(normalized, module.linear_no_bias_a.weight))
                                * F.linear(normalized, module.linear_no_bias_b.weight),
                                module.linear_no_bias.weight)

            def reject_native():
                raise RuntimeError("native fallback")

            for operation, native, candidate in (
                ("layer_norm", native_norm, lambda: recycle.dispatch_layer_norm(
                    module=norm, x=x, native=reject_native)),
                ("gated_transition", native_transition, lambda: recycle.dispatch_transition(
                    module=module, x=x, native=reject_native)),
            ):
                reference = native()
                for implementation, function in (("native", native), ("hip", candidate), ("native_repeat", native)):
                    result = None
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    values = []
                    for iteration in range(args.warmups + args.repetitions):
                        result = None
                        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                        start.record()
                        result = function()
                        end.record()
                        end.synchronize()
                        if iteration >= args.warmups:
                            values.append(start.elapsed_time(end))
                    peak = torch.cuda.max_memory_allocated()
                    torch.testing.assert_close(result, reference, atol=1e-5, rtol=2e-5)
                    record = dict(operation=operation, implementation=implementation,
                                  median_ms=statistics.median(values),
                                  p95_ms=sorted(values)[(95 * len(values) + 99) // 100 - 1],
                                  values_ms=values, peak_allocated_bytes=peak,
                                  max_abs_delta=(result-reference).abs().max().item())
                    report["records"].append(record)
                    print(json.dumps({k: v for k, v in record.items() if k != "values_ms"}), flush=True)
                    del result
                del reference
        report.update(status="ok", torch=torch.__version__, hip=torch.version.hip,
                      fast_paths=recycle.fast_path_manifest(),
                      source_commit=os.environ.get("FIREPROT_SOURCE_COMMIT") or _git_commit(REPO_ROOT),
                      source_sha256={name: _sha256(REPO_ROOT / name) for name in (
                          "benchmarks/protenix/benchmark_epilogues.py",
                          "python/fireprot/backends/amd/recycle.py",
                          "kernels/amd/fireprot_amd_ops.cu",
                      )})
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["gpu_process_monitor"] = monitor.stop()
        if not report["gpu_process_monitor"]["passed"]:
            report["status"] = "contaminated"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
