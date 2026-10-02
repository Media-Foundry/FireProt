#!/usr/bin/env python3
"""Benchmark observed Protenix GEMM signatures in isolated BLAS processes."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any

from benchmarks.protenix.profile_inference import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    _git_commit,
    _json_load,
    _rocm_preflight,
    _sha256,
)
from fireprot.backends.amd.blas import (
    BlasMeasurement,
    configure_process_blas,
    select_blas_backend,
)


BACKENDS = ("auto", "hipblaslt", "hipblas", "ck")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "count": len(ordered),
        "min_us": ordered[0],
        "median_us": statistics.median(ordered),
        "p95_us": ordered[p95_index],
        "max_us": ordered[-1],
        "mean_us": statistics.fmean(ordered),
    }


def _shape_id(dtype: str, kind: str, shape: list[int]) -> str:
    return f"{dtype}:{kind}:" + "x".join(str(value) for value in shape)


def _measure(torch: Any, function: Any, warmups: int, iterations: int) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    return _stats(
        [start.elapsed_time(end) * 1000 for start, end in pairs]
    )


def _worker(spec_path: pathlib.Path) -> int:
    spec = _json_load(spec_path)
    output = pathlib.Path(spec["output"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": _utc_now(),
        "spec": spec,
    }
    try:
        preflight = _rocm_preflight(spec["measurement"])
        result["rocm_preflight"] = preflight
        exclusive = preflight["exclusive_check"]
        if exclusive["required"] and not exclusive["passed"]:
            raise RuntimeError(
                "Exclusive-device preflight failed: "
                f"{exclusive.get('reason', 'unknown reason')}"
            )

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-compatible PyTorch device is required")
        policy = configure_process_blas(spec["backend"], torch_module=torch)
        torch.manual_seed(spec["seed"])
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        measurements = []
        for dtype_name in spec["benchmark"]["dtypes"]:
            dtype = dtype_map[dtype_name]
            for shape_record in spec["benchmark"]["shapes"]:
                kind = shape_record["kind"]
                shape = shape_record["shape"]
                record = {
                    "id": _shape_id(dtype_name, kind, shape),
                    "dtype": dtype_name,
                    "kind": kind,
                    "shape": shape,
                    "backend": spec["backend"],
                }
                try:
                    if kind == "mm":
                        m, k, n = shape
                        left = torch.randn(m, k, device="cuda", dtype=dtype)
                        right = torch.randn(k, n, device="cuda", dtype=dtype)
                        function = lambda: torch.mm(left, right)
                        iterations = spec["benchmark"]["iterations"]
                    elif kind == "bmm":
                        batch, m, k, n = shape
                        left = torch.randn(
                            batch, m, k, device="cuda", dtype=dtype
                        )
                        right = torch.randn(
                            batch, k, n, device="cuda", dtype=dtype
                        )
                        function = lambda: torch.bmm(left, right)
                        iterations = spec["benchmark"]["bmm_iterations"]
                    else:
                        raise ValueError(f"Unknown operation kind: {kind}")
                    record.update(
                        {
                            "supported": True,
                            "timing": _measure(
                                torch,
                                function,
                                spec["benchmark"]["warmups"],
                                iterations,
                            ),
                        }
                    )
                except Exception as error:
                    record.update(
                        {
                            "supported": False,
                            "error": {
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        }
                    )
                    torch.cuda.empty_cache()
                measurements.append(record)
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        result.update(
            {
                "status": "ok",
                "completed_at": _utc_now(),
                "blas": policy,
                "software": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                },
                "hardware": {
                    "hostname": platform.node(),
                    "device_name": properties.name,
                    "gcn_arch_name": getattr(properties, "gcnArchName", None),
                    "total_memory_bytes": properties.total_memory,
                },
                "measurements": measurements,
            }
        )
    except Exception as error:
        result.update(
            {
                "status": "error",
                "completed_at": _utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def _select_summary(
    reports: dict[str, dict[str, Any]], minimum_speedup: float
) -> list[dict[str, Any]]:
    by_backend = {
        backend: {
            measurement["id"]: measurement
            for measurement in report["measurements"]
        }
        for backend, report in reports.items()
        if report["status"] == "ok"
    }
    native_records = by_backend.get("auto", {})
    decisions = []
    for identifier, native_record in sorted(native_records.items()):
        if not native_record["supported"]:
            continue
        native = BlasMeasurement(
            backend="auto",
            median_us=native_record["timing"]["median_us"],
            p95_us=native_record["timing"]["p95_us"],
        )
        candidates = {}
        for backend in BACKENDS[1:]:
            record = by_backend.get(backend, {}).get(identifier)
            if record is None:
                continue
            candidates[backend] = BlasMeasurement(
                backend=backend,
                median_us=(
                    record["timing"]["median_us"]
                    if record["supported"]
                    else float("inf")
                ),
                p95_us=(
                    record["timing"]["p95_us"]
                    if record["supported"]
                    else float("inf")
                ),
                supported=record["supported"],
                reason=record.get("error", {}).get("message"),
            )
        decision = select_blas_backend(
            native, candidates, minimum_speedup=minimum_speedup
        )
        decisions.append(
            {
                "id": identifier,
                "native": native_record,
                "candidates": {
                    key: {
                        "supported": value.supported,
                        "median_us": value.median_us,
                        "p95_us": value.p95_us,
                        "reason": value.reason,
                    }
                    for key, value in candidates.items()
                },
                "decision": decision,
            }
        )
    return decisions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--_worker-spec", type=pathlib.Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args._worker_spec is not None:
        return _worker(args._worker_spec)
    config_path = args.config.resolve()
    config = _json_load(config_path)
    benchmark = config["blas_benchmark"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backends = args.backend or list(BACKENDS)
    reports = {}
    for backend in backends:
        output = output_dir / f"{backend}.json"
        spec = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "backend": backend,
            "seed": args.seed,
            "output": str(output),
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "source": {
                "fireprot_commit": _git_commit(REPO_ROOT)
                or os.environ.get("FIREPROT_SOURCE_COMMIT"),
                "benchmark_script_sha256": _sha256(
                    pathlib.Path(__file__).resolve()
                ),
            },
            "measurement": config["measurement"],
            "benchmark": benchmark,
        }
        spec_path = output_dir / f"{backend}-spec.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--output-dir",
                str(output_dir),
                "--_worker-spec",
                str(spec_path),
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
        )
        if completed.returncode != 0:
            report = _json_load(output)
            if report.get("error", {}).get("message", "").startswith(
                "Exclusive-device preflight failed:"
            ):
                raise RuntimeError(report["error"]["message"])
        reports[backend] = _json_load(output)

    summary = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "backends": backends,
        "reports": {
            backend: str((output_dir / f"{backend}.json").resolve())
            for backend in backends
        },
        "decisions": _select_summary(
            reports, float(benchmark["minimum_speedup"])
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
