#!/usr/bin/env python3
"""Reproducible Protenix inference profiler for AMD GPU systems.

The public entry point launches one isolated worker per workload.  A worker
loads Protenix once, runs warm-up and measured records through the native
inference path, and writes a structured JSON manifest.  Model latency uses
CUDA events; preprocessing, serialization, and process time use wall clocks.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import contextvars
import copy
import datetime as dt
import functools
import hashlib
import json
import os
import pathlib
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Callable


SCHEMA_VERSION = 1
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTENIX_ROOT = REPO_ROOT / "Protenix"
DEFAULT_CONFIG = REPO_ROOT / "benchmarks/protenix/configs/mi250.yaml"


def _json_load(path: pathlib.Path) -> Any:
    """Load JSON, which is also valid YAML, without adding a YAML dependency."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: pathlib.Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _run_json_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
    except OSError as error:
        return {
            "command": command,
            "returncode": None,
            "error": f"{type(error).__name__}: {error}",
        }
    record: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }
    try:
        record["data"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        record["stdout"] = completed.stdout.strip()
    return record


def _rocm_preflight(measurement: dict[str, Any]) -> dict[str, Any]:
    command = shutil.which("rocm-smi")
    if command is None:
        result = {"available": False, "reason": "rocm-smi was not found"}
    else:
        result = {
            "available": True,
            **_run_json_command(
                [
                    command,
                    "--showpids",
                    "--showuse",
                    "--showmeminfo",
                    "vram",
                    "--showclocks",
                    "--json",
                ]
            ),
        }
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "ROCR_VISIBLE_DEVICES"
    )
    physical_index = None
    if visible:
        first = visible.split(",", 1)[0].strip()
        if first.isdigit():
            physical_index = int(first)
    result["visible_devices"] = visible
    result["selected_physical_index"] = physical_index
    result["exclusive_check"] = {"required": False, "passed": None}
    if measurement.get("device_policy") != "exclusive":
        return result

    check = result["exclusive_check"]
    check["required"] = True
    if not result.get("available") or result.get("returncode") != 0:
        check.update({"passed": False, "reason": "rocm-smi snapshot unavailable"})
        return result
    if physical_index is None:
        check.update(
            {
                "passed": False,
                "reason": "exclusive policy requires HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES",
            }
        )
        return result
    card = result.get("data", {}).get(f"card{physical_index}")
    if not isinstance(card, dict):
        check.update(
            {"passed": False, "reason": f"card{physical_index} missing from rocm-smi"}
        )
        return result
    try:
        gpu_use = int(card["GPU use (%)"])
        vram_used = int(card["VRAM Total Used Memory (B)"])
    except (KeyError, TypeError, ValueError) as error:
        check.update(
            {"passed": False, "reason": f"invalid rocm-smi card fields: {error}"}
        )
        return result
    max_use = int(measurement.get("exclusive_max_gpu_use_percent", 5))
    max_vram = int(
        measurement.get("exclusive_max_vram_used_bytes", 256 * 1024 * 1024)
    )
    check.update(
        {
            "gpu_use_percent": gpu_use,
            "vram_used_bytes": vram_used,
            "max_gpu_use_percent": max_use,
            "max_vram_used_bytes": max_vram,
            "passed": gpu_use <= max_use and vram_used <= max_vram,
        }
    )
    if not check["passed"]:
        check["reason"] = "selected device is already active"
    return result


def _stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in ("warmup", "measure"):
        selected = [record for record in records if record["phase"] == phase]
        stage_names = sorted(
            {
                stage
                for record in selected
                for stage in record["stages_gpu_ms"].keys()
            }
        )
        summary[phase] = {
            "model_gpu_ms": _stats(
                [record["model_gpu_ms"] for record in selected]
            ),
            "model_wall_ms": _stats(
                [record["model_wall_ms"] for record in selected]
            ),
            "stages_gpu_ms": {
                stage: _stats(
                    [
                        sum(record["stages_gpu_ms"].get(stage, []))
                        for record in selected
                    ]
                )
                for stage in stage_names
            },
        }
    return summary


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    if "layer_norm" in lowered or "layernorm" in lowered:
        return "layernorm"
    if "softmax" in lowered:
        return "softmax"
    if any(token in lowered for token in ("gemm", "cijk_", "rocblas", "bmm")):
        return "gemm"
    if any(
        token in lowered
        for token in ("transpose", "permute", "copy", "contiguous", "direct_copy")
    ):
        return "copy_or_layout"
    if "reduc" in lowered:
        return "reduction"
    if any(token in lowered for token in ("elementwise", "pointwise", "vectorized")):
        return "elementwise"
    return "other"


def _import_rocprof(path: pathlib.Path, output: pathlib.Path | None) -> int:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "name": row["Name"],
                    "calls": int(row["Calls"]),
                    "total_duration_ns": int(row["TotalDurationNs"]),
                    "average_ns": float(row["AverageNs"]),
                    "percentage": float(row["Percentage"]),
                    "min_ns": int(row["MinNs"]),
                    "max_ns": int(row["MaxNs"]),
                    "stddev": float(row["StdDev"]),
                    "category": _kernel_category(row["Name"]),
                }
            )
    total_duration = sum(row["total_duration_ns"] for row in rows)
    categories: dict[str, dict[str, int | float]] = {}
    for row in rows:
        category = categories.setdefault(
            row["category"], {"kernels": 0, "calls": 0, "total_duration_ns": 0}
        )
        category["kernels"] += 1
        category["calls"] += row["calls"]
        category["total_duration_ns"] += row["total_duration_ns"]
    for category in categories.values():
        category["percentage"] = (
            100 * category["total_duration_ns"] / total_duration
            if total_duration
            else 0
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source": str(path.resolve()),
        "source_sha256": _sha256(path),
        "kernel_count": len(rows),
        "dispatch_count": sum(row["calls"] for row in rows),
        "total_duration_ns": total_duration,
        "categories": dict(sorted(categories.items())),
        "top_kernels": sorted(
            rows, key=lambda row: row["total_duration_ns"], reverse=True
        )[:50],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if output is None:
        print(rendered, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    return 0


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config schema {config.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    measurement = config.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("measurement must be an object")
    for key in ("warmups", "repetitions"):
        if not isinstance(measurement.get(key), int) or measurement[key] < 0:
            raise ValueError(f"measurement.{key} must be a non-negative integer")
    if measurement["repetitions"] < 1:
        raise ValueError("measurement.repetitions must be at least one")
    workloads = config.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("workloads must be a non-empty list")
    required = {
        "id",
        "input",
        "token_count",
        "model_name",
        "dtype",
        "cycles",
        "diffusion_steps",
        "samples",
        "seeds",
        "cache",
        "use_msa",
        "use_template",
    }
    ids: set[str] = set()
    for workload in workloads:
        missing = sorted(required - workload.keys())
        if missing:
            raise ValueError(f"workload is missing fields: {', '.join(missing)}")
        if workload["id"] in ids:
            raise ValueError(f"duplicate workload id: {workload['id']}")
        ids.add(workload["id"])
        if not isinstance(workload["seeds"], list) or not workload["seeds"]:
            raise ValueError(f"{workload['id']}.seeds must be a non-empty list")


def _resolve_input(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input JSON does not exist: {path}")
    return path


def _safe_id(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"Unsafe workload id: {value!r}")
    return value


def _build_spec(
    *,
    config_path: pathlib.Path,
    config: dict[str, Any],
    workload: dict[str, Any],
    backend: str,
    blas_backend: str,
    output_root: pathlib.Path,
    warmups: int,
    repetitions: int,
    profile_one: bool,
    torch_profile: bool,
) -> dict[str, Any]:
    workload_id = _safe_id(workload["id"])
    run_dir = output_root / workload_id / backend
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "repo_root": str(REPO_ROOT),
        "protenix_root": str(PROTENIX_ROOT),
        "run_dir": str(run_dir),
        "backend": backend,
        "blas_backend": blas_backend,
        "profile_one": profile_one,
        "torch_profile": torch_profile,
        "source": {
            "fireprot_commit": os.environ.get("FIREPROT_SOURCE_COMMIT")
            or _git_commit(REPO_ROOT),
            "protenix_commit": os.environ.get("FIREPROT_PROTENIX_COMMIT")
            or _git_commit(PROTENIX_ROOT),
        },
        "measurement": {
            **config["measurement"],
            "warmups": warmups,
            "repetitions": repetitions,
        },
        "workload": {
            **workload,
            "input": str(_resolve_input(workload["input"])),
        },
    }


def _run_worker(spec: dict[str, Any]) -> pathlib.Path:
    run_dir = pathlib.Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "worker-spec.json"
    log_path = run_dir / "worker.log"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    python_path = [str(REPO_ROOT), str(PROTENIX_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "PROTENIX_ROOT_DIR": str(REPO_ROOT / "runtime"),
            "LAYERNORM_TYPE": "torch",
            "OMP_NUM_THREADS": environment.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": environment.get("OPENBLAS_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": environment.get("MKL_NUM_THREADS", "1"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    command = [sys.executable, str(pathlib.Path(__file__).resolve()), "--_worker-spec", str(spec_path)]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=PROTENIX_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker failed for {spec['workload']['id']} with exit code "
            f"{completed.returncode}; see {log_path}"
        )
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Worker did not write {manifest_path}")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=("native", "fireprot"), default="native")
    parser.add_argument(
        "--blas-backend",
        choices=("auto", "hipblaslt", "hipblas", "ck"),
        default="auto",
        help="Process-wide PyTorch BLAS preference. Per-op routing is added separately.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--profile-one", action="store_true")
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--import-rocprof", type=pathlib.Path)
    parser.add_argument("--import-output", type=pathlib.Path)
    parser.add_argument("--_worker-spec", type=pathlib.Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def _main() -> int:
    args = _parse_args()
    if args._worker_spec is not None:
        return _worker_main(args._worker_spec)
    if args.import_rocprof is not None:
        return _import_rocprof(args.import_rocprof.resolve(), args.import_output)

    config_path = args.config.resolve()
    config = _json_load(config_path)
    _validate_config(config)
    selected = [
        workload
        for workload in config["workloads"]
        if not args.workload or workload["id"] in set(args.workload)
    ]
    if args.workload and len(selected) != len(set(args.workload)):
        known = {item["id"] for item in config["workloads"]}
        missing = sorted(set(args.workload) - known)
        raise ValueError(f"Unknown workloads: {', '.join(missing)}")

    measurement = config["measurement"]
    warmups = measurement["warmups"] if args.warmups is None else args.warmups
    repetitions = (
        measurement["repetitions"] if args.repetitions is None else args.repetitions
    )
    if args.profile_one:
        selected = selected[:1]
        warmups = 0
        repetitions = 1
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be >= 0 and repetitions must be >= 1")

    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    output_root = (
        args.output_dir.resolve()
        if args.output_dir
        else REPO_ROOT
        / config.get("output_root", "outputs/benchmarks/protenix")
        / f"{timestamp}-{config['suite']}"
    )
    specs = [
        _build_spec(
            config_path=config_path,
            config=config,
            workload=workload,
            backend=args.backend,
            blas_backend=args.blas_backend,
            output_root=output_root,
            warmups=warmups,
            repetitions=repetitions,
            profile_one=args.profile_one,
            torch_profile=args.torch_profile,
        )
        for workload in selected
    ]
    if args.dry_run:
        print(json.dumps({"output_root": str(output_root), "workers": specs}, indent=2))
        return 0

    manifests = []
    for spec in specs:
        print(f"[fireprot] profiling {spec['workload']['id']} ({args.backend})")
        manifests.append(str(_run_worker(spec)))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "suite": config["suite"],
        "backend": args.backend,
        "blas_backend": args.blas_backend,
        "manifests": manifests,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    return 0


class _Recorder:
    def __init__(self, torch_module: Any, run_labels: list[dict[str, Any]]) -> None:
        self.torch = torch_module
        self.run_labels = run_labels
        self.current_call: contextvars.ContextVar[int | None] = contextvars.ContextVar(
            "fireprot_profile_call", default=None
        )
        self.active_gpu_stages: contextvars.ContextVar[frozenset[str]] = (
            contextvars.ContextVar("fireprot_active_gpu_stages", default=frozenset())
        )
        self.call_count = 0
        self.model_failures: list[dict[str, Any]] = []
        self.gpu_events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.wall_events: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def _new_event(self) -> Any:
        return self.torch.cuda.Event(enable_timing=True)

    def wrap_model_call(self, original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            call_id = self.call_count
            self.call_count += 1
            token = self.current_call.set(call_id)
            start = self._new_event()
            end = self._new_event()
            wall_start = time.perf_counter()
            start.record()
            try:
                return original(*args, **kwargs)
            except BaseException as error:
                self.model_failures.append(
                    {
                        "call_id": call_id,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                raise
            finally:
                end.record()
                self.gpu_events["model"].append(
                    {"call_id": call_id, "start": start, "end": end}
                )
                self.wall_events["model"].append(
                    {
                        "call_id": call_id,
                        "wall_ms": (time.perf_counter() - wall_start) * 1000,
                    }
                )
                self.current_call.reset(token)

        return wrapped

    def wrap_gpu_stage(self, original: Callable[..., Any], label: str) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            active = self.active_gpu_stages.get()
            if label in active:
                return original(*args, **kwargs)
            token = self.active_gpu_stages.set(active | {label})
            start = self._new_event()
            end = self._new_event()
            start.record()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                self.gpu_events[label].append(
                    {
                        "call_id": self.current_call.get(),
                        "start": start,
                        "end": end,
                    }
                )
                self.active_gpu_stages.reset(token)

        return wrapped

    def wrap_wall_stage(
        self, original: Callable[..., Any], label: str, *, skip: bool = False
    ) -> Callable[..., Any]:
        @functools.wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            if skip:
                result = None
            else:
                result = original(*args, **kwargs)
            self.wall_events[label].append(
                {
                    "call_id": self.current_call.get(),
                    "wall_ms": (time.perf_counter() - start) * 1000,
                    "skipped": skip,
                }
            )
            return result

        return wrapped

    def finalize(self) -> dict[str, Any]:
        self.torch.cuda.synchronize()
        records: list[dict[str, Any]] = []
        for call_id in range(self.call_count):
            label = (
                self.run_labels[call_id]
                if call_id < len(self.run_labels)
                else {"phase": "unknown", "iteration": None}
            )
            model_wall = next(
                (
                    item["wall_ms"]
                    for item in self.wall_events["model"]
                    if item["call_id"] == call_id
                ),
                None,
            )
            model_gpu = next(
                (
                    item["start"].elapsed_time(item["end"])
                    for item in self.gpu_events["model"]
                    if item["call_id"] == call_id
                ),
                None,
            )
            stage_values: dict[str, list[float]] = {}
            for stage, events in self.gpu_events.items():
                if stage == "model":
                    continue
                values = [
                    item["start"].elapsed_time(item["end"])
                    for item in events
                    if item["call_id"] == call_id
                ]
                if values:
                    stage_values[stage] = values
            records.append(
                {
                    "call_id": call_id,
                    **label,
                    "model_gpu_ms": model_gpu,
                    "model_wall_ms": model_wall,
                    "stages_gpu_ms": stage_values,
                }
            )
        return {
            "records": records,
            "wall_events": self.wall_events,
            "unassigned_gpu_events": {
                stage: sum(1 for item in events if item["call_id"] is None)
                for stage, events in self.gpu_events.items()
                if stage != "model"
            },
        }


def _make_repeated_input(
    source: pathlib.Path,
    destination: pathlib.Path,
    warmups: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    payload = _json_load(source)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(
            "Benchmark inputs must contain exactly one Protenix record; "
            f"{source} contains {len(payload) if isinstance(payload, list) else 'non-list'}"
        )
    base = payload[0]
    original_name = base.get("name", source.stem)
    records = []
    labels = []
    for phase, count in (("warmup", warmups), ("measure", repetitions)):
        for iteration in range(count):
            record = copy.deepcopy(base)
            record["name"] = f"{original_name}__fireprot_{phase}_{iteration:03d}"
            records.append(record)
            labels.append(
                {
                    "phase": phase,
                    "iteration": iteration,
                    "input_name": original_name,
                    "record_name": record["name"],
                }
            )
    destination.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return labels


def _hardware_metadata(torch: Any) -> dict[str, Any]:
    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "device_index": device_index,
        "device_name": properties.name,
        "device_arch": torch.cuda.get_device_capability(device_index),
        "gcn_arch_name": getattr(properties, "gcnArchName", None),
        "total_memory_bytes": properties.total_memory,
    }


def _worker_main(spec_path: pathlib.Path) -> int:
    spec = _json_load(spec_path.resolve())
    run_dir = pathlib.Path(spec["run_dir"])
    manifest_path = run_dir / "manifest.json"
    started = time.perf_counter()
    run_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "spec": spec,
        "source": {
            **spec.get("source", {}),
            "profile_script_sha256": _sha256(pathlib.Path(__file__).resolve()),
        },
    }
    try:
        rocm_preflight = _rocm_preflight(spec["measurement"])
        run_manifest["rocm_preflight"] = rocm_preflight
        exclusive = rocm_preflight["exclusive_check"]
        if exclusive["required"] and not exclusive["passed"]:
            raise RuntimeError(
                "Exclusive-device preflight failed: "
                f"{exclusive.get('reason', 'unknown reason')}"
            )

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-compatible PyTorch device is required")
        fireprot_manifest = None
        if spec["backend"] == "fireprot":
            from fireprot.adapters.protenix.runtime import (
                install,
                manifest as runtime_manifest,
            )

            install(spec["blas_backend"])
            fireprot_manifest = runtime_manifest
        elif spec["blas_backend"] != "auto":
            torch.backends.cuda.preferred_blas_library(spec["blas_backend"])

        from protenix.model.modules.confidence import ConfidenceHead
        from protenix.model.modules.diffusion import DiffusionModule
        from protenix.model.modules.pairformer import PairformerStack
        from protenix.model.modules.transformer import (
            AtomAttentionDecoder,
            AtomAttentionEncoder,
            DiffusionTransformer,
        )
        from protenix.model.protenix import Protenix
        from protenix.model.triangular.triangular import (
            TriangleAttention,
            TriangleMultiplicativeUpdate,
        )
        from runner import batch_inference
        from runner.dumper import DataDumper
        from runner.inference import InferenceRunner

        repeated_input = run_dir / "repeated-input.json"
        labels = _make_repeated_input(
            pathlib.Path(spec["workload"]["input"]),
            repeated_input,
            spec["measurement"]["warmups"],
            spec["measurement"]["repetitions"],
        )
        recorder = _Recorder(torch, labels)
        InferenceRunner.predict = recorder.wrap_model_call(InferenceRunner.predict)
        for owner, name, label in (
            (Protenix, "get_pairformer_output", "pairformer_output"),
            (Protenix, "sample_diffusion", "sample_diffusion"),
            (Protenix, "run_confidence_head", "confidence"),
            (PairformerStack, "forward", "pairformer_stack"),
            (
                TriangleMultiplicativeUpdate,
                "forward",
                "triangle_multiplication",
            ),
            (TriangleAttention, "forward", "triangle_attention"),
            (DiffusionModule, "f_forward", "diffusion_module"),
            (DiffusionTransformer, "forward", "diffusion_transformer"),
            (AtomAttentionEncoder, "forward", "atom_attention_encoder"),
            (AtomAttentionDecoder, "forward", "atom_attention_decoder"),
            (ConfidenceHead, "forward", "confidence_head"),
        ):
            setattr(owner, name, recorder.wrap_gpu_stage(getattr(owner, name), label))

        original_preprocess = batch_inference.preprocess_input
        batch_inference.preprocess_input = recorder.wrap_wall_stage(
            original_preprocess, "preprocess"
        )
        dump_outputs = bool(spec["measurement"].get("dump_outputs", False))
        DataDumper.dump = recorder.wrap_wall_stage(
            DataDumper.dump, "serialization", skip=not dump_outputs
        )

        workload = spec["workload"]
        trimul_kernel = (
            "fireprot"
            if spec["backend"] == "fireprot"
            else workload.get("trimul_kernel", "torch")
        )
        triatt_kernel = (
            "fireprot"
            if spec["backend"] == "fireprot"
            else workload.get("triatt_kernel", "torch")
        )
        inference_started = time.perf_counter()
        torch_profiler = None
        profiler_context: Any = contextlib.nullcontext()
        if spec.get("torch_profile", False):
            from torch.profiler import ProfilerActivity, profile

            profiler_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
        with profiler_context as active_profiler:
            torch_profiler = active_profiler
            batch_inference.inference_jsons(
                json_file=str(repeated_input),
                out_dir=str(run_dir / "predictions"),
                use_msa=workload["use_msa"],
                seeds=workload["seeds"],
                n_cycle=workload["cycles"],
                n_step=workload["diffusion_steps"],
                n_sample=workload["samples"],
                dtype=workload["dtype"],
                model_name=workload["model_name"],
                trimul_kernel=trimul_kernel,
                triatt_kernel=triatt_kernel,
                enable_cache=workload["cache"],
                enable_fusion=workload.get("enable_fusion", True),
                enable_tf32=workload.get("enable_tf32", False),
                use_template=workload["use_template"],
                use_rna_msa=workload.get("use_rna_msa", False),
                use_seeds_in_json=False,
                need_atom_confidence=workload.get("need_atom_confidence", False),
            )
        timing = recorder.finalize()
        if recorder.model_failures:
            first = recorder.model_failures[0]
            raise RuntimeError(
                f"{len(recorder.model_failures)} model call(s) failed; first failure "
                f"at call {first['call_id']}: {first['type']}: {first['message']}"
            )
        if recorder.call_count != len(labels):
            raise RuntimeError(
                f"Expected {len(labels)} model calls but observed "
                f"{recorder.call_count}; inspect worker.log for inference errors"
            )
        timing["summary"] = _summarize_records(timing["records"])
        profiler_artifacts = None
        if torch_profiler is not None:
            trace_path = run_dir / "torch-trace.json.gz"
            torch_profiler.export_chrome_trace(str(trace_path))
            operator_rows = []
            for item in torch_profiler.key_averages(group_by_input_shape=True):
                device_time = getattr(
                    item,
                    "self_device_time_total",
                    getattr(item, "self_cuda_time_total", 0),
                )
                operator_rows.append(
                    {
                        "key": item.key,
                        "input_shapes": item.input_shapes,
                        "count": item.count,
                        "self_cpu_time_total_us": item.self_cpu_time_total,
                        "self_device_time_total_us": device_time,
                        "self_device_memory_usage_bytes": getattr(
                            item, "self_device_memory_usage", 0
                        ),
                    }
                )
            operator_rows.sort(
                key=lambda item: item["self_device_time_total_us"], reverse=True
            )
            operators_path = run_dir / "torch-operators.json"
            operators_path.write_text(
                json.dumps(operator_rows[:500], indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            profiler_artifacts = {
                "trace": str(trace_path),
                "operators": str(operators_path),
                "operator_rows_written": min(500, len(operator_rows)),
            }
        run_manifest.update(
            {
                "status": "ok",
                "completed_at": _utc_now(),
                "software": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "miopen": (
                        getattr(torch.backends.miopen, "version")()
                        if callable(getattr(torch.backends.miopen, "version", None))
                        else None
                    ),
                    "miopen_available": bool(
                        getattr(torch.backends.miopen, "is_available", lambda: False)()
                    ),
                    "preferred_blas": str(
                        torch.backends.cuda.preferred_blas_library()
                    ),
                },
                "hardware": _hardware_metadata(torch),
                "input": {
                    "path": workload["input"],
                    "sha256": _sha256(pathlib.Path(workload["input"])),
                    "token_count": workload["token_count"],
                    "atom_count": workload.get("atom_count"),
                },
                "timing": timing,
                "torch_profiler": profiler_artifacts,
                "fireprot_runtime": (
                    fireprot_manifest() if fireprot_manifest is not None else None
                ),
                "inference_wall_ms": (time.perf_counter() - inference_started)
                * 1000,
                "process_wall_ms": (time.perf_counter() - started) * 1000,
            }
        )
    except Exception as error:
        run_manifest.update(
            {
                "status": "error",
                "completed_at": _utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "process_wall_ms": (time.perf_counter() - started) * 1000,
            }
        )
        manifest_path.write_text(
            json.dumps(run_manifest, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        raise

    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
