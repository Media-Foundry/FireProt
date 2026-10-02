#!/usr/bin/env python3
"""Measure observed FP32 triangle attention layouts and optional direct CK tiles."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _GpuProcessMonitor, _git_commit, _json_load, _rocm_preflight,
    _selected_physical_device, _sha256,
)


def _build_ck(rocm_root: pathlib.Path, ck_set: str) -> tuple[list[tuple[str, pathlib.Path]], dict]:
    """Build isolated C ABI probes; no CK candidate is registered in model dispatch."""
    source = REPO_ROOT / "benchmarks/protenix/csrc/ck_attention_probe.cpp"
    headers = rocm_root / "include/ck"
    compiler = rocm_root / "bin/hipcc"
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    digest = hashlib.sha256()
    files = sorted(p for p in headers.rglob("*") if p.is_file() and p.suffix in {".h", ".hpp"})
    if not files:
        raise RuntimeError(f"No CK headers under {headers}")
    for path in files:
        digest.update(str(path.relative_to(headers)).encode() + b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    header_digest = digest.hexdigest()
    digest.update(source.read_bytes())
    digest.update(pathlib.Path(__file__).read_bytes())
    patch = REPO_ROOT / "patches/ck/attention-cache-q-fp32.patch"
    digest.update(patch.read_bytes())
    digest.update(version.encode())
    build_root = REPO_ROOT / "runtime/ck_attention_probes" / digest.hexdigest()[:16]
    build_root.mkdir(parents=True, exist_ok=True)
    variants = [(m, n, mf32, False, False) for mf32 in (False, True) for m in (64, 128) for n in (64, 128)]
    variants.append((64, 64, False, True, False))
    if ck_set == "tuned":
        variants = [(64, 64, False, False, False)]
    variants += [(m, n, False, False, True) for m, n in ((64, 64), (64, 32), (128, 32))]
    relative = pathlib.Path("ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp")
    original = rocm_root / "include" / relative
    expected = "03661397734c09cc5abcd0c2587623a145b6bf95cfae5cd5ece3203ba0406a43"
    if _sha256(original) != expected:
        raise RuntimeError("CK attention overlay requires the measured header SHA256; refusing a different version")
    overlay = build_root / "overlay"
    target = overlay / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(original.read_bytes())
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p0", "-i", str(patch)], cwd=overlay, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    libraries, builds = [], []
    for m, n, mf32, fast, cached_q in variants:
        name = f"ck_m{m}_n{n}_mf{32 if mf32 else 16}" + ("_fast" if fast else "")
        if cached_q:
            name += "_cached_q"
        output = build_root / f"{name}.so"
        command = [str(compiler), "-O3", "-std=c++17", "-fPIC", "-shared",
                   "--offload-arch=gfx90a", f"-I{rocm_root / 'include'}", f"-DMB={m}", f"-DNB={n}"]
        if mf32:
            command.append("-DMF32")
        if fast:
            command.append("-ffast-math")
        if cached_q:
            command[1:1] = [f"-I{overlay}", "-DFULL_G1K"]
        command += [str(source), "-o", str(output)]
        reused = output.is_file()
        if not reused:
            temporary = output.with_suffix(".pending.so")
            command[-1] = str(temporary)
            with (build_root / f"{name}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            temporary.replace(output)
        libraries.append((name, output))
        builds.append(dict(name=name, command=command, reused=reused, library_sha256=_sha256(output)))
    return libraries, dict(
        compiler_version=version, installed_version=(headers / "version.h").read_text(),
        header_files=len(files), header_tree_sha256=header_digest,
        header_digest_method="sorted relative path + NUL + raw SHA256 per .h/.hpp file",
        overlay=dict(original_sha256=expected, patch_sha256=_sha256(patch), result_sha256=_sha256(target)),
        source_sha256=_sha256(source), builds=builds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--include-ck", action="store_true")
    parser.add_argument("--ck-set", choices=("all", "tuned"), default="all")
    parser.add_argument("--rocm-root", type=pathlib.Path, default=pathlib.Path("/opt/rocm"))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be nonnegative and repetitions positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, status="running", seed=419, shape=[1000, 4, 1000, 32],
                  dtype="float32", tf32=False, warmups=args.warmups, repetitions=args.repetitions,
                  cache_state="warm within process; compilation excluded",
                  scope="random QKV and bias, observed strides; no checkpoint or model speedup",
                  source_commit=os.environ.get("FIREPROT_SOURCE_COMMIT") or _git_commit(REPO_ROOT),
                  source_sha256={name: _sha256(REPO_ROOT / name) for name in (
                      "benchmarks/protenix/benchmark_attention.py",
                      "benchmarks/protenix/csrc/ck_attention_probe.cpp",
                      "patches/ck/attention-cache-q-fp32.patch",
                      "fireprot/backends/amd/triangle_attention.py",
                  )}, records=[])
    monitor = None
    try:
        libraries, report["ck_build"] = _build_ck(args.rocm_root, args.ck_set) if args.include_ck else ([], None)
        measurement = _json_load(REPO_ROOT / "benchmarks/protenix/configs/mi250-long.json")["measurement"]
        report["preflight"] = _rocm_preflight(measurement)
        if not report["preflight"]["exclusive_check"]["passed"]:
            raise RuntimeError("Exclusive-device preflight failed")
        _, physical = _selected_physical_device(os.environ)
        monitor = _GpuProcessMonitor(physical, os.getpid(), 1.0)
        monitor.start()
        import torch
        from fireprot.backends.amd.triangle_attention import _direct_sdpa, _efficient_sdpa

        if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx90a":
            raise RuntimeError("This experiment requires gfx90a HIP")
        report.update(torch=torch.__version__, hip=torch.version.hip)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(419)
        with torch.inference_mode():
            q, k, v = [torch.randn(1000, 1000, 4, 32, device="cuda").transpose(1, 2) for _ in range(3)]
            q.div_(32**0.5)
            bias = torch.randn(1000, 1000, 4, device="cuda").permute(2, 0, 1).unsqueeze(0)
            report.update(q_stride=list(q.stride()), bias_stride=list(bias.stride()))
            keywords = dict(module=None, q=q, k=k, v=v, biases=[bias], native=lambda: None)
            reference = _efficient_sdpa(**keywords)
            functions = [("chunked_sdpa", lambda: _efficient_sdpa(**keywords)),
                         ("direct_sdpa", lambda: _direct_sdpa(**keywords))]
            handles = []
            for name, library in libraries:
                handle = ctypes.CDLL(str(library))
                handles.append(handle)
                entry = handle.ck_attention
                entry.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
                entry.restype = ctypes.c_int

                def ck_call(entry=entry):
                    packed_bias = bias.contiguous()
                    output = torch.empty_like(q)
                    status = entry(q.data_ptr(), k.data_ptr(), v.data_ptr(), packed_bias.data_ptr(),
                                   output.data_ptr(), 1000, 1000, torch.cuda.current_stream().cuda_stream)
                    if status:
                        raise RuntimeError(f"CK probe returned {status}")
                    return output.transpose(1, 2)

                functions.append((name, ck_call))
            functions.append(("direct_sdpa_repeat", lambda: _direct_sdpa(**keywords)))
            for name, function in functions:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                result = None
                for iteration in range(args.warmups + args.repetitions):
                    result = None
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    start.record()
                    result = function()
                    end.record()
                    end.synchronize()
                    if iteration >= args.warmups:
                        timings.append(start.elapsed_time(end))
                peak = torch.cuda.max_memory_allocated()
                torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)
                ordered = sorted(timings)
                record = dict(name=name, median_ms=statistics.median(timings),
                              p95_ms=ordered[(95 * len(ordered) + 99) // 100 - 1], values_ms=timings,
                              peak_allocated_bytes=peak, max_abs_delta=(result-reference).abs().max().item(),
                              output_stride=list(result.stride()))
                report["records"].append(record)
                print(json.dumps({key: value for key, value in record.items() if key != "values_ms"}), flush=True)
                del result
        report["status"] = "ok"
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if monitor is not None:
            report["gpu_process_monitor"] = monitor.stop()
            if not report["gpu_process_monitor"]["passed"]:
                report["status"] = "contaminated"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
