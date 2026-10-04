"""Compare fused QKV/gate preparation with the validated QKV path and CK attention."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import pathlib
import statistics
import subprocess

from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _additional_device_preflight, _GpuProcessMonitor,
    _rocm_preflight, _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--scope", choices=("preparation", "attention", "both"), default="both")
    args = parser.parse_args()
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible not in {"6", "7"}:
        parser.error("Select physical GPU 6 or 7")
    os.environ["LAYERNORM_TYPE"] = "torch"
    os.environ["FIREPROT_ENABLE_LONG1536"] = "1"
    os.environ["FIREPROT_ENABLE_QKV_PROJECTION"] = "1"
    import torch
    from fireprot.backends.amd import ck_attention as ck
    from fireprot.backends.amd import qkv_projection as qkv
    from fireprot.backends.amd.blas import configure_process_blas
    from fireprot.backends.amd.extension import load_amd_extension

    policy = {"device_policy": "exclusive"}
    preflight = _rocm_preflight(policy)
    assert preflight["exclusive_check"]["passed"]
    sibling = _additional_device_preflight([7 if visible == "6" else 6], policy)
    monitors = [_GpuProcessMonitor(d, os.getpid(), 1) for d in (6, 7)]
    for monitor in monitors:
        monitor.start()
    source = REPO_ROOT / "benchmarks/protenix/csrc/qkvg_projection_probe.cpp"
    source_paths = [source, pathlib.Path(__file__),
                    REPO_ROOT / "python/fireprot/backends/amd/qkv_projection.py",
                    REPO_ROOT / "kernels/amd/fireprot_amd_ops.cpp",
                    REPO_ROOT / "kernels/amd/fireprot_amd_ops.cu",
                    REPO_ROOT / "Protenix/protenix/model/triangular/layers.py"]
    report = dict(scope=args.scope, status="running", seed=1027, dtype="float32", tf32=False,
                  warmups=5, repetitions=20, model="Protenix Attention shape replay; random weights",
                  cycle_step_sample_count="Not applicable to operator replay",
                  cache_state="warm; allocations included; compilation excluded",
                  parameters="Normal inputs and normal(0,0.1) weights; first rows set to 0,+100,-100",
                  memory_scope="Whole probe process including references; not model memory",
                  preflight=preflight, sibling_preflight=sibling,
                  torch=torch.__version__, hip=torch.version.hip,
                  source_sha256={str(p.relative_to(REPO_ROOT)): _sha256(p) for p in source_paths},
                  builds=[], records=[], rejections=[])
    try:
        report["blas"] = configure_process_blas("hipblas")
        from protenix.model.triangular.layers import Attention

        assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx90a"
        assert load_amd_extension() is not None
        if args.scope != "preparation":
            assert ck.load_extension() is not None and ck._exact1536_enabled and ck._vector1000_enabled
            report["ck_attention"] = ck.extension_status()
        compiler = subprocess.check_output(["/opt/rocm/bin/hipcc", "--version"], text=True)
        report["compiler"] = compiler
        digest = hashlib.sha256(source.read_bytes() + compiler.encode()).hexdigest()[:16]
        directory = REPO_ROOT / "runtime/qkvg_projection" / digest
        directory.mkdir(parents=True, exist_ok=True)
        candidates = []
        for rows, channels, waves in ((128, 64, 16), (128, 64, 8), (64, 64, 4), (64, 128, 8), (64, 64, 8)):
            name = f"r{rows}c{channels}k16w{waves}"
            binary = directory / (name + ".so")
            pending = binary.with_suffix(".pending.so")
            command = ["/opt/rocm/bin/hipcc", "-O3", "-std=c++17", "-shared", "-fPIC",
                       "--offload-arch=gfx90a", f"-DROW_TILE={rows}",
                       f"-DCHANNEL_TILE={channels}", "-DK_TILE=16", f"-DWAVES={waves}",
                       str(source), "-o", str(pending)]
            if not binary.is_file():
                subprocess.run(command, check=True)
                pending.replace(binary)
            library = ctypes.CDLL(str(binary))
            project = library.qkvg_projection
            project._library_handle = library
            project.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int, ctypes.c_void_p]
            project.restype = ctypes.c_int
            assert project(*([None] * 6), 1, None) == 2
            assert project(*([None] * 6), 1000000, None) == 3
            candidates.append((name, project))
            report["builds"].append(dict(name=name, command=command, library_sha256=_sha256(binary)))

        torch.manual_seed(1027)
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            for batch, n in ((1000, 1000), (512, 1536)):
                module = Attention(128, 128, 128, 32, 4).cuda().eval()
                for parameter in module.parameters():
                    parameter.normal_(0, .1)
                x = torch.randn(batch, n, 128, device="cuda")
                x[0, 0] = 0
                x[0, 1] = 100
                x[0, 2] = -100
                bias = torch.randn(1, 4, n, n, device="cuda") * .1

                def baseline_qkv():
                    def unexpected_fallback():
                        raise RuntimeError("Validated QKV path unexpectedly fell back")
                    return qkv.prepare(module=module, q_x=x, kv_x=x, apply_scale=True,
                                       native=unexpected_fallback)

                def baseline_prepare():
                    return (*baseline_qkv(), torch.sigmoid(module.linear_g(x)))

                def attention(q, k, v):
                    return ck.attention(module=module, q=q, k=k, v=v, biases=[bias],
                                        native=lambda **_: None)

                def baseline_attention():
                    return (module._wrap_up(attention(*baseline_qkv()), x),)

                preparation = [("baseline_before", baseline_prepare)]
                complete = [("baseline_before", baseline_attention)]
                reference = baseline_prepare()
                for name, project in candidates:
                    def fused_prepare(project=project, guard=False):
                        count = 4 * batch * n * 128
                        storage = (torch.full((count + 64,), 123.5, device="cuda") if guard
                                   else torch.empty(count, device="cuda"))
                        output = storage[32:-32] if guard else storage
                        status = project(x.data_ptr(), module.linear_q.weight.data_ptr(),
                                         module.linear_k.weight.data_ptr(), module.linear_v.weight.data_ptr(),
                                         module.linear_g.weight.data_ptr(), output.data_ptr(), batch * n,
                                         torch.cuda.current_stream().cuda_stream)
                        if status:
                            raise RuntimeError(f"QKVG HIP status {status}")
                        if guard:
                            assert bool((storage[:32] == 123.5).all())
                            assert bool((storage[-32:] == 123.5).all())
                        q, k, v, gate = output.view(4, batch, n, 128).unbind(0)
                        return (*(t.view(batch, n, 4, 32).transpose(1, 2) for t in (q, k, v)), gate)

                    actual = fused_prepare(guard=True)
                    try:
                        for a, b in zip(actual, reference):
                            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                    except AssertionError as error:
                        rejection = dict(scope="preparation_guard", batch=batch, length=n,
                                         implementation=name, error=str(error))
                        report["rejections"].append(rejection)
                        print(json.dumps(rejection), flush=True)
                        del actual, a, b
                        continue
                    del actual, a, b
                    preparation.append((name, fused_prepare))

                    def fused_attention(prepare=fused_prepare):
                        q, k, v, gate = prepare()
                        output = attention(q, k, v).reshape(batch, n, 128)
                        return (module.linear_o(output * gate),)
                    complete.append((name, fused_attention))
                preparation.append(("baseline_after", baseline_prepare))
                complete.append(("baseline_after", baseline_attention))
                for scope, functions in (("preparation", preparation), ("attention", complete)):
                    if args.scope not in (scope, "both"):
                        continue
                    expected = functions[0][1]()
                    for name, function in functions:
                        # Assertion tracebacks can retain comparison tensors until cyclic GC.
                        gc.collect()
                        actual = function()
                        delta = max((a - b).abs().max().item() for a, b in zip(actual, expected))
                        try:
                            for a, b in zip(actual, expected):
                                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                        except AssertionError as error:
                            rejection = dict(scope=scope, batch=batch, length=n, implementation=name,
                                             max_abs_delta=delta, error=str(error))
                            report["rejections"].append(rejection)
                            print(json.dumps(rejection), flush=True)
                            del actual, a, b
                            continue
                        strides = [list(t.stride()) for t in actual]
                        if scope == "preparation":
                            assert all(s == [n * 128, 32, 128, 1] for s in strides[:3])
                        del actual, a, b
                        torch.cuda.reset_peak_memory_stats()
                        values = []
                        for iteration in range(25):
                            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                            start.record()
                            output = function()
                            end.record()
                            end.synchronize()
                            del output
                            if iteration >= 5:
                                values.append(start.elapsed_time(end))
                        row = dict(scope=scope, batch=batch, length=n, implementation=name,
                                   median_ms=statistics.median(values), p95_ms=sorted(values)[18],
                                   values_ms=values, max_abs_delta=delta, strides=strides,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
                        report["records"].append(row)
                        print(json.dumps({k: v for k, v in row.items() if k != "values_ms"}), flush=True)
                    del expected
                del reference, x, module, bias
                gc.collect()
                torch.cuda.empty_cache()
        report["qkv_dispatch"] = qkv.manifest()
        report["status"] = "completed_with_rejections" if report["rejections"] else "ok"
    except Exception as error:
        report.update(status="error", error=str(error))
        raise
    finally:
        report["monitors"] = [monitor.stop() for monitor in monitors]
        if not all(m["passed"] for m in report["monitors"]):
            report["status"] = "contaminated"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] in {"ok", "completed_with_rejections"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
