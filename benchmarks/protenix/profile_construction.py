"""Profile Mini construction and test a private checkpoint-covered initializer skip."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import inspect
import json
import os
import pathlib
import pickle
import pstats
import random
import time

from benchmarks.protenix.profile_inference import (
    REPO_ROOT, _additional_device_preflight, _GpuProcessMonitor, _rocm_preflight, _sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--skip-truncated", action="store_true")
    args = parser.parse_args()
    visible = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible not in {"6", "7"}:
        parser.error("Select physical GPU 6 or 7")
    os.environ["LAYERNORM_TYPE"] = "torch"
    os.environ.setdefault("PROTENIX_ROOT_DIR", str(REPO_ROOT / "runtime"))
    checkpoint = pathlib.Path(os.environ["PROTENIX_ROOT_DIR"]) / "checkpoint/protenix_mini_default_v0.5.0.pt"
    if not checkpoint.is_file():
        parser.error("The existing Mini-default checkpoint is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = {"device_policy": "exclusive"}
    preflight = _rocm_preflight(policy)
    assert preflight["exclusive_check"]["passed"]
    sibling = _additional_device_preflight([7 if visible == "6" else 6], policy)
    monitors = [_GpuProcessMonitor(d, os.getpid(), 1) for d in (6, 7)]
    for monitor in monitors:
        monitor.start()
    report = dict(status="running", skip_truncated=args.skip_truncated,
                  model="protenix_mini_default_v0.5.0", dtype="FP32", cycles=4, steps=1,
                  samples=1, request_seed=101, construction_seed=1031,
                  scope="Construction and checkpoint load only; no prediction or model speedup claim",
                  cache_state="Fresh process; warm filesystem caches from preceding inference runs",
                  preflight=preflight, sibling_preflight=sibling,
                  initializer_calls=0, initializer_elements=0, initializer_wall_ms=0.0)
    try:
        import numpy as np
        import torch
        import scipy
        from scipy.stats import truncnorm
        from scipy.stats._distn_infrastructure import rv_continuous
        from runner import batch_inference as batch
        from runner import inference
        from protenix.model.triangular import layers
        from protenix.model.modules import primitives

        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.preferred_blas_library("hipblas")
        random.seed(1031)
        np.random.seed(1031)
        torch.manual_seed(1031)
        report.update(torch=torch.__version__, hip=torch.version.hip, torch_threads=torch.get_num_threads())
        assert truncnorm._rvs.__func__ is rv_continuous._rvs
        sampling_source = inspect.getsource(truncnorm._rvs)
        report["scipy_sampling"] = dict(version=scipy.__version__, source=sampling_source,
                                        source_sha256=hashlib.sha256(sampling_source.encode()).hexdigest(),
                                        rng_advancement="one uniform float64 draw per initialized element")
        sources = [pathlib.Path(__file__), pathlib.Path(inference.__file__),
                   pathlib.Path(layers.__file__), pathlib.Path(primitives.__file__)]
        report["source_sha256"] = {str(p.relative_to(REPO_ROOT)): _sha256(p) for p in sources}

        def random_state_digest():
            return {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                    "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
                    "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}

        original_initializer = layers.trunc_normal_init_
        assert primitives.trunc_normal_init_ is original_initializer

        def measured_initializer(weights, *positional, **keywords):
            started = time.perf_counter()
            report["initializer_calls"] += 1
            report["initializer_elements"] += weights.numel()
            if not args.skip_truncated:
                original_initializer(weights, *positional, **keywords)
            else:
                # Match the verified SciPy inverse-CDF sampler's RNG advancement.
                # Checkpoint coverage is mandatory before accepting this private experiment.
                truncnorm.random_state.uniform(size=weights.numel())
            report["initializer_wall_ms"] += (time.perf_counter() - started) * 1000

        layers.trunc_normal_init_ = measured_initializer
        primitives.trunc_normal_init_ = measured_initializer
        original_model = inference.Protenix
        profiler = cProfile.Profile()

        def measured_model(configs):
            report["rng_before_construction"] = random_state_digest()
            started = time.perf_counter()
            model = profiler.runcall(original_model, configs)
            report["cpu_construction_ms"] = (time.perf_counter() - started) * 1000
            report["rng_after_construction"] = random_state_digest()
            return model

        inference.Protenix = measured_model
        original_init_model = inference.InferenceRunner.init_model

        def measured_init_model(self):
            torch.cuda.synchronize()
            started = time.perf_counter()
            original_init_model(self)
            torch.cuda.synchronize()
            report["construction_and_transfer_ms"] = (time.perf_counter() - started) * 1000

        inference.InferenceRunner.init_model = measured_init_model
        original_load = inference.InferenceRunner.load_checkpoint

        def measured_load(self):
            assert "load_state_dict" not in self.model.__dict__
            native_load = self.model.load_state_dict

            def check_coverage(*positional, **keywords):
                result = native_load(*positional, **keywords)
                report["checkpoint_coverage"] = dict(missing_keys=list(result.missing_keys),
                                                     unexpected_keys=list(result.unexpected_keys))
                return result

            self.model.load_state_dict = check_coverage
            started = time.perf_counter()
            try:
                original_load(self)
                torch.cuda.synchronize()
            finally:
                del self.model.load_state_dict
            report["checkpoint_load_ms"] = (time.perf_counter() - started) * 1000

        inference.InferenceRunner.load_checkpoint = measured_load
        batch.inference_configs["dump_dir"] = str(args.output.parent / "construction-output")
        started = time.perf_counter()
        runner = batch.get_default_runner(seeds=[101], n_cycle=4, n_step=1, n_sample=1,
                                          dtype="fp32", model_name=report["model"], use_msa=False,
                                          trimul_kernel="torch", triatt_kernel="torch",
                                          enable_cache=True, enable_fusion=True, enable_tf32=False,
                                          use_template=False, use_rna_msa=False, use_seeds_in_json=False)
        report["runner_creation_ms"] = (time.perf_counter() - started) * 1000
        report["checkpoint_sha256"] = _sha256(checkpoint)
        report["parameter_elements"] = sum(p.numel() for p in runner.model.parameters())
        report["state_entries"] = len(runner.model.state_dict())

        def tensor_digest(items):
            digest = hashlib.sha256()
            for name, tensor in sorted(items):
                digest.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        report["loaded_state_sha256"] = tensor_digest(runner.model.state_dict().items())
        report["all_buffers_sha256"] = tensor_digest(runner.model.named_buffers())
        profile_path = args.output.with_suffix(".pstats")
        profiler.dump_stats(str(profile_path))
        statistics = pstats.Stats(profiler)
        rows = [dict(file=key[0], line=key[1], function=key[2], primitive_calls=value[0],
                     calls=value[1], own_seconds=value[2], cumulative_seconds=value[3])
                for key, value in statistics.stats.items()]
        report["top_cumulative"] = sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True)[:40]
        report["top_own_time"] = sorted(rows, key=lambda row: row["own_seconds"], reverse=True)[:30]
        report["profile_file"] = str(profile_path)
        if args.skip_truncated:
            assert not report["checkpoint_coverage"]["missing_keys"], "Skipping requires complete checkpoint coverage"
        report["status"] = "ok"
        print(json.dumps({key: report[key] for key in ("status", "cpu_construction_ms",
              "construction_and_transfer_ms", "checkpoint_load_ms", "runner_creation_ms",
              "initializer_calls", "initializer_elements", "initializer_wall_ms", "checkpoint_coverage")}), flush=True)
    except Exception as error:
        report.update(status="error", error=str(error))
        raise
    finally:
        report["monitors"] = [monitor.stop() for monitor in monitors]
        if not all(m["passed"] for m in report["monitors"]):
            report["status"] = "contaminated"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
