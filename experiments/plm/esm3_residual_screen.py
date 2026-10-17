"""Actual-activation and whole-model ESM3 residual fusion diagnostic."""
import argparse
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    parser.add_argument("profile")
    parser.add_argument("library")
    parser.add_argument("output")
    parser.add_argument("--variants", default="1,4,8,16")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    profile = json.loads(Path(args.profile).read_text())
    os.environ.update(profile["environment"])
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    sys.path.insert(0, request["source"]["root"])
    import torch
    from fireprot.adapters.plm.runtime import optimized_scope
    from fireprot_workers.plm import load_model
    from fireprot_workers.plm_batch import prepare_forward
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    input_path = Path(request["input"]["path"])
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == request["input"]["sha256"]
    sequences = [item["sequence"] for item in json.loads(input_path.read_text())["sequences"]]
    torch.set_num_threads(8)
    torch.manual_seed(request["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1100"
    library = ctypes.CDLL(str(Path(args.library).resolve()))
    kernel = library.fireprot_esm3_residual
    kernel.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    kernel.restype = ctypes.c_int
    report = dict(schema_version=1, model="esm3-open", batch_size=4, length=1000,
                  seed=request["seed"], precision="native BF16", torch=str(torch.__version__),
                  hip=torch.version.hip, architecture="gfx1100", input_sha256=request["input"]["sha256"],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
                  boundary="resident warm forward; excludes load/export; not E2E", variants=[])

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def exact(expected, actual):
        return {k: torch.equal(v.contiguous().view(torch.uint8), actual[k].contiguous().view(torch.uint8))
                for k, v in expected.items()}

    @contextmanager
    def fused_scope(model, unroll, checks):
        changes = []
        calls = []

        def add_residual(x, r, scale):
            assert x.dtype == r.dtype == torch.bfloat16, (x.dtype, r.dtype)
            assert x.shape == r.shape == (4, 1002, 1536)
            assert x.is_contiguous() and r.is_contiguous()
            result = torch.empty_like(x)
            code = kernel(x.data_ptr(), r.data_ptr(), result.data_ptr(), x.numel(),
                          1.0 / scale, unroll, torch.cuda.current_stream().cuda_stream)
            assert code == 0, code
            if checks:
                expected = x + r / scale
                check = torch.equal(expected.view(torch.uint8), result.view(torch.uint8))
                calls.append(dict(exact=check, shape=list(x.shape), scale=scale))
                if not check:
                    report["failed_operator"] = dict(unroll=unroll, call=len(calls),
                        unequal=int(torch.count_nonzero(expected != result)),
                        max_abs=float((expected.float() - result.float()).abs().max()))
                    save()
                    raise AssertionError("Residual operator is not bitwise exact")
            return result

        try:
            for block in model.transformer.blocks:
                native = block.forward

                def run(x, sequence_id, frames, frames_mask, chain_id,
                        output_attentions=False, _block=block):
                    attn = None
                    if _block.use_plain_attn:
                        r, attn = _block.attn(x, sequence_id, output_attentions=output_attentions)
                        x = add_residual(x, r, _block.scaling_factor)
                    if _block.use_geom_attn:
                        r = _block.geom_attn(x, frames, frames_mask, sequence_id, chain_id)
                        x = add_residual(x, r, _block.scaling_factor)
                    x = add_residual(x, _block.ffn(x), _block.scaling_factor)
                    return x, attn

                changes.append((block, native))
                block.forward = run
            yield calls
        finally:
            for block, native in reversed(changes):
                block.forward = native

    def invoke(unroll=0, checks=False):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with optimized_scope(model, request, 1000, batch_size=4):
            if unroll:
                with fused_scope(model, unroll, checks) as calls:
                    actual = forward()
            else:
                actual = forward()
                calls = []
        torch.cuda.synchronize()
        return actual, time.perf_counter() - start, calls

    report["isolation_before"] = json.loads(subprocess.check_output(
        [profile["amd_smi"], "snapshot-v1", "--json"], text=True))
    save()
    with torch.inference_mode():
        model, _, _ = load_model(request, sequences[0])
        forward = prepare_forward(model, "esm3-open", sequences)
        reference, _, _ = invoke()
        reference = {k: v.clone() for k, v in reference.items()}
        invoke()
        for unroll in map(int, args.variants.split(",")):
            result, _, checks = invoke(unroll, True)
            row = dict(unroll=unroll, operator_checks=checks, initial_exact=exact(reference, result))
            report["variants"].append(row)
            save()
            assert len(checks) == 97 and all(row["initial_exact"].values())
            invoke(unroll)
            samples = {"baseline": [], "candidate": []}
            row["exact_runs"] = []
            for order in ((0, unroll), (unroll, 0)) * 4:
                for variant in order:
                    actual, seconds, _ = invoke(variant)
                    arm = "candidate" if variant else "baseline"
                    samples[arm].append(seconds)
                    comparison = exact(reference, actual)
                    row["exact_runs"].append(dict(arm=arm, exact=comparison))
                    assert all(comparison.values())
            row["samples_seconds"] = samples
            row["medians_seconds"] = {k: statistics.median(v) for k, v in samples.items()}
            row["speedup"] = row["medians_seconds"]["baseline"] / row["medians_seconds"]["candidate"]
            save()
            print(json.dumps({k: v for k, v in row.items() if k not in ("operator_checks", "exact_runs", "samples_seconds")}), flush=True)
    report["isolation_after"] = json.loads(subprocess.check_output(
        [profile["amd_smi"], "snapshot-v1", "--json"], text=True))
    save()


if __name__ == "__main__":
    main()
