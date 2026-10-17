"""Precision OpenFold3 wave32 diagnostic; requires no model checkpoint."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("output")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    sys.path.insert(0, str(repo / "runtime/repos/openfold3"))
    import torch
    import openfold3.core.model.primitives.attention as attention
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert str(torch.__version__) == "2.13.0+rocm10.0.0"
    assert torch.version.hip == "7.15.26333"
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1100"
    assert hashlib.sha256(Path(attention.__file__).read_bytes()).hexdigest() == "30fe5433982ee73e3d7ce5341fa6b510b1ea332dbde072d56129632cc53f0746"
    library = ctypes.CDLL(str(Path(args.library).resolve()))
    entry = library.fireprot_softmax_gfx1100
    entry.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2 + [ctypes.c_int64] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
    entry.restype = ctypes.c_int
    report = dict(schema_version=1, model="OpenFold3/OpenBind-0 operator diagnostic", seed=101,
        torch=str(torch.__version__), hip=torch.version.hip, architecture="gfx1100",
        precision="strict deterministic FP32; deterministic memory filling retained",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
        attention_source_sha256=hashlib.sha256(Path(attention.__file__).read_bytes()).hexdigest(),
        boundary="synthetic QKV with upstream native attention and layouts; no checkpoint or whole-model claim",
        records=[], timings=[])

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def snapshot():
        return json.loads(subprocess.check_output([str(repo / "runtime/fireprot-smi-probe"), "snapshot-v1", "--json"], text=True))

    def fused(scores, mask, bias, waves, reuse, packed):
        if packed:
            bias = bias.contiguous()
        actual = torch.empty_like(scores)
        code = entry(scores.data_ptr(), mask.data_ptr(), bias.data_ptr(), actual.data_ptr(),
                     scores.shape[0], scores.shape[-1], mask.stride(0) if mask.shape[0] > 1 else 0,
                     mask.stride(-1), *bias.stride()[-3:], waves, reuse,
                     torch.cuda.current_stream().cuda_stream)
        if code:
            raise RuntimeError(f"HIP launch status: {code}")
        return actual

    def compare(expected, actual):
        a, b = expected.contiguous().view(torch.int32), actual.contiguous().view(torch.int32)
        exact = torch.equal(a, b)
        return dict(exact=exact, unequal_bits_elements=0 if exact else int(torch.count_nonzero(a != b)),
                    nan_elements=int(torch.count_nonzero(torch.isnan(actual))))

    def measure(callback):
        for _ in range(2):
            callback()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(3):
            callback()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) / 3

    report["isolation_before"] = snapshot()
    configs = ((1000, 1), (1536, 1)) if args.quick else ((1000, 8), (1536, 8), (1536, 128))
    with torch.inference_mode():
        for n, batch in configs:
            q, k, v = [torch.randn(batch, n, 4, 32, device="cuda").transpose(1, 2) * 0.125 for _ in range(3)]
            scores = torch.einsum("...qc,...kc->...qk", q, k)
            for ending in (False, True):
                pair = torch.randn(n, n, 4, device="cuda") * 0.1
                bias = pair.permute((2, 1, 0) if ending else (2, 0, 1)).unsqueeze(0)
                mask_storage = torch.zeros(batch, 1, 1, 2 * n, device="cuda")
                mask = mask_storage[..., ::2]
                cases = ("zero", "partial", "nonzero", "all_masked") if batch < 128 else ("partial",)
                for mask_case in cases:
                    mask.zero_()
                    if mask_case == "partial":
                        mask[..., -7:] = -float("inf")
                    if mask_case == "nonzero":
                        mask.normal_(0, 0.1)
                    if mask_case == "all_masked":
                        mask.fill_(-float("inf"))
                    def native_softmax():
                        temporary = scores.clone()
                        temporary += mask
                        temporary += bias
                        return torch.softmax(temporary, -1)
                    expected = native_softmax()
                    good = []
                    for waves in ((4,) if args.quick else (1, 2, 4, 8)):
                        for reuse in ((0,) if args.quick else (0, 1)):
                            for packed in (False, True):
                                actual = fused(scores, mask, bias, waves, reuse, packed)
                                record = dict(n=n, batch=batch, ending=ending, mask=mask_case,
                                              waves=waves, reuse=reuse, packed=packed,
                                              bias_stride=list(bias.stride()), mask_stride=list(mask.stride()),
                                              **compare(expected, actual))
                                report["records"].append(record)
                                if record["exact"]:
                                    good.append((waves, reuse, packed))
                                del actual
                    print(json.dumps(dict(n=n, batch=batch, ending=ending, mask=mask_case,
                                          exact_variants=len(good), tested=len(report["records"]))), flush=True)
                    save()
                    if not good:
                        report["status"] = "rejected_not_exact"
                        save()
                        return
                    if mask_case in ("zero", "partial") and not args.quick:
                        # Screening only: precomputed scores, allocation/filling and packing included.
                        baseline_ms = statistics.median(measure(native_softmax) for _ in range(3))
                        variant_times = []
                        for waves, reuse, packed in good:
                            callback = lambda: fused(scores, mask, bias, waves, reuse, packed)
                            ms = statistics.median(measure(callback) for _ in range(3))
                            variant_times.append(dict(waves=waves, reuse=reuse, packed=packed, ms=ms))
                        best = min(variant_times, key=lambda row: row["ms"])
                        timing = dict(n=n, batch=batch, ending=ending, mask=mask_case,
                                      baseline_ms=baseline_ms, variants=variant_times, best=best,
                                      isolated_speedup=baseline_ms / best["ms"],
                                      isolated_boundary="native includes scores.clone() to restore input; use complete-attention timing for a realistic speedup")
                        report["timings"].append(timing)
                        print(json.dumps({k: value for k, value in timing.items() if k != "variants"}), flush=True)
                        # Check the complete upstream attention using the selected exact operator.
                        reference = attention._attention(q, k, v, [mask, bias])
                        p = fused(scores, mask, bias, best["waves"], best["reuse"], best["packed"])
                        candidate = torch.einsum("...qk,...kc->...qc", p, v)
                        timing["complete_attention"] = compare(reference, candidate)
                        assert timing["complete_attention"]["exact"]
                        del reference, p, candidate
                        def candidate_attention():
                            fresh = torch.einsum("...qc,...kc->...qk", q, k)
                            probabilities = fused(fresh, mask, bias, best["waves"], best["reuse"], best["packed"])
                            return torch.einsum("...qk,...kc->...qc", probabilities, v)
                        callbacks = {"native": lambda: attention._attention(q, k, v, [mask, bias]),
                                     "candidate": candidate_attention}
                        samples = {"native": [], "candidate": []}
                        for order in (("native", "candidate"), ("candidate", "native")) * 2:
                            for arm in order:
                                samples[arm].append(measure(callbacks[arm]))
                        timing["attention_samples_ms"] = samples
                        timing["attention_medians_ms"] = {arm: statistics.median(values) for arm, values in samples.items()}
                        timing["attention_speedup"] = timing["attention_medians_ms"]["native"] / timing["attention_medians_ms"]["candidate"]
                        print(json.dumps(dict(n=n, batch=batch, ending=ending, mask=mask_case,
                                              complete_attention=timing["complete_attention"],
                                              medians_ms=timing["attention_medians_ms"],
                                              speedup=timing["attention_speedup"])), flush=True)
                        save()
                    del expected
                del pair, bias, mask, mask_storage
            del q, k, v, scores
        report["status"] = "operator_checks_complete"
    report["isolation_after"] = snapshot()
    save()


if __name__ == "__main__":
    main()
