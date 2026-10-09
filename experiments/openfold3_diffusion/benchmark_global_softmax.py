"""Synthetic probability and complete D48 attention gates on the pinned runtime."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32
import openfold3.core.model.primitives.attention as upstream

from global_softmax import GlobalSoftmaxDispatcher


def same(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


def run(destination):
    source = Path(upstream.__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == "30fe5433982ee73e3d7ce5341fa6b510b1ea332dbde072d56129632cc53f0746"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101)
    kernel = SoftmaxFP32()
    dispatch = GlobalSoftmaxDispatcher(lambda native, *a, **k: native(*a, **k), kernel)
    records = []
    with torch.inference_mode():
        for case in ("packed", "masked", "strided_bias", "uniform", "all_masked"):
            q, k, v = [torch.randn(1, 1, 1536, 16, 48, device="cuda").transpose(-3, -2)
                       for _ in range(3)]
            q = q / (48 ** .5)
            mask = torch.zeros(1, 1, 1, 1, 1536, device="cuda")
            bias = torch.randn(1, 1, 16, 1536, 1536, device="cuda")
            if case == "masked":
                mask[..., ::3] = -1e9
            elif case == "strided_bias":
                bias = bias.transpose(-1, -2)
            elif case == "uniform":
                q.zero_()
                bias.zero_()
            elif case == "all_masked":
                mask.fill_(-float("inf"))
            biases = [mask, bias]
            scores = torch.einsum("...qc, ...kc->...qk", q, k)
            expected_probabilities = scores.clone()
            for term in biases:
                expected_probabilities += term
            expected_probabilities = upstream.softmax_no_cast(expected_probabilities, dim=-1)
            actual_probabilities = dispatch.probabilities(scores, biases)
            torch.cuda.synchronize()
            assert same(actual_probabilities, expected_probabilities), (case, "probabilities")
            expected = upstream._attention(q, k, v, biases, use_high_precision=True)
            samples = {"native": [], "candidate": []}
            checks = {"native": 0, "candidate": 0}
            for iteration in range(14):
                order = ("native", "candidate") if iteration % 2 == 0 else ("candidate", "native")
                for arm in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    if arm == "native":
                        result = upstream._attention(q, k, v, biases, use_high_precision=True)
                    else:
                        result = dispatch(upstream._attention, q, k, v, biases, use_high_precision=True)
                    end.record()
                    end.synchronize()
                    assert same(result, expected), (case, arm, iteration)
                    checks[arm] += 1
                    if iteration >= 2:
                        samples[arm].append(start.elapsed_time(end))
            records.append(dict(case=case, q_shape=list(q.shape), q_strides=list(q.stride()),
                bias_strides=list(bias.stride()), probabilities_exact=True, checks=checks,
                samples_ms=samples, medians_ms={a: statistics.median(x) for a, x in samples.items()}))
            del q, k, v, mask, bias, biases, scores, expected_probabilities, actual_probabilities, expected, result
    report = dict(schema_version=1, correctness_passed=True, strict=True, precision="float32", seed=101,
        records=records, dispatch=dispatch.manifest(), torch=torch.__version__, hip=torch.version.hip,
        source_pins=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in (source, Path(__file__), Path(__file__).with_name("global_softmax.py"))],
        scope="Five synthetic probability checks and140 complete native-D48 attention checks",
        limits=["Synthetic only: no trained-module, whole-model or E2E qualification.",
                "Two warmups and twelve alternating samples per arm/case; checks affect cache state.",
                "All-masked case compares NaN bits, not finite-coordinate model behavior."])
    with destination.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
