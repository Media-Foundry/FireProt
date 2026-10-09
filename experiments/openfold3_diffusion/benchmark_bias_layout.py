"""Strict N1536 synthetic bias-add layout screen; no model/E2E claim."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

from bias_cache import exact, pack_projection


def run(destination):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101)
    records = []
    with torch.inference_mode():
        for case in ("random", "masked_scores", "cancellation", "special_bits"):
            projection = torch.randn(1, 1, 1536, 1536, 16, device="cuda", dtype=torch.float32)
            original = projection.permute(0, 1, 4, 2, 3)
            scores = torch.randn(original.shape, device="cuda", dtype=torch.float32)
            if case == "masked_scores":
                scores[..., ::5] = -1e9
            elif case == "cancellation":
                scores.copy_(-original)
            elif case == "special_bits":
                bits = torch.tensor([0, -2147483648, 2139095040, -8388608,
                                     2143289345, 1065353216], device="cuda", dtype=torch.int32)
                projection.view(-1)[:6].copy_(bits.view(torch.float32))
                scores.view(-1)[:6].copy_(bits.flip(0).view(torch.float32))
            packed_value = pack_projection(projection)
            packed = packed_value.permute(0, 1, 4, 2, 3)
            assert packed.is_contiguous() and exact(projection, packed_value)
            reference = scores.clone()
            reference.add_(original)
            output = torch.empty_like(scores)
            samples = {"original": [], "packed": []}
            checks = {"original": 0, "packed": 0}
            for iteration in range(14):
                order = ("original", "packed") if iteration % 2 == 0 else ("packed", "original")
                for arm in order:
                    bias = original if arm == "original" else packed
                    output.copy_(scores)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    output.add_(bias)
                    end.record()
                    end.synchronize()
                    assert exact(output, reference), (case, arm, iteration)
                    checks[arm] += 1
                    if iteration >= 2:
                        samples[arm].append(start.elapsed_time(end))
            records.append(dict(case=case, shape=list(original.shape),
                original_strides=list(original.stride()), packed_strides=list(packed.stride()),
                checks=checks, samples_ms=samples,
                medians_ms={arm: statistics.median(values) for arm, values in samples.items()}))
            del projection, original, packed_value, packed, scores, reference, output, bias
    result = dict(schema_version=1, correctness_passed=True, strict=True, seed=101,
                  records=records, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  cache_source_sha256=hashlib.sha256(Path(__file__).with_name("bias_cache.py").read_bytes()).hexdigest(),
                  torch=torch.__version__, hip=torch.version.hip,
                  scope="Four synthetic N1536 FP32 cases; only in-place bias add is timed",
                  limits=["Packing is done outside timing and would be amortized only by a qualified cache.",
                          "Native score copy and exact checks occur outside timing and affect cache state.",
                          "Two warmups and twelve alternating samples per arm/case; no model/E2E result."])
    with destination.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    run(Path(sys.argv[1]))
