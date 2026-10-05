"""Pending GPU acceptance for the prepared published loader and scoped dispatch."""
import json
import os
from pathlib import Path
import sys

import torch

from fireprot.adapters.openfold3.attention import AttentionDispatcher, attention_scope
from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32


def main(destination):
    import openfold3.core.model.primitives.attention as attention
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(101)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    kernel = SoftmaxFP32()
    dispatcher = AttentionDispatcher(kernel)
    original = attention._attention
    records = []
    completed = False
    try:
        with torch.no_grad():
            for n, batch in ((1000, 8), (1536, 8), (1536, 128), (23, 2)):
                # Head-last storage mirrors projected QKV before head permutation.
                q, k, v = [torch.randn(batch, n, 4, 32, device="cuda", dtype=torch.float32)
                           .transpose(1, 2) * 0.125 for _ in range(3)]
                for ending in (False, True):
                    pair = torch.randn(n, n, 4, device="cuda", dtype=torch.float32)
                    bias = pair.permute((2, 1, 0) if ending else (2, 0, 1)).unsqueeze(0) * 0.1
                    mask = torch.zeros(batch, 1, 1, n, device="cuda", dtype=torch.float32)
                    cases = ("zero", "partial", "nonzero") if batch == 8 else ("zero",)
                    for case in cases:
                        if case == "partial":
                            mask[..., -7:] = -float("inf")
                        elif case == "nonzero":
                            mask.normal_(0, 0.1)
                        for high_precision in (False, True):
                            expected = original(q, k, v, [mask, bias], use_high_precision=high_precision)
                            launches_before = kernel.launches_submitted
                            with attention_scope(dispatcher):
                                actual = attention._attention(q, k, v, [mask, bias],
                                                              use_high_precision=high_precision)
                            passed = torch.equal(expected.view(torch.int32), actual.view(torch.int32))
                            record = dict(n=n, batch=batch, ending=ending, case=case,
                                          high_precision=high_precision, passed=passed,
                                          q_strides=list(q.stride()), bias_strides=list(bias.stride()),
                                          original_restored=attention._attention is original,
                                          launches_submitted=kernel.launches_submitted - launches_before,
                                          expected_route="native" if n == 23 else "hip")
                            records.append(record)
                            print(json.dumps(record), flush=True)
                            del expected, actual
                            if (not passed or not record["original_restored"]
                                    or record["launches_submitted"] != (0 if n == 23 else 1)):
                                raise RuntimeError("Scoped attention numerical/restoration check failed")
                    del pair, bias, mask
                del q, k, v
        completed = True
    finally:
        report = dict(schema_version=1, completed=completed, records=records,
                      runtime=dispatcher.manifest(), seed=101,
                      torch=str(torch.__version__), hip=str(torch.version.hip),
                      visible_device=os.environ.get("HIP_VISIBLE_DEVICES"),
                      scope="Operator correctness only; no whole-model or E2E claim")
        (destination / "screen.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
