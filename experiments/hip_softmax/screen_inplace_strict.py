"""Strict deterministic operator screen using the maintained pinned HIP loader.

Private independent allocations only; not a maintained in-place API or E2E test.
"""
import json
import argparse
import os
from pathlib import Path
import sys

import torch

from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32
from owned_scores import OwnedScoreSoftmax
import torch.utils.deterministic


def main(destination, *, large=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(101)
    kernel = SoftmaxFP32()
    owned_kernel = OwnedScoreSoftmax()
    rows = []
    with torch.inference_mode():
        cases = ((1536, 256), (1536, 512)) if large else (
            (1000, 32), (1000, 256), (1536, 32), (1536, 128))
        for n, batch in cases:
            for ending in (False, True):
                # Independent allocations guarantee no score/mask/bias overlap.
                scores = torch.randn(batch, 4, n, n, device="cuda", dtype=torch.float32) * 0.5
                work = torch.empty_like(scores)
                bias = torch.randn(n, n, 4, device="cuda", dtype=torch.float32).permute(
                    2, 1, 0) if ending else torch.randn(n, n, 4, device="cuda", dtype=torch.float32).permute(2, 0, 1)
                bias = bias.unsqueeze(0) * 0.1
                mask = torch.zeros(batch, 1, 1, n, device="cuda", dtype=torch.float32)

                def inplace():
                    return owned_kernel(work, [mask, bias])

                checks = []
                for case in ("zero", "partial", "nonzero", "special"):
                    if case == "partial":
                        mask[..., -7:] = -float("inf")
                    elif case == "nonzero":
                        mask.normal_(0, 0.1)
                    elif case == "special":
                        mask.zero_()
                        scores[0, 0, 0, :] = -0.0
                        scores[0, 0, 1, :] = -float("inf")
                        scores[0, 0, 2, 0] = float("nan")
                        scores[0, 0, 3, 0] = float("inf")
                    work.copy_(scores).add_(mask).add_(bias)
                    reference = torch.softmax(work, dim=-1)
                    work.copy_(scores)
                    actual = inplace()
                    passed = torch.equal(reference.view(torch.int32), actual.view(torch.int32))
                    checks.append(dict(case=case, passed=passed,
                                       same_output_pointer=actual.data_ptr() == work.data_ptr()))
                    del reference, actual
                    if not passed:
                        (destination / "failed-check.json").write_text(json.dumps(dict(
                            n=n, batch=batch, ending=ending, checks=checks,
                            kernel=kernel.receipt), indent=2) + "\n")
                        raise RuntimeError(f"In-place numerical mismatch: {n}/{ending}/{case}")

                # Fresh finite values for latency; allocation and bias packing
                # inside operations, owned-score reset outside every event pair.
                scores.normal_(0, 0.5)
                mask.zero_()
                operations = {"out_of_place": lambda: kernel(work, [mask, bias]),
                              "in_place": inplace}
                observations = {name: [] for name in operations}
                extra_allocated = {name: [] for name in operations}
                for repeat in range(7):
                    names = list(operations)
                    if repeat % 2:
                        names.reverse()
                    for name in names:
                        work.copy_(scores)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        before = torch.cuda.memory_allocated()
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        output = operations[name]()
                        end.record()
                        end.synchronize()
                        if repeat >= 2:
                            observations[name].append(start.elapsed_time(end))
                            extra_allocated[name].append(torch.cuda.max_memory_allocated() - before)
                        del output
                row = dict(n=n, batch=batch, ending=ending, checks=checks,
                           observations_ms=observations, extra_peak_allocated_bytes=extra_allocated)
                rows.append(row)
                print(json.dumps(row), flush=True)
                del scores, work, bias, mask
    report = dict(schema_version=1, rows=rows,
                  inplace=owned_kernel.experiment_manifest(), seed=101, warmups=2, repetitions=5,
                  kernel=kernel.receipt, torch=str(torch.__version__), hip=str(torch.version.hip),
                  strict_deterministic=torch.are_deterministic_algorithms_enabled(),
                  warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                  fill_uninitialized_memory=torch.utils.deterministic.fill_uninitialized_memory,
                  inference_mode=True, allocator_cache="warm, not flushed between measurements",
                  fresh_process=True, model=None, cycles=None, diffusion_steps=None, samples=None,
                  visible_device=os.environ.get("HIP_VISIBLE_DEVICES"),
                  scope="Owned score overwrite; GPU-event operator timings, not model/E2E")
    (destination / "strict-inplace-screen.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--large", action="store_true",
                        help="Screen observed N1536 B256/B512 calls (large GPU memory use)")
    arguments = parser.parse_args()
    main(arguments.destination, large=arguments.large)
