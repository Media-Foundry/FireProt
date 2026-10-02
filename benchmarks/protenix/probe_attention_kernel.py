#!/usr/bin/env python3
"""Isolate one attention kernel for rocprof; use benchmark_attention for latency.

Inputs and additive bias use the observed Protenix layout. Bias packing and
compilation happen before the measured launches. This probe deliberately has
no subprocess-based GPU monitor: profiler injection can corrupt its JSON.
Check device isolation externally before and during the profiler process.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pathlib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("sdpa", "ck"), required=True)
    parser.add_argument("--length", type=int, choices=(1000, 1536), default=1536)
    parser.add_argument("--library", type=pathlib.Path,
                        help="Prebuilt CK C ABI library; required for CK to exclude compilation")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if os.environ.get("HIP_VISIBLE_DEVICES") not in {"6", "7"}:
        parser.error("select exactly one authorized physical GPU with HIP_VISIBLE_DEVICES=6 or 7")
    if args.iterations < 1 or (args.implementation == "ck" and args.library is None):
        parser.error("iterations must be positive and CK requires --library")

    import torch
    import torch.nn.functional as functional

    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx90a":
        raise RuntimeError("This probe requires gfx90a PyTorch HIP")
    torch.manual_seed(419)
    torch.backends.cuda.matmul.allow_tf32 = False
    n, batch = args.length, 1000 if args.length == 1000 else 512
    report = dict(scope="kernel resource/counter probe; not a latency benchmark or model result",
                  implementation=args.implementation, shape=[batch, 4, n, 32], seed=419,
                  dtype="float32", tf32=False, iterations=args.iterations,
                  torch=torch.__version__, hip=torch.version.hip,
                  physical_device=os.environ["HIP_VISIBLE_DEVICES"],
                  cache_state="first launch cold; subsequent launches warm",
                  source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    entry = None
    if args.library is not None:
        library = ctypes.CDLL(str(args.library.resolve()))
        entry = library.ck_attention
        entry.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        entry.restype = ctypes.c_int
        report.update(library=str(args.library.resolve()),
                      library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    with torch.inference_mode():
        q, k, v = [torch.randn(batch, n, 4, 32, device="cuda").transpose(1, 2) for _ in range(3)]
        q.div_(32**0.5)
        bias = torch.randn(n, n, 4, device="cuda").permute(2, 0, 1).unsqueeze(0).contiguous()
        output = torch.empty_like(q)
        torch.cuda.synchronize()
        for _ in range(args.iterations):
            if args.implementation == "sdpa":
                output = functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, dropout_p=0.0, scale=1.0)
            else:
                status = entry(q.data_ptr(), k.data_ptr(), v.data_ptr(), bias.data_ptr(),
                               output.data_ptr(), batch, n, torch.cuda.current_stream().cuda_stream)
                if status:
                    raise RuntimeError(f"CK launch returned HIP status {status}")
            torch.cuda.synchronize()
        report["finite"] = bool(torch.isfinite(output).all().item())
        if not report["finite"]:
            raise RuntimeError("nonfinite probe output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
