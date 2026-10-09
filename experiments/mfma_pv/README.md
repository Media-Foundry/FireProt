# OpenFold3 PV accumulation-order probe

Status: GPU numerical screen passed on HIP7. Only order 1 (even then odd
within each K8 group) passed all six bounded cases and all four full-output
B128/B512 random/probability checks. No model or performance claim.

The current attention path materializes FP32 probabilities between QK/softmax
and PV. A future fusion must preserve the softmax results and native PV reduction
order. This probe tests that prerequisite independently, without changing model
attention or its defaults.

## Bounded contract

The native reference is packed Torch `bmm` on gfx90a with the `cublas` preference
(hipBLAS on ROCm): `[B,1536,1536] @ [B,1536,32]`. Precision is FP32, strict
algorithms and memory filling remain enabled, and seed is 101. `native-pin.json`
records the previously observed native PV kernel and rocBLAS code-object digest.
The random B128 reference is profiled only to verify that exact kernel identity;
profiler durations are not performance evidence.

The handwritten HIP probe uses FP32 16x16x4 MFMA. It checks three K-order
hypotheses: contiguous groups of four; even then odd within groups of eight;
and odd then even within groups of eight. It first compares the first 16 query
rows of the full native B128 signature on random, probability, cancellation,
subnormal, signed-zero and nonfinite inputs. Untouched destination rows must
retain their sentinel. Only hypotheses passing every bounded case proceed to
all output rows at B128/B512 on random/probability inputs.

This screen establishes only those synthetic PV comparisons. It does not
establish arbitrary-input equivalence, full attention or model parity.
The probe is deliberately untuned and must not be benchmarked as a candidate
inference implementation.

## Run on a prepared host

Use the validated OpenFold3 environment and Rust-backed HIP builder. Run only
after all existing timing sequences have ended; use a fresh result/cache root.

```bash
HIP_VISIBLE_DEVICES='' python build.py /fresh/result/build
HIP_VISIBLE_DEVICES=7 python screen.py /fresh/result
```

The result directory must contain these scripts, `pv_probe.cpp`,
`native-pin.json` and `trace_probe.py`. Configure `PYTHONPATH` for the maintained
FireProt package, `FIREPROT_BRIDGE_LIBRARY` for its verified Rust bridge and
`FIREPROT_EXTENSION_CACHE` for an isolated build cache. Clear conflicting
`CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES`, and verify the selected physical GPU.
`trace_probe.py` is an unchanged copy of the existing correlation observer in
`experiments/attention_trace/probe.py` so this frozen experiment is standalone.

DiamondHill queue: `outputs/experiments/openfold3-pv-order-20261009`.
`run_after_pairs.py` there waits for the existing isolated-data driver, checks
all six output receipts, collects/archives that series, verifies the current
environment inventory, builds, probes GPU identity and runs this screen on HIP7.
A missing or failed predecessor stops the queue. Use existing status records;
do not restart the same experiment directory.

## Static LDS address audit

`analyze_native_lds.py EXCERPT OUTPUT.json` reconstructs the native main-loop
operand coordinates from the SHA-pinned disassembly. It checks 4,096 operand
address relationships across 256 threads, two accumulator groups and eight
MFMA instructions per DepthU=32 tile. The resulting hypothesis is even/odd K8
ordering, independently for both operands. This supports probe order 1; it does
not establish floating-point equality or replace GPU checks.

The excerpt is `pv-0-44f200.asm` inside the previously archived
`native-mfma-review-v2.tar.gz`; its hash and archive location are in
`provenance/openfold3_native_mfma_review.json`. The audit result is
[recorded separately](../../provenance/openfold3_pv_lds_address_audit.json).
