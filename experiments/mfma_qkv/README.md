# Packed FP32 QK / softmax / PV fusion experiment

Status: uncompiled and unqualified; queued behind the PV accumulation-order
prerequisite. No model integration, acceptance or speedup claim.

## Proposed execution

The kernel extends the qualified QK/softmax source whose SHA-256 is
`b89129b5efabb20e3563279a39c1ab4c4f47bed6909cbcb5678e367ad30c4405`.
This is a **source digest**, not a compiled-library digest. It retains the
16-query, 1,024-thread layout, mask/bias additions, full-row maximum, exponentials,
summation and division order. It does not introduce online softmax.

Normalized FP32 probabilities remain in registers, then reuse the 16 KiB LDS
staging area in six 256-key bands. Two waves accumulate the 32 value channels
using the even/odd K8 MFMA sequence. The hypothesis is that avoiding global
probability writes and reads will offset added staging and limited PV wave
parallelism. This is a performance hypothesis, not a measured benefit.

The ABI accepts packed Q `[B*4,1536,32]`, K `[B*4,32,1536]`, V
`[B*4,1536,32]`, the existing broadcast/per-outer mask and dense/transposed bias
strides, and a separate packed output. B is restricted to 32 or 128. It is a raw
experimental ABI, with no maintained tensor dispatch or workspace policy.

A diagnostic template additionally writes probabilities for comparison. The
ordinary template omits those writes. The numerical screen tests both variants
against native Torch probabilities and PV output, with output-buffer guards.
It covers random inputs, transposed bias, partially/fully masked inputs,
cancellation, subnormal, signed-zero and nonfinite values. Strict FP32 and memory
filling remain enabled. The screen runs only after PV order 1 passes the separate
bounded and full-output checks in `experiments/mfma_pv`.

The handwritten softmax retains the repository's
[PyTorch attribution](../../THIRD_PARTY.md#pytorch-bsd-style-notices).

## Queued execution and limits

DiamondHill root: `outputs/experiments/openfold3-qkv-fused-20261009`.
The frozen `run_after_pv.py` coordinator waits for the preceding PV task to
succeed, verifies its output gate and the current full environment inventory,
checks device identity/idle state, then invokes the Rust-backed HIP builder
and this numerical screen on HIP7. It stops when the prerequisite fails.

No training capture, complete attention module, model or E2E comparison has run.
Packing costs, unsupported-input fallback, ownership, memory accounting and
launch resources still need evaluation before any model integration. A passing
synthetic screen would not establish those properties or a speedup.
