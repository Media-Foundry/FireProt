# Packed FP32 QK / softmax / PV fusion experiment

Status: both initial and index32 kernels passed the GPU synthetic numerical
screen on HIP7: eight cases, ordinary/capture variants, exact probabilities and
outputs, unchanged guards. Complete trained attention replay also passes exactly. The index32 variant is
3.3–3.5% faster in this operator replay; no model or E2E speedup claim.

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

## GPU execution and limits

DiamondHill root: `outputs/experiments/openfold3-qkv-fused-20261009`.
The frozen `run_after_pv.py` coordinator waits for the preceding PV task to
succeed, verifies its output gate and the current full environment inventory,
checks device identity/idle state, then invokes the Rust-backed HIP builder
and this numerical screen on HIP7. It stops when the prerequisite fails.

Complete trained attention replay completed separately under
`outputs/experiments/openfold3-qkv-replay-20261010`. No full-model or E2E
comparison has run.
Packing costs, unsupported-input fallback, ownership, memory accounting and
launch resources still need evaluation before any model integration. The passing
synthetic screen does not establish those properties or a speedup.


## Local resource screening (2026-10-10)

The local gfx90a compiler reports 16 KiB LDS and zero scratch for both template
variants. The initial four-slot PV prefetch uses 68 VGPRs. One-slot prefetch uses
66; two-slot prefetch still uses 68. These alternatives have not run on the GPU.

`index32.patch` narrows internal element-index arithmetic for the fixed packed
shapes and adds explicit host stride guards: mask stride 1 with outer stride
0/1536, bias head stride 1536² and row/key strides 1536/1 or 1/1536. The C ABI
retains 64-bit stride arguments; only values passing those guards are narrowed.
At B128, the largest probability element index remains below 2³¹; pointer byte
offsets retain normal pointer-width arithmetic. The local ordinary template
uses **58 VGPRs**, while the probability-capture template uses **55**, with zero
scratch. This does not prove target-machine occupancy or a speedup.

The index32 candidate passed separately under
`outputs/experiments/openfold3-qkv-index32-20261010`, after the initial fusion
passes every numerical case. It uses a distinct Rust build-cache identity and
repeated the same output, probability and guard checks. Neither GPU source
is replaced by a local compile artifact.
See [compiler/resource evidence](../../provenance/openfold3_qkv_local_compile_20261010.json).

## Complete trained attention replay

`benchmark.py` compares baseline, initial fusion and index32 using trained trunk
TriangleAttention start/end captures, canonical HIP normalization and HIP K
packing. All arms retain the same score allocation and Rust lease. Packing,
strict output allocation, PV, gating and projection are inside the timed module.
The fused arms do not write the retained probability workspace.

Two warmups and twelve balanced samples per arm/capture produced:

| Capture | Baseline | Initial fusion | Index32 | Baseline / index32 |
| --- | ---: | ---: | ---: | ---: |
| Ending | 267.045 ms | 357.777 ms | 258.433 ms | 1.0333x |
| Starting | 256.973 ms | 347.628 ms | 248.387 ms | 1.0346x |

All 84 complete calls match native trained outputs bitwise. Precision is strict
FP32, seed101, N1536, HIP7, caches unflushed, with equality checks between samples.
The initial fusion is rejected for latency. `model_scope.py` is the opt-in index32
full-model trial: it preserves existing allocation/lease/stream guards and writes
a separate library/call receipt. The original workspace manifest still describes
the fallback QK library and is not evidence that fused calls used that library.
This experimental scope does not support probability-reference inspection.

Evidence: [complete replay](../../provenance/openfold3_qkv_replay_20261010.json)
and [synthetic qualification](../../provenance/openfold3_qkv_synthetic_20261010.json).
Full-model and paired E2E gates remain required before promotion.
