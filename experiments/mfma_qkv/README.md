# Packed FP32 QK / softmax / PV fusion experiment

Status: both initial and index32 kernels passed the GPU synthetic numerical
screen on HIP7: eight cases, ordinary/capture variants, exact probabilities and
outputs, unchanged guards. Complete trained attention replay also passes exactly. The index32 variant is
3.3–3.5% faster in this operator replay; the N1536 full-model exactness gate also passes. Three paired E2E measurements
found no gain, so this initial index32 fusion is rejected for promotion.

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
`outputs/experiments/openfold3-qkv-replay-20261010`. The full-model and paired correctness gates passed subsequently; paired E2E
found no gain.
Packing costs, unsupported-input fallback, ownership, memory accounting and
launch resources still need evaluation before maintained-profile promotion. The passing
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
The full-model gate passed: 5,064 index32 calls, all 11 arrays bitwise equal and
full CIF bytes equal against the qualified isolated-data anchor. Strict FP32
N1536 R3/four trunk passes, D200/S1, seed101, HIP7; unchanged normalization/gate
counts and runner configuration. The single invocation took 335.676655 s, model
forward 249.373962 s. This is numerical qualification, not a paired speedup.
See [model evidence](../../provenance/openfold3_qkv_model_n1536.json).
Three AB/BA/AB pairs completed under
`outputs/experiments/openfold3-qkv-pairs-20261010`; both arms use identical
source and isolated-data preparation, differing only in the fusion flag/profile
digest. Complete E2E medians were 341.625922 ->342.071571 s (0.9987x); model medians
254.552379 ->253.474888 s (1.0043x). All six outputs match all11 arrays and full
CIF bytes. The small module gain did not produce a stable E2E benefit. Retain
the qualified isolated-data baseline; do not enable this fusion by default.
See [paired evidence](../../provenance/openfold3_qkv_e2e_n1536.json).

## PV scheduling follow-up

`pv_variants.py INDEX32_SOURCE DESTINATION` requires frozen source SHA
`12467c8bb72ab7786039a864bc463a4cac5750ad0ba9fc8778eb1b0dd7bc8a24`
and generates PV group-unroll4, unroll16 and one-slot-prefetch candidates.
It preserves ascending MFMA fragment order and every floating-point operation.
Source-level preservation does not replace the GPU gates.

Local gfx90a compilation found:

| Variant | Ordinary VGPR | Capture VGPR | LDS | Scratch | Selection |
| --- | ---: | ---: | ---: | ---: | --- |
| Qualified index32 | 58 | 55 | 16 KiB | 0 | Control |
| Unroll4 | 58 | 58 | 16 KiB | 0 | Next GPU trial |
| Unroll16 | 69 | 82 | 16 KiB | 0 | Deprioritized by resource pressure |
| One-slot prefetch | 58 | 55 | 16 KiB | 0 | Identical disassembled instructions to control |

The unroll4 candidate waits under
`outputs/experiments/openfold3-qkv-unroll4-20261010` until the existing index32
paired E2E coordinator completes and all six outputs pass. It then verifies the
full environment inventory, builds through Rust with no visible GPU, screens
eight packed cases/two variants on HIP7 and runs complete trained attention
against both baseline and index32. No additional remote build or GPU work
overlaps the E2E series. Unroll4 subsequently passed its synthetic/module gates; results are below.
See [static evidence and archive](../../provenance/openfold3_qkv_pv_scheduling_20261010.json).

## Separate QK index-width control

`qk_index32.patch` applies the same guarded 32-bit internal addressing to the
qualified QK/softmax source b89129b5... while retaining native PV. The C ABI
keeps 64-bit strides and bounds B to32/128 and known mask/bias layouts. This
control distinguishes index-width benefits from removing probability traffic.

Local compile: QK+softmax VGPR62 ->44; raw-QK VGPR51 ->44. Both retain16KiB
LDS and zero scratch. This is not a measured occupancy or speedup result.
GPU root `outputs/experiments/openfold3-qk-index32-20261010` waits for the
unroll4 replay, then checks raw scores/probabilities for eight synthetic cases
and compares complete trained attention for baseline, QK-index32, QKV-index32
and QKV-unroll4. Two warmups plus sixteen balanced samples per arm/capture:
144 full calls with exact checks. All queued work remains serial on HIP7.
See [local evidence](../../provenance/openfold3_qk_index32_local_20261010.json).

### Paired E2E decision

| Pair order | Baseline | Index32 fusion | Baseline / fusion |
| --- | ---: | ---: | ---: |
| AB | 341.625922 s | 364.559315 s | 0.9371x |
| BA | 342.858610 s | 342.071571 s | 1.0023x |
| AB | 340.641587 s | 339.245448 s | 1.0041x |

All observations are retained. The first candidate's slowdown is inside model
forward; chunk choices, operator counts and recorded peak memory match. These
records do not identify a clock/power or external-load cause. Median and p95
are both worse. PV schedule and QK-only index-width trials remain separate
candidates, with their own numerical and performance gates.

## PV-unroll4 and QK-only control results

The unroll4 candidate passes all eight synthetic cases in both templates and
all 84 full trained calls in its first replay. It then passes all calls in the
independent four-arm replay, which also checks the QK-only index32 control.

| Replay / direction | Baseline | QK-only index32 | QKV index32 | QKV unroll4 |
| --- | ---: | ---: | ---: | ---: |
| Three-arm / ending | 267.023 ms | — | 258.413 ms | 240.266 ms |
| Three-arm / starting | 256.957 ms | — | 248.351 ms | 230.792 ms |
| Four-arm / ending | 266.899 ms | 271.389 ms | 258.322 ms | 239.579 ms |
| Four-arm / starting | 256.866 ms | 261.595 ms | 248.301 ms | 232.345 ms |

Both replays use strict FP32 N1536, seed101, canonical HIP normalization and K
packing, unchanged retained workspace, HIP7 and unflushed caches. The three-arm
series has two warmups/twelve balanced samples per arm/capture; the four-arm
series has two warmups/sixteen balanced samples (144 exact calls). No profiler
runs inside timing. Output comparisons between samples affect cache state.

Reject QK-only index32 for latency: fewer registers did not make it faster.
Unroll4 consistently improves the complete module by about1.11x. Its subsequent
whole-model qualification passed:5,064 fused calls, all11 arrays and full CIF
bytes exact, unchanged gate/normalization counts, strict N1536 R3/D200/S1
seed101 on HIP7. The single invocation took 333.859071 s and model forward
241.828975 s. Three matched E2E pairs are running; this single observation is
not a new speedup ratio. See [model evidence](../../provenance/openfold3_qkv_unroll4_model_n1536.json).

Evidence: [unroll4 replay](../../provenance/openfold3_qkv_unroll4_replay_20261010.json),
[QK-only control](../../provenance/openfold3_qk_index32_replay_20261010.json).
The completed first QKV family is preserved in a
[pinned archive](../../provenance/openfold3_qkv_family_archive_20261010.json).

`unroll8.patch` is a further scheduling candidate against the same frozen index32
source. Local compilation reports58 VGPR for ordinary output,77 for probability
capture,16KiB LDS and zero scratch. It has no GPU result. Its numerical/module
screen is queued only after the unroll4 model and paired E2E gates complete.
See [local unroll8 evidence](../../provenance/openfold3_qkv_unroll8_local_20261010.json).
