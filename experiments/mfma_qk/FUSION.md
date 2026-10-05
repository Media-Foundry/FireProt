# Packed QK/softmax fusion prototype

Status: local HIP compilation and static resource inspection only. No GPU
execution, numerical acceptance, timing result or model integration yet.
Do not build or run on DiamondHill while the D16 paired benchmark is active.

`fused_qk_softmax.cpp` combines the experimentally validated K8 even/odd MFMA
grouping with the maintained FP32 softmax evaluation order. It accepts packed
Q `[B*4,1536,32]`, transposed K `[B*4,32,1536]`, broadcast mask and head-specific
pair bias, writing `[B,4,1536,1536]`. Native PV remains outside this prototype.
Caller-provided storage must cover those layouts and the output must not alias
any input. These low-level pointer requirements need a future tensor boundary.

## Layout and arithmetic

- A block has1024 threads/16 wave64 waves and covers16 query rows.
- In each of six256-column bands, every wave computes a16x16 MFMA tile. Its
  eight K4 issues use `k=(fragment/2)*8+2*k_lane+(fragment%2)`.
- A16x256 FP32 LDS tile transposes MFMA accumulator coordinates into one query
  per wave. Its column XOR swizzle is applied identically on writes and reads.
- Producers write each LDS element once. A block barrier precedes reads; a
  second barrier precedes the next band's overwrite. No early thread return
  crosses those barriers.
- Computing and transposing each band immediately avoids keeping all six old
  accumulator tiles alongside the complete per-row softmax values. Query K
  operands are cached across bands.
- Each softmax wave owns columns `lane+64*t`. Mask addition precedes bias
  addition. The32-slot local max, XOR reduction, `std::exp`, local sum, XOR sum,
  division and zero-sum NaN branch follow the maintained implementation, including
  the eight padded slots. No online-softmax rescaling or reduced precision.

The C ABI `fireprot_qk_softmax_f32` takes five pointers (Q, K, mask, bias, output),
outer batch count, five int64 mask/bias strides, mode and stream. Mode0 writes
raw scores after LDS transposition; mode1 includes mask/bias/softmax. Sequence
length1536, four heads and K32 are fixed. The raw batch bound is a storage/launch
contract, not numerical acceptance for all batch sizes or library algorithms.

## Local compiler observation

With HIP7.14, O3/C++17/shared/gfx90a and no requested fast-math option:

| Variant | VGPR | SGPR | LDS | Reported scratch/spills | Compiler occupancy estimate |
| --- | ---: | ---: | ---: | --- | ---: |
| QK plus softmax | 70 | 28 | 16 KiB | zero | 7 waves/SIMD |
| QK plus transpose only | 56 | 16 | 16 KiB | zero | 8 waves/SIMD |

The occupancy remarks are compiler estimates, not measured residency. Workgroup
granularity, runtime launch feasibility, barriers, register lifetimes, LDS bank
behavior and memory traffic still need GPU validation. See the pinned local
artifacts in `provenance/openfold3_qk_softmax_local_compile.json`.

## Required next gates

1. Finish the active paired benchmark before any remote build or GPU trial.
2. Build via `build_fusion.py` in a fresh Rust cache; publish a separate bundle
   with C ABI `experimental-qk-softmax-n1536-f32-v1`. Never replace the maintained
   softmax or the accepted score-only artifact.
3. First compare mode0 against full native hipBLAS scores, retaining actual
   native GEMM dimensions. This tests the new block/LDS mapping independently.
4. Compare mode1 against native QK, ordered mask/bias additions and softmax for
   random/sparse/adversarial values, both bias layouts, broadcast/per-outer masks,
   and the actual submitted batch sizes. Preserve strict deterministic execution
   and memory filling. Require every output bit, including NaNs and signed zeros.
5. Add a guarded tensor boundary only after operator acceptance. Include input
   packing, bias packing and output allocation/filling when comparing operators.
   Validate original QKV layouts and both complete long-input models before E2E.

This prototype does not yet eliminate Q/K packing or the probability tensor
written for native PV. Its block order does not implement the maintained
softmax's outer-batch bias-reuse schedule. Both costs remain optimization
questions, not established bottlenecks or promised gains for this kernel.

## Pending numerical screen

`probe_fusion.py DESTINATION --batch 32` loads only an explicitly pinned bundle
through `FIREPROT_QK_SOFTMAX`. Each invocation uses a fresh destination. The
optional outer batches64/128/256/512/1024 must each be validated independently;32 means
128 packed head matrices. Larger screens require enough free device memory for
multiple full score tensors and are correctness runs, not timing measurements.

The screen first checks raw scores after LDS transposition for random,
adjacent cancellation, subnormal, negative-zero and nonfinite inputs. It then
checks native-order additions and Torch softmax, using contiguous/transposed
bias and zero, per-outer partial, additive and entirely masked rows. It records
actual native QK kernel correlation for the random case at the full GEMM shape.
Every comparison uses integer views of all FP32 outputs, including NaN payloads.
An error writes an incomplete report and exits without accepting later cases.
Fresh output allocation and restricted dense input layouts bound the pointer ABI.
This is an experimental harness, not a maintained model dispatch boundary.

Local syntax, wrong-environment report generation and four malformed/non-GPU
input rejection checks pass. No GPU numerical screen has run yet. Neither the
harness nor local compilation establishes fused-kernel correctness or speed.

## Separate LDS address candidate

`lds-swizzle.patch` applies only to a separate copy of the base fusion source.
It replaces the column XOR with `(query & 3) | ((query & 12) << 1)`, preserving
all stage coordinates and arithmetic. The inspected base compiler output uses
four scalar `ds_write_b32` instructions per band, not one vector store.

CPU enumeration checks all4096 stage entries and producer/consumer coordinates.
Under the explicit model `bank = FP32_word_address % 32`, the maximum number of
distinct producer addresses per bank in each contiguous32-lane group changes
from4 to1. Consumer groups stay at1. These are address-model results, not measured
hardware servicing, latency or bank-conflict counters.

The separate candidate compiles locally: full fusion remains70VGPR/28SGPR,
raw mode changes56->57VGPR with16SGPR; both retain16KiB LDS and zero reported
scratch/spills. Evidence and local archive are pinned in
`provenance/openfold3_qk_softmax_lds_candidate.json`. The base kernel and prepared
v1 numerical package remain unchanged. Validate base fusion first, then build
and validate this candidate separately before any comparative timing.

## Separate vector LDS store candidate

Apply `lds-vector-store.patch` after `lds-swizzle.patch` in a fresh source copy.
This explicitly aligns the stage to16 bytes and permutes the four accumulator
registers to match the low-two-bit column XOR, then writes an `acc4` vector.
CPU checks exhaust all4096 element mappings and vector destination alignment.
Sequential patch application reproduces the locally compiled source exactly.

Two register-permutation spellings were compiled locally. A four-way switch
produces12 static EXEC-zero branches and raises full-fusion VGPR usage to100.
The retained patch uses two conditional permutations: no such branches,68VGPR
and32SGPR for full fusion;53VGPR/16SGPR for raw scores. Both variants emit six
`ds_write_b128` instructions per kernel instead of24 scalar stage writes,
retain24 scalar reads,48 MFMA and11 barriers, and report no scratch/spills.
See `provenance/openfold3_qk_softmax_vector_candidate.json` for both artifacts.
These are static instruction counts, not executed instruction or latency claims.
The switch version is deprioritized based on compiler resources; it has not
failed a numerical or performance test. Neither vector variant has run on GPU.

The read-only ISA review confirms the manual's32 four-byte LDS banks (PDF pages16
and94, printed pages8 and86) and four consecutive dwords for DS_WRITE_B128 (PDF
page188, printed page180). Lane-to-cycle arbitration is not specified precisely
enough there to establish a measured conflict or throughput reduction. The local
CDNA2 reference PDF remains external to the committed distribution.

## Large-batch reference storage

A single B1024/H4/N1536 FP32 score tensor occupies36GiB. Use the explicit
`--reference-batch 32` option for this case; the default full-GPU reference mode
rejects batches above512. Native QK still runs once at its entire declared GEMM
shape and its output moves to CPU. The candidate runs at full shape. Comparison
then moves bounded outer-batch slices back to GPU, applies mask then bias and
Torch softmax on complete rows, and counts every differing FP32 bit. The report
labels this reference method and batch bound. It is not a claim that native
full-batch and partitioned softmax select identical GPU kernels. It cannot
replace full-model acceptance. Check this reference mode at B32 against the
ordinary full-GPU screen before using it for B1024; neither has run on GPU yet.

The host mode checks available CPU memory before allocation and releases the
cached full score allocation after staging. It is a numerical diagnostic, never
a timing benchmark. CPU checks pass for tail batches, per-outer masks, transposed
bias, all-masked NaNs, signed-zero/NaN bit differences and exact mismatch counts.
The earlier v1 prepared archive retains the original full-GPU harness; its bytes
are immutable. Use a newly pinned package for the added host-reference mode.

## First GPU observations and comparison-memory correction

The separate Rust-built/published base fusion passes all21 records at outer
B32, B64, B128 and B256; B32 with the staged reference also passes. The B32
native QK symbol equals the previously accepted packed-QK observation. These
are synthetic packed-operator checks, not model or performance acceptance.

The first full-GPU B512 screen fails inside `torch.count_nonzero` before it can
record the raw-score comparison: the reduction requests a36GiB temporary with
only22.04GiB free. This is a validation-memory failure, not evidence of an
arithmetic mismatch. Its report/log remain in the original remote directory.
`bitwise_mismatch_count` now accumulates exact counts over at most16 outer
batches per reduction; native reference QK and full-GPU softmax shapes stay
unchanged. Eight CPU tests pass, including a regression that checks reduction
batch bounds and differences across the final partial batch.

The corrected probe is deployed as a new file, preserving the old script and
failure. It must repeat both B32 reference modes before retrying B512 and the
host-staged B1024 screen. No tolerance, input, arithmetic or kernel change is
introduced by this count-buffer correction.
