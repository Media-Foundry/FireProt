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
