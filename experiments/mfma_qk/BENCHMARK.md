# Complete attention benchmark

Status: B32 and B256 completed, with every warm and measured output bitwise
identical. Fusion is slower than the existing split HIP path and is not promoted.
This is an operator experiment, not full-model or E2E acceptance.

`benchmark_attention.py CONTROLS NEW_DESTINATION --batch 32` compares native
Torch attention, the accepted private-score HIP softmax path, and explicitly
pinned fusion bundles. Each fusion arm requires a hashed successful21-record
numerical screen for the exact batch and matching bundle manifest. The controls
contain `split_hip` (bundle, manifest_sha256) and `fusion` entries (label, bundle,
manifest_sha256, numerical_screen with path and sha256).

The native route uses full-shape QK einsum, mask then bias additions, Torch
softmax and native PV. The split HIP route retains native QK/PV and includes
its actual bias-contiguity conversion. Each fusion includes Q/KT reshape copies
from head-last input storage, strict output allocation/filling, fused arithmetic
and native PV. Both contiguous and transposed bias are measured. Inputs use
FP32, seed101, a per-outer partial mask, strict deterministic execution with
memory filling, and the pinned OpenFold3 hipBLAS preference. Input layouts,
source hashes, bundle receipts and controls are recorded.

Every arm is warmed twice. Cyclic and reversed cyclic orderings put each arm in
every ordinal position twice. Each timing covers one function call plus device
synchronization; input generation and exact comparison are outside the interval.
Every warm and measured output must match the native complete-attention result
bitwise. Difference counting uses bounded buffers. Warmed artifacts/allocators
are retained; caches are not flushed, and the post-call checks may influence
subsequent device cache contents. Selected-device/host isolation must be recorded
by the launch controller; this standalone script is not a continuous monitor.

With the three planned fusion arms there are10 rounds,10 samples per arm and
layout, and100 measured calls total. The in-place baseline must have zero storage
fallbacks and the expected submitted-launch count. Report distributions and
ratios against both native and split HIP; do not interpret a kernel-only or
complete-attention ratio as E2E speedup. The first planned batch sizes are32 and
256. B1024 needs a separate comparison scope because full native Torch attention
requires more simultaneous score storage than a64GiB GCD can hold.

Local Python syntax, order-balance and CPU head-last packing correspondence
checks pass. These checks do not establish GPU PV parity or performance.

## Measured complete-attention medians

N1536, four heads, head dimension32. Times in milliseconds:

| Batch / bias | Native Torch | Split HIP | Fusion base | Swizzle | Vector |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 / contiguous | 9.629 | 5.522 | 9.139 | 9.109 | 9.128 |
| 32 / transposed | 16.519 | 5.817 | 10.939 | 10.914 | 11.024 |
| 256 / contiguous | 76.340 | 43.482 | 72.036 | 71.852 | 71.989 |
| 256 / transposed | 131.387 | 43.785 | 86.363 | 86.107 | 87.079 |

All200 measured calls pass bitwise comparison. Ten samples per arm/layout;
all medians independently recomputed. The LDS variants show no established
meaningful gain. Fusion is not a candidate for model promotion at this stage.
All eight GCDs were observed idle before launch; no concurrent FireProt GPU
work ran, but there was no continuous device isolation monitor.

Evidence: `provenance/openfold3_qk_attention_benchmark.json`. Full remote archive
`qk-attention-bench-v1-results.tar.gz`, SHA256
`86a6c3be25ec0c55f4de7521be62662f4d534dc84cd62a284ab7a20f8737d428`.

`profile_attention.py` is a separate warmed GPU trace diagnostic. It retains
packing, strict allocation/filling, native PV and exact output gates. Its
profiled kernel durations and Torch allocator peaks are diagnostic observations,
not a replacement latency benchmark or a total GPU-memory measurement.

## Kernel-duration diagnostic

A separate B256 trace completed for all four optimized arms and both layouts;
all eight outputs match bitwise. For contiguous bias, the fused core takes
50.47–50.70ms versus native QK9.95ms plus split HIP softmax19.17ms. PV stays
11.80–11.96ms across arms. Transposed bias increases fused-core time to
64.85–65.48ms, while split QK/softmax remain9.88/19.15ms. These are single
profiled observations, not a new statistical latency comparison.

Torch peak allocation increments are9.375GiB for split and9.750GiB for fusion;
this excludes memory outside the Torch allocator. The evidence points to the
fused core as the first optimization target. It does not establish LDS bank
conflicts as the cause. See `provenance/openfold3_qk_attention_profile.json`.

## Wave scalar address experiment

A fresh five-arm interleaved benchmark replaces the base fusion arm with the
explicit wave-scalar candidate, retaining native, split HIP, swizzle and vector.
B32 and B256 complete all200 measured calls with bitwise agreement, under the
same strict execution/input/cache contract described above.

| Batch / bias | Split HIP (ms) | Vector fusion (ms) | Wave scalar fusion (ms) |
| --- | ---: | ---: | ---: |
| 32 / contiguous | 5.528 | 9.140 | 6.899 |
| 32 / transposed | 5.819 | 11.015 | 9.532 |
| 256 / contiguous | 43.491 | 71.970 | 54.934 |
| 256 / transposed | 43.770 | 87.037 | 75.259 |

B256 contiguous complete-attention time falls23.67% versus vector fusion,
but remains26.31% above split HIP. This candidate is not promoted to model
execution. Actual compiled VGPR count falls66->60; actual hardware occupancy
has not been measured, so occupancy is not established as the cause.
Evidence: `provenance/openfold3_qk_attention_wave_benchmark.json`.

## Deferred additions experiment

A new five-arm interleaved comparison retains native/split/vector/wave and adds
the deferred mask/bias candidate. Same shapes, seed, strict execution, cache
conditions and exact-output gates; all200 measured calls pass bitwise.

| Batch / bias | Split HIP (ms) | Wave scalar (ms) | Deferred additions (ms) |
| --- | ---: | ---: | ---: |
| 32 / contiguous | 5.525 | 6.989 | 6.637 |
| 32 / transposed | 5.822 | 9.535 | 14.105 |
| 256 / contiguous | 43.498 | 55.057 | 52.187 |
| 256 / transposed | 43.791 | 75.260 | 117.302 |

B256 contiguous time improves5.21%, but transposed time increases55.86% versus
wave scalar. Deferred additions are therefore not a generic improvement and
remain experimental. Neither layout beats the existing split HIP path. No
model/E2E gain is established. Evidence and archive pin:
`provenance/openfold3_qk_attention_deferred_benchmark.json`.
