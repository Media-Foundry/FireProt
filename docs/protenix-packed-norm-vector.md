# Vectorized packed LayerNorm

Status: promoted for exact N1000 and opt-in N1536 after model validation.

The packed triangle output is stored as `[128,N,N]`. The existing HIP
kernel loads a scalar channel-first tile, normalizes each 128-channel row
with a wave64 reduction, and emits two scalar channel halves. The new
candidate uses float4 reads, a 32-row LDS tile, and float4 output stores.

## Reduction-order gate

A naive 32-lane reduction of four adjacent channels is fast but fails the
unchanged `atol=rtol=2e-5` gate on a low-variance row near one. Maximum
absolute differences reach 1.05e-4 at N1000 and 9.17e-5 at N1536. Random
inputs alone miss this failure. This mode is retained only as rejected
benchmark evidence.

The selected mode first pairs channels separated by 64, then follows the
old wave64 reduction tree using subgroup shuffles and the final local
channel pairs. It retains the pairing for squared deviations as well as
the mean. Floating-point results are not claimed bitwise identical: the
maximum random/edge absolute delta is 2.87e-6. Both constant and
low-variance rows pass the original tolerance. FP32 arithmetic is retained.

## Operator result

GPU 7 performed the work with GPU 6 idle; both continuous monitors pass.
Seed 967, FP32, TF32 off, five warmup groups and twenty measured groups of
eight calls. Timing includes output allocation and excludes compilation.

| N | Original before / after | Vector load, old wave64 | Selected vector output |
| --- | ---: | ---: | ---: |
| 1000 | 1.984 / 1.988 ms | 1.611 ms | 1.190 ms |
| 1536 | 6.421 / 6.417 ms | 5.868 ms | 4.088 ms |

All three columns use 32-row tiles. 64-row alternatives were slower.
The selected operator is about 1.67x faster at N1000 and 1.57x at N1536.
Output-buffer sentinels before and after the logical tensor remain intact.
[Evidence](../provenance/protenix_packed_norm_vector_probe_7daf6df.json)
contains every candidate, including rejected numerical results.

## Integration

`FIREPROT_ENABLE_VECTOR_PACKED_NORM=1` selects the new kernel in the exact
N1000 and opt-in N1536 packed triangle path. It is independent of the
preceding vector packing/residual flag, which stays enabled in the planned
model A/B. The shared extension fingerprints its C++/HIP source. Bindings
check exact input dimensions, FP32, contiguity, matching devices, positive
epsilon, affine size and pointer alignment; output allocation is internal.

The runtime `triangle_vector` manifest adds `packed_norm_enabled` and
`packed_norm_counts`. Full model validation must confirm expected counts,
unchanged input/configuration, GPU isolation, coordinate/confidence gates,
and repeated performance before promotion. The completed model comparison is reported below. Set the flag to `0` to
restore the preceding packed LayerNorm; the new kernel is now the default
inside the validated capability buckets.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m benchmarks.protenix.benchmark_packed_norm_vector \
  --output outputs/benchmarks/packed-norm-vector.json
```

## Same-source model result

Candidate `a5af61b` used Mini-default v0.5.0, FP32/TF32 off, C4/S1, one
sample, seed 101, hipBLAS, no MSA/templates and expandable allocator segments.
Physical GPU 7 executed every run with GPU 6 idle and monitored. Each side
used one warmup and five measured predictions. N1536 ran disabled then
enabled; N1000 ran enabled then disabled. CK attention and vector packing/
residual fusion remained enabled on both sides.

| Metric | N1000 previous / new | N1536 previous / new |
| --- | ---: | ---: |
| Warm model median | 10.466 / 10.360 s | 30.237 / 29.859 s |
| Warm model p95 | 10.527 / 10.418 s | 30.390 / 29.960 s |
| Triangle multiplication median | 3.632 / 3.536 s | 10.011 / 9.676 s |
| Peak allocated bytes | 10,011,042,304, unchanged | 22,721,185,792, unchanged |
| Complete benchmark worker wall time | 116.085 / 115.865 s | 264.130 / 254.939 s |

Warm model gains are **1.010x** and **1.013x**, respectively. In each
five-sample comparison the candidate maximum is below the control minimum.
This is a small model increment despite the larger isolated operator gain.
Model time excludes load, preprocessing, serialization and first compilation.
Worker wall time includes one warmup and five predictions, with CPU/I/O
variation; it is not a repeated single-request cold-start measurement.

Each candidate records 864 new packed LayerNorm calls; each control records
zero. All four runs pass both continuous process monitors. All ten prediction
pairs pass the unchanged 1e-3 aligned-coordinate RMSD and confidence gates.
Maximum aligned RMSD / confidence delta are 9.97e-5 Angstrom / 4.30e-6 at
N1000 (7,979 atoms) and 9.91e-5 Angstrom / 7.63e-6 at N1536 (12,139 atoms).
These are agreement checks against the preceding implementation on two
inputs, not experimental structure accuracy.

[Model evidence](../provenance/protenix_packed_norm_vector_model_a5af61b.json)
contains input/configuration/build identity, timings, dispatch counts and
output comparisons. Raw manifests remain on DiamondHill under
`outputs/benchmarks/packed-norm-model-a5af61b/`. Promotion changes only the
Python default; the model-tested C++/HIP sources remain unchanged. N1536
still requires `FIREPROT_ENABLE_LONG1536=1`.

## Next investigation

The historical N1000 projection inventory measures plain direct CK GEMM
slower than hipBLAS for the large 128-to-128 projection. A useful next
candidate should fuse the gate and channel-first output with that GEMM. The installed `DeviceGemmMultipleD_Xdl_CShuffle` accepts
only row-major D and E layouts in its capability checks. Expressing the
projection as weights times transposed features could produce the desired
channel-first output in logical row-major form, but both its projection
layout has now been measured in a separate [projection fusion investigation](protenix-ck-projection-gate.md). Complete-triangle gains are small; its model benefit remains unmeasured.
