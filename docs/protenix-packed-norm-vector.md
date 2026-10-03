# Vectorized packed LayerNorm

Status: experimental integration; model performance and output gates pending.

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

## Integration and next gate

`FIREPROT_ENABLE_VECTOR_PACKED_NORM=1` selects the new kernel in the exact
N1000 and opt-in N1536 packed triangle path. It is independent of the
preceding vector packing/residual flag, which stays enabled in the planned
model A/B. The shared extension fingerprints its C++/HIP source. Bindings
check exact input dimensions, FP32, contiguity, matching devices, positive
epsilon, affine size and pointer alignment; output allocation is internal.

The runtime `triangle_vector` manifest adds `packed_norm_enabled` and
`packed_norm_counts`. Full model validation must confirm expected counts,
unchanged input/configuration, GPU isolation, coordinate/confidence gates,
and repeated performance before promotion. No model speedup is claimed yet.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m benchmarks.protenix.benchmark_packed_norm_vector \
  --output outputs/benchmarks/packed-norm-vector.json
```
