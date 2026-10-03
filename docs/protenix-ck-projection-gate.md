# CK fused projection and gate investigation

## Status

Experimental benchmark only. No production dispatch changes and no new model
or end-to-end speedup claim. The latest measured default warm model remains
10.360 s at N1000 and 29.859 s at N1536 for Mini-default FP32 C4/S1.

## Design

The existing triangle projection computes two head-last linear projections,
then packs `sigmoid(gate) * value` into channel-first storage. The candidate
computes `value = Wp @ X.T` with hipBLAS and uses CK
`DeviceGemmMultipleD_Xdl_CShuffle` for `Wg @ X.T`, fusing sigmoid and the
value multiplication in its epilogue. Both D and E are logical row-major
`[128, N*N]`, satisfying the installed CK capability checks.

The benchmark explores four M/N tiles: 64/64, 64/128, 128/64, and 128/128.
Channel padding of 0, 16, or 64 floats changes only physical storage, never
protein length. A separate private entry supports row chunks with independent
D/E strides, writing directly into final channel-first output. Chunk sizes
262144, 1000000, and 1048576 preserve the mathematical projection.

Compiled artifacts are cached by source, compiler, and installed CK header
fingerprints. These private C entry points are benchmark interfaces and are
not general tensor-validation or production fallback APIs.

## Projection measurements

All measurements use physical MI250 gfx90a GPU 7; GPU 6 is monitored as idle.
Reports pass both continuous process monitors. FP32, TF32 disabled, seed 983,
random inputs and normal(0, 0.1) weights, five warmups, twenty repetitions,
synchronized GPU events. Compilation is excluded; allocations and the value
GEMM are included. There is no model, cycle, diffusion step, or sample count
in these isolated tests.

| Experiment | Length | Baseline before / after (ms) | Best candidate (ms) |
| --- | ---: | ---: | ---: |
| Unpadded replay | 1000 | 4.197 / 4.192 | 3.606, M128/N64 |
| Unpadded replay | 1536 | 11.669 / 11.550 | 12.370, M128/N64 |
| Padding 64 | 1536 | 11.806 / 11.616 | 11.236, M128/N64 |
| Chunk sweep | 1536 | 12.036 / 11.443 | 11.082, M128/N128, 1000000 rows |

Padding 16 regresses both lengths. Padding 64 helps the long CK path but
is close to transposed BLAS (11.182 ms in that run). Chunking does not remove
all performance sensitivity to the full output stride. The chunk sweep's
baseline drift also prevents a precise projection speedup claim. All full
output comparisons pass atol=rtol=2e-5; maximum absolute difference is
4.53e-6. Padding sentinels are checked in the padded full-row experiments.

## Complete triangle operator

The complete operator comparison retains current vector packing and packed
LayerNorm in the control. The candidate replaces only the A/B projections,
using M128/N64 at N1000 and M128/N128 at N1536, with 1000000-row chunks and
no channel padding. Both contraction directions and residual-add/overwrite
modes are checked, including a nondefault stream. Input reset is outside the
timed interval. Parameters are random normal(0, 0.1), not checkpoint weights.

| Length | Direction | Update | Baseline before / after (ms) | Candidate (ms) |
| --- | --- | --- | ---: | ---: |
| 1000 | Outgoing | Add | 25.040 / 25.070 | 24.252 |
| 1000 | Outgoing | Overwrite | 24.455 / 24.556 | 23.683 |
| 1000 | Incoming | Add | 23.691 / 23.709 | 22.647 |
| 1000 | Incoming | Overwrite | 23.211 / 23.221 | 22.118 |
| 1536 | Outgoing | Add | 68.183 / 68.380 | 67.340 |
| 1536 | Outgoing | Overwrite | 67.325 / 67.587 | 66.131 |
| 1536 | Incoming | Add | 68.380 / 68.320 | 66.409 |
| 1536 | Incoming | Overwrite | 67.206 / 67.150 | 65.187 |

All eight comparisons pass atol=rtol=2e-5, with maximum absolute difference
4.77e-7. At N1536 the complete-operator gain is only about 1.01–1.03x.
Peak allocated storage falls by 512,000,000 bytes at N1000 and 1,207,959,552
bytes at N1536 within this benchmark. These are not full-model memory gains.

## Decision and next work

Keep the candidate out of default dispatch. A large isolated projection gain
at N1000 becomes a small complete-triangle increment, and the long bucket
remains the priority. The next useful investigation is to combine both
projections in a shared-input kernel, or target the remaining attention and
contraction cost. Any integration must add capability/fallback checks,
checkpoint-weight tests, and a same-source model A/B with fixed C4/S1 and
unchanged coordinate/confidence gates before promotion.

[Evidence](../provenance/protenix_ck_projection_gate_probe.json) records raw
timing distributions, source/build fingerprints, runtime identity, numerical
errors and isolation checks. Raw reports remain under DiamondHill
`outputs/benchmarks/ck-projection-gate-*.json`.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_projection_gate.py \
  --chunks 262144 1000000 1048576 --output outputs/benchmarks/projection.json

HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix FIREPROT_ENABLE_LONG1536=1 python \
  benchmarks/protenix/benchmark_projection_gate.py \
  --full-triangle --output outputs/benchmarks/projection-triangle.json
```

The subsequent [dual projection HIP investigation](protenix-dual-projection.md)
shares both projections in one kernel and improves complete-triangle timings.
It remains experimental pending model validation.
