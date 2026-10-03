# Fused dual projection HIP investigation

## Status

The handwritten gfx90a FP32 MFMA implementation is now the default inside
the validated packed triangle paths. Same-source model validation improves
N1000 from 10.370 to 9.932 s (1.044x) and N1536 from 29.930 to 28.916 s
(1.035x). These are warm model times; see the model comparison below.
The earlier random-weight operator investigation is retained in this document.

## Design and tuning

The kernel computes both `X @ Wg.T` and `X @ Wp.T`, applies
`sigmoid(gate) * value`, and writes channel-first `[128,N,N]` output in one
launch. It uses FP32 `mfma_f32_16x16x4f32`, four wave64 groups, a 64-row
spatial tile, and shared input/weight tiles. It does not reduce cycles or
change precision. It is handwritten HIP using CK-style tiled computation;
it does not call the CK library.

A simpler concatenated-weight hipBLAS GEMM was measured first. It helps
N1000 (3.858 ms versus baseline 4.218–4.223 ms), but regresses N1536
(13.696 ms versus 11.439–11.969 ms). Weight concatenation is cached outside
timing in that experiment.

The first MFMA prototype had an incorrect output-register mapping and was
rejected by numerical validation. The corrected mapping has output rows
`lane/16*4 + element` and columns `lane%16`. Its direct weight loads and
unpartitioned LDS input were slow: approximately 36.55 ms at N1536.

K32 tiles with shared weight loads and four-float LDS row padding reduced
that to 9.76 ms with 64 channels per block. A subsequent C/K sweep selected
C64/K16. Its LDS working set is 15,360 bytes, including separate gate and
value weights. C128/K16 offers more input reuse but was slower; C128/K32
was substantially worse. These measurements motivate balancing reuse with
resource occupancy rather than maximizing tile size alone. No hardware
counter attribution was collected.

## Isolated projection

| Length | Baseline before / after | C64/K16 | Maximum absolute delta |
| --- | ---: | ---: | ---: |
| 1000 | 4.223 / 4.222 ms | 2.747 ms | 4.30e-6 |
| 1536 | 11.950 / 11.503 ms | 8.874 ms | 4.53e-6 |

The N1536 control drifts during the sweep, so the speedup is approximately
1.30–1.35x rather than a single precise ratio. All full-output comparisons
pass atol=rtol=2e-5. The candidate removes both projection intermediates.

## Complete triangle operator

The control retains the promoted vector packing, residual update and packed
LayerNorm. The candidate replaces only the A/B input projections. Both use
the same hipBLAS contraction, LayerNorm and output projections. Parameters
are random normal(0,0.1), not checkpoint weights. Both directions and
residual modes pass full-output comparison on a nondefault stream.

| Length | Direction | Update | Baseline before / after (ms) | Candidate (ms) |
| --- | --- | --- | ---: | ---: |
| 1000 | Outgoing | Add | 25.070 / 25.066 | 22.195 |
| 1000 | Outgoing | Overwrite | 24.601 / 24.569 | 21.587 |
| 1000 | Incoming | Add | 23.717 / 23.725 | 20.720 |
| 1000 | Incoming | Overwrite | 23.229 / 23.235 | 20.299 |
| 1536 | Outgoing | Add | 68.677 / 68.597 | 61.052 |
| 1536 | Outgoing | Overwrite | 67.426 / 67.406 | 59.725 |
| 1536 | Incoming | Add | 67.810 / 67.745 | 60.571 |
| 1536 | Incoming | Overwrite | 66.726 / 66.572 | 59.316 |

Maximum absolute delta is 4.77e-7 across all eight checks. The operator's
peak allocation falls by 512,000,000 bytes at N1000 and 1,207,959,552 bytes
at N1536. Full-model memory impact remains unmeasured.

All successful reports use physical GPU 7 with GPU 6 monitored as idle,
MI250 gfx90a, FP32, TF32 disabled, seed 983, five warmups and twenty measured
repetitions, synchronized GPU events and warm compilation caches. Timings
include allocations; complete-triangle input reset is outside timing.
Both continuous process monitors pass in every report. These isolated
workloads have no model cycle, diffusion step or sample setting.

## Validation gates

The integration and model gates below are complete for the two measured
lengths. Additional lengths, dtypes, architectures, model weights and
production diffusion-step counts require independent validation.

[Evidence](../provenance/protenix_dual_projection_probe.json) includes the
rejected mapping experiment and subsequent successful reports, source and
build fingerprints, distributions and monitor results. Raw reports remain
under DiamondHill `outputs/benchmarks/dual-projection-*.json`.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_projection_gate.py --dual-hip \
  --output outputs/benchmarks/dual-projection.json

HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix FIREPROT_ENABLE_LONG1536=1 python \
  benchmarks/protenix/benchmark_projection_gate.py --dual-hip --full-triangle \
  --output outputs/benchmarks/dual-projection-triangle.json
```

## Integration

`FIREPROT_ENABLE_DUAL_PROJECTION` defaults to `1` inside the existing packed
triangle route. Set it to `0` to restore the preceding projection path. N1536 additionally
requires `FIREPROT_ENABLE_LONG1536=1`. Only gfx90a, FP32, contiguous aligned
N1000/N1536 inputs and bias-free contiguous aligned 128x128 projection
weights are accepted. Training/autograd and unsupported weights use the
preceding projection path. Extension unavailability preserves the existing
native fallback. The runtime `triangle_vector` manifest records the flag,
per-length triangle call counts, and candidate fallback counts.

The integrated tests cover input/weight preservation, three queued launches,
nondefault streams, zero/positive/negative constant input rows, malformed
weights, dtype/layout/alignment rejection, autograd rejection, both triangle
directions and residual modes, and bias fallback. CPU suite: 129 passed,
39 HIP skips. GPU regression subset: 21 passed, both GPU process monitors
passed. The C++/HIP sources used for model validation are unchanged by
default promotion. Historical projection, vector packing and packed-norm
benchmarks/tests explicitly disable dual projection in their controls.


## Same-source model comparison

Candidate commit `67b274aee128c064da5522bd38bc6a482b26080e`, independent Protenix
integration `77218da6f5ccf8562a76577dd568869785e6b45b`. Both sides use
Mini-default v0.5.0, FP32, TF32 disabled, four cycles, one diffusion step,
one sample, seed 101, no MSA/template search, hipBLAS, one warmup and five
measured predictions with warm in-process caches. GPU 7 runs inference;
GPU 6 is monitored as idle. Expandable allocator segments are enabled.
Other experimental attention/two-GCD/hipBLASLt routes are disabled. All
previously promoted vector packing, residual and LayerNorm paths stay on.

N1536 runs disabled then enabled; N1000 runs enabled then disabled. In each
comparison the candidate maximum is below the control minimum. Both GPU
process monitors pass in all four runs. Candidate manifests record 864
accelerated triangle calls per length (two projection kernels per triangle),
with zero fallback; controls record zero dual-projection calls.

| Metric | N1000 disabled / enabled | N1536 disabled / enabled |
| --- | ---: | ---: |
| Warm model median | 10.370 / 9.932 s | 29.930 / 28.916 s |
| Warm model p95 | 10.393 / 9.980 s | 30.084 / 28.939 s |
| Triangle multiplication median | 3.524 / 3.100 s | 9.717 / 8.621 s |
| Peak allocated bytes | 10,011,042,304, unchanged | 22,721,185,792, unchanged |
| Complete worker wall time | 115.106 / 112.589 s | 253.361 / 247.573 s |

Warm model gains are **1.044x** and **1.035x**. Whole-model improvements are
smaller than the isolated operator gain because attention and other stages
remain unchanged. Model time excludes loading, preprocessing, serialization
and compilation. Worker wall time includes one warmup and five predictions;
it is not a repeated single-request cold-start measurement.

All ten same-seed prediction comparisons pass the locked 1e-3 aligned RMSD
and confidence gates. Maximum aligned RMSD / confidence delta are 9.95e-5
Angstrom / 1.15e-5 at N1000 (7,979 atoms), and 9.73e-5 Angstrom / 7.63e-6
at N1536 (12,139 atoms). These establish agreement with the preceding
implementation on two inputs, not experimental structure accuracy.

[Model evidence](../provenance/protenix_dual_projection_model_67b274a.json)
records input/configuration identity, source and binary fingerprints,
all timing samples, dispatch counters and output comparisons. Raw reports
remain under DiamondHill `outputs/benchmarks/dual-projection-model-67b274a/`.

## Further tile tuning

A subsequent [spatial/wave/stride investigation](protenix-dual-tiles.md)
finds an additional complete-triangle improvement with C128/R128/K16/W8
and padded channel storage. It remains experimental pending model validation;
the model result above uses the promoted C64/R64/K16/W4 implementation.
