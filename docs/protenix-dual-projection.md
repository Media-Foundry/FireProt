# Fused dual projection HIP investigation

## Status

A handwritten gfx90a FP32 MFMA candidate now accelerates the complete
triangle multiplication operator by approximately 1.12–1.15x on the tested
random-weight workloads. It remains a benchmark candidate. No production
dispatch change or model/E2E speedup is established by this investigation.

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

## Next gate

Integrate the C64/K16 candidate behind an explicit opt-in, with tensor/device/
alignment guards and native fallback. Add checkpoint-weight and edge-value
validation, repeated asynchronous launches and source-preservation checks.
Then run a same-source Mini-default FP32 C4/S1 model A/B at both lengths,
with unchanged 1e-3 coordinate/confidence gates, before default promotion.

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
