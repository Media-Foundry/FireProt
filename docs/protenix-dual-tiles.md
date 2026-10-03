# Dual projection spatial tiles, waves and output strides

## Status

Experimental follow-up to the promoted C64/R64/K16/W4 dual projection at
`30019c00c76500c7324e5e0959b10cf25d1aba74`. The best new candidate is
C128/R128/K16/W8 with 128 floats of output channel padding. It improves the
complete N1536 triangle operator by approximately 1.06–1.07x in these probes.
Production dispatch is unchanged; no additional model/E2E gain is claimed.

C is the channel tile, R the spatial row tile, K the reduction tile, and W
the number of wave64 groups per block. The mathematical operation, FP32
accumulation, sigmoid, protein length and cycle policy are unchanged.

## Measurements and validation

All runs use physical GPU 7, with GPU 6 monitored as idle, MI250 gfx90a,
FP32, TF32 disabled, random seed 1009, normal inputs and normal(0,0.1)
weights. These are random-weight operator workloads, not model predictions;
there is no cycle, diffusion step or sample count. Timings use five warmups,
twenty measurements, synchronized GPU events and warm build caches.
Allocations are included; compilation and complete-triangle source resets
are excluded.

All nine reports pass both continuous process monitors. Every full output
comparison has maximum absolute delta **zero** versus the promoted kernel
or complete operator for these inputs. Projection tests check allocation
boundary sentinels and every padded channel region. N1000 exercises a
partial final R128/R256 tile. Complete-triangle tests cover both directions,
add/overwrite, and a nondefault stream. None establishes general bitwise
identity across arbitrary inputs or model weights.

## Spatial tile and wave findings

With four waves, C128/R128/K16 becomes very slow: 18.378 ms at N1000 and
44.717 ms at N1536. Eight waves reduce it to 2.531 and 7.275 ms in the next
sweep. The corresponding promoted controls are 2.773–2.775 and
8.845–8.890 ms. C128/R64/K8 is another long-sequence improvement, but loses
to C128/R128/K16/W8 and regresses N1000.

Compiler resource analysis explains the large-tile failure:

| C128/R128/K16 configuration | VGPRs | Scratch bytes/lane | VGPR spills | Reported waves/SIMD |
| --- | ---: | ---: | ---: | ---: |
| W4, implicit launch bound | 128 | 348 | 154 | 4 |
| W8, implicit launch bound | 124 | 0 | 0 | 4 |
| W4, explicit 256-thread bound | 194 | 0 | 0 | 2 |

All three use 30,720 bytes of LDS per block. Specifying the actual block
size removes spilling from W4, but W8 remains faster in the measured
padding128 sweep: 6.677 versus 6.839 ms at N1536. Compiler occupancy is a
resource estimate, not a measured runtime occupancy counter.

## Output stride

The logical output stays `[128,N,N]`. Padding adds space between channels,
with strides `[N*N+padding,N,1]`; it changes neither protein length nor
values. The existing hipBLAS-backed `torch.bmm` path consumes this strided
batch directly in the complete-operator tests.

| Channel padding (floats) | N1000 C128/R128/K16/W8 | N1536 C128/R128/K16/W8 | N1536 production before / after |
| --- | ---: | ---: | ---: |
| 0 | 2.549 ms | 7.287 ms | 8.491 / 8.493 ms |
| 16 | 2.653 ms | 7.621 ms | 8.442 / 8.368 ms |
| 64 | 2.546 ms | 7.053 ms | 8.509 / 8.405 ms |
| 128 | 2.542 ms | 6.634 ms | 8.498 / 8.387 ms |

The winning padded projection is about 1.26–1.28x faster than its controls
at N1536. Padding16 regresses. No hardware memory-counter attribution was
collected, so the results establish sensitivity to physical output stride,
not a proven memory-partition mechanism.

## Complete triangle result with padding128

The control uses the current promoted dual projection, vector LayerNorm and
residual update. The candidate changes only the A/B projections and their
physical output stride.

| Length | Direction | Update | Production before / after (ms) | Candidate (ms) |
| --- | --- | --- | ---: | ---: |
| 1000 | Outgoing | Add | 22.259 / 22.240 | 21.727 |
| 1000 | Outgoing | Overwrite | 21.729 / 21.719 | 21.210 |
| 1000 | Incoming | Add | 20.844 / 20.844 | 20.356 |
| 1000 | Incoming | Overwrite | 20.371 / 20.367 | 19.864 |
| 1536 | Outgoing | Add | 62.703 / 62.632 | 58.787 |
| 1536 | Outgoing | Overwrite | 61.333 / 60.925 | 57.603 |
| 1536 | Incoming | Add | 60.782 / 60.777 | 57.377 |
| 1536 | Incoming | Overwrite | 60.228 / 60.111 | 56.191 |

The unpadded C128/R128/K16/W8 candidate also improves complete N1536 triangle
latency, by approximately 1.04–1.06x. Padding128 preserves and modestly
extends that benefit. The two padded projection outputs add 131,072 bytes
to peak allocated storage in this operator harness at both lengths. This
is not a full-model memory result.

## Next validation

Integrate the W8 candidate behind a separate switch while retaining the
current C64/R64 fallback. Verify tensor strides, asynchronous use, and
model numerical agreement before promotion. Run a same-source model A/B
with the existing FP32 C4/S1 configuration and locked 1e-3 output gates.

[Evidence](../provenance/protenix_dual_tiles_probe_30019c0.json) contains all
nine reports, timing samples, source/build fingerprints, guards, monitors
and compiler resource remarks. Raw reports remain in DiamondHill
`outputs/benchmarks/dual-*-30019c0.json`.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_dual_tiles.py --compact --padding 128 \
  --output outputs/benchmarks/dual-tiles.json

HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix FIREPROT_ENABLE_LONG1536=1 \
FIREPROT_ENABLE_LT_BMM_1536=0 python \
  benchmarks/protenix/benchmark_dual_tiles.py --full-triangle --padding 128 \
  --output outputs/benchmarks/dual-tiles-triangle.json
```

## Experimental model integration

`FIREPROT_ENABLE_TILED_DUAL_PROJECTION=1` selects the C128/R128/K16/W8
candidate with 128 floats of channel padding. It defaults to off and only
applies when the existing dual-projection capability predicate succeeds.
`FIREPROT_ENABLE_DUAL_PROJECTION=0` disables both dual implementations.
N1536 still requires its existing length opt-in. The runtime manifest
records the tiled flag and per-length counts separately from total dual
projection calls.

The extension returns logical `[128,N,N]` with stride `[N*N+128,N,1]` and
owns the complete padded storage. Both entry points reject unsupported
shape/dtype/layout/alignment/device/autograd combinations. Tests cover
three queued projections, nondefault streams, preserved inputs, repeated
complete-triangle calls in both directions and residual modes, and total
switch precedence. CPU suite: 132 passed / 43 HIP skips. GPU regression
subset: 28 passed, both GPU process monitors passed. Model A/B is pending;
the tiled route remains opt-in.
