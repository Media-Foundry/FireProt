# Long triangle packing and residual probes

Status: measured candidates, not registered or promoted. The default model
path remains unchanged by these probes.

After the N1000 CK attention promotion, triangle multiplication remains
about 3.91 seconds of the 10.745-second warm model. The packed path already
avoids repeated normalization and contraction layout copies. This study
checks the surrounding memory kernels before another contraction-library
search.

## Changes

The sigmoid packing probe uses float4 global reads and writes on both sides
of the `[N*N,128]` to `[128,N*N]` transpose, retaining a padded LDS tile.
It compares 32- and 64-row tiles. Both supported lengths have row counts
that divide 64, and the ABI rejects other lengths and unaligned pointers.

The residual probe combines sigmoid multiplication and the final residual
addition into one HIP kernel. Compilation disables FP contraction so the
multiply and addition retain separate rounding. Its standalone ABI also
supports overwrite, but this study measures only the add mode used here;
overwrite validation is still required before integration.

Neither candidate changes FP32 precision, mathematical contractions,
recycle counts, masks, or sequence length. These are handwritten HIP
kernels, not CK library kernels.

## Packing measurements

Physical GPU 7 executed all work; GPU 6 remained idle. Both GPUs were
continuously monitored. FP32, TF32 disabled, seed 937, five warmup groups,
twenty measured groups of eight calls. Timings include output allocation
and exclude compilation. Inputs include extreme and infinite gate values.

| N | Existing pack before / after | Vector 32 rows | Vector 64 rows |
| --- | ---: | ---: | ---: |
| 1000 | 2.356 / 2.358 ms | 1.721 ms | 1.712 ms |
| 1536 | 7.699 / 7.699 ms | 6.218 ms | 4.849 ms |

All candidates produce zero maximum absolute delta against the current
HIP pack on these inputs. The N1536 64-row kernel improves about 1.59x.

## Complete triangle operator

The full operator comparison uses randomly initialized nonzero weights
(normal standard deviation 0.1), seed 929, one sample, no mask, FP32 and
hipBLAS. These are isolated operator invocations, not model recycle runs.
Each implementation receives the same restored input outside its timed
interval, with five warmups and twenty repetitions. The order is original,
candidate, original; compilation is excluded. Numerical comparison runs
on a non-default stream with an explicit producer-stream handoff.

| N | Direction | Original before / after | Vector pack + fused residual |
| --- | --- | ---: | ---: |
| 1000 | Outgoing | 27.953 / 27.956 ms | 25.856 ms |
| 1000 | Incoming | 26.535 / 26.509 ms | 24.519 ms |
| 1536 | Outgoing | 79.805 / 79.810 ms | 70.242 ms |
| 1536 | Incoming | 78.211 / 78.189 ms | 68.649 ms |

This is approximately 1.08x for N1000 and 1.14x for N1536. Complete output
comparisons are bitwise equal on these random inputs. Peak allocated
memory is unchanged within each controlled comparison. Every monitor
passes with no external GPU PID. The residual-only candidate provides
about 2-3% operator improvement; vectorized packing contributes the larger
part of the combined improvement.

Diagnostic events place N1000 contraction at roughly 9-10.5 ms, the two
packing calls at about 4.8 ms, and six projections at about 7.5 ms. For
N1536, contraction takes about 27-29 ms and scalar packing roughly 18-19 ms
in the instrumented operator. These extra-event measurements are diagnostic;
they should not be added to independent full-model measurements.

## Reproduce and next gate

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m benchmarks.protenix.benchmark_triangle_pack_vector \
  --output outputs/benchmarks/triangle-pack-vector.json

HIP_VISIBLE_DEVICES=7 FIREPROT_ENABLE_LONG1536=1 PYTHONPATH=.:Protenix \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python -m benchmarks.protenix.benchmark_triangle_residual \
  --length 1536 --vector-pack \
  --output outputs/benchmarks/triangle-combined-n1536.json
```

The next gate is production integration behind an experimental flag,
including pointer/shape/device guards, add and overwrite semantics, input
preservation, queued calls, and source-fingerprinted builds. Then run
same-source Mini model A/B with coordinate/confidence comparisons for
both lengths. No end-to-end speedup is established for these candidates.

[Evidence](../provenance/protenix_triangle_vector_probe_1f7b777.json) preserves
source/library hashes, build commands, distributions, stage diagnostics and
isolation results. Earlier residual-only reports predate the benchmark's
optional vector-pack argument; their recorded script hashes therefore
intentionally differ from the final combined reports.
