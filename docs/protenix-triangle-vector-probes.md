# Long triangle packing and residual probes

Status: promoted for exact N1000 and opt-in N1536 after same-source model
validation. The initial probes below are retained as historical evidence.

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

The initial follow-up gate was production integration behind an experimental flag,
including pointer/shape/device guards, add and overwrite semantics, input
preservation, queued calls, and source-fingerprinted builds. Then run
same-source Mini model A/B with coordinate/confidence comparisons for
both lengths. The later model results below complete these gates for the measured Mini workloads.

[Evidence](../provenance/protenix_triangle_vector_probe_1f7b777.json) preserves
source/library hashes, build commands, distributions, stage diagnostics and
isolation results. Earlier residual-only reports predate the benchmark's
optional vector-pack argument; their recorded script hashes therefore
intentionally differ from the final combined reports.

## Production integration

`FIREPROT_ENABLE_TRIANGLE_VECTOR=1` enables the 64-row vector pack and
fused gated residual through the existing AMD extension for exact N1000
and opt-in N1536 triangles. The path is now enabled by default in these validated buckets; set
`FIREPROT_ENABLE_TRIANGLE_VECTOR=0` to restore the previous implementation. The shared extension fingerprints both edited C++/HIP sources.
Bindings check dtype, contiguity, exact dimensions, device consistency,
16-byte alignment and residual-target storage aliasing. The residual
helper explicitly disables floating-point contraction to preserve separate
multiply/add rounding. Existing operation capability gates and native
build fallback continue to apply.

The runtime manifest adds `triangle_vector.enabled` and separate N1000,
N1536 and fallback call counts. Model comparisons must check these counts
in addition to the existing registry selections, since both implementations
share the same registered triangle operation.

## Same-source model validation

Candidate `bffe312` ran on physical GPU 7, with GPU 6 idle and continuously
monitored. Each length used Mini-default v0.5.0, FP32 with TF32 disabled,
C4/S1, one sample, seed 101, hipBLAS, no MSA/templates, expandable allocator
segments, one warmup and three measured predictions. The candidate ran
first and the disabled control second. CK attention stayed enabled on both
sides. All runs completed with both isolation monitors passing.

| Metric | N1000 previous / vector | N1536 previous / vector |
| --- | ---: | ---: |
| Warm model median | 10.759 / 10.465 s | 31.524 / 30.096 s |
| Warm model p95 | 10.788 / 10.488 s | 31.735 / 30.221 s |
| Triangle multiplication median | 3.928 / 3.629 s | 11.229 / 9.988 s |
| Peak allocated memory | 10,011,042,304 B, unchanged | 22,721,185,792 B, unchanged |
| Complete worker wall time | 89.988 / 88.610 s | 187.125 / 181.297 s |

Warm model gains are **1.028x for N1000** and **1.047x for N1536**. Model
timing excludes load, preprocessing, serialization and first compilation.
Worker wall time includes one warmup and three measured predictions; it is
not single-request cold latency. These increments must not be multiplied
by unrelated historical comparisons to claim a controlled total speedup.

Each enabled run records 576 vector triangle calls and zero fallbacks.
The controls record zero vector calls and 576 fallbacks to the preceding
packed implementation. Six output pairs pass the fixed 1e-3 coordinate
RMSD and confidence gates. Maximum aligned RMSD / confidence delta are
9.58e-5 Angstrom / 3.82e-6 for N1000 (7,979 atoms) and 1.011e-4 Angstrom /
7.63e-6 for N1536 (12,139 atoms). This demonstrates agreement with the
preceding implementation, not experimental structure accuracy.

[Model evidence](../provenance/protenix_triangle_vector_model_bffe312.json)
records configuration, inputs, build identity, timing summaries, monitor
results, dispatch counts and output comparisons. Raw manifests remain on
DiamondHill under `outputs/benchmarks/triangle-vector-model-bffe312/`.
Promotion changes only the Python flag default; measured C++/HIP sources
remain unchanged. N1536 still requires `FIREPROT_ENABLE_LONG1536=1`.
