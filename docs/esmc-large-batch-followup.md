# ESMC larger batches and refreshed profiling

This follow-up keeps the published B4 route unchanged. B8 provides essentially
no additional warm throughput over optimized B4. B16 improves warm forward by
about 2.24%, but the complete invocation comparison is **1.00246x**, with one
regressing pair. The prototype is retained for experiments, not promoted.
The published direct-native B4 result remains **1.978x**.

## Conditions and correctness

HIP6 / PCI `0000:8e:00.0`, bundled ESMC 6B, native BF16, seed101, N1000 per
input. Throughput comparisons process the same 16 distinct development sequences
(rotations from the existing manifest); these are not an independent test set.
All routes retain the 80 layers, native attention and every requested output.

The strict reference is the official model with identical grouping. B4, B8 and
B16 each have their own frozen native outputs. Changing grouping can change
GEMM rounding: maximum per-element embedding differences relative to native B4
are **1.75 for B8** and **1.25 for B16**, while tokens match. These are embedding
values, not structure errors. Cross-group equality is not asserted.

All 48 HIP screening archives match the 48 corresponding native archives
byte-for-byte. Every timed forward is checked against its group's reference.
The full invocation comparison independently rehashes and checks all **96
prediction archives**. Both startup diagnostics add four exact outputs each.
No tolerance was relaxed. CPU meta-construction experiments are separate and
have no full-inference qualification.

## Warm resident throughput

Each observation executes all 16 inputs on one resident model. Loading,
preflight, tokenization, export and validation are outside this boundary. Caches
are unflushed. Each route receives two warmups and nine timed observations,
in alternating route order. Validation follows each timed observation.

| First screen | Seconds for 16 inputs | Sequences/s |
| --- | ---: | ---: |
| Native, four B4 groups | 2.566689 | 6.234 |
| Native, two B8 groups | 2.418966 | 6.614 |
| Native, one B16 group | 2.359910 | 6.780 |
| Current optimized B4, including lazy fixed FFN | 2.128590 | 7.517 |

The larger-batch HIP prototype changes only admission bounds for the existing
ESMC RoPE and pointwise kernels. It retains intermediate BF16 rounding and the
same compile flags. It fuses residual scaling/addition and uses vectorized
SwiGLU, with 80 RoPE, 80 SwiGLU and 160 residual calls per group. Residual
intermediates are also audited. B8/B16 retain native GEMM; the B4-only fixed-FFN
algorithm is explicitly disabled for those groups.

| Second screen | Seconds for 16 inputs | Ratio against optimized B4 |
| --- | ---: | ---: |
| Current optimized B4 | 2.129119 | 1.000x |
| HIP B8 | 2.128551 | 1.00027x |
| HIP B16 | 2.082488 | 1.02239x |

Do not combine medians across the two screens to manufacture a larger gain.

## Complete invocation

A frozen experimental deployment admits B8/B16 only with an explicit
`hip_esmc_large_batch` resource, N1000, the pinned runtime and gfx90a. Published optimized
admission remains B1/B2/B4. Both arms use the same frozen worker and Rust CLI,
same 16 inputs, checkpoints and snapshot monitor. B4 retains its lazy fixed FFN;
B16 uses native GEMM. Each invocation loads the model once.

Rust measures and seals each invocation. After measurement, an independent
collector checks it against its own official grouping reference. The aggregate
comparison intentionally changes grouping and is not submitted to the
identical-workload comparator. Three AB/BA/AB pairs retain all original hashes,
all exports and isolation checks; workers are fresh, caches unflushed, offline
compilation excluded.

| Pair | Optimized B4 | HIP B16 | Ratio |
| --- | ---: | ---: | ---: |
| 1, B4 then B16 | 14.058182 s | 14.023623 s | 1.00246x |
| 2, B16 then B4 | 14.074871 s | 13.977752 s | 1.00695x |
| 3, B4 then B16 | 13.966921 s | 14.028632 s | 0.99560x |
| Median | **14.058182 s** | **14.023623 s** | **1.00246x** |

Forward medians are 2.824826 -> 2.772342 s; supervisor medians are
10.238797 -> 10.192512 s. Both have median 18 isolation samples. This limited,
mixed E2E gain is insufficient to change the default. It is not a new comparison
against original native inference, and it is not a folding speedup.

## Remaining measured hotspots

Separate profiler traces instrument one B4 group. The following fractions count
only actual device `kernel` events. They do not double-count CPU `aten` events
that attribute the same device work to operators.

| GPU category | Native duration / fraction | Current optimized duration / fraction |
| --- | ---: | ---: |
| GEMM | 452.896 ms / 70.85% | **425.058 ms / 79.91%** |
| Attention | 48.530 ms / 7.59% | 48.530 ms / 9.12% |
| LayerNorm | 23.379 ms / 3.66% | 23.327 ms / 4.39% |
| Other kernels | 114.446 ms / 17.90% | 35.006 ms / 6.58% |

These are diagnostic kernel times, not hardware-counter proof of compute or
memory saturation. GEMM is the largest remaining GPU target. Reopening attention
alone has a smaller measured ceiling for this workload.

A current-route startup diagnostic measures Torch import 1.659 s, upstream
imports 1.716 s, model load 2.228 s (including placement 1.318 s), first forward
1.290 s and exports 0.037 s. A second diagnostic splits loading into meta model
construction **0.664 s**, checkpoint placement 1.164 s, strict assignment
0.015 s and model conversion/eval 0.186 s. Tokenizer import is already warm;
its measured cost is negligible. Phases are nested or overlapping and must not
be summed as independent E2E savings. Both diagnostics match official B4 outputs.

A separate three-pair cold-process CPU screen skips only initialization on meta
tensors; real tensor initialization remains native. Constructor medians improve
0.656569 -> 0.579517 s, saving about **77 ms**. All module/state metadata, real
buffer bytes and RNG state match; CUDA is never initialized. This small
constructor-only gain has not been qualified on loaded model inference and is
not enabled in a published profile.

Next candidates should target exact GEMM behavior, including B16 FFN algorithm
selection, before further broad batch admission. That direction is unmeasured;
this report does not imply that CK or another GEMM will satisfy native byte
parity. Historical CK failures remain valid for their tested shapes.

## Reproduction

Remote experiment roots under `/media/IntelSSD/fireprot/outputs/experiments`:

- `esmc-current-profile-20261016`: current-route startup diagnostic.
- `esmc-load-substeps-20261016`: finer loading diagnostic.
- `esmc-large-batch-20261016`: native batching and GPU traces.
- `esmc-large-batch-ops-20261016`: bound-only HIP prototype and same-group checks.
- `esmc-large-batch-e2e-20261016`: Rust-measured full calls and independent collector.
- `esmc-meta-construction-20261016`: CPU-only construction screen.

Measured drivers are frozen in each root's `measured-scripts` directory. Maintained
reproduction scripts are under `experiments/plm`; the HIP build recipe uses the
Rust build service. Full trace/receipt identities, every output archive pin,
source transformations and measured data are in the
[evidence record](../provenance/esmc_large_batch_followup_20261016.json).
