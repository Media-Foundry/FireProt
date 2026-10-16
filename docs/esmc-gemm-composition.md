# ESMC FFN composition and lazy initialization

This follow-up tests direct hipBLASLt FFN up/gate dispatch on top of the
qualified pinned loader and import overlap. It remains experimental: the public
profile and its **1.440x direct-native B4 E2E** result are unchanged.
That statement describes this historical GEMM cycle; the subsequent
[host scheduling update](esmc-host-scheduling.md) publishes 1.911x without
enabling either fixed-GEMM candidate.

## Complete invocation comparisons

Each row is a separate three-pair AB/BA/AB comparison against the current exact
FireProt route. Inputs have 1,000 residues, native BF16, B4 grouping, seed101 and
HIP6 / PCI `0000:8e:00.0`. Workers are fresh, caches are unflushed, and complete
original checkpoint hashes and output exports are included. All paired arms
declare the same extra kernel resource; only the fixed-GEMM flag and profile
hash differ. No offline build or tuning time is included.

| Candidate / workload | Baseline E2E | Candidate E2E | Ratio | Paired ratios |
| --- | ---: | ---: | ---: | --- |
| Eager FFN plan, four inputs / one B4 | 16.841608 s | 16.930650 s | 0.994741x | 0.9808x, 1.0467x, 0.9728x |
| Eager FFN plan, 64 inputs / sixteen B4 | 27.716254 s | 27.757428 s | 0.998517x | 1.0004x, 0.9985x, 0.9993x |
| Lazy FFN plan, four inputs / one B4 | 16.601334 s | 16.323932 s | **1.016994x** | 1.0170x, 1.0136x, 0.9992x |

The eager composition does not improve E2E. Lazy creation improves the median,
but the third pair is 13.963 ms slower and the maximum is slightly worse.
This is mixed evidence, not a consistent E2E improvement. Do not multiply these
incremental ratios by the published native ratio or mix rows' medians.

Forward medians within these invocations are:

| Candidate / workload | Baseline forward | Candidate forward |
| --- | ---: | ---: |
| Eager, four inputs | 1.384410 s | 1.480894 s |
| Eager, 64 inputs | 10.103926 s | 9.700392 s |
| Lazy, four inputs | 1.370878 s | 1.337036 s |

Thus lazy B4 forward improves by about 1.025x. The resident eager candidate also
has a forward saving, but it does not yield a measured complete-call saving in
this comparison. A warm-kernel result alone is insufficient for promotion.

## Initialization diagnosis and implemented candidate

The eager plan is created when the optimization scope is entered, before the
first native QKV projection. Its host initialization medians are 262.935 ms for
one B4 group and 265.331 ms for the resident workload.

The diagnostic HIP/C++ library times handle creation, descriptor setup, algorithm
lookup and support checking separately. Two fresh-process observations attribute
251.565 and 254.814 ms to `getAlgosFromIndex`; handle creation takes about 1.5 ms.
These diagnostic timers exclude model loading and are not E2E comparisons.

The installed header describes same-version serialization of
`hipblasLtMatmulAlgo_t`. We retain the version/revision guards and the
[documented algorithm support check](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-10.1.0/reference/ext-reference.html).
Restoring the same 24-byte descriptor in two fresh processes fails that check
(our wrapper returns `-9`), before model execution. Restoring it after the library
has been used succeeds: plan creation takes 1.823 and 1.878 ms, and all 80 real
projections plus final native tokens/embeddings remain byte-identical. This is
consistent with an initialization dependency; the private library root cause
has not been established. The support check is never bypassed.

The useful implementation avoids serialization entirely. It delays ordinary
indexed plan creation until the first matching FFN projection, after the native
QKV has run. This reduces measured initialization to **2.124 ms median** across
the three E2E candidate invocations. It uses the original pinned kernel and
algorithm 4139, preserves all shape/dtype/device checks, and reuses the plan for
later groups. The cold-restore route is rejected.

## Correctness and retained artifacts

Independent collectors rehash and compare **432 output archives from 18 complete
invocations**. All arrays match bytewise. B4 is also checked against the retained
official same-batch output; resident64 is checked against the previous exact
FireProt resident64 output, not presented as a new direct-native comparison.
Every candidate group records 80 direct projections and 160 existing HIP calls.

The startup probes additionally check every one of the 80 projections for each
accepted indexed or warm-restored forward. Rejected cold restores produce no
model output and are not counted as successful correctness trials. The original
failed diagnostic driver and its log are retained alongside the explicit
accepted/rejected follow-up reports.

Evidence: [sealed E2E, initialization and output checks](../provenance/esmc_composed_gemm_20261016.json).
Remote root:
`/media/IntelSSD/fireprot/outputs/experiments/esmc-composed-gemm-20261016`.
The eager deployment is `qualified-fixed`; the lazy deployment is
`lazy/qualified-fixed`. Both retain pinned requests, profiles and Rust benchmark
plans. The public profile continues to point to the earlier startup deployment.

Sources are under `experiments/plm`: `prepare_esmc_composed_gemm_pairs.py`,
`esmc_lt_serialized.hip`, `probe_esmc_lt_startup.py`,
`run_esmc_lt_startup_probe.py`, `esmc_lt_lazy_scope.py`,
`prepare_esmc_lazy_gemm_pairs.py` and the corresponding collectors.

The next independent hypothesis is CPU/NUMA placement for loading. HIP6 is local
to NUMA node1 with CPU list `64-127,192-255` on DiamondHill. No affinity speedup
has been measured yet; keep full verification and bitwise acceptance when
testing it.
