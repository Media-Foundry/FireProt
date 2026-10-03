# Current original-Torch comparison

## Protocol

Both backends use FireProt source `b74161cb162b61e8cb4e10e9337801d326fcc462`
and independent Protenix integration `77218da6f5ccf8562a76577dd568869785e6b45b`,
based on upstream v1.1.0. The native backend runs the original Torch model
paths without FireProt runtime hooks. The candidate uses current default
FireProt operators, including CK attention and fused HIP QKV preparation.

Mini-default v0.5.0, FP32, TF32 disabled, four cycles, one diffusion step, one
sample, seed 101, no MSA/template search, hipBLAS process policy, expandable
allocator segments, one warmup and three measured predictions per side.
The same real 1000- and 1536-residue inputs are used on both sides; manifests
record their hashes. These performance inputs have no experimental-coordinate
mapping and establish native agreement, not structural accuracy.

All GPU work runs serially on MI250 gfx90a physical GPU 7; GPU 6 remains idle.
Both devices pass continuous process isolation checks. N1536 runs native then
FireProt; N1000 runs FireProt then native. Caches are warm within each process;
compiled extension caches were already populated. No recycle reduction or
lower-precision substitution is used.

## Warm model results

| Length | Original Torch median | FireProt median | Speedup |
| --- | ---: | ---: | ---: |
| 1000 | 45.248 s | 9.744 s | 4.644x |
| 1536 | 137.168 s | 28.096 s | 4.882x |

These are complete model-forward GPU timings with synchronization, excluding
loading, preprocessing, serialization and compilation. They are a fresh
same-source comparison, not products of historical incremental speedups.

| Length | Native peak allocated | FireProt peak allocated | Reduction |
| --- | ---: | ---: | ---: |
| 1000 | 39,475,042,816 B | 10,011,042,304 B | 74.64% |
| 1536 | 53,379,450,880 B | 22,721,185,792 B | 57.43% |

## Complete worker time

| Length | Native worker | FireProt worker | Speedup |
| --- | ---: | ---: | ---: |
| 1000 | 227.509 s | 85.430 s | 2.663x |
| 1536 | 609.943 s | 173.790 s | 3.510x |

Each worker includes initialization, one warmup, three measured predictions,
feature preparation, and output serialization. This is a four-prediction
benchmark process, not a single-request cold-start latency. The lower worker
speedup reflects substantial setup and CPU work outside the accelerated model.

## Numerical and dispatch checks

All six prediction pairs pass the fixed `1e-3` aligned-coordinate-RMSD and
confidence thresholds. Maximum aligned RMSD is 0.00011138 A at N1000 and
0.00012874 A at N1536. Maximum confidence delta is 7.62939e-6 and 3.81470e-6
respectively. All coordinates and confidence values are finite and schemas
match. Comparisons cover 7,979 and 12,139 output atoms respectively.

The native manifests contain no installed FireProt runtime. FireProt selects
576 QKV calls at N1000 and 1,728 at N1536 across warmup and measurements, with
zero QKV fallbacks. Manifests also retain the selected attention, triangle and
epilogue implementations.

## Remaining work

At N1536, triangle attention decreases from 95.328 to 13.291 s and triangle
multiplication from 33.316 to 8.346 s. These stage scopes overlap other reported
module scopes and must not all be added together. Attention remains the largest
operator target. A separate [QKV/gate fusion probe](protenix-qkvg-projection.md)
improves preparation with one smaller tile, but fails the complete-attention
numerical stress gate and remains unregistered.

The FireProt log shows approximately 18.6 s between environment initialization
and checkpoint loading. The constructor initializes weights before the
checkpoint overwrites them, including SciPy truncated-normal sampling. This is
an optimization lead, not an attribution of all 18.6 s to sampling. Constructor
profiling, checkpoint coverage checks, and random-state validation are required
before changing initialization. A subsequent [construction profile](protenix-construction.md)
attributes most of that time to truncated-normal initialization and tests a
checkpoint-covered candidate that preserves random-state advancement. It is
not yet integrated into the inference runtime.

[Evidence](../provenance/protenix_native_current_b74161c.json) preserves
configuration, source/input identities, individual model samples, stage totals
and call counts, memory, worker times, device monitoring and output comparisons.
