# Latest native-Torch versus FireProt single-request comparison

## Isolation evidence correction (2026-10-04)

The historical launcher below monitors SMI indices derived directly from HIP
visibility ordinals. These are not interchangeable: the current DiamondHill
mapping is HIP 7 / PCI `0000:93:00.0` / SMI 5. Historical monitor pass flags
therefore do not independently establish isolation of the executing device.
Retain the archived timings and numerical checks with this limitation; they are
not a newly verified isolated performance comparison. Rust runtime monitoring
maps by PCI and rechecks the mapping. The operator migration validation records
an explicit PCI mapping and monitors the corresponding SMI index; see
[Rust operator validation](rust-operators.md).

## Protocol and timing boundary

Framework source `fa7cc555ccd8eb6cb09f3f9b30081878d5918688`, independent
Protenix `bb3be993b52713795796c72cf0a46ef8a7323442`. Mini-default v0.5.0,
FP32, TF32 off, hipBLAS, C4/S1, one sample, seed 101, no MSA/RNA
MSA/templates/ESM. Three pairs of fresh workers per length, one prediction
per worker, zero warmups, alternating execution order. Physical GPU 7
computes serially while the launcher also monitors index 6. The historical
isolation flags pass subject to the mapping limitation above. Only the authorized
HIP visibility ordinals are exposed.

Both sides use the same checkout, including direct-int64 bond-mask
construction. The reference is the native Torch route, not an unmodified
upstream checkout: it loads no FireProt runtime and uses native model
construction and template feature preparation. FireProt uses the current
promoted CK/HIP kernels plus pair-layout views, subgroup LayerNorm, fused
transition projections, fast model construction and unused-template omission.
This comparison measures the combined current inference stack rather than
attributing its gain to an individual kernel.

Filesystem, checkpoint-file and compiled-kernel caches are warm from earlier
runs; each worker starts with fresh model, CCD and allocator process state.
`process_wall_ms` includes worker setup, model loading, feature preparation,
prediction and output writing. It excludes parent launcher/preflight,
interpreter startup before the worker timer, and service/network overhead.
These are fresh-process single-request timings, not cold-storage results.

## Complete single-request E2E

| Length | Native Torch median | FireProt median | E2E speedup | Native / FireProt ranges |
| --- | ---: | ---: | ---: | --- |
| 1000 aa | **82.074 s** | **27.949 s** | **2.937x** | 82.015–82.129 / 27.866–27.955 s |
| 1536 aa | **177.345 s** | **45.596 s** | **3.889x** | 177.254–177.680 / 45.481–46.785 s |

All three paired ratios are 2.934–2.947x at N1000 and 3.791–3.907x at
N1536. These are measured medians from three pairs, not a broad production
latency-distribution claim. The narrower `inference_wall_ms` medians are
78.765 → 24.599 s and 174.014 → 42.253 s.

## Model time and memory

| Length | Native first-forward model | FireProt first-forward model | Model ratio | Native / FireProt model-call peak allocated |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 50.968 s | 15.391 s | 3.312x | 38.945 / 7.961 GB |
| 1536 | 143.157 s | 32.302 s | 4.432x | 52.131 / 14.225 GB |

These GPU-event model times include first-use overhead and are distinct from
previous warm-model medians. Memory values are decimal GB and model-call
allocated peaks, not reserved memory or process-wide CPU/GPU peaks. Do not
multiply earlier incremental ratios to replace this measured comparison.

## Numerical validation and evidence

All six output comparisons pass the locked 1e-3 Å aligned-RMSD and 1e-3
confidence gates. Maximum aligned RMSD is 0.00010709 Å at N1000 and
0.00013498 Å at N1536; maximum confidence delta is 7.63e-6 and 2.289e-5.
The native workers report no FireProt runtime. Each FireProt worker selects
fast construction once, skips unused template preparation once, and records
72 fused transition-projection calls. Four recycle cycles are retained.

These inputs are development performance cases without independent
experimental-coordinate mappings; output agreement is a regression check,
not a structure-accuracy evaluation. Model/dtype/length coverage must not be
extended from these results without separate verification.

[Complete evidence](../provenance/protenix_native_requests_fa7cc55.json)
contains the exact serial launcher, all twelve worker manifests, six output
comparisons, source and input hashes, configuration, cache state, every timing
sample, stage totals, memory, dispatch counters and isolation reports.
