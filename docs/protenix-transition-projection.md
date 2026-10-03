# Fused pair-transition projections

## Candidate and scope

The pair transition normalizes 128 channels, computes two independent
128-to-512 linear projections, applies `SiLU(a) * b`, then projects 512-to-128.
The existing HIP path fuses only SiLU and multiplication; both expansion
projections still use hipBLAS and retain separate large activation tensors.

The candidate shares input tiles across two FP32 MFMA accumulators and writes
only the SiLU-gated product. It preserves K order, uses
`(a / (1 + expf(-a))) * b`, and retains the existing LayerNorm and final
hipBLAS projection. Production specialization is R128/C128/K16/W8 on gfx90a,
with padded shared-memory rows and no fast-math compiler option. It is a
handwritten HIP kernel using the existing CK-style tiling approach, not a CK
library instantiation.

`FIREPROT_ENABLE_TRANSITION_PROJECTION` now defaults to `1` within the existing
long-transition capability gates after the model validation below. Set it to
`0` to restore the preceding hipBLAS projections and HIP SiLU gate. Inputs must be contiguous FP32
`[1000000,128]` or `[2359296,128]` matrices from the existing long-transition
path, inference/no-grad, with 16-byte alignment. N1536 retains its length
opt-in. Both expansion weights must be contiguous same-device FP32
`[512,128]`, from standard Torch or Protenix linear modules with no bias,
precision override, instance-forward override, or local/global forward hooks.
Non-FP32 autocast falls back; the model's FP32 autocast context is accepted.
C++ independently checks shape, dtype, contiguity, alignment, device,
inference mode and architecture, and launches on the current HIP stream.
Unavailable extensions preserve the preceding path. Manifests count actual
selections as `amd-gfx90a-transition-projection` under gated transitions.
No Protenix source patch, recycle reduction or precision change is involved.

## Standalone operator measurements

Framework base `6c8160f`; random weights and inputs, seed 1029, FP32, TF32 off,
hipBLAS, no checkpoint/cycles/diffusion/sample count. Normal inputs and
normal(0,0.1) weights; input rows include zero and positive/negative 100.
Five warmups, twenty measurements; allocations included, compilation excluded.
Both 6 and 7 are continuously monitored, compute uses 7, and isolation passes.
All four tile choices pass preparation and complete-output checks at
`atol=rtol=2e-5`, including a 131-row tail and guarded output boundaries on a
nondefault stream. The 131-row signature exists only in the private probe.

| N | Scope | Baseline before / after | R128/C128/W8 |
| --- | --- | ---: | ---: |
| 1000 | Two projections + SiLU gate | 14.542 / 14.553 ms | 10.124 ms |
| 1000 | Complete transition | 19.990 / 19.992 ms | **15.529 ms (1.287x)** |
| 1536 | Two projections + SiLU gate | 33.960 / 33.929 ms | 23.842 ms |
| 1536 | Complete transition | 46.871 / 46.846 ms | **36.425 ms (1.286x)** |

R128/C64/W8 is close (15.637 / 36.497 ms complete); R64/C64/W8 and
R128/C64/W16 are slower. Complete-output maximum absolute differences for
N1000/N1536 are 1.19e-7 and 1.94e-7. The complete probe's allocated peak drops
by one `[N*N,512]` FP32 tensor: 2,048,000,000 / 4,831,838,208 bytes. Those
peaks include retained reference tensors and are not full-model peaks.

[Probe evidence](../provenance/protenix_transition_projection_6c8160f.json)
retains all timing samples, before/after controls, guard/precision results,
compiler commands and binary/source fingerprints, memory and GPU monitors.
The production interface and capability tests are tracked separately. No
model-level numerical agreement, model memory reduction or E2E speedup is
established by this random-weight operator replay.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_transition_projection.py \
  --output outputs/benchmarks/transition-projection.json
```


## Same-source model validation

Candidate `c5a24ce`, independent Protenix `bb3be99`. Mini-default v0.5.0 FP32,
TF32 off, hipBLAS, C4/S1, one sample, seed 101, no MSA/RNA MSA/templates/ESM.
Each worker performs one warmup and three measured predictions. Pair-layout
views, subgroup transposed LayerNorm, fast model construction and unused-template
omission are enabled on both sides; direct int64 bond masks are shared.
Only the transition-projection flag changes. N1536 runs enabled then disabled;
N1000 runs disabled then enabled. GPU 7 computes serially and GPU 6 is monitored;
all isolation checks pass. Filesystem/compiled caches are warm, and measured
model calls also have warm process caches.

| Length | Disabled warm model median | Enabled median | Speedup | Allocated peak disabled / enabled |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 9.515 s | **9.184 s** | **1.036x** | 8.987 / 8.491 GB |
| 1536 | 27.188 s | **26.408 s** | **1.030x** | 20.305 / 15.473 GB |

Model ranges are 9.502–9.555 versus 9.174–9.230 s at N1000 and
27.137–27.365 versus 26.222–26.460 s at N1536. Model allocated peaks drop
5.52% and 23.80%, respectively. Unlike the standalone probe, these are actual
model-call peaks; the N1000 reduction is limited by other live tensors.

Complete worker wall time improves 60.111 → 58.572 s (1.026x) at N1000 and
135.073 → 131.482 s (1.027x) at N1536. Each worker includes initialization,
features/output writing, one warmup and three predictions. This is not a
single-request cold E2E comparison and excludes parent launch/preflight.

All six measured output comparisons pass the locked aligned-RMSD/confidence
thresholds of 1e-3. Maximum aligned RMSD is 0.00010660 Å at N1000 and
0.00010366 Å at N1536; maximum confidence delta is 7.63e-6 and 3.82e-6.
Enabled workers select the fused projection 288 times (72 per prediction),
disabled workers zero times. Both sides select subgroup LayerNorm 288 times
and fast construction once. These are regression results for development
performance cases, not independent experimental-coordinate accuracy metrics.

The full local suite passes 158 tests with 55 HIP-gated skips. Ten GPU tests
cover the new production interface and preceding N1536 epilogues. Default
selection is also exercised by leaving the new flag unset in the production
numerical test. No weights, cycle count or diffusion count change at promotion.

[Model evidence](../provenance/protenix_transition_projection_model_c5a24ce.json)
contains exact source/input hashes, the serial launcher, all four worker
manifests, per-call stage totals, memory, dispatch/isolation records and six
output comparisons. The incremental model ratios must not be multiplied by
historical original-Torch results to claim a new controlled baseline.
