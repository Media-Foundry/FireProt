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

`FIREPROT_ENABLE_TRANSITION_PROJECTION=1` selects the experimental path;
default is off pending model validation. Inputs must be contiguous FP32
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
