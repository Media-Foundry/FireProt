# AtlasFold and AtlasLM-3B performance work

## Scope

The first integration is the published AtlasFold 260703 monomer together with
AtlasLM-3B base, using the independent clean source commit
`8ab3aca0e18c8b814d5ca6756b2617a07d72c68d`. See the
[DiamondHill asset inventory](diamondhill-checkpoints.md) for both weight paths,
hashes, source location, and Python environment.

`benchmarks.atlasfold.profile_inference` runs the actual upstream loader and
runner. It retains every LM call, random MLM mask, native precision boundary,
recycle, diffusion step, and requested sample. This is an initial monomer
benchmark integration; a shared adapter coordinator is not implemented yet.

## Reproduce the native baseline

On DiamondHill, from `/media/IntelSSD/fireprot`:

```bash
env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 \
  PYTHONPATH=/media/IntelSSD/fireprot \
  /media/PM982/engramfold/folding_e2e_20260921/venvs/atlasfold-rocm/bin/python \
  -m benchmarks.atlasfold.profile_inference \
  --input benchmarks/protenix/inputs/umem_L01000_462bb697ecdcce2f.json \
  --out outputs/atlasfold/native-n1000-new-run \
  --kernel torch --recycles 4 --samples 1 --seed 101 --repeats 2
```

Use a new output directory for every run. `--model-path` and `--lm-path`
override the recorded DiamondHill defaults. Single-record FASTA is also accepted.
The upstream clean source pin is checked before inference. The two existing
1000/1536-residue inputs are performance development cases, not an independent
quality evaluation set.

The manifest records full checkpoint hashes, source commit, input sequence hash,
native configuration, real PCI/SMI identity, environment, stage CUDA events,
wall times, allocated/reserved memory, and process isolation. All sample
coordinates and confidence fields are saved as NPZ plus native mmCIF outputs.
The first prediction is labeled `first`; later same-process predictions are
`warm`. Stages are inclusive and must not be added across parent/child scopes.

The first-request timer includes model loading, feature preparation, prediction,
and serialization. It excludes interpreter/import startup, provenance hashing,
device preflight, and parent launch. Checkpoint hashing warms filesystem caches.
Event instrumentation adds overhead; final performance comparisons must also
confirm gains without dense operator instrumentation.

## Correctness and optimization policy

- Compare each candidate operator against the native path before integration.
- Then compare full outputs at identical weights, input, random seed, recycles,
  sampling configuration, dtype, and sample count. Include pLDDT, PAE, pTM, and
  coordinates; finite outputs alone do not establish correctness.
- Establish repeated-native variation before choosing a numerical gate. Do not
  loosen a gate after seeing an optimization fail it.
- Never cache stochastic LM features across recycles or omit required LM logits.
- Profile triangle attention, triangle multiplication, LM attention/logit feature
  projection, and diffusion separately. CK, hipBLAS/hipBLASLt, and handwritten
  HIP are candidates only after measured operator and full-model validation.
- Continue measured optimization until representative long-sequence gains
  flatten or a material unresolved issue prevents meaningful progress.

## Initial measured baseline

N1000, native mixed precision, seed 101, four recycles (five trunk passes),
30 diffusion steps, one sample, SMI GPU 5 / PCI `0000:93:00.0`:

The native runner pads 1000 residues to a **1024-residue bucket** with 24 masked
positions. N1536 uses bucket1536 without padding. Initial operator screening at
an unpadded N1000 therefore does not reproduce the complete model's dimensions
or mask. Preserve native buckets when comparing models.

| Measurement | Seconds |
| --- | ---: |
| First complete request, as scoped above | 184.476 |
| First prediction | 179.710 |
| Second, warm prediction | 177.577 |

Coordinates, pLDDT, PAE, and pTM are bitwise equal between both predictions.
Selected-device isolation passed. These are two observations, not a stable
multi-run median. Brief operator screening on a different GCD overlapped;
the host was shared. No optimized E2E speedup is established yet.

At N1536 the native path fails in the first LM-stack triangle-attention softmax:
it requests a 54 GiB allocation with only 22.89 GiB free on a 63.98 GiB GCD.
This is a capacity failure, not a completed timing measurement or a quality
failure. An exact chunked reference is needed for later long-input validation.

## Initial candidate screening

Random BF16 activations, actual first main-block weights, N1000, all-valid mask,
seed 1029, one warmup and three event-timed calls:

- Ordinary outer-axis chunking preserves bitwise outputs but gives limited
  starting-node gains and often regresses ending-node performance.
- Explicit contiguous input does not resolve the dominant overhead.
- SDPA changes native BF16 arithmetic: observed maximum output deltas are 2.25
  (starting) and 1.875 (ending), and performance regresses. Reject this candidate.
- Upstream Triton cannot import in this environment (`ModuleNotFoundError`);
  no Triton numerical or performance claim follows.
- Skipping the all-valid mask's additive zero preserves bitwise outputs in
  both tested operators. Chunk128 measures 185.678 to 164.731 ms (starting)
  and 263.186 to 164.245 ms (ending). Full-model validation is still required.

An operation-level profile explains the next target: one native ending-node
attention spends about 98.156 ms adding the broadcast mask, 37.425 ms adding
bias, 34.049 ms in QK BMM, and 27.279 ms in softmax. Eliminating or fusing
large intermediate-tensor passes is a higher priority than isolated projections.

All manifests, failed trials, probe sources, and the profiler table are in
[`provenance/atlasfold_initial_baseline.json`](../provenance/atlasfold_initial_baseline.json).

## Exact padding-mask candidate

The first unpadded-N1000 integration experiment selected **zero** optimized
calls and fell back 1056 times across two predictions. Its unchanged latency
and matching outputs do not validate any acceleration.

The corrected candidate accepts the actual `[1,1024,1024,128]` BF16 pair tensor
and verifies the exact outer-product mask for 1000 valid residues. It chunks
the outer triangle axis by 128, applies the original negative mask value only
to padded keys/outer rows, then performs the original bias addition, softmax,
value contraction, gate, and projection. Other inputs use native fallback.

| N1000 measurement | Native | Padding candidate | Ratio |
| --- | ---: | ---: | ---: |
| First complete request | 184.476 s | 136.834 s | 1.348x |
| First prediction | 179.710 s | 132.086 s | 1.361x |
| Warm prediction | 177.577 s | 129.794 s | 1.368x |
| First peak allocated | 54.084 GB | 16.737 GB | 69.05% lower |

All 1056 attention calls select the corrected candidate. Both prediction
comparisons have zero coordinate, pLDDT, PAE, and pTM deltas. All 240 AtlasLM
block calls per prediction remain present. Selected-device isolation passes.
These observations are experimental first/warm pairs, not repeated medians;
the optimized adapter is not enabled by default.

A handwritten HIP mask/bias fusion additionally passes the initial operator
comparison with zero output deltas, measuring about 132/136 ms for starting/
ending attention versus 178/323 ms native at the actual padded shape. It
retains BF16 rounding between the two additions. Probe source, all
failed/unselected trials, and manifests:
[`atlasfold_padding_mask_trials.json`](../provenance/atlasfold_padding_mask_trials.json).

### Handwritten HIP complete-model validation

The HIP fusion now also passes both complete-model comparisons with identical
raw bytes for coordinates, pLDDT, PAE, and pTM. All 1056 attention calls select
the candidate, all LM calls remain, and selected-device isolation passes.

| N1000 measurement | Native | HIP candidate | Ratio |
| --- | ---: | ---: | ---: |
| First complete request | 184.476 s | 123.126 s | 1.498x |
| First prediction | 179.710 s | 118.396 s | 1.518x |
| Warm prediction | 177.577 s | 116.362 s | 1.526x |
| First peak allocated | 54.084 GB | 16.746 GB | 69.04% lower |

Twelve direct-kernel checks also pass with exact numerical equality (matching
NaNs), covering fully/partially/unmasked rows, chunk offsets, infinities/NaNs,
and a non-default stream. The first build failed because an unavailable Torch
HIP error-checking macro was used; replacing it with the installed equivalent
resolved compilation. No scientific computation was changed by that fix.

The extension cache is warm and compilation is excluded from request timing.
This is still one first/warm process pair per route; repeat the measurements
without dense instrumentation before promotion. The candidate scripts and HIP
source are archived in provenance; packaging the optimized adapter, validating
N1536 against the bounded-memory reference, and broader inputs remain active
work. Neither candidate is a production default yet.
