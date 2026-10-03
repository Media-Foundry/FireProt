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
