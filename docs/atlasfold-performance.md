# AtlasFold and AtlasLM-3B performance work

## Scope

The first integration is the published AtlasFold 260703 monomer together with
AtlasLM-3B base, using the independent clean source commit
`8ab3aca0e18c8b814d5ca6756b2617a07d72c68d`. See the
[DiamondHill asset inventory](diamondhill-checkpoints.md) for both weight paths,
hashes, source location, and Python environment.

`benchmarks.atlasfold.profile_inference` runs the actual upstream loader and
runner. It retains every LM call, random MLM mask, native precision boundary,
recycle, diffusion step, and requested sample. An explicit instance adapter
is available under `fireprot.adapters.atlasfold.runtime`; a shared adapter
coordinator is not implemented yet.

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
the host was shared. Later sections report separately measured candidates.

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
source are archived in provenance. The mask/bias adapter is now packaged as an
explicit opt-in; native remains the default. Broader inputs and repeated
uninstrumented process comparisons remain active work.

## Packaged adapter and N1536 validation

Use the same benchmark command with `--backend hip` for the packaged HIP
mask/bias path. `--backend chunked` selects bounded-memory native arithmetic,
including both original additions. `--backend native` remains the default.
For N1536 also pass `--allow-long1536`. `--stage-events off` removes stage-event
instrumentation for latency comparisons; `stages` retains only major boundaries.

The adapter checks upstream source hashes and uses reversible instance methods.
It supports native BF16 autocast, batch size one, channel width 128, four heads,
gfx90a, and exactly the declared sequence/bucket mask. Training, autograd,
unexpected child hooks or precision overrides, and unsupported calls use the
original method. HIP build failure selects the chunked native-math reference
and records that fallback. Model semantics and stochastic LM features are
unchanged. Final manifests include actual backend counts and implementation
source hashes captured during installation.

N1536 now completes with five trunk passes, 240 AtlasLM block calls, 100
diffusion steps, and one sample. On the same SMI2 / PCI `0000:32:00.0` GCD:

| N1536 measurement | Chunked native-math reference | Packaged HIP | Ratio |
| --- | ---: | ---: | ---: |
| First complete request | 551.132 s | 334.643 s | 1.647x |
| Prediction | 546.336 s | 329.536 s | 1.658x |
| Peak allocated | 28.837 GB | 28.840 GB | Essentially unchanged |

All coordinates, pLDDT, PAE, and pTM bytes match. All 528 attention calls select
HIP and process isolation passes. The unchunked upstream route OOMs, so these
ratios are explicitly against a bounded-memory native-arithmetic reference.
Each route has one process observation; this is not a stable repeated median.
Final added capability guards also pass the GPU regression suite (11 tests),
including actual pretrained triangle weights, reinstall, masks, streams, and
precision/hook fallback. Local regression: 165 passed, 63 GPU tests skipped.
The final packaged source also completes N1000 with stage events disabled:
123.910 s request / 118.709 s prediction, all 528 calls selecting HIP and output
bytes matching native. Its manifest hashes match the committed adapter and
kernel source. This run validates packaging and instrumentation removal; a
matched uninstrumented native process is still needed for that timing ratio.

## Further softmax fusion experiment

A separate handwritten HIP kernel fuses sequential BF16 mask/bias addition,
the installed Torch wave64 softmax reduction, and the BF16 probability cast
consumed by the following value BMM. It does not use online softmax or change
the requested inference work. It is still a probe, separate from `--backend hip`.

N1000 same-device first request improves from 184.476 to **102.953 s (1.792x)**;
prediction improves from 179.710 to 98.194 s, and the second warm prediction
from 177.577 to **95.635 s (1.857x)**. Peak allocation is 14.347 GB. Both full
predictions are byte-identical to native for all saved coordinate/confidence
fields. This remains a first/warm process pair, not repeated medians.

N1536 full-model softmax fusion also matches every output byte on the same
SMI2 GCD: first request **551.132 to 253.377 s (2.175x)** against the chunked
native-math reference, prediction **546.336 to 248.554 s (2.198x)**, allocated
peak 23.432 GB. All 528 calls select the candidate. This is one process pair,
not a repeated median or a comparison with the unchunked route that OOMs.
Twelve direct probability checks also pass across N1024/N1536, padding
boundaries, fully masked rows, nonfinite inputs, and a non-default stream.
No numerical gate was relaxed.

The next measured target is triangle-multiplication layout conversion: at
N1024, two BMM input copies take about 27.25 ms while BMM itself takes 2.32 ms.
The initial tiled BF16 transpose is faster, but complete operator outputs
differ by up to 0.125 at N1024 and 0.0625 for N1536 incoming multiplication.
It is not integrated. Explicitly materializing the transposed BMM operand in
the native physical layout removes every observed delta for both lengths and
directions. Input/output packing then measures about 64.5 to 36.9 ms at N1024
and 151.3 to 84.3 ms at N1536. Full-model validation is recorded below. Layout-copy
correctness alone does not establish equivalent GEMM rounding.

The packaged-adapter, N1536, and softmax records and probe sources are in
[`atlasfold_adapter_validation.json`](../provenance/atlasfold_adapter_validation.json).

## Packaged softmax and multiplication fusion

`--backend hip-fused` explicitly selects the sequential mask/bias/softmax
fusion plus tiled BF16 input/output packing for triangle multiplication.
Native remains the default. Both N1000 (bucket1024, 24 masked residues) and
explicit `--allow-long1536` retain all five trunk passes, AtlasLM-3B calls,
30/100 native diffusion steps, one sample, and native mixed precision.
The softmax reduction is gated to the tested Torch `2.12.0a0+git78258b9`,
HIP `7.14.60850`, and gfx90a. Other versions or failed compilation fall back
to chunked native attention and original multiplication, with manifest counts.
The new kernels are handwritten HIP, not CK library calls.

Combined probe results, seed101, stage events enabled:

| Length | Reference first request | Combined first request | Ratio |
| --- | ---: | ---: | ---: |
| 1000 | 184.476 s native | 88.388 s | 2.087x |
| 1536 | 551.132 s chunked native math | 219.493 s | 2.511x |

All saved fields match raw bytes; N1000 also passes its second prediction
(81.104 s warm prediction). N1536 unchunked upstream still OOMs, so its ratio
must not be described as a completed unmodified upstream comparison.

The packaged backend additionally passes full N1536 validation: 220.848 s
first request, 215.485 s prediction, 23.432 GB peak allocated, 528 fused
attention and 528 packed multiplication calls, zero fallbacks. This run has
stage events disabled, so it validates packaging and does not form a matched
uninstrumented ratio against the earlier instrumented reference.

A second random seed (102), same physical SMI5 GCD, fresh model per process,
and **stage events disabled on both sides** also passes every output byte:

| N1000 measurement | Native | Packaged hip-fused | Ratio |
| --- | ---: | ---: | ---: |
| First complete request | 184.581 s | 88.998 s | 2.074x |
| Prediction | 179.677 s | 83.675 s | 2.147x |

These request timers include loading, adapter setup, features, prediction, and
serialization, with warm filesystem/extension caches; imports, hashing, and
preflight are excluded. Selected-device isolation passes. These remain single
process pairs, not repeated medians or independent structural-quality tests.
All coordinates, pLDDT, PAE, and pTM are compared at identical random state.
Final validation: 32 GPU tests pass; local suite 167 passed / 82 skipped.
The N1536 packaged run preceded only the final Torch/HIP version guard; final
GPU tests and the seed102 N1000 run include it.

### Continuing targets

A further batched transpose preserves native physical BMM input layouts and
passes four exact operator comparisons: about 34.0 ms at bucket1024 and
78.1 ms at N1536, versus the preceding packing candidate's 36.9/84.3 ms.
It is not packaged or full-model validated yet.

A diagnostic N1536 fused ending-attention profile attributes about 149.6 ms
of 278.4 ms to the fused softmax and 79.1 ms to QK/PV BMM. This makes softmax
throughput the next major target. The profiler reported a duplicate ROCTracer
flow; the sample guides investigation and is not an E2E timing result.
Neither numerical tolerances nor requested inference work have been reduced.

Raw reports and probe sources:
[`atlasfold_fused_pair_validation.json`](../provenance/atlasfold_fused_pair_validation.json).
