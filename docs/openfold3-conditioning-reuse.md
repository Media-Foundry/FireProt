# OpenFold3 diffusion conditioning reuse investigation

Status: source audit and sampled native observation completed, 2026-10-05.
The native observation passes all 11 full-model arrays bitwise at N1536.
A subsequent experimental reuse diagnostic is implemented with mandatory native
recomputation; no unchecked cache or measured conditioning speedup is claimed.

## Audited source and scope

The DiamondHill checkout is
`/media/PM982/engramfold/folding_e2e_20260921/repos/openfold3`, recorded upstream
revision `7de748b7bc93adb5af0a4032d5e9f208b6e1325f`.
The following file hashes identify the bytes read, independently of Git metadata:

| File below `openfold3/core/model/` | SHA-256 |
| --- | --- |
| `layers/diffusion_conditioning.py` | `5bf5170d836c3b56cf665a9dc68bb935018ef21f779f516bc97ee0be5b1c6305` |
| `structure/diffusion_module.py` | `3b5ed9fb090d53df2b5ad16be517cfdca7f5290a6bb8274e71408a3ff465155d` |

Initial target remains inference-only FP32, strict deterministic execution,
N1000/N1536, R3, D200, S1, seed101 and the existing input/weight pins.
Recycles, timesteps, random draws and model outputs must remain unchanged.

## Dependency evidence

`SampleDiffusion._sample_rollout` passes the same `batch`, `si_input`, `si_trunk`
and `zij_trunk` to `DiffusionModule` on every executed timestep. It updates noisy
coordinates and `t`. `DiffusionModule.forward` calls `diffusion_conditioning`
before atom attention and the diffusion transformer.

| Expression in `DiffusionConditioning` | Dependencies | Reuse candidate |
| --- | --- | --- |
| Relative-position features | Batch residue/token/chain/entity/cyclic features and relative-position limits | Yes, within a rollout with immutable features |
| Pair concatenation, LayerNorm, `linear_z` | Relative-position features, trunk pair tensor, parameters and dtype | Yes |
| Two pair transitions and residual additions | Embedded pair, token mask, parameters and actual chunk size | Yes |
| Single concatenation, LayerNorm, `linear_s` | Input/trunk single tensors and parameters | Yes, before time embedding |
| Fourier embedding and addition to single representation | `t`, sigma and parameters | No across timesteps |
| Two single transitions | Time-conditioned single representation | No across timesteps |

For `use_conditioning=False`, upstream multiplies trunk tensors by zero before
embedding. Reuse must preserve that operation, including NaN behavior; replacing
it with zero allocation is not equivalent for arbitrary inputs.

The pair branch contains no random operation in the inspected implementation.
This supports a reuse experiment; it does not by itself establish whole-model
equivalence or safe lifetime management.

## Chunking and ownership constraints

`DiffusionConditioning._chunk_forward` tunes the combined pair/single `_forward`
on clones, then invokes it again with the selected chunk size. The tuner caches
shape/item-size metadata. Skipping its first execution or changing available
memory can select different GEMM shapes and affect bitwise results. Preserve
the first native call and record its effective chunk size before considering
reuse. Do not inject partially computed cache entries during tuning attempts.

Inspected immediate consumers produce new tensors: the atom encoder normalizes
and projects the conditioned pair before converting it to local blocks; the
diffusion transformer normalizes it into another tensor. Pair bias projection
also produces a new tensor. This is encouraging but not a complete alias audit.
Runtime validation must compare cached tensor bits before and after consumers.
Tensor version counters alone are insufficient in `inference_mode`, where
version tracking may be unavailable.

Cache scope must be a single sampling invocation with explicit teardown in
`finally`. Do not use a global pointer-keyed cache or persist across requests,
recycles, weight changes, devices or precision contexts. Preserve the original
tensor layouts and use native operations for all newly computed values.
Training, gradients, unsupported execution modes and uncertain dependencies
must use the upstream path. Initial experiments should exclude pocket refinement
and additional sampling passes until their cache lifetime is separately checked.

## Experiment sequence

1. After the active softmax benchmark finishes, instrument a fresh isolated
   worker copy. Record conditioning calls, actual chunks, tensor shapes/strides,
   conditioning mode and time spent in pair versus single operations. Keep
   instrumentation out of final timing runs.
2. Observe native pair output equality across distinct timesteps. Audit input
   mutations and output aliasing; record RNG state before/after the candidate
   region. Include nonzero/partial masks and conditioning-disabled inputs.
3. Introduce a per-rollout experimental cache of the final pair conditioning
   output, preserving the first complete native call and single branch on every
   timestep. Validate each returned pair against recomputation before full-model
   trials. Measure retained memory: FP32 `[1,1536,1536,128]` alone is 1.125 GiB.
4. Run native-versus-cache full N1000/N1536 comparisons with all 11 arrays
   bitwise equal. Add repeated-request invalidation and exceptional-exit cleanup
   tests. Any mismatch rejects the trial; no tolerance relaxation.
5. Measure three sequential alternating pairs under Rust with identical scientific
   work, environment, cache policy and selected GPU. Report model and complete
   invocation time, memory, actual hits/misses and every pair's correctness.
6. Only after that gate, consider caching transformer pair normalization and
   individual layer bias projections. Account for every retained tensor; caching
   all projected biases can consume substantial memory. Test each extension
   separately before combining it with the HIP softmax candidate.

## Priority and performance limits

The earlier instrumented native N1536 run recorded 957.943 seconds in `run_trunk`
and 85.310 seconds in `sample_diffusion`; N1000 recorded 296.006 and 41.509 seconds.
These are diagnostic stage timings from
`provenance/rocm_coverage_firstpass.json`, not controlled E2E baselines. They
include profiling effects and cannot predict the current candidate's speedup.
Conditioning is only part of diffusion. Trunk attention and pair operations
therefore remain the main optimization workstream while this reuse candidate is
investigated. No 200x model gain follows from a potential 200-to-1 computation
count reduction in one invariant branch.

## Accepted native observation

`provenance/openfold3_conditioning_observation.json` records one complete sampling
scope with 200 conditioning calls. The first call uses `_forward` twice with
chunk 1024 (tuning and execution); later calls use it once with the same chunk.
The final pair has shape `[1,1,1536,1536,128]`, contiguous FP32 storage.
Calls 0, 1, 99 and 199 have identical pair bytes despite different noise levels;
CPU/device RNG is unchanged within those sampled conditioning calls, and pair
bytes are unchanged after their consumers return. This is sampled evidence, not
an exhaustive proof of input immutability or equality across all 200 calls.
The unchanged full-model output passes all 11 native array comparisons. This
instrumented run has no performance interpretation. Its control archive is
pinned in the evidence record. The next gate is a scoped cache experiment with
native recomputation checks before timing.

The synthetic upstream GPU screen now passes all 12 scenarios covering both
lengths, three masks and conditioning enabled/disabled. The diagnostic checks
all four calls in each scope, including pair reuse and an independently evaluated
single branch. See `provenance/openfold3_conditioning_reuse_screen.json`.
N1000/N1536 full-model diagnostics are running with recomputation on every call;
those results remain pending. Retaining pair values and doing redundant native
work changes memory/timing, so these runs cannot supply performance evidence.
