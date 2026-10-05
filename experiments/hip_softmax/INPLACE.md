# Pending experiment: owned score buffer reuse

Status: GPU operator screen complete; no model/E2E validation or integration.
The completed three-pair N1536 experiment uses the original out-of-place boundary.
Keep its copied workers and immutable bundle intact.

The current kernel loads each score row into lane-local values before any output
store. Its wave64 reductions keep all row lanes participating. Independent rows
are mapped either directly or through a bijection over groups of eight outer
rows. No row consumes another score row, and the C++ pointers are not `restrict`.
These source properties support testing identical score/output pointers when
the score allocation is privately owned and disjoint from both biases.

`screen_inplace.py` calls the existing pinned C ABI with a separately allocated
work buffer. Both mask and pair bias use independent allocations. It compares
FP32 output bits against native mask-add, bias-add and softmax at N1000/B256 and
N1536/B128, with both bias layouts and zero/partial/nonzero/special-value cases.
It then alternates the selected packed HIP out-of-place and in-place variants,
including bias packing and output allocation but excluding score reset and QK/PV.
Five observations follow two warmups; peak additional Torch allocations are
recorded separately from latency.

The N1536/B128 FP32 score contains 1,207,959,552 elements (4.5 GiB); N1000/B256
contains 1,024,000,000 elements (about 3.815 GiB). Avoiding a second allocation of
this size is a theoretical workspace saving, not a measured whole-model peak
reduction. Score reads and probability writes still occur, so lower memory
traffic or latency must not be assumed.

The screen ran after the E2E experiment finished, through the same pinned
published loader on HIP5/PCIb3/SMI7. All 16 comparisons pass bitwise. Five
observations follow two warmups, with alternating variant order:

| Shape/layout | Out-of-place median | In-place median | Saved extra allocation |
| --- | ---: | ---: | ---: |
| N1000/B256 starting | 10.970 ms | 11.095 ms | 4,096,000,000 bytes |
| N1000/B256 ending | 10.964 ms | 11.071 ms | 4,096,000,000 bytes |
| N1536/B128 starting | 9.882 ms | 9.863 ms | 4,831,838,208 bytes |
| N1536/B128 ending | 10.268 ms | 10.256 ms | 4,831,838,208 bytes |

These observations establish operator allocation savings, not a latency win.
N1000 is about 1% slower; the N1536 differences are only 0.1-0.2%. Keep the
out-of-place path for maintained integration. Full observations, source/library
pins and limits are in `provenance/openfold3_hip_softmax_inplace_screen.json`.
Isolation checks were point-in-time, not continuous monitoring.

A later
worker trial must ensure QK scores have no other consumers and reject aliases
with mask/bias. Full-model validation must record effective chunk sizes: the
memory reduction can alter upstream autotuning, GEMM shapes and numerical
results. Require all 11 arrays bitwise equal at both lengths before any paired
E2E or default-dispatch change. This script adds no maintained in-place API.

## Strict-mode follow-up candidate

The accepted bounded N1536 attention trace records `aten::empty_like` followed
by `aten::fill_` on the `[32,4,1536,1536]` softmax output allocation. The maintained
loader creates that allocation in `SoftmaxFP32.__call__`. The earlier in-place
screen above uses `no_grad` and does not explicitly enable strict deterministic
algorithms or deterministic uninitialized-memory filling. Its latency result
therefore does not close the question for the current strict model requests.

A follow-up should repeat the operator screen under the actual strict settings,
retain all numerical and alias guards, then validate both full-model lengths and
effective chunk sizes. Reusing the private score buffer could avoid this output
allocation and fill; no latency gain is yet established. Do not disable global
determinism or its fill policy to improve a benchmark. The actual settings and
the allocation being eliminated must be recorded explicitly.

## Completed strict operator screen and model candidate

`screen_inplace_strict.py` now uses the maintained `SoftmaxFP32` loader and
`owned_scores.py`, under `inference_mode`, strict deterministic algorithms,
`warn_only=False`, and `fill_uninitialized_memory=True`. All 32 native bitwise
comparisons pass across N1000/N1536, B32 and B256/B128, both bias layouts and
four mask/special-value cases. A repeat using the shared model-candidate helper
confirms the initial independent screen. Both versions are archived.

| Shape/layout | Out-of-place median | In-place median | Ratio |
| --- | ---: | ---: | ---: |
| N1000/B256 starting | 14.017 ms | 11.144 ms | 1.258x |
| N1000/B256 ending | 14.110 ms | 11.132 ms | 1.268x |
| N1536/B128 starting | 13.418 ms | 9.918 ms | 1.353x |
| N1536/B128 ending | 13.793 ms | 10.291 ms | 1.340x |

These are GPU-event operator timings, including output allocation and bias
packing but excluding score reset/QK/PV. Two warmups precede five alternating
observations; the allocator cache stays warm. All GCDs were observed idle before
the screens, but isolation was not continuously monitored. Peak extra allocation
falls by exactly the score storage size. This is not a whole-model peak or E2E
claim. Evidence: `provenance/openfold3_hip_softmax_inplace_strict_screen.json`.
Global determinism and memory filling remain enabled.

The helper requires strict inference, full-score storage and disjoint bias
storage, falling back to the maintained allocation for unsafe storage. The caller
must guarantee private score ownership; these checks alone cannot prove it.
The audited attention adapter creates QK scores locally and consumes them only
through softmax/PV. Three CPU tests reject partial storage and aliases.

Complete-model numerical checks are active in a new frozen copied worker at
`/tmp/fireprot-inplace-model-v2`, over the already validated atom/pair caches.
The candidate records actual submitted softmax shapes, storage fallbacks and
final module chunk-tuner values without overriding upstream tuning. The first
v1 control failed execution preflight before worker creation because its copy
omitted the root import shim and native source tree; its failed run IDs remain
reserved. Source/archive pins and the exclusion are recorded in
`provenance/openfold3_inplace_model_preparation.json`.
No maintained dispatch changes or new E2E result are claimed yet.

N1000 has now completed and matches all 11 original-native arrays. It submitted
1,702 in-place calls with zero storage fallbacks; final recorded module chunk
tuners are all 1024. N1536 remains active. See
`provenance/openfold3_inplace_models.json` for the current gate state.

A separate same-source AB/BA/AB N1536 comparison against original Torch is
prepared, **not launched**, at `/tmp/fireprot-combined-native-paired-v1`.
The native arm disables pair/atom caches and calls upstream attention; the
candidate enables both and uses the owned-score HIP softmax. The worker delta
from the accepted atom-cache control is `worker-owned-scores-overlay.patch`;
copy `owned_scores.py` into the worker package too. Profiles explicitly bind the
cache flags, backend difference and Rust bridge identity. All scientific settings
stay FP32/R3/D200/S1/seed101. Evidence and archive:
`provenance/openfold3_combined_native_paired_preparation.json`.
Launch only after both in-place model gates pass and all diagnostic GPU jobs
terminate. Keep all other FireProt GPU workloads stopped during that sequential
benchmark. Do not multiply earlier speedup ratios to predict its result.

## N1536 rejection and confidence-stage diagnostic

The all-scope N1536 model completed but **fails the required bitwise gate**.
Six of eleven arrays, including coordinates, match. Five confidence arrays differ:
chain_ptm/ptm up to 1.49e-8, PAE up to 1.7166e-5, PDE up to 1.3352e-5 and
pLDDT up to 1.9073e-5. This is rejected regardless of the small deltas.
The runtime's finite-output acceptance is not an exact-parity pass.
Final recorded chunk tuners are 512 except diffusion conditioning at 1024;
5,502 in-place launches have zero storage fallbacks. The earlier baseline
submitted 5,592 HIP launches, but that alone cannot distinguish tuning attempts
from steady-state changes. Allocation/tuning is a hypothesis, not a proven cause.
Completed outputs, controls and comparisons are archived in the model evidence.
**Do not launch the prepared combined-native benchmark with this candidate.**

`scope_softmax.py` prepares a diagnostic boundary around the native `aux_heads`
forward. Mode `none` preserves out-of-place HIP everywhere; mode `trunk` uses
in-place HIP only outside that confidence scope. Both retain atom/pair caches,
unchanged arithmetic and upstream tuning. The scope restores on return/error.
Fresh concurrent trials: N1536/none on HIP5, N1536/trunk on HIP6, N1000/trunk on
HIP7. They record final tuner values and per-scope submitted shapes. This
localizes the failure before deciding whether to preserve confidence allocation
or fix a more specific cause. Source/request/archive pins:
`provenance/openfold3_inplace_scope_preparation.json`.
No new model speedup or promotion is established.

## Large submitted shapes

The strict screen now accepts `--large` for N1536/B256 and B512. These sizes
occurred in the rejected complete-model invocation but were absent from the
initial B32/B128 screen. All 16 comparisons pass bitwise across both bias layouts
and zero/partial/nonzero/special masks, with zero storage fallbacks. B512 uses an
18 GiB score allocation. The screen ran on HIP0 concurrently with model numerical
diagnostics; its timing samples are not performance evidence. See
`provenance/openfold3_inplace_large_screen.json` and the archived sources/results.
This closes these synthetic shape checks, not the full-model confidence gate.

Source inspection confirms that the upstream Pairformer tuner probes cloned
single/pair inputs, catches RuntimeError as an unsuccessful size trial, and sets
non-flash attention chunks to `max(1, tuned_chunk_size // 4)`. The bounded trace
and current recorded final sizes do not establish the earlier baseline's
confidence chunk size. The live `none` diagnostic records that missing value.
Audited source hashes: `core/utils/chunk_utils.py`
`4c9ab484854868e3c3d9a28a0d2e8fe0f9b1c75a35dca4984e0631591489fac5`;
`core/model/latent/pairformer.py`
`cdb66c9553ef42b93835dec92f3e6dcf711dbf139a5a59aa1f2561216435af9f`.
A possible next experiment is preserving the measured baseline's confidence
chunk size while keeping in-place softmax; do not select a cap before the baseline
measurement or claim this mechanism is established.

## Confidence scope results and measured chunk ceiling

All three scope diagnostics completed and pass every original-native output
array: N1536 `none`, N1536 `trunk`, and N1000 `trunk`. N1536's out-of-place
baseline and guarded candidate both select confidence chunk **256**, versus
**512** in the rejected all-in-place model. All other N1536 final tuner sizes
match (512, except diffusion conditioning 1024). N1000 tuners remain 1024.
These are measured differences; a controlled chunk experiment is still needed
to attribute the numerical change. Evidence and archive:
`provenance/openfold3_inplace_scope_models.json`.

`confidence_chunk.py` now prepares that experiment. It keeps in-place softmax
for all eligible calls, including confidence, while bounding the fresh native
confidence tuner to the observed 256 ceiling only for N1536 FP32 inputs. N1000
and unmatched precision/length retain the original maximum. Native tuning still
runs and may choose a smaller value; the helper records requested/bounded/selected
sizes, verifies both upstream source hashes, rejects an already-used tuner and
restores the original method on exit. Three CPU tests check shape/precision scope,
native callback execution, cleanup, fresh-state and source rejection.

Fresh complete-model checks are active under `/tmp/fireprot-inplace-chunk-v1`,
N1000/HIP5 and N1536/HIP6, with unchanged R3/D200/S1/seed101/FP32 and the existing
pair/atom caches. Require all 11 arrays before acceptance. A new direct native
versus combined AB/BA/AB E2E spec is prepared at
`/tmp/fireprot-combined-native-paired-v2` but MUST NOT run before those gates and
GPU-idle checks pass. The old v1 E2E spec remains rejected. See
`provenance/openfold3_inplace_chunk_preparation.json` and
`provenance/openfold3_combined_native_paired_v2_preparation.json`.
No controlled performance gain from atom reuse or in-place softmax is claimed yet.
