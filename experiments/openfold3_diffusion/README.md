# Steady diffusion-call diagnostic

Status: diagnostic completed with exact whole-model outputs; global pair-bias reuse is under validation.
Historical post-normalization profiling recorded about44.35 s in the complete
sampling module, but its operator trace covered the trunk. This diagnostic
collects the missing steady denoiser detail on the qualified isolated-data
baseline; the later accepted unroll4 fusion was not enabled for this diagnostic.

`probe.py` wraps `model.sample_diffusion.diffusion_module.forward` only within an
explicit context. The upstream source digest must match the audited diffusion
module. All200 denoiser calls retain the qualified baseline inputs and arithmetic. CUDA/HIP
events measure each call; only call5, after four normal calls, receives Torch
CPU/GPU profiling with input shapes. Immediate child modules receive named ranges
and shape/stride/dtype metadata for that call. No tensor values are copied out
for metadata. Instance methods are restored on normal exit or failure.

Outputs are `diffusion-profile.json` and `diffusion-call5.chrome.json`. The
profile includes operator self/inclusive CPU/device statistics, seven immediate
child ranges and metadata, per-call events and the trace digest. The whole-model
run must still match all11 arrays and full CIF bytes; unchanged runner settings,
normalization/gate counts and200 denoiser calls are checked independently.

The trace excludes outer random augmentation, noise generation and scheduler
arithmetic. Its timings contain instrumentation overhead, nested times overlap,
and shapes alone do not prove CU saturation or a memory/computation bound.
Neither this run's invocation time nor its model time is an E2E speedup result.
It does not enable graph capture, lower precision or change diffusion steps.

DiamondHill coordinator:
`outputs/experiments/openfold3-diffusion-probe-queue-20261010/run_after_unroll8.py`.
It waits for the existing unroll8 replay to finish successfully, verifies the
selected physical GPU is idle, prepares a new frozen diagnostic root, and uses
Rust `measure-inference` for environment checks, execution and acceptance.
Physical scope remains HIP7/PCI0000:93:00.0/SMI5. No profiler overlaps paired E2E
or the queued attention replay. The model runs from
`outputs/experiments/openfold3-diffusion-profile-20261010` once prepared; output
is `outputs/openfold3-diffusion-profile-n1536-20261010`.


## Measured steady-call result

All200 denoiser calls completed and the whole model matches all11 arrays and
full CIF bytes. Call5 is profiled; the other199 event timings have a median of
218.334 ms (they still include event instrumentation). N1536, strict FP32,
R3/four trunk passes, D200/S1, seed101, HIP7, unflushed caches.

`analyze_trace.py TRACE PROFILE OUTPUT` attributes each kernel's duration to its
CPU operation and enclosing immediate child range through the external ID.
It excludes GPU annotation durations from sums to avoid double counting.
One HIP normalization launch lacks an external ID and remains unassigned
(1.796 ms). All2874 sampled kernels are accounted for.

| Sampled region / operation | Kernel duration sum |
| --- | ---: |
| Global diffusion transformer | 166.156 ms |
| Atom encoder | 27.316 ms |
| Atom decoder | 21.635 ms |
| Global projected bias addition, 24 calls | 63.052 ms |
| Global pair projection GEMM `[2359296,128] x [128,16]`, 24 calls | 39.286 ms |

The last two rows are included in the global-transformer total. These are sums
of sampled kernel execution durations, not complete stage latency or E2E time.
[Diagnostic qualification and analysis](../../provenance/openfold3_diffusion_profile_20261010.json).

## Global bias reuse experiment

`bias_cache.py` scopes reuse to a single `_sample_rollout` and restores every
instance method on success or failure. It wraps only the global transformer's
pair normalization and each of its24 `linear_z` projections. Time-dependent
activations, single conditioning, masks, attention and atom stacks retain their
existing execution. Values keep their original layouts. Upstream source hashes,
24 blocks, strict inference and N1536 FP32 tensor shapes are required.

The prior qualified pair cache returns an invariant pair embedding across the
rollout. Native transformer code normalizes it once per call, and each block
projects the result; those operations depend on the pair and fixed parameters.
Identity/shape/stride/parameter guards invalidate on replacements and retained
owners prevent pointer reuse. They do not detect in-place inference mutations:
this is an explicitly source-pinned experiment, not a general-purpose cache.

Verification mode recomputes every hit with native operations, checks all bytes,
and compares retained results again after each denoiser's consumers. Expected
N1536 counts are200 calls,25 misses,4975 hits,4975 native comparisons and5000
post-consumer comparisons. The timing version has no redundant computations or
comparisons and requires a separate full-model all11/fullCIF gate. Retained
result tensors total4,831,838,208 bytes (4.5 GiB), excluding diagnostic reference
copies; this is not measured peak-memory overhead. Repeated-rollout invalidation,
input replacement, consumer mutation and exception cleanup pass six CPU tests;
those tests do not replace GPU qualification.

Coordinator `outputs/experiments/openfold3-global-bias-followup-20261010/driver.py`
runs recomputation qualification before compute-skipping qualification. Both
start from the qualified unroll4 composition on HIP7. No E2E ratio exists yet.
The projected bias is currently a permuted view after `linear_z`; packing it
for the later addition is a separate candidate, requiring separate bitwise and
performance checks. No layout optimization is enabled by this cache experiment.


## Global reuse verification and packing follow-up

The first N1536 diagnostic passes all200 steps:25 initial entries,4975 native
recomputation comparisons and5000 checks after denoiser consumers. All11 model
arrays and complete CIF bytes equal the qualified unroll4 anchor. No fallback
or invalidation occurred. The native output layouts were preserved. This run
contains redundant work and cannot establish a speedup.
[Verification evidence](../../provenance/openfold3_global_bias_verified_n1536.json).
The compute-skipping model gate is running under the original frozen source.

`pack_bias=True` is a separate opt-in candidate. At each projection cache miss,
it copies the native `[...,N,N,H]` result into head-first storage and restores
the logical shape as a view. Upstream's subsequent permutation then produces a
contiguous `[...,H,N,N]` bias. No floating-point arithmetic is introduced, and
the normalized pair retains its original layout. The cache size remains4.5GiB.
This targets the sampled strided bias addition; its actual benefit requires
measurement. Default `pack_bias=False` preserves the previous experiment.

`benchmark_bias_layout.py OUTPUT` screens random, masked-score, cancellation
and special-bit cases on GPU, with112 exact checks. It times only the in-place
bias addition (two warmups/twelve alternating samples per arm/case); packing,
score reset and checks are outside timing. It does not model complete attention
or establish E2E improvement. Nine CPU tests now pass, including packed cache
values/strides and bit preservation for signed zero, infinities and a NaN payload.

The serial coordinator
`outputs/experiments/openfold3-packed-bias-followup-20261010/run_after_cache.py`
waits for both original-layout full-model qualifications, verifies the complete
environment and idle HIP7/PCI93/SMI5, screens the layout, and only then runs
packed recomputation and compute-skipping whole-model gates. It stops on a
failed prerequisite or a layout screen with no measured improvement. The synthetic GPU result is recorded below; no E2E claim is available yet.


## Original-layout cache gate and measured bias packing screen

The compute-skipping original-layout cache passes all11 arrays and full CIF
bytes, with200 denoiser calls,25 misses/4975 hits, no fallbacks/invalidations,
and no redundant native/consumer comparisons. Single qualification is
323.335369 s complete invocation /235.354797 s model forward. This is not a
paired speedup. [Cache qualification](../../provenance/openfold3_global_bias_cache_n1536.json).

The packing screen subsequently passes all112 GPU checks:

| Synthetic case | Original bias add median | Packed bias add median |
| --- | ---: | ---: |
| Random | 2.659921 ms | 0.375601 ms |
| Masked scores | 2.645360 ms | 0.373680 ms |
| Cancellation | 2.640801 ms | 0.375280 ms |
| Special float bits | 2.641120 ms | 0.374960 ms |

This approximately7x gain is **only the isolated in-place bias add**, with
packing and score reset outside timing. It supports the memory-layout hypothesis
without establishing a hardware memory/computation bound. Packed whole-model
recomputation and compute-skipping gates are still in progress.
[Raw screen evidence](../../provenance/openfold3_packed_bias_layout_20261010.json).

A separate serial coordinator waits for both packed model gates before preparing
three AB/BA/AB E2E pairs: qualified unroll4 versus global normalization/projection
reuse plus packing. Both arms share one frozen worker tree and scientific
settings; only the cache flag and profile digest differ. It checks all11 arrays,
complete CIF bytes, actual cache/normalization/QKV counts and current sidecar
hashes. Queue root: `outputs/experiments/openfold3-packed-bias-pairs-queue-20261010`;
benchmark root, once prepared: `outputs/experiments/openfold3-packed-bias-pairs-20261010`.
The completed paired result and promotion decision are recorded below.


## Packed model gates and next HIP softmax experiment

Both packed whole-model gates pass. The diagnostic recomputes4975 hits and
checks5000 retained results after consumers; all200 denoiser calls complete.
The compute-skipping run records25 misses/4975 hits and no redundant comparisons,
with all11 arrays and complete CIF bytes exact. Single complete invocation is
321.920080 s, model234.036924 s. These are qualifications, not matched ratios;
the approximately7x isolated-add result must not be presented as whole-model
speedup. Three AB/BA/AB pairs subsequently completed; see the decision below.
[Packed verification](../../provenance/openfold3_packed_bias_verified_n1536.json),
[packed compute-skipping gate](../../provenance/openfold3_packed_bias_cache_n1536.json).

Global diffusion Q/K/V have shape`[1,1,16,1536,48]`. Existing attention eligibility
covers four-dimensional H4 tensors; global diffusion therefore uses native
bias addition and softmax. The source-pinned native operation is FP32 einsum QK,
mask then pair-bias additions, softmax and einsum PV. The published HIP binary
already implements the required row arithmetic but its C ABI fixes four heads.

`global_softmax.py` is a new experimental dispatcher around the existing scoped
dispatcher. It preserves native QK/PV and strict FP32, slices16 heads into four
contiguous H4 groups, and uses the existing published binary with batch1,
waves2 and the same row order. One fresh output allocation holds all groups.
Biases are made contiguous only when necessary. Nonmatching tensor/device/mode
contracts delegate to the previous dispatcher. It excludes graph capture and
reports calls, submitted groups, copies, layout fallbacks and the binary receipt;
the shared kernel launch counter includes the grouped calls. Source and binary
validation remain in the existing attention scope and Rust artifact loader.
This adapted arithmetic retains Apache-2.0 attribution in its file and
[THIRD_PARTY.md](../../THIRD_PARTY.md).

`benchmark_global_softmax.py OUTPUT` compares seven synthetic probability
arrays and196 complete D48 attention results, including strided bias, masking,
uniform scores and an all-masked NaN case. Two warmups/twelve alternating
samples per arm/case retain all observations. Only a correct screen with at
least1% improvement in both packed and partially masked cases advances to a
full N1536 model gate. Expected model counts are4800 attention/probability calls,
19200 grouped launches, no bias copies or score-layout fallbacks, unchanged
cache/norm/QKV behavior, all11 arrays and full CIF bytes exact.

Two CPU fallback-contract tests pass. The first dependent screen did not run after its E2E prerequisite failed.
An independent screen subsequently completed, as recorded below. The serial coordinator is
`outputs/experiments/openfold3-global-softmax-screen-20261010/run_after_pairs.py`.
It first waits for the packed-bias E2E collector, requires exact outputs and
improvement in every pair, then validates the complete environment, idle physical
HIP7/PCI93/SMI5 and source pins before screening. A passed screen precedes
`outputs/experiments/openfold3-global-softmax-model-20261010`; no new build or
GPU task overlaps the current E2E measurements.


## Packed bias E2E decision and independent global softmax screen

The strict N1536/R3/D200/S1 seed101 experiment completed three AB/BA/AB
pairs on HIP7 (PCI93, SMI5), with fresh processes and unflushed caches.
All six outputs match all11 arrays bitwise and complete CIF bytes.

| Pair | Qualified unroll4 | Packed global bias cache | Ratio |
| --- | ---: | ---: | ---: |
| AB0 | 328.562508 s | 320.447457 s | 1.0253x |
| BA1 | 331.768884 s | 332.515964 s | 0.9978x |
| AB2 | 329.492405 s | 314.023158 s | 1.0493x |

Complete invocation medians are **329.492405 ->320.447457 s (1.0282x)**;
model medians are 242.326938 ->228.748256 s (1.0594x). The candidate's
p95/max regresses from331.768884 to332.515964 s. With only three samples,
p95 here equals the maximum. **Do not promote this cache composition** under
the predeclared all-pairs-faster gate. No observation is excluded and no cause
is asserted for the slower second candidate. Retain the qualified unroll4
composition. This is an incremental comparison, not a direct-native ratio.

The cache-dependent global-softmax queue therefore exited before any GPU
screen. A separate screen evaluated the grouped HIP softmax over the accepted
unroll4 path, including the actual uncached `linear_z` bias layout. Seven
probability checks and196 complete attention checks all match native bits.
Packed attention improves1.381991 ->1.110792 ms, but the actual interleaved
layout regresses3.651414 ->3.821254 ms (masked:3.653495 ->3.831414 ms).
**Reject this implementation for the native interleaved layout.** Its Torch
contiguous copy remains inside timing; the packed result alone is insufficient.
No global-softmax full-model job was launched by this screen.

The next experiment adds an explicit HIP bit transpose (`pack_bias.cpp`),
with64/128-row tiles and swizzled shared memory. The subclass
`packed_global_softmax.py` accepts only the exact N1536/H16 interleaved FP32
layout, preserves native QK/PV and delegates other layouts to the qualified
parent. Library hashes, same-device guards and graph exclusion remain enforced.
The screen includes allocation and packing in full-attention timing and checks
arbitrary bits, including NaN payloads. It does not establish E2E speedup.

The completed diffusion experiment family is archived remotely as
`outputs/experiments/openfold3-diffusion-family-20261010.tar.gz`:
1,334,486,133 bytes,33 roots, SHA256
`84f57c3b43bf2675029e2f0f0ad127173cb9aae145282af611a11f08254a79e8`.
It precedes the independent softmax screen, which is outside that archive.
Weights and external data remain separate dependencies.

Evidence: [paired E2E](../../provenance/openfold3_packed_bias_e2e_n1536.json),
[softmax screen](../../provenance/openfold3_global_softmax_screen_20261010.json),
[archive](../../provenance/openfold3_diffusion_family_archive_20261010.json).


### HIP packing screen

The64-row HIP transpose changes complete attention medians for the actual
interleaved bias from **3.650305 to1.469834 ms (2.4835x)**, including the
fresh output allocation and packing. The masked case is3.643425 to1.467114 ms.
All seven probability cases,196 complete attention results and the separate
arbitrary-bit transpose check pass exactly. The new kernel library SHA256 is
`ed9a81589fb906cba2f572d13ae5895e04b203a13fe347fb70f4f95547ff8efd`.
The64-row candidate advances to strict full-model qualification over unroll4,
without global bias caching. This is a synthetic attention result, not E2E.
[Screen receipt](../../provenance/openfold3_hip_bias_pack_screen_20261010.json).

Three host guard tests reject tampered binaries, CPU operands and unsupported
tiles before native launch. Together with the existing softmax and direct-native
router tests, ten CPU tests pass. GPU qualification is separate from these tests.


### HIP packing full-model qualification

The strict N1536 R3/four-trunk-pass D200/S1 seed101 full model passes all11
arrays bitwise and full CIF bytes against the qualified unroll4 anchor. It
executes4800 global attention/probability calls,4800 HIP bias copies and19200
published softmax groups, with zero Torch bias copies or score-layout fallbacks.
The existing5064 fused triangle attention calls and normalization counts remain
unchanged. No global bias cache is enabled.

Single complete invocation is **317.785213 s**, model forward **231.075384 s**.
This is a numerical qualification with timing, not a paired speedup. Three
AB/BA/AB pairs subsequently completed against the unroll4 composition, with one shared
worker tree and only `FIREPROT_GLOBAL_SOFTMAX` plus the profile digest changing.
Both arms include the same packer library validation and runtime receipts.
New softmax, QKV and gate sidecars are included in the completed transport
artifact hashes. All11 arrays and complete CIF bytes are required on every run.
[Model receipt](../../provenance/openfold3_hip_bias_pack_model_n1536.json).

Paired coordinator: `outputs/experiments/openfold3-hip-bias-pack-pairs-queue-20261010`.
Paired benchmark: `outputs/experiments/openfold3-hip-bias-pack-pairs-20261010`.
The recovered128-row packing receipt also confirms arbitrary-bit equality;
its actual-layout attention medians1.471994/1.472234 ms provide no meaningful
advantage over64 rows. Both raw variant summaries are retained in the screen
receipt. Maintained-profile promotion remains pending; the later direct result is below.


## Direct-native follow-up preparation

The direct-native router now has explicit preparation, full-model qualification
and paired collection scripts in this directory. The follow-up waits for all
three HIP-packing incremental pairs to pass and improve before selecting that
candidate. No GPU jobs overlap. Both model routes must then match all 11 arrays
and full CIF bytes before a fresh direct-native three-pair comparison starts.

Native worker source and data defaults are preserved; all optimized selection
flags are zero in the native profile. One common worker tree prevents source
layout differences from being hidden as protected-input changes. Route/runner
and optimization receipts are sealed with outputs, and the collector checks their
current bytes. Twelve CPU tests pass. Both routed GPU qualifications and the
three-pair direct comparison now pass (2.8644x E2E). See the [workflow and evidence boundary](../../docs/openfold3-deployment-audit.md#frozen-direct-native-comparison-workflow).


## Completed hardware diagnostics

`hardware_profile.py trace` and `hardware_profile.py counters` prepare two
separate diagnostic profiles around the exact HIP-packing model worker. The
serial coordinator waits for the direct-native study to succeed and verifies
that physical SMI5 is idle before each run. Requests retain HIP7, strict FP32,
N1536, R3/four trunk passes, D200/S1 and seed101. Neither the existing E2E profiles
nor their scientific settings are modified.

The script obtains the installed rocprofv3 injection environment, applies it to
the Rust-supervised worker profile, and pins profiler libraries/configuration.
The ordinary full environment inventory remains present. The trace collects all
kernel dispatches. The counter run samples kernel occurrences10–12 for QKV,
GEMM, normalization, softmax, packing, gate and copy families, requesting
`GRBM_COUNT`, `GRBM_GUI_ACTIVE`, `SQ_BUSY_CU_CYCLES`, `MeanOccupancyPerCU`,
`OccupancyPercent`, `MfmaUtil`, `VALUBusy`, `FetchSize` and `WriteSize`.

Both runs must preserve all11 arrays, full CIF bytes and the qualified dispatch
receipts. Raw CSV files and their hashes are retained. Per-grid coverage and
metric completeness must be examined before any hardware conclusion. Counter
sampling is not a complete model timeline, and instrumented times are not E2E
performance measurements. Both diagnostic runs now pass all11/full-CIF gates.
[Per-grid hardware analysis](../../docs/openfold3-hardware.md) identifies fused
QKV as the next scheduling target; sampled coverage is explicitly bounded.

Coordinator: `outputs/experiments/openfold3-hardware-followup-20261010`.
The diagnostic jobs completed after all direct-native timed calls; no profiling
overlapped the timing study.112 counter dispatches have all nine metrics.


## HIP packing plus grouped softmax: accepted incremental E2E

Three AB/BA/AB pairs compare the qualified unroll4 composition with the same
composition plus HIP interleaved-bias packing and grouped global softmax. They
use strict deterministic FP32, N1536, R3/four trunk passes, D200/S1, seed101,
HIP7/PCI93/SMI5, fresh processes and unflushed caches. Both arms share one frozen
worker tree; only `FIREPROT_GLOBAL_SOFTMAX` and the profile digest differ. Global
bias caching remains disabled.

| Pair | Unroll4 baseline | HIP packing + softmax | Incremental speedup |
| --- | ---: | ---: | ---: |
| AB0 | 327.198516 s | 320.345378 s | 1.0214x |
| BA1 | 331.932364 s | 323.498000 s | 1.0261x |
| AB2 | 330.160733 s | 322.417190 s | 1.0240x |

Complete invocation medians are **330.160733 ->322.417190 s (1.0240x)**,
a **2.35% time reduction**. Model medians are **242.935593 ->233.159512 s
(1.0419x)**. p95/max also improves,331.932364 ->323.498000 s; with three samples,
the reported p95 equals the maximum. No observation is discarded.

All six outputs match all11 arrays bitwise and complete CIF bytes. Each candidate
executes4800 global attention calls,4800 HIP packing calls and19200 grouped HIP
softmax launches, with no Torch bias copies or layout fallbacks. Each baseline
reports zero global HIP calls. Existing normalization, gate, template, isolated
data and5064 fused QKV calls retain their qualified behavior.

**Retain HIP packing plus grouped global softmax in the experimental composition.**
The narrower synthetic attention gain2.48x does not describe model or E2E speedup.
The earlier packed projection-cache proposal remains unselected because its
paired stability gate failed. Maintained-profile integration is still pending.

The direct-native follow-up subsequently passed both routed model gates and all
three fresh pairs:950.891469 ->331.969765 s E2E medians (2.8644x). Its native worker
SHA remains `c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e`;
the optimized worker body is copied unchanged from the qualified paired tree.
This direct result is independent of the incremental ratio above.

[Complete paired evidence](../../provenance/openfold3_hip_bias_pack_e2e_n1536.json).


### Reproduce the accepted experimental composition

On DiamondHill, after the active benchmark queue has finished, copy the qualified
request and give it a fresh `run_id` and unused `output_dir`. Keep its input,
checkpoint, strict workload, seed and `hip_device: 7` for this reproduction:

```bash
cp /media/IntelSSD/fireprot/outputs/experiments/openfold3-hip-bias-pack-pairs-20261010/hip-bias-pack-pairs-n1536-candidate-0.json /tmp/my-openfold3-hip-pack.json
# Edit run_id and output_dir in the copied request.
fireprot_cli=/media/IntelSSD/fireprot/runtime/recovered-openfold3-20261007/fireprot-triangle-e2e-v1/target/release/fireprot
"$fireprot_cli" inspect /tmp/my-openfold3-hip-pack.json
"$fireprot_cli" infer /tmp/my-openfold3-hip-pack.json /media/IntelSSD/fireprot/outputs/experiments/openfold3-hip-bias-pack-pairs-20261010/candidate-profile.json
```

This uses the frozen worker and kernel dependencies recorded in the paired
preparation receipt. It is not a portable installer or the maintained profile's
default. The input/checkpoint/cache resources remain on DiamondHill; the
preparation and source pins must remain valid. A fresh inference alone does not
reproduce the three-pair performance study; that study additionally requires
the pinned AB/BA/AB specification, isolated GPU, numerical/CIF gates and collector.


## Routed native and optimized model gates completed

Both shared-tree routes pass the strict N1536 full-model gate: all 11 arrays,
complete CIF bytes, actual route/source identity, and sealed execution receipts.
Native data-module defaults are preserved and native outputs contain no optimized
operator receipts. Qualification E2E observations are966.949545 s native and
321.596325 s optimized; model forward is872.441172 and233.799366 s. These are
single model-gate observations, not completed three-pair timing results.

The direct-native coordinator completed all three AB/BA/AB pairs:950.891469 ->
331.969765 s E2E medians (2.8644x), all six all11/full-CIF exact. Subsequent
hardware diagnostics also passed their exact gates. [Qualification receipt](../../provenance/openfold3_direct_native_routes_n1536.json).

## Narrow global projection experiment

The earlier diffusion-call trace identifies 24 `linear_z` calls of
`[2359296,128] @ [128,16]` per denoiser call. `narrow_projection.cpp` is an
experimental handwritten MFMA implementation with cooperative input staging.
It can also write the head-major layout directly, avoiding the subsequent
separate HIP bias transpose. It is not installed in maintained dispatch.

The initial two-elements-per-lane reduction order matches the smaller synthetic
matrices, but fails the actual large shape: 30,921,559 output elements differ in
each layout. That version is rejected before timing. The
`narrow_projection_k4.patch` changes the MFMA lane grouping to four K elements
and restricts its C ABI to exactly 2,359,296 rows. This demonstrates why a small
matrix gate cannot qualify the model's GEMM shape.

The K4 variant passes eight full-size cases in both output layouts, including
cancellation, subnormal, signed-zero and nonfinite inputs. All output canaries
remain intact. A separate balanced timing sequence includes strict output
allocation, two warmups and 16 samples for each of four arms; every result is
checked bitwise between samples. FP32, seed101, HIP7, unflushed caches and the
same refreshed full environment inventory before/after apply.

| Synthetic projection path | Median |
| --- | ---: |
| Original Torch linear | 1.664797 ms |
| Original Torch linear + qualified HIP packing | 2.019596 ms |
| HIP K4, ordinary output layout | 1.517517 ms |
| HIP K4, direct head-major output | 1.655597 ms |

Direct head-major output is 1.2199x faster than linear plus HIP packing in this
screen. This is not a trained-module or E2E speedup. The full-model follow-up
uses `narrow_scope.py`, validates upstream source/library hashes, and restricts
selection to the 24 global projections with exact dtype/shape/layout/inference
guards. Other inputs call the original method. Verification mode compares the
first invocation of every trained projection against the original module;
both verification and compute-only runs must then pass all 11 arrays/full CIF.

Only after those two model gates pass may the queued three-pair comparison run.
Both arms use a shared frozen worker tree and the refreshed full environment;
only `FIREPROT_NARROW_PROJECTION` and the profile digest differ. Verification and
profiling are disabled during timing. No observations may be discarded.

- [Initial numerical rejection](../../provenance/openfold3_narrow_projection_initial_20261010.json).
- [K4 synthetic gates and complete raw timings](../../provenance/openfold3_narrow_projection_k4_screen_20261010.json).
- [Complete model qualification](../../provenance/openfold3_narrow_projection_models_20261010.json).
- Current model root: `outputs/experiments/openfold3-narrow-model-recovery2-20261010`.
- Current model coordinator: `outputs/experiments/openfold3-narrow-model-recovery2-followup-20261010`.
- Current paired coordinator: `outputs/experiments/openfold3-narrow-pairs-recovery2-queue-20261010`.
- Current paired root: `outputs/experiments/openfold3-narrow-pairs-recovery2-20261010`.

The initial model attempt selected zero calls because its guard rejected the
upstream FP32-target autocast context. The corrected guard admits FP32 autocast
while retaining FP16/BF16 fallback. Actual sealed metadata confirms FP32 inputs,
weights and autocast target. The first recovery passed trained/full-output checks
but its collector rejected an unsealed runner receipt; that failed audit is
retained. Recovery2 seals runner and operator receipts and passes both modes:
4,800 projection calls, zero fallbacks, 24 native checks in verification mode,
all eleven arrays bitwise and full CIF bytes equal. The redundant 4,800 HIP bias
copies disappear, while 4,800 global-softmax calls and 19,200 groups remain.
The completed three-pair E2E comparison rejects promotion: baseline median
322.474732 s versus candidate 324.589092 s, **0.993486x** (0.66% more time).
All three paired ratios are below one: 0.996547x, 0.987414x and 0.993146x.
Model-forward medians likewise regress from 231.829614 to 234.538317 s.
All six outputs retain all eleven arrays bitwise and full CIF bytes exactly;
numerical correctness does not establish a performance improvement. Retain the
previous qualified HIP-packed composition. [Full paired evidence](../../provenance/openfold3_narrow_projection_e2e_20261015.json).
