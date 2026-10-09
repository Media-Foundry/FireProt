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
No new E2E result exists yet.


## Packed model gates and next HIP softmax experiment

Both packed whole-model gates pass. The diagnostic recomputes4975 hits and
checks5000 retained results after consumers; all200 denoiser calls complete.
The compute-skipping run records25 misses/4975 hits and no redundant comparisons,
with all11 arrays and complete CIF bytes exact. Single complete invocation is
321.920080 s, model234.036924 s. These are qualifications, not matched ratios;
the approximately7x isolated-add result must not be presented as whole-model
speedup. Three AB/BA/AB pairs are now running.
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

`benchmark_global_softmax.py OUTPUT` will compare five synthetic probability
arrays and140 complete D48 attention results, including strided bias, masking,
uniform scores and an all-masked NaN case. Two warmups/twelve alternating
samples per arm/case retain all observations. Only a correct screen with at
least1% improvement in both packed and partially masked cases advances to a
full N1536 model gate. Expected model counts are4800 attention/probability calls,
19200 grouped launches, no bias copies or score-layout fallbacks, unchanged
cache/norm/QKV behavior, all11 arrays and full CIF bytes exact.

Two CPU fallback-contract tests pass. **No global-softmax GPU or E2E result is
available yet.** The serial coordinator is
`outputs/experiments/openfold3-global-softmax-screen-20261010/run_after_pairs.py`.
It first waits for the packed-bias E2E collector, requires exact outputs and
improvement in every pair, then validates the complete environment, idle physical
HIP7/PCI93/SMI5 and source pins before screening. A passed screen precedes
`outputs/experiments/openfold3-global-softmax-model-20261010`; no new build or
GPU task overlaps the current E2E measurements.
