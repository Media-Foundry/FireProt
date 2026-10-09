# OpenFold3 / OpenBind-0 AMD performance work

## Latest accepted result and final-cycle decision (2026-10-06)

Three complete N1536 pairs accept the triangle-memory combination at
**452.471 → 416.226 seconds E2E (1.0871x incremental)** and
**356.595 → 321.699 seconds model time (1.1085x)**. All six outputs match all
11 arrays and full CIF bytes against three anchors; N1000 fallback passes.
See [the complete comparison](#triangle-memory-accepted-incremental-e2e).

The requested one additional attention cycle is complete. Both alternatives
preserve exact raw scores/probabilities and trained attention outputs, but lose
complete-operator performance with the same accepted HIP K packing:

| Alternative | Ending attention, baseline → candidate | Starting attention, baseline → candidate | Decision |
| --- | ---: | ---: | --- |
| Native BMM into retained scores + owned HIP softmax | 276.495 → 306.176 ms | 266.272 → 296.169 ms | Reject; about 11% slower |
| Double-LDS fused QK/softmax | 276.268 → 279.788 ms | 266.095 → 269.559 ms | Reject; about 1.3% slower |

Each passes 21 numerical cases and real trained start/end replay, with six
measured calls per arm after first-call exclusion. The double-buffer variant
reduces barriers from11 to6 and doubles LDS16 to32KiB without spills, but that
static improvement does not translate to lower latency. Its first publication
attempt was correctly rejected after an inspection command rewrote the
unpublished experimental ELF; a fresh isolated build and separate inspection
output resolved the tooling error without changing arithmetic or acceptance.

Neither rejected candidate proceeds to model promotion or E2E measurement.
Evidence: `provenance/openfold3_attention_final_split_result.json` and
`provenance/openfold3_attention_final_double_result.json`. The accepted
triangle-memory combination is retained. This closes the requested cycle,
not a claim of a global performance optimum. The historical development
sequence below keeps its original timing boundaries.

## Workspace-era whole-model bottlenecks (2026-10-06)

The latest N1536 workspace candidate passes a coarse stage-event diagnostic
under strict FP32/R3/D200/S1/seed101. All 11 output arrays and full CIF bytes
match original native, accepted D16 and latest-workspace references.

| Inclusive GPU stage | Seconds | Calls |
| --- | ---: | ---: |
| Trunk, including the following three sub-stages | 295.922 | 1 |
| Pairformer stack | 221.906 | 4 |
| Template embedding | 49.194 | 4 |
| MSA module | 20.929 | 4 |
| Diffusion sampling | 46.514 | 1 |
| Confidence heads | 7.658 | 1 |

Trunk accounts for approximately 85% of trunk plus rollout time; its Pairformer
stack alone accounts for approximately 63%. These nested, instrumented intervals
must not be added indiscriminately or interpreted as new E2E results. The next
diagnostic splits Pairformer triangle multiplication, attention and transitions,
and separately observes the template stack. Lack of template search does not
justify omitting the template network: its source consumes the current pair
representation and returns a learned update.

The preceding direct three-pair E2E attempt stopped during its fourth invocation
because the host library inventory changed. A fresh full inventory identifies
only additions of libiperf/libsctp and a changed loader cache; no inventoried
Torch or model files changed. The diagnostic uses the new pinned inventory and
retains full numerical acceptance. The failed E2E run remains rejected.
See [stage evidence](../provenance/openfold3_latest_stage_profile.json).

The finer diagnostic also passes all 11 arrays and full CIF bytes against the
three references. Its 228.209-second Pairformer interval breaks down as follows:

| Pairformer operation | Inclusive seconds |
| --- | ---: |
| Triangle attention, starting + ending | 110.790 |
| Triangle multiplication, outgoing + incoming | 97.205 |
| Pair transition | 13.798 |
| Single attention with pair bias | 3.556 |
| Single transition | 0.059 |

Each leaf records 193 calls, including upstream tuning. Residual work and
instrumentation prevent these rows from being an exact partition of the parent.
The next targeted diagnostic captures one trained outgoing and incoming
multiplication, including actual inputs, masks, weights and bounded Torch
operator attribution. This prioritizes the remaining native multiplication cost
alongside the already optimized attention path; no new kernel gain is claimed.
See [fine-grained evidence](../provenance/openfold3_latest_pairformer_profile.json).

## Workspace increment before triangle copies (2026-10-06)

Three isolated N1536 AB/BA/AB pairs compare the accepted D16/cache combination
against qualified transposed-mask normalization and one retained 4.5 GiB HIP
QK/softmax workspace. Both arms use the same frozen worker and Rust binaries,
strict FP32 with memory filling, R3/four trunk passes, D200, S1 and seed101.
Diagnostics are disabled; filesystem caches are not flushed.

| Scope | Previous combination median | Workspace combination median | Speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 540.302 s | 449.723 s | 1.20141x |
| Model forward | 445.333 s | 355.204 s | 1.25374x |

Complete invocation time decreases by 16.76%. All six outputs match both the
original-native and accepted-D16 anchors in all 11 arrays bitwise and complete
CIF bytes. Every run retains the same 51,757,716,992-byte peak Torch allocation.
The ordinary allocator is used without explicit cache release. Complete runtime,
lease, cache and reference checks are in
[`openfold3_workspace_mask_e2e_n1536.json`](../provenance/openfold3_workspace_mask_e2e_n1536.json).

This is an incremental result, not the latest direct original-Torch speedup.
The first subsequent direct comparison stopped because its treatment declaration
omitted the native-versus-HIP difference in Rust bridge loading. Correcting that
declaration in a separate retrospective audit passes the existing pair's Rust
comparison and all 11 arrays; the failed benchmark remains unchanged. A fresh
three-pair comparison must complete before publishing an updated direct ratio.
Do not multiply this increment with historical benchmark ratios.

## Post-migration experiment (2026-10-05)

Rust migration acceptance is complete. The subsequent
[ROCm coverage and zero-mask experiment](../experiments/rocm_coverage/README.md)
preserves strict FP32/R3/D200/S1/seed101 and passes all 11 output arrays exactly
at N1000 and N1536. Verified-zero-mask elimination improves three-pair N1000
E2E arm medians from 332.802 to 305.387 seconds (1.090x). The single N1536 pair
improves 952.871 to 880.036 seconds (1.083x), with insufficient repeats.
These shared-host measurements use an experimental worker overlay; no accelerated
OpenFold3 production route is promoted. Detailed controls and limitations are in
`provenance/openfold3_zero_mask_experiment.json`. The sections below retain the
historical pre-migration baseline and first operator screen.

## Scope and current status

OpenFold3 is a separate implementation from Protenix. Existing FireProt
Protenix results do not establish OpenFold3 correctness or performance.
OpenFold2 work is deferred at the user's request.

The initial target is the published OpenBind-0 checkpoint, using the existing
clean OpenFold3 checkout at commit `7de748b7bc93adb5af0a4032d5e9f208b6e1325f`.
No trained adapter is loaded. Checkpoint:
`/media/PM982/engramfold/folding_e2e_20260921/weights/openbind0/of3-ob-2025-06-30-174k.pt`.
The upstream repository documents this checkpoint family as compatible with
OpenFold3 releases starting at 0.5.0. Runtime loading and long-input results
must still be validated for this exact checkout and environment.

Environment: `/media/PM982/engramfold/folding_e2e_20260921/venvs/openfold3-amd/bin/python`,
Torch `2.12.0a0+git78258b9`, HIP `7.14.60850`, gfx90a on DiamondHill.
The upstream entry point explicitly selects hipBLAS on ROCm. This differs
from the observed native AtlasFold hipBLASLt preference.

## Baseline protocol

`benchmarks/openfold3/profile_inference.py` wraps the upstream CLI, retaining
native feature preparation, model loading, sampling, and output writing.
It verifies the source commit and clean tracked files, hashes the complete
checkpoint, resolves HIP to SMI through PCI identity, and continuously checks
selected-device isolation. It rejects missing predictions even when upstream
catches an inference exception and returns without output.

The N1000/N1536 inputs reuse the existing FireProt long protein sequences.
Both are single-chain development workloads with query-only MSA and no
template search, not an independent structure-quality set. Initial runs use
seed101, one diffusion sample, the native `predict` preset, and explicit
`32-true` trainer precision. The complete effective model configuration is
saved. Source defaults specify three recycles/four trunk passes and 200 full
rollout steps; the actual run configuration is authoritative.

Request timing excludes imports, checkpoint hashing, and GPU preflight;
it includes upstream setup/loading, feature preparation, inference, output
writing, and diagnostic raw coordinate/confidence capture. Filesystem caches
are warmed by hashing. Inclusive GPU stage events are optional; final latency
comparisons must disable them on both sides. Raw output capture converts BF16
values losslessly to FP32 for NPZ compatibility. Later comparisons must also
check original precision and configuration before claiming equality.

The baseline explicitly disables Triton, DeepSpeed and cuEquivariance routes.
Although the pinned source includes AMD Triton kernels, the current environment
reports Triton unavailable. No Triton speedup or compatibility claim follows.
The first native N1000 and N1536 runs completed; see the observations below.

## Optimization sequence

1. Complete both native runs and repeat native output checks. Record effective
   chunk sizes/tuning, model stages, output fields, peak memory, and cold versus
   warm cache conditions. Preserve native recycle and diffusion work.
2. Screen the existing upstream AMD path in an isolated environment if its
   dependencies can be established. Measure actual route selection and
   correctness; an enabled flag alone does not establish acceleration.
3. Target the measured long-input hotspots: Pairformer triangle attention and
   multiplication, MSA/outer-product updates, diffusion attention, and confidence
   computation. Reuse FireProt HIP/CK implementations only where contracts match.
4. Validate each operator candidate before full-model comparisons. Do not relax
   numerical gates after failures. Promote only measured end-to-end gains and
   continue until practical gains flatten or a material issue blocks progress.

No OpenFold3 optimization or completed performance result is claimed yet.


## First N1000 observation

The first instrumented native request completes in 253.375 s, with 227.676 s
inside the model and 38,636,892,160 bytes peak allocated memory. Inclusive GPU
stages include 145.103 s across four Pairformer calls, 13.195 s across four MSA
module calls, 38.977 s for diffusion, and 3.786 s for auxiliary heads. These
inclusive times must not all be added. The effective configuration retains
three recycles/four trunk passes and 200 diffusion steps with one sample.
All exported coordinates/confidence fields are finite and device isolation
passes; this does not yet establish repeated-native or optimized correctness.
Evidence: `provenance/openfold3_initial_n1000.json`.

The user has now prioritized a Rust framework rebuild immediately after the
current baseline and first operator-screening cycle. See `rust-migration.md`.
Do not start another OpenFold3 optimization cycle before that migration.

## First operator screen

Trained first Pairformer triangle weights, synthetic FP32 pair activations,
all-valid masks, chunk128, one warmup and three GPU-event repetitions:

| Length | Direction | Native | Contiguous-bias SDPA | Max absolute output delta |
| --- | --- | ---: | ---: | ---: |
| 1000 | Starting | 199.018 ms | 54.005 ms | 0.0008202 |
| 1000 | Ending | 276.605 ms | 54.699 ms | 0.0012627 |
| 1536 | Starting | 651.204 ms | 144.084 ms | 0.0013046 |
| 1536 | Ending | 1004.338 ms | 152.189 ms | 0.0014172 |

The native Q scaling is retained and SDPA uses scale1. The synthetic mask is
known all-valid, so the candidate only passes the pair bias. The first attempt
failed because native bias has a non-unit final stride; explicit contiguous
materialization is included in the subsequent timing. This candidate is not
bitwise exact and is not promoted. These are operator results, not E2E gains.

An isolated import of the existing DS Triton3.7.1 package preserved the original
Torch import but failed with zero active drivers. No Triton kernel ran and no
shared environment was modified. Reports and exact probe sources are archived
in `provenance/openfold3_initial_operator_screen.json`.


## First N1536 observation and cycle closure

The instrumented native N1536 request completes in 874.002 s, with 838.465 s
inside the model and 51,757,716,992 bytes peak allocated memory. Inclusive GPU
stages include 550.591 s across four Pairformer calls, 57.775 s across four MSA
module calls, 80.960 s for diffusion, and 20.640 s for auxiliary heads.
Configuration retains three recycles/four trunk passes and 200 diffusion steps,
seed101 and one sample. Exported coordinates/confidence fields are finite and
selected-device isolation passes. This single observation is not a repeated
baseline or an optimized correctness result. Evidence:
`provenance/openfold3_initial_n1536.json`.

Both initial baselines and the first operator screen are now archived. No
OpenFold3 candidate had been promoted at that stage. Per the user's updated priority,
framework migration to Rust took precedence at that point; it has since passed
the acceptance gates in `rust-acceptance.md`.

## Exact FP32 HIP softmax: three-pair N1536 result

The post-migration handwritten HIP experiment fuses mask addition, pair bias and
native-order softmax while preserving upstream QK/PV arithmetic. N1000/N1536
full-model checks pass all 11 arrays bitwise. Three sequential N1536 pairs retain
FP32, three recycles/four trunk passes, 200 diffusion steps, one sample and seed101:

| Scope | Native median | HIP median | Ratio |
| --- | ---: | ---: | ---: |
| Complete invocation | 956.112 s | 694.973 s | 1.37575x |
| Model computation | 859.950 s | 599.567 s | 1.43429x |

All three pairs pass exact array and CIF atom-identity checks. Runs use fresh
processes, published kernels and strict environment inventories on HIP7/PCI93/
SMI5, with filesystem cache unflushed. No other FireProt GPU job was launched
during the sequence; other host jobs were not continuously audited. These results
cover one fixed long input and an experimental worker, not a promoted maintained
OpenFold3 backend or experimental-coordinate accuracy.

See `provenance/openfold3_hip_softmax_e2e_n1536.json` and
`experiments/hip_softmax/README.md` for raw-report pins and integration gates.
An early in-place screen without explicit strict-deterministic memory filling
passes 16 exact comparisons and saves 4.5 GiB at N1536/B128, with negligible
latency change. The strict-mode follow-up and model gates are recorded in `experiments/hip_softmax/INPLACE.md`.

## Maintained HIP request integration

OpenFold3 now admits opt-in `hip_fused` with FP32 and explicit strict determinism.
The worker loads the published `softmax-fp32-v1` bundle and scopes dispatch to one
model invocation, preserving native QK/PV and falling back on unsupported tensor
signatures. Native requests do not load HIP assets. The N1000 maintained request
passes all 11 arrays bitwise and submits 1,702 HIP calls. N1536 also passes all
11 arrays bitwise; submitted-launch and selection counts are retained separately.
See `provenance/openfold3_hip_fused_requests.json` for both pinned acceptances.

```bash
target/release/fireprot infer configs/requests/openfold3-n1000-hip-fused.json \
  configs/diamondhill/openfold3-hip-fused.json
```

Use fresh run IDs/output directories. The checked-in profile pins DiamondHill's
published kernel and environment inventory. These concurrent integration checks
do not establish maintained-route E2E speedup; the 1.37575x result above remains
attributed to the experimental worker used for those three measured pairs.

## Additional pair-conditioning reuse

The maintained HIP worker now has a separate three-pair N1536 comparison with
an experimental rollout-local pair cache. E2E medians are 692.373 s (HIP) and
656.659 s (HIP plus cache), or 1.05439x. Model medians are 597.684 s and 562.362 s,
or 1.06281x. All three pairs pass all 11 arrays and CIF identity; the first HIP
anchor also matches the retained strict native reference exactly.

See [conditions and limits](openfold3-conditioning-reuse.md) and
`provenance/openfold3_conditioning_cache_e2e_n1536.json`. This incremental ratio
and the earlier 1.37575x softmax ratio are separate measurements; their product
is not a directly measured combined-versus-original-Torch result. The cache
remains experimental, scoped to the verified immutable rollout.

## Combined reuse: direct original-Torch N1536 E2E result

The current experimental combination retains the maintained HIP kernel, reuses
private QK score storage, caches static pair/atom conditioning within each rollout,
and preserves the measured N1536 confidence chunk ceiling256. N1000 and N1536
both pass all11 original-native arrays bitwise under FP32/strict deterministic,
R3 (four trunk passes), D200, S1 and seed101. No cycles or diffusion steps are cut.
Each conditioning cache has one native miss and199 hits; softmax has zero storage
fallbacks. Source, full request and output pins:
`provenance/openfold3_inplace_chunk_models.json`.

An earlier all-in-place variant enlarged the N1536 confidence chunk from256 to512
and failed five confidence arrays while coordinates matched. That variant remains
rejected. Keeping the measured ceiling restores the full exact-array gate; this
is not a license to change arbitrary numerical chunking without verification.
See `experiments/hip_softmax/INPLACE.md` for both rejected and accepted evidence.

A direct original-Torch versus combined AB/BA/AB N1536 comparison is complete:

| Timing scope | Original strict Torch median | Combined median | Ratio |
| --- | ---: | ---: | ---: |
| Complete invocation | 949.999 s | 633.712 s | 1.49910x |
| Model forward | 854.616 s | 538.416 s | 1.58728x |

Complete-invocation time decreases by 33.29%. The three individual E2E pair
ratios are 1.46939x, 1.51339x and 1.50073x. All three pairs pass all11 arrays
bitwise and CIF atom identities. Repeated outputs agree with the first native
anchor, which separately matches all11 arrays of the historical native reference.
Each candidate records5472 in-place launches, zero storage fallbacks, confidence
chunk256, and pair/atom caches each with200 calls/199 hits/one miss/no fallback.

Both arms use a shared frozen worker on HIP7/PCI93, FP32, strict determinism with
memory filling, R3/four trunk passes, D200, S1 and seed101. Native bypasses HIP and
both caches. Every invocation starts a fresh model process; filesystem caches are
not flushed. No other FireProt GPU work runs during the sequential measurement.
Selected-device isolation is continuously monitored; unrelated host jobs are not
continuously audited. All eight GCDs are observed idle after completion.

This is a directly measured combination, not a product of earlier ratios. It
does not isolate the contribution of each optimization and does not establish
performance at other lengths, seeds or workloads. Timing-boundary differences
are retained but are not independently measured startup phases.
Evidence: `provenance/openfold3_combined_native_e2e_n1536.json`.
Launch/source/control pins:
`provenance/openfold3_combined_native_paired_v2_preparation.json`.

## D16 coverage: incremental N1536 E2E result

Three additional same-source AB/BA/AB pairs compare the preceding combined
backend with D16 softmax disabled/enabled. Both arms retain pair/atom reuse,
private score storage and confidence ceiling256. Only the head-dimension flag
and corresponding profile digest differ as declared treatment settings.

| Timing scope | D32-only combination median | D16+D32 combination median | Ratio |
| --- | ---: | ---: | ---: |
| Complete invocation | 628.689 s | 540.848 s | 1.16241x |
| Model forward | 534.758 s | 446.844 s | 1.19674x |

E2E time decreases by13.97%; individual pair ratios are1.16186x,1.16305x and
1.16305x. All three pairs, repeated-reference checks and CIF identities pass.
The first baseline separately matches all11 arrays of the historical original
Torch output. Model, checkpoint, input, FP32 strict determinism with memory
filling, R3/four trunk passes, D200, S1, seed101 and HIP7/PCI93 remain fixed.
Fresh processes and unflushed filesystem caches apply as in the preceding test;
no other FireProt GPU work or heavy remote build runs during the sequence.

D32-only records5472 D32 launches; D16-enabled records400 D16 and2708 D32
launches. Both have zero storage fallbacks and both caches record200 calls,
199 hits and one miss. Native main-stack tuners select512 versus1024;
confidence stays256 and diffusion conditioning1024. The result includes that
observed tuning change; it does not isolate a fixed-chunk D16 kernel effect.

This is an incremental comparison over the combined backend, not a new direct
original-Torch speedup. Do not multiply it by the separate1.49910x measurement.
D16 remains experimental. Evidence:
`provenance/openfold3_head16_e2e_n1536.json`; controls:
`provenance/openfold3_head16_paired_preparation.json`.

## Triangle memory experiments (2026-10-06)

The bounded multiplication and attention diagnostics both pass all 11 arrays
and full CIF bytes against original native, accepted D16 and latest workspace
anchors. Their detailed operation tables are recorded in
`provenance/openfold3_triangle_trimul_profile.json` and
`provenance/openfold3_triangle_triatt_profile.json`. These instrumented intervals
are not E2E measurements. Attention's current QK+softmax and PV remain larger
costs than its key packing.

The first handwritten HIP bit-copy experiment preserves native multiplication
arithmetic, chunk256 and in-place residual behavior. On the actual captured
N1536/C128 trained inputs, incoming multiplication measures **238.182 → 160.799
ms (1.481x)** and outgoing **239.371 → 167.584 ms (1.428x)**. All repeated outputs
match bitwise. FP32 strict deterministic mode and memory filling remain enabled;
first calls are excluded, with six measured calls per arm in balanced order.
Copy-specific tests also pass NaN/signed-zero bits, offset destination canaries,
nondefault streams and fallback checks. See
`provenance/openfold3_triangle_copy_operator_v1.json`.

This is an experimental complete-operator result, not a new model or E2E
speedup. Full-model acceptance, N1000 fallback and paired E2E are separate
requirements. Sources and reproduction commands are in
`experiments/triangle_memory/README.md`.

The multiplication-only full N1536 model gate subsequently passed all 11 arrays
bitwise and full CIF bytes against all three anchors. It selected 430 supported
module calls and submitted 7,740 exact copies; 72 other calls used native code.
Its two trunk multiplication directions total 64.401 seconds in diagnostic
stage events, versus 97.205 seconds in the earlier fine profile. This is a
separate diagnostic observation, not a paired E2E claim. See
`provenance/openfold3_triangle_copy_model_v1.json`.

The independent attention key pack screen measured approximately 0.771 → 0.265
ms including ordinary strict output allocation. Captured complete attention
measured ending 282.995 → 276.846 ms (1.0222x) and starting
272.867 → 266.748 ms (1.0229x), with all repeats bitwise equal. See
`provenance/openfold3_triangle_attention_copy_v1.json`. QK+softmax and PV remain
the primary attention targets; these small packing gains do not establish a
performance plateau. Combined full-model, N1000 fallback and paired E2E gates
are tracked separately by the guarded controller.

### Triangle memory: accepted incremental E2E

Three AB/BA/AB pairs at N1536, strict deterministic FP32 with memory filling,
R3/four trunk passes, D200/S1, seed101 compare the latest accepted workspace
combination against the same combination plus exact multiplication copies and
attention key packing. Both use the same frozen worker; diagnostic stage
profiling is disabled. All six outputs match all 11 arrays bitwise and full CIF
bytes against original native, accepted D16 and latest workspace anchors.

| Scope | Previous workspace median | Triangle memory median | Speedup |
| --- | ---: | ---: | ---: |
| Complete Rust invocation E2E | 452.471 s | 416.226 s | **1.0871x** |
| Model forward | 356.595 s | 321.699 s | **1.1085x** |

E2E time decreases by 8.01%; all paired ratios are 1.0847–1.0975x.
Observed p95 improves from 464.711 to 428.428 seconds. Three observations per arm
are not a production tail-latency estimate. Fresh processes retain warm filesystem
and published-kernel caches; the environment inventory and declared treatment
are verified. Each candidate selects 430 multiplication calls/7,740 exact copies
and 5,064 key packs; N1000 native fallback separately passes.

This is **incremental**, not a direct original-Torch ratio. Do not multiply
historical improvements into a new direct E2E claim. Complete evidence:
`provenance/openfold3_triangle_memory_e2e_n1536.json`.

## Latest combination: refreshed hotspot profile (2026-10-07)

The accepted triangle-memory combination was profiled again on HIP7 with the
same strict FP32 N1536, R3/four trunk passes, D200/S1 and seed101 workload.
All 11 arrays and the complete CIF match its qualified reference exactly;
frozen input/source/library pins were rechecked. The environment inventory
matches the preceding accepted deployment.

| Inclusive diagnostic boundary | Seconds |
| --- | ---: |
| Trunk, including all four passes | 262.373 |
| Pairformer stack | 190.612 |
| Pairformer triangle attention, both directions | 105.698 |
| Pairformer triangle multiplication, both directions | 65.081 |
| Template embedder | 49.415 |
| MSA module | 18.466 |
| Diffusion sampling | 46.465 |
| Confidence heads | 6.891 |

These GPU event intervals overlap hierarchically and include native tuning calls;
**do not add the rows or treat this instrumented run as a new E2E comparison**.
The starting combination's earlier uninstrumented E2E median was 416.226 seconds.
Evidence: `provenance/openfold3_current_hotspots_20261007.json`.

Template-stack inputs have four slots, each with 1536-by-1536 pair features.
The next bounded experiment checks whether actual slot tensors and masks are
bitwise identical before reusing a native block's result. It retains all four
output slots and the native final normalization/reduction. No-template input
alone is insufficient to skip the network: it also consumes the evolving pair
representation. Distinct slots and unsupported calls must use native execution.
The first full-model diagnostic qualification passed all 11 arrays and complete
CIF bytes. Nine block calls reused three slots each; no distinct or unsupported
call was selected. Template-embedder events decreased from 49.415 to 14.192
seconds. Other stage times varied, so this is a diagnostic stage comparison,
not an E2E speedup. A separate guard screen verifies different masks and
signed-zero input differences fall back, training/non-inplace calls fall back,
and the original method and output aliasing are preserved.

Evidence: `provenance/openfold3_template_reuse_model_20261007.json` and
`provenance/openfold3_template_reuse_guard_screen.json`. Sources and reproduction
details are in [the experiment notes](../experiments/triangle_memory/TEMPLATE_REUSE.md).
The implementation is not enabled by the maintained profile. Its completed
uninstrumented comparison is described below.

## Identical template-slot reuse: accepted incremental E2E

Three fresh AB/BA/AB pairs compare the accepted triangle-memory combination with
the same combination plus exact duplicate-template block reuse:

| Scope | Previous combination median | Template reuse median | Speedup |
| --- | ---: | ---: | ---: |
| Complete Rust invocation E2E | 419.239 s | 387.985 s | **1.0806x** |
| Model forward | 323.661 s | 291.027 s | **1.1121x** |

Complete E2E time decreases by 7.45%. Individual pair ratios are 1.0806x,
1.0543x and 1.1096x. The three candidate observations span 379.140–395.816
seconds; three repetitions do not establish production tail latency.

All six outputs match all 11 arrays bitwise and complete CIF bytes against the
qualified triangle-memory anchor. Each candidate records nine eligible block
calls and 27 reused slots. The original four-slot normalization/reduction,
FP32 strict determinism with memory filling, N1536, R3/four trunk passes,
D200/S1, seed101 and HIP7/PCI93 remain fixed. Processes are fresh, filesystem
caches are unflushed, and model profiling is disabled. Rust verifies the declared
treatment, run order, device isolation and pair/reference comparisons.

A preceding series was interrupted by a host reboot and has damaged output
files; it is excluded. The replacement series uses persistent dependency paths
restored from hash-verified archives and an identical environment inventory.
Evidence: `provenance/openfold3_template_reuse_e2e_n1536.json` and
`provenance/openfold3_reboot_recovery_20261007.json`.

This is an incremental result, not a fresh original-Torch comparison. Do not
multiply historical ratios. N1000 eligibility and maintained-profile promotion
are not established by this N1536 experiment.


## Exact HIP C128 LayerNorm: accepted incremental E2E

Three fresh AB/BA/AB pairs compare the qualified template-reuse combination
with the same combination plus the exact Welford HIP normalization kernel.

| Scope | Previous combination median | HIP normalization median | Speedup |
| --- | ---: | ---: | ---: |
| Complete Rust invocation E2E | 391.162 s | 364.159 s | **1.0742x** |
| Model forward | 287.825 s | 260.434 s | **1.1052x** |

E2E time decreases by 6.90%. Individual pair ratios are 1.0624x, 1.0805x and
1.0675x. Candidate E2E observations span 363.943–366.417 seconds. These three
repetitions do not establish production tail latency or a direct speedup against
original Torch; historical incremental ratios must not be multiplied.

All six runs pass unchanged Rust comparison gates. All 11 arrays and complete
CIF bytes also match the qualified template-reuse anchor, re-read after completion.
Each candidate selects 11,202 normalization calls and preserves 7,846 native
fallbacks; 4,730 noncontiguous-input copies remain. FP32 strict determinism with
memory filling, N1536, R3/four trunk passes, D200/S1, seed101 and HIP7/PCI93 are
unchanged. Processes are fresh, filesystem caches are unflushed, and model
profiling is disabled. The declared treatment is `FIREPROT_NORM128` and its profile
digest only. The maintained profile does not yet enable this experimental hook.

Evidence: `provenance/openfold3_norm_e2e_n1536.json`, including the hashed archive
of all six outputs and frozen experiment sources. The kernel's rejected rounding
variants and successful operator/model checks remain separately documented in
[the normalization experiment](../experiments/openfold3_norm/README.md).


## Direct strided normalization: no E2E benefit

The subsequent HIP candidate reads three noncontiguous layouts directly, removing
4,730 normalization input copies per model run. Operator and full-model numerical
qualification pass, but three fresh AB/BA/AB pairs do not improve performance:

| Scope | Preceding HIP normalization | Direct strided candidate | Ratio |
| --- | ---: | ---: | ---: |
| Complete invocation E2E median | 365.069 s | 368.236 s | 0.9914x |
| Model-forward median | 261.807 s | 264.061 s | 0.9915x |

All three candidate E2E observations are slower. All six outputs match all
11 arrays bitwise and full CIF bytes, with unchanged N1536 strict FP32,
R3/D200/S1, seed101 and HIP7. Only the strided dispatch flag and profile digest
differ; filesystem caches are unflushed, processes fresh, and profiling disabled.

This path is **not promoted**. The preceding normalization optimization remains
the selected combination. The microbenchmark gain and eliminated copy count do
not establish model E2E benefit. A new diagnostic targets the remaining large
modules; these results do not establish an overall optimization plateau.
Evidence: `provenance/openfold3_norm_strided_e2e_n1536.json`; see
[the strided experiment](../experiments/openfold3_norm/STRIDED.md).


## Post-normalization hotspot diagnostic

After rejecting the strided candidate, a fresh full-model diagnostic retains the
preceding HIP normalization path. It passes all 11 arrays bitwise and full CIF
bytes with the same strict N1536 R3/D200/S1 seed101 contract on HIP7.

| Inclusive GPU region | Seconds |
| --- | ---: |
| Complete trunk | 203.470 |
| Trunk pairformer | 169.992 |
| Trunk triangle attention, start + end | 101.772 |
| Trunk triangle multiplication, outgoing + incoming | 50.239 |
| Trunk pair transition | 13.502 |
| Template embedder | 12.826 |
| MSA module | 16.407 |
| Diffusion sampling | 44.345 |
| Confidence heads | 6.322 |

Parent regions include their child regions: **do not sum these rows**. Timing
instrumentation changes execution overhead; this is not an E2E comparison or a
measured speedup against the older profile. The pairformer runs four passes;
`run_trunk` wraps them in one call. Triangle attention remains the largest named
trunk component. The next diagnostic replays the two retained trained attention
inputs to separate projections, workspace attention and PV costs.

Evidence: `provenance/openfold3_post_norm_hotspots_20261007.json`. The hashed archive
also preserves the rejected strided E2E series and its full outputs.


## Fused triangle gates and channel packing: N1536 paired E2E

The handwritten HIP kernel fuses projection sigmoid, projection multiplication,
mask multiplication and channel-first packing. It preserves FP32 arithmetic order.
Canonical LayerNorm output strides are used in both arms; this control is newer
than the preceding normalization comparison, so their ratios must not be multiplied.

Three AB/BA/AB pairs retain strict deterministic FP32 N1536, R3/four trunk passes,
D200/S1, seed101, HIP7, fresh processes and unflushed caches. Only
`FIREPROT_GATE_PACK=0/1` and the declared profile hash differ. No profiler runs
inside either measured arm.

| Pair | Baseline E2E (s) | Candidate E2E (s) | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 351.629 | 348.149 | 1.0100x |
| 2 | 352.949 | 350.103 | 1.0081x |
| 3 | 354.483 | 348.258 | 1.0179x |

Complete invocation medians are **352.949 → 348.258 s**, **1.0135x**,
**1.33% less time**. Model-forward medians are **258.904 → 254.220 s**
(**1.0184x**). All three E2E pairs are faster. All six outputs pass all 11
bitwise array comparisons and complete CIF byte comparisons against the qualified
model anchor. Candidate selection is confirmed by 430 accepted triangle modules
and 5,160 fused calls, with 18 unsupported calls retaining native behavior.

[Raw measurements, pins and archive receipt](../provenance/openfold3_gate_pack_e2e_n1536.json)
include independent re-reading of current output arrays, CIFs and source pins.
The archive is `openfold3-gate-pairs-epilogue-20261007.tar.gz`, SHA-256
`63c213999cf40505d16d17ddff6498da1f44a7e1a340807afe371a05df40d18f`.
It also retains subsequent epilogue operator experiments and their rejected
launcher attempts; those are not part of this E2E treatment.

This remains a frozen experimental deployment, not a maintained-profile default.
A direct original-Torch comparison of the latest complete combination is still
pending. See [kernel scope and frozen entry](../experiments/openfold3_gate/README.md).


## Output epilogue and data-loader follow-ups

The output epilogue combines sigmoid, projection multiplication and residual
writeback in one handwritten HIP kernel. Multiplication and addition retain
separate FP32 rounding (`#pragma clang fp contract(off)`). It writes directly
into the original strided destination. Twelve numerical cases check the entire
backing storage, including untouched slices and extreme values. All pass.

Four-arm trained-module replay passes all 80 comparisons (two warmups and eight
balanced timed calls per arm and capture). Against the already qualified
projection gate/packing route, adding the epilogue changes median triangle
multiplication time from **116.201 to 112.740 ms** for incoming and
**121.050 to 117.631 ms** for outgoing updates. These are operator measurements.

Full-model qualification also passes all eleven arrays bitwise and complete CIF
bytes under strict FP32 N1536 R3/D200/S1, seed101, HIP7. The worker records 2,580
fused epilogue calls, 430 accepted triangle modules and 18 native fallbacks.
The single observation is **349.815 s complete invocation / 255.063 s forward**;
it does not establish an E2E speedup. The separate three-pair AB/BA/AB experiment
completed in `outputs/experiments/openfold3-epilogue-pairs-20261007`, using
`FIREPROT_EPILOGUE=0/1` on shared frozen sources. No maintained default changes.

The first post-run collector incorrectly expected the default data-worker count
to be explicit in `runner.json` and raised `KeyError('data_module_args')` after
array/CIF checks. A separate verifier reread all outputs and pins, compared the
entire runner configuration with the qualified anchor, and checked route counts.
The original failure is preserved; inference was not rerun or changed.
See [full-model and operator evidence](../provenance/openfold3_epilogue_model_n1536.json).

A separate loader experiment changes only the number of data workers from the
upstream default of ten. Zero workers fails nine of eleven exact-array checks
and is rejected. One worker passes every array and the complete CIF, but its
single **351.665 s invocation / 257.647 s forward** observation provides no
confirmed performance improvement. The default remains ten. A changed RNG or
feature-generation context is a possible explanation for the zero-worker
mismatch, not an established diagnosis.
See [both loader trials](../provenance/openfold3_dataworkers_n1536.json).

These full outputs, frozen workers and scripts are archived on DiamondHill in
`openfold3-epilogue-model-dataworkers-20261007.tar.gz`, SHA-256
`7eee4f1d7d3fdcb483f01169074f801149929ef074a082d3ae904909a8ed55f5`.


## Output epilogue paired decision (2026-10-09)

All six N1536 outputs pass eleven bitwise array comparisons and complete CIF
byte checks. The collector independently rechecks outputs, source pins and
actual dispatch counts. Scientific settings remain strict FP32, R3/D200/S1,
seed101, HIP7, fresh processes and unflushed caches.

| Pair | Gate/packing baseline E2E (s) | Added epilogue E2E (s) | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 357.640 | 375.849 | 0.9516x |
| 2 | 369.436 | 371.197 | 0.9953x |
| 3 | 372.737 | 347.422 | 1.0729x |

Complete invocation medians are **369.436 → 371.197 s (0.9953x)**. Model-forward
medians are **270.866 → 269.711 s (1.0043x)**. E2E median and p95 regress,
and only one pair improves. The spread prevents treating isolated operator
wins or the fastest candidate observation as a stable E2E gain.

**Decision: retain the gate/packing composition without the additional epilogue.**
The epilogue remains a numerically qualified experiment. These results do not
change the earlier gate/packing comparison, and are not a direct original-Torch
benchmark. See [raw paired evidence](../provenance/openfold3_epilogue_e2e_n1536.json).
Archive: `openfold3-epilogue-pairs-20261009.tar.gz`, SHA-256
`99ebf832434a1ff4f298564fae9cc2b207831bddea8f8b8f9a72747dacedd505`.


## Startup diagnostic and next trial (2026-10-09)

A diagnostic run retains all eleven arrays and complete CIF bytes against the
qualified gate/packing anchor. Inclusive CPU wall-clock intervals are:

| Stage | Seconds |
| --- | ---: |
| Checkpoint read | 1.367 |
| Model construction | 2.914 |
| State loading, including construction | 3.147 |
| Runner setup | 4.522 |
| Data setup | 0.001 |
| DataLoader construction | 0.001 |
| First prediction start minus DataLoader construction end | 23.422 |
| Predict step, including output | 262.051 |

Intervals overlap and must not be summed. The 23.422 s gap includes worker
startup and first-batch preparation; it is not a measured breakdown of those
components. Full invocation is 355.373 s and model forward is 260.542 s; these
instrumented observations establish no new speedup.

The old environment inventory rejected this session before worker launch.
An audit found 48 changed system-library identities, including alias entries:
FreeRDP/WinPR, FreeType, Poppler, librsvg/pixbuf and libsoup. Inventoried
Python/Torch/ROCm files did not change. The new experiment pins a newly captured
manifest with the same coverage and preserves the old manifest and full delta.
See [stage measurements and environment audit](../provenance/openfold3_startup_profile_n1536.json).

The next isolated experiment emulates worker 0 initialization during single-item
CPU preparation in the main process, restoring Python/NumPy/Torch RNG, thread
count and determinism settings before model execution. It addresses a concrete
semantic difference in the rejected plain zero-worker attempt. Full-model exact
validation now passes; matched timing is still required before selection. It is not enabled
in maintained profiles and makes no multi-input or multi-seed claim.


The worker-0 isolated preparation trial now passes all eleven arrays bitwise and
complete CIF bytes. CPU regression coverage confirms parent-state restoration on
success and dataset failure, including Lightning's wrapped batch sampler.
A single observation is **351.541 s E2E / 256.046 s forward**, with 15.877 s inside
the isolated CPU preparation context. No stable speedup is inferred from this
observation. Three matched pairs are now complete with the same audited
2026-10-09 environment in both arms (results below).
See [experimental implementation](../experiments/openfold3_data/README.md) and
[qualification](../provenance/openfold3_isolated_data_n1536.json).


## Isolated data preparation: three-pair N1536 E2E result

Strict deterministic FP32 N1536 R3/D200/S1, seed101, HIP7, fresh processes,
unflushed caches, AB/BA/AB order. Both arms use the qualified gate/packing
composition and the same newly audited environment. Only isolated single-item
CPU preparation differs; no GPU operator changes enter this comparison.

| Pair | Ten-worker baseline E2E (s) | Isolated preparation E2E (s) | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 349.126 | 341.782 | 1.0215x |
| 2 | 349.971 | 339.599 | 1.0305x |
| 3 | 351.146 | 344.159 | 1.0203x |

Complete invocation medians: **349.971 → 341.782 s**, **1.0240x**, **2.34% less
time**. Model-forward medians: **254.622 → 254.178 s**. Every pair improves in
complete invocation time. All six outputs match all eleven arrays bitwise and
complete CIF bytes against the qualified model anchor. Independent collection
rechecks current arrays, files and source pins.

**Decision: retain the isolated single-item route in the frozen experimental
composition.** Maintained defaults are not changed. This result is incremental,
with no new direct-original ratio inferred from older measurements.
[Raw results and pins](../provenance/openfold3_isolated_data_e2e_n1536.json) include
the archive `openfold3-isolated-data-pairs-20261009.tar.gz`, SHA-256
`ea825abf416ab4745fe920d7328839d608e77672c3eca616affbed8c08e1449d`.

## Exact QK/softmax/PV fusion replay (2026-10-10)

Native PV reduction order was reconstructed and tested independently: even/odd
K8 is the only passing hypothesis across six bounded synthetic cases and four
full-output B128/B512 cases. Both initial and 32-bit-index fusion then passed
eight packed QK/softmax/PV cases in ordinary and probability-capture modes.

Complete trained TriangleAttention replay includes canonical normalization,
Q/K/V packing, strict allocation, output gating and projection. With two warmups
and twelve balanced samples for each of three arms on each of two captures:

| Direction | Qualified baseline | Initial fusion | Index32 fusion | Index32 speedup |
| --- | ---: | ---: | ---: | ---: |
| Ending | 267.045 ms | 357.777 ms | 258.433 ms | 1.0333x |
| Starting | 256.973 ms | 347.628 ms | 248.387 ms | 1.0346x |

All 84 calls match native trained outputs bitwise. N1536, strict FP32, seed101,
HIP7, unflushed caches. Existing retained-score allocation and Rust leases are
shared across arms; fused kernels leave that score storage unused. The initial
fusion is rejected. The index32 kernel passed full-model qualification but was subsequently rejected
by the paired E2E comparison below.
These are module timings, not E2E speedups; no recycles or diffusion steps change.
See [implementation and limitations](../experiments/mfma_qkv/README.md) and
[pinned replay evidence](../provenance/openfold3_qkv_replay_20261010.json).

The index32 full-model gate subsequently passed: 5,064 fused calls, all 11 arrays
bitwise equal and full CIF bytes equal against the qualified isolated-data anchor.
The strict N1536 R3/D200/S1 seed101 workload and gate/normalization counts are
unchanged. Single invocation 335.676655 s, model forward 249.373962 s; this is
qualification, not a new speedup ratio. Three paired E2E comparisons completed; see the decision below.
See [model qualification](../provenance/openfold3_qkv_model_n1536.json).

## QKV index32: three-pair N1536 E2E rejection

The candidate retains isolated data preparation, caches, template reuse, HIP
normalization and gate/packing in both arms. Only `FIREPROT_QKV_INDEX32=0/1` and
the corresponding profile digest differ. Strict FP32 N1536 R3/four trunk passes,
D200/S1, seed101, HIP7; fresh processes, unflushed caches, AB/BA/AB.

| Measurement | Baseline median | QKV index32 median | Speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 341.625922 s | 342.071571 s | 0.9987x |
| Model forward | 254.552379 s | 253.474888 s | 1.0043x |

All six outputs match all 11 arrays bitwise and full CIF bytes. All candidates
execute 5,064 fused calls; existing gate/normalization/key-copy counts and chunk
choices remain unchanged. First-pair candidate latency rises to364.559315 s; the
remaining pair ratios are only1.0023x and1.0041x. No observations are excluded.

**Decision: reject this index32 QKV composition for promotion.** Its module
replay gain did not produce a stable E2E improvement; retain the isolated-data
baseline. The first slowdown occurred inside model forward with the same peak
allocated/reserved memory. Available records do not establish its cause.
Subsequent PV scheduling and the QK-only control are evaluated separately below.

[Paired evidence](../provenance/openfold3_qkv_e2e_n1536.json) includes the six raw
measurements and refreshed output/source checks. This is not an original-Torch
comparison; no new direct-native speedup is claimed.

## PV scheduling follow-up and QK-only control

Unroll4 preserves the MFMA accumulation order and passes synthetic plus complete
trained attention checks. First three-arm replay gives baseline267.023/256.957
ms (ending/starting) versus unroll4 240.266/230.792 ms. A second four-arm replay
gives baseline266.899/256.866 ms versus unroll4 239.579/232.345 ms. All calls
are bitwise exact, including two warmups plus12/16 balanced samples respectively.

The four-arm control shows that QK-only index32 is slower:271.389/261.595 ms,
despite reduced local register counts. Reject that control. Unroll4 advances to
whole-model qualification and paired E2E; the final E2E result is below. [Details and evidence](../experiments/mfma_qkv/README.md#pv-unroll4-and-qk-only-control-results).

Unroll4 subsequently passed the full-model gate with5,064 calls, all11 arrays
bitwise equal and full CIF bytes equal. Strict N1536 R3/D200/S1 seed101 on HIP7;
single invocation 333.859071 s, model forward241.828975 s. Three paired E2E
runs subsequently completed. [Qualification evidence](../provenance/openfold3_qkv_unroll4_model_n1536.json).


## PV-unroll4: accepted incremental N1536 E2E

Three AB/BA/AB pairs use the qualified isolated-data composition in both arms;
only the fusion flag and profile digest differ. Strict deterministic FP32,
N1536, R3/four trunk passes, D200/S1, seed101, HIP7/PCI93/SMI5, fresh processes
and unflushed caches. All six outputs match all 11 arrays bitwise and complete
CIF bytes. Every candidate executes 5,064 fused calls without bounded-batch
fallback; normalization, gate and template-reuse counts are unchanged.

| Measurement | Baseline median | Unroll4 median | Speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 345.752829 s | 330.998593 s | 1.0446x |
| Model forward | 254.904648 s | 243.406797 s | 1.0472x |

Individual E2E pairs are 339.417633 ->329.915480, 345.752829 ->330.998593
and 347.449672 ->338.770763 seconds: 1.0288x, 1.0446x and 1.0256x.
No observation is discarded. **Retain unroll4 in the experimental composition.**
Complete invocation median falls by 4.27%; maintained deployment integration
and a fresh direct-original-Torch comparison remain pending.
[Full paired evidence](../provenance/openfold3_qkv_unroll4_e2e_n1536.json).

The next unroll8 screen passes eight cases in both templates and all84 trained
module calls, but loses to unroll4: ending257.943 versus240.712 ms,
starting247.681 versus229.770 ms. Reject unroll8 at module latency; no unroll8
full-model/E2E claim. This closes the current tested loop-unroll sweep, without
claiming that all attention optimization opportunities are exhausted.
[Unroll8 evidence](../provenance/openfold3_qkv_unroll8_replay_20261010.json).


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


See [complete experiment and receipts](../experiments/openfold3_diffusion/README.md#packed-bias-e2e-decision-and-independent-global-softmax-screen).


## HIP packing plus grouped global softmax: model qualification

The64-row HIP bit transpose passes the strict N1536 full-model gate: all11
arrays and complete CIF bytes are exact,4800 HIP bias copies and19200 grouped
softmax launches complete with no copies through Torch and no layout fallback.
Single invocation is317.785213 s, model231.075384 s, over the accepted unroll4
composition without global bias caching. Three matched AB/BA/AB E2E pairs subsequently completed; the accepted result
is below. This individual qualification timing is not a paired ratio.
[Evidence and paired roots](../experiments/openfold3_diffusion/README.md#hip-packing-full-model-qualification).


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

The direct-native follow-up has passed CPU preparation and started the unchanged
native worker's full-model qualification. Its native worker SHA remains
`c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e`; the optimized
worker body is copied unchanged from the qualified paired tree. Both routed model
gates must pass before the fresh direct-native pairs run. **No new direct-native
speedup is available yet; do not combine older native timings with these results.**

[Complete paired evidence](../provenance/openfold3_hip_bias_pack_e2e_n1536.json).
