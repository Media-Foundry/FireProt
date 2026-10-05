# OpenFold3 / OpenBind-0 AMD performance work

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
