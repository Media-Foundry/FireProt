# Project context

FireProt is an AMD GPU protein structure inference framework. Its architecture separates model-specific adapters from semantic operation contracts and AMD implementations.

## Current state: Rust framework validated (2026-10-05)

The maintained Rust migration has passed the model, operator, CPU and independent
installation gates described in `docs/rust-acceptance.md`. Consult that audit and
README for current scope; the dated entries below are chronological evidence and
may describe gates that were open at the time.

Rust owns requests, shared registry policy, process/device supervision, artifact
and environment verification, native build/cache publication, output acceptance,
archive replay, invocation timing and paired execution. Python owns explicit
upstream/Torch invocation, tensor metadata/capability predicates, operator calls,
Torch build descriptors and serialization. Historical Python benchmark and
validation helpers remain reproduction tools. New framework policy belongs in Rust.

DiamondHill: `pc@10.120.16.9`, checkout `/media/IntelSSD/fireprot`, validated
installation `/media/IntelSSD/fireprot-install-v2`. All eight GCDs are authorized;
verify HIP-to-PCI-to-SMI mapping and isolation for each run. Checkpoints are listed
in `docs/diamondhill-checkpoints.md`. Preserve the independent Protenix integration
HEAD `bb3be993b52713795796c72cf0a46ef8a7323442` and its separate Git history.

AtlasFold/AtlasLM-3B N1000/N1536 exact array gates pass. Protenix Mini-default
N1000/N1536, Mini-ESM N1000 and Tiny L124 pass their declared gates; accelerated
full confidence arrays are not claimed bitwise equal. OpenFold3 strict deterministic
N1000 passes exact parity; legacy nondeterministic runs remain unpassed. Do not
weaken gates, reduce work or label concurrent acceptance timings as E2E speedup.

Next work is OpenFold3 optimization through the Rust framework, preserving
baseline correctness after every trial. OpenFold2/ESMFold integration, distributed
execution and approximation policies remain separate future work.

The user explicitly directs optimization toward handwritten HIP and CK-style
operators; do not pursue Triton enablement. Keep Rust orchestration and the
existing numerical gates. The post-migration zero-mask experiment passes all
11 OpenFold3 arrays exactly at N1000/N1536. Shared-host E2E measurements show
1.090x at N1000 (three pairs) and 1.083x at N1536 (one pair, insufficient repeats).
This is an experimental worker overlay, not a promoted production route. See
`experiments/rocm_coverage/README.md` and its pinned evidence.

The handwritten FP32 HIP mask/bias/softmax experiment passes 36 operator
configurations and all 11 model arrays exactly at N1000 and N1536. Three
sequential N1536 AB/BA/AB pairs have now completed successfully under Rust:
FP32/R3/D200/S1/seed101, HIP7/PCI93/SMI5, fresh processes, filesystem cache not
flushed, published kernel and strict environment inventories. E2E medians are
956.112 -> 694.973 s (1.37575x), model medians 859.950 -> 599.567 s (1.43429x).
All 33 pair/sample array comparisons and CIF atom identity checks pass.
Evidence: `provenance/openfold3_hip_softmax_e2e_n1536.json`. Individual pair
records `openfold3_hip_softmax_pair_000.json` and `pair_001.json` are historical
partial observations, superseded by that aggregate.

The completed benchmark is at DiamondHill
`/tmp/fireprot-hip-softmax-model-v2/paired-n1536`. Rust PID 1596375 and all six
model workers have exited; all eight GCDs were observed idle before the next
trial. Its old SSH transport session 48828 remained open after scientific
completion; local SSH PID 1715960 was then terminated (transport exit 255).
Do not restart the successful benchmark. Keep copied workers, inventories and
published artifacts intact for reproduction. The valid bundle is
`runtime/kernels/experimental-softmax-f32-v2`. The v1 bundle was accidentally
rewritten by an ISA extraction command and is invalid; both initial model trials
were rejected/cancelled and excluded. Always give objcopy a separate output ELF.
The user supplied local `docs/isa/CDNA2.pdf`; GPT-6-luna xhigh review is complete.
The user PDF stays untracked. Further optimization remains an active goal.
Control/source records and the valid v2 bundle are archived at
`outputs/experiments/hip-softmax-20261005/experimental-control-records.tar.gz`
on DiamondHill, SHA-256
`a7aff023b313b12ccca2cdf2a4c7f40f8105d6449e96fd85f139ce408652b87c`.

The owned-score in-place screen has now completed on HIP5/PCIb3/SMI7 at
`/tmp/fireprot-hip-softmax-inplace-v1/results-001`: 16 bitwise checks pass.
N1536/B128 saves 4.5 GiB of extra operator allocation, with only 0.1-0.2% latency
changes; N1000 is about 1% slower. Evidence:
`provenance/openfold3_hip_softmax_inplace_screen.json`. This establishes memory
savings for this operator only, not model peak or E2E gains. Keep out-of-place
execution for maintained integration; no in-place model route is enabled.

Maintained FP32 boundary (formal request validation passed):
`python/fireprot/backends/amd/softmax_fp32.py`, with byte-identical kernel source
copy `kernels/amd/fireprot_softmax_fp32.cpp`. The package/ABI
`fireprot_softmax_fp32`/`softmax-fp32-v1` is now built and published at
`runtime/kernels/softmax-fp32-v1`, manifest SHA-256
`4914cb71a64c42bf54495f1c9ee0eeb27808f0571b6a50fed81e8067f56f01be`.
`python/fireprot/adapters/openfold3/attention.py` adds source-pinned, context-
scoped dispatch through the Rust registry. Sixteen CPU/meta checks pass across
both components. Strict deterministic FP32 OpenFold3 `hip_fused` request admission
is implemented; both maintained-route long-input validations pass.

Prepared validation entry points in `experiments/hip_softmax` are
`build_runtime.py`, `screen_runtime.py`, and `worker-runtime-overlay.patch`.
Thirty-two GPU operator checks pass: 28 submitted HIP calls and four native
fallbacks, all bitwise exact. See `provenance/openfold3_softmax_runtime_operator.json`.
The new worker overlay enables
candidate dispatch only with `FIREPROT_RUNTIME_SOFTMAX=1`; both arms load the
same new pinned bundle. A separate submitted-launch counter distinguishes Rust
selection from immediate-success HIP launch submission. See that directory's
README for the remaining model and E2E gates. The reviewed narrow source sync
has been applied. New copied worker root `/tmp/fireprot-softmax-runtime-model-v1`
has completed both full-model validations. N1000 HIP6
passes all 11 arrays bitwise against the strict native reference, with 1,702
submitted HIP calls; tool session 50267 exited 0. See
`provenance/openfold3_softmax_runtime_models.json`. N1536 HIP7 also passes all 11
arrays exactly, with 5,592 submitted HIP calls; session 49710 exited 0 and Rust
PID 1657022 has exited. Six scientific request fields match each native reference.
Output run IDs are `softmax-runtime-v1-n1000` and `softmax-runtime-v1-n1536`.
Environment inventory SHA-256 is
`70afed797c43f1f99f3742015f4106fac3dd030b56e38f0dc12a5ef2031dbf86`.
Keep those copied workers for reproduction. The bridge/CLI, inventory and new
published bundle are retained for reproduction.
Parallel model timings do not establish E2E gains. Maintained-route model acceptance and paired timing remain to be done.
Latest completed CI for the recorded experiment is `7b70465` (success).

The source audit in `docs/openfold3-conditioning-reuse.md` identifies timestep-
independent pair conditioning and defines chunking, aliasing, lifetime and exact-
output gates. `experiments/conditioning_reuse` contains a diagnostic overlay for
sampled pair equality, downstream mutation, RNG and chunk observations. Five CPU
tests pass. The N1536 GPU diagnostic completed successfully from copied worker
`/tmp/fireprot-conditioning-probe-v1`, run `conditioning-probe-v1-n1536`, HIP5.
All 11 arrays match native bitwise. One scope contains 200 conditioning calls;
first actual chunk sequence is [1024,1024], all later calls [1024]. Calls
0/1/99/199 have identical pair bytes, unchanged CPU/GPU RNG and no observed
consumer mutation. Actual pair shape is [1,1,1536,1536,128]. Only four calls were
sampled; input immutability is not proven. Evidence and durable control archive
are pinned in `provenance/openfold3_conditioning_observation.json`.
The original observation enables no cache. Diagnostic timings are not E2E results.

Formal request admission now accepts OpenFold3 `hip_fused` only with FP32 and
`deterministic=true`. Native invocation does not load HIP assets. Worker scope
restoration and manifest writing also cover exceptions. Rust workspace tests,
Clippy and 29 focused Python tests pass. The formal N1000 request
`openfold3-n1000-hip-fused-v1` (HIP6, session 80558 exited 0) passed all 11 arrays
bitwise, with 1,702 submitted HIP calls. The formal N1536 process exited 0;
its final array comparison still needs to be collected:
`openfold3-n1536-hip-fused-v1` (HIP7, Rust PID 1682017, session 92978 exited 0).
Do not mutate deployed python/, CLI, bridge, environment or kernel bundle while
these run. Parallel acceptance timing is not a speedup measurement.
Remote Cargo failed downloading crates.io config after retries. The CLI was
built locally with `--release --locked --offline` and deployed after the remote
build exited; both request inspections and remote version checks pass.
Current CLI SHA-256: dfbbe3038421fc5f68dcaba80f4ad95aa8ad93960f29580ee990f5e68b69c817.
Old CLI is preserved under runtime/framework-snapshots; the bridge is unchanged.
Historical worker-overlay patches target the pre-admission worker source; pin
that source or update the patch explicitly when preparing new experiments.

## Historical implemented state

Latest default: CK Tile FP32 attention for exact `[1000,4,1000,32]` QKV with head-last storage on gfx90a. Candidate `af43d8e`, promotion `8fbdddc`. The controlled reverse-order model comparison improves 13.120 to 12.580 seconds (1.043x), attention 6.712 to 6.183 seconds, with unchanged model peak allocation. The native-agreement gate passes over 7,979 atoms (aligned RMSD 9.37e-5 A, confidence delta 3.81e-6). `FIREPROT_DISABLE_CK_ATTENTION=1` returns to direct SDPA; the older long-direct disable also disables CK and restores chunking. The generic CK attention header patch remains a separate unregistered experiment. See `docs/protenix-ck-tile.md`.

The top-level repository now provides a Protenix benchmark harness, same-seed numerical validation, a model-neutral capability registry, process-start BLAS policy, Protenix runtime bootstrap, short-sequence gfx90a FP32 paths, and promoted long-sequence triangle, LayerNorm, and transition paths for the exact 1,000-token Mini bucket. The benchmark harness checks the selected physical GPU before launch and continuously monitors it for other processes. Unsupported signatures and unavailable extension builds fall back to native Protenix.

The packed long-sequence comparison uses FireProt code commit `14eb9f7`; the latest direct attention candidate and promotion use `3df2e91` and `5236d0c`. The independent Protenix integration commit is `77218da6f5ccf8562a76577dd568869785e6b45b`, based on upstream v1.1.0 `d3b4db6a121dd4584edd85e93744239325a2b72e`. Earlier records misreported the full integration SHA suffix. A complete audit of 248 tracked files confirms that local and deployed source trees match; archived labels and the correction are preserved in `provenance/protenix_amd_attention_2ad81b6.json`. The historical short comparison uses FireProt `773999ae404c561309391fa72902641e5be3fb95`; incremental direct CK and channel-first LayerNorm studies use `619db00e21d942f3ff4f3d202b0b02b93bfe5695` and `47850407da303a21ec84a6327300424082ac4875`.

On continuously monitored DiamondHill MI250 gfx90a GCD 6, freshly repeated native and default FireProt Mini/1,000-token/FP32/C4/S1 model medians were 45.350 and 13.501 seconds (`3.359x`); p95 improved by `3.350x`. Pairformer improved by `3.334x`, triangle attention by `4.159x`, and triangle multiplication by `3.195x`. Peak allocated and reserved memory fell by `74.59%` and `67.46%`. Full benchmark-worker wall time improved by `2.261x`; this includes one warmup and three measured calls, not a single cold request. The final default-path comparison over 7,979 atoms passed the locked `1e-3` aligned-RMSD and confidence gates (RMSD `9.46e-5 A`, max confidence delta `7.63e-6`). The performance-only input has no experimental-coordinate mapping.

On continuously monitored DiamondHill MI250 gfx90a GCDs 6 and 7, Mini-default FP32 C4/S1 and C4/S5 warm model medians improved by 1.369x and 1.359x. Both p95 values improved, maximum peak allocated and reserved memory decreased by 10.83% and 10.59%, and same-seed coordinate/confidence checks passed the locked 1e-3 thresholds. These are development results for one 124-token case.

The preceding packed-triangle implementation combines a hipBLAS process policy, direct short-sequence triangle multiplication, ROCm efficient triangle attention, and handwritten HIP kernels. In that historical comparison, memory-efficient SDPA uses outer chunks of 64 and avoids the native full attention matrix; the Protenix seam omits an implicit all-ones mask. Packed triangle multiplication normalizes once, fuses sigmoid projection with channel-first packing, uses a library BMM in each direction, and fuses unpacking with output LayerNorm. Its incremental same-source A/B improved model median from 20.865 to 13.495 seconds (`1.546x`) without increasing whole-model allocated peak memory. The extension also accelerates the measured long 128-wide LayerNorm row counts and the exact long pair transition. For 124 tokens, it provides contiguous 128-wide pair LayerNorm, exact channel-first pair LayerNorm, and the gated transition epilogue. The HIP extension builds lazily into excluded `runtime/torch_extensions` and has native fallback.

The complete observed C4/S1 matrix contains 57 FP32 BLAS signatures and 3,132 calls. Process-wide CK won five isolated signatures, was slower on the dominant 1,588-call signature, and did not support fifteen signatures. A direct CK wrapper supported 29 signatures and won only the eight-call OuterProductMean `124x64 @ 64x32` bucket. Its small isolated advantage did not improve end-to-end medians, so it requires `FIREPROT_ENABLE_CK_OPS=1` and is not promoted. Global CK is rejected for this short bucket. A separate long inventory contains 62 FP32 signatures: CK supported 48, direct CK 31, and dominant projection GEMMs were slower than hipBLAS. After packing, hipBLASLt was only 0.5% faster in a three-sample model comparison and used about 78 MB more allocated memory; the reported primary policy remains hipBLAS. The fused packing kernels are handwritten HIP, not CK library calls.

## Current paths

| Path | Role |
| --- | --- |
| `python/fireprot/ops/` | Model-neutral operation registry and capability signatures |
| `python/fireprot/backends/amd/`, `kernels/amd/` | BLAS policy and promoted AMD operator paths |
| `python/fireprot/adapters/protenix/` | FireProt bootstrap for the independent Protenix checkout |
| `python/fireprot/validation/` | Coordinate, confidence, and ground-truth validation |
| `benchmarks/protenix/` | Repeated inference, rocprof import, BLAS matrix, and output comparison |
| `Protenix/` | Independent upstream repository with local FireProt integration commits |
| `cases/` | Eight short single-chain development cases |
| `runtime/` | DiamondHill-only weights and CCD assets, excluded from Git |
| `outputs/` | Predictions, manifests, and profiler traces, excluded from Git |
| `provenance/` | Compact versioned evidence records |

## Runtime defaults

The archived `.agents/legacy_protenix_infer.sh` uses Mini-ESM, four cycles, one diffusion step, one sample, FP32, seed 101, no MSA/template search, TF32 disabled, and torch LayerNorm before FireProt hooks are installed. `FIREPROT_BACKEND=native` selects upstream torch paths. `FIREPROT_BACKEND=fireprot` installs the registry and runtime hooks. `FIREPROT_BLAS_BACKEND=hipblas` is the fixed policy in the 1,000-token comparison, but only the 124-token all-call matrix establishes it as the best measured process-wide option. `FIREPROT_DISABLE_LONG_TRIANGLE_PACKED=1` returns to the previous chunked gate path. `FIREPROT_DISABLE_LONG_TRIANGLE_GATES=1` disables both long multiplication paths. Other diagnostic switches are `FIREPROT_DISABLE_LONG_TRIANGLE_ATTENTION=1`, `FIREPROT_DISABLE_LONG_LAYERNORM=1`, and `FIREPROT_DISABLE_LONG_TRANSITION=1`. `FIREPROT_DISABLE_CHANNEL_FIRST_LAYERNORM=1` disables the short exact-stride kernel. Direct CK requires `FIREPROT_ENABLE_CK_OPS=1`.

Recycle-cut measurements produced C1/C2/C3/C4 medians of 123.86, 189.20, 259.57, and 334.48 ms. C1/C2/C3 differed from C4 by 5.760, 2.691, and 4.221 Angstrom aligned RMSD. The result is not monotonic on the single development example, so C4 remains the default and lower cycle counts are explicit approximate requests.

## Next measurement

The attention profile and first CK tuning pass are complete. AOTriton `attn_fwd` accounts for about 5.333 seconds of the preceding attention stage. Direct SDPA removes outer chunk assignments and layout copies for the exact implicit-mask bucket: the final model median is 13.124 seconds, attention 6.726 seconds, with unchanged allocated peak. Bidirectional same-source A/B gives 1.026–1.028x incremental model gains. Native agreement passes the unchanged coordinate/confidence gates. `FIREPROT_DISABLE_LONG_DIRECT_ATTENTION=1` restores chunked SDPA.

CK Tile QRKSVS is now compiled, measured, integrated, and promoted for the exact N1000 bucket. Its M128/N64, four-wave configuration measures 32.711 ms including bias packing and output allocation, versus direct SDPA's 36.35–36.36 ms. K16 is required by the installed pipeline's two-stage prefetch constraint at head dimension 32. M32 tiles and fast exp2 were slower; the old generic CK M64/N32 cache-Q probe remains separate. The new loader pins the measured CK version, hashes all CK Tile headers and build inputs, and falls back on unavailable builds. Next extend to multiple real inputs longer than 1,000 residues with explicit workspace and model numerical gates, then revisit dominant projections and triangle multiplication. Base weights and production step counts remain unmeasured. Any recycle-cut policy still needs independent quality validation.

## Latest evidence

- Packed path: `docs/protenix-packed-triangle.md` and `provenance/protenix_amd_long1000_14eb9f7.json`.
- Long BLAS inventory: `provenance/protenix_amd_long1000_blas_a8bbfef.json`; contiguous dimension replay does not preserve real strides.
- Reproducible operator benchmark: `python -m benchmarks.protenix.benchmark_triangle` with explicit GPU 6 or 7.
- Latest attention source, tuning, and model evidence: `docs/protenix-attention-tuning.md` and `provenance/protenix_amd_attention_2ad81b6.json`.
- Latest CK Tile implementation and model results: `docs/protenix-ck-tile.md`.
- Final tests: local 99 passed / 13 skipped; DiamondHill GPU 7 with HIP tests enabled 111 passed / 1 skipped.
- Use `PYTHONPATH=.:Protenix python -m pytest -q tests`; omitting the test directory also collects independent upstream tests. For synchronized remote inference set `FIREPROT_PROTENIX_COMMIT` to the verified integration commit because remote Git metadata remains at upstream.

## Independent installation (2026-10-05)

Rust `install` and `verify-installation` create a Linux prefix independently of the
source/build tree. DiamondHill `/media/IntelSSD/fireprot-install-v1` contains the
validated 57-file installation; its generated profiles point to installed worker,
bridge, HIP probe and compiler guard paths. Five installed model runs pass the
existing AtlasFold/Protenix N1000/N1536 and deterministic OpenFold3 N1000 gates.
See `docs/installation.md` and `provenance/rust_installation_integration.json`.
Do not move a prefix without regenerating its absolute profiles. Environments,
weights, sources, published kernels and writable caches remain external.

## Explicit environment inventories (2026-10-05)

`capture-environment` and execution-profile `environment_inventory` pin declared
Python/import/library trees in Rust. `configs/diamondhill/*-inventoried.json` refer
to immutable snapshots in `runtime/environments` on DiamondHill. Three N1000 runs
pass existing correctness and isolation gates. Coverage is not yet a complete
runtime lock: OS libraries, plugins and external resources still need audit.
Full scans take 17-25 seconds in the recorded captures and are repeated at launch
and acceptance, so default profiles remain unchanged. See
`docs/environment-inventories.md`. Output validation now rechecks cancellation and
timeout before accepting a run; the new cancellation regression failed before the
fix and passes afterward. Final environment acceptance remains outstanding.

## Prediction CIF acceptance (2026-10-05)

The initial CIF migration required prediction bundle schema 2 with explicit sample/CIF
associations; archive replay preserves schema 1 compatibility. Rust parses native
prediction CIFs and compares Protenix structures with the unchanged numerical
gates. Four N1000 requests (AtlasFold, Protenix one/two samples, deterministic
OpenFold3) pass acceptance and archive replay. AtlasFold/OF3 arrays retain exact
native parity; Protenix single-sample passes established gates. AtlasFold CIF
rounding and omitted atom14 slots prevent claiming a complete CIF-to-tensor
mapping. See `docs/cif-validation.md` and `provenance/rust_cif_integration.json`.

## Mini-ESM migration (2026-10-05)

Protenix Mini-ESM FP32 is connected with independently pinned folding, ESM2-3B
and contact-regression weights. N1000 native/FireProt preserve exact input ESM
features against an independent upstream invocation and pass established
coordinate/summary gates. All CLI workers now use fresh per-run working
directories. Mini-ESM recomputes embeddings per request and rejects upstream
truncation above 4094 residues; that bound is not a tested capacity claim.
See `docs/protenix-mini-esm.md`. The maintained shell entry point now uses Rust;
this worker addition does not complete the framework rebuild.


## Rust batch and maintained entry points (2026-10-05)

`infer-batch` executes pinned requests sequentially with a shared cancellation
flag, explicit batch deadline and per-job accepted archives. The three-model
L124 smoke uses a 600-second deadline and retains C4/step1/sample1/FP32/seed101.
Tiny-default native and FireProt pass the existing coordinate/confidence gates
against an independent upstream reference on that development case. No long
Tiny performance claim is established. `scripts/infer.sh REQUEST PROFILE` and
`scripts/smoke.sh [MANIFEST]` now exec Rust; original launchers are archived under
`.agents/legacy_*` with `provenance/entrypoint_path_map.json`. See
`docs/entrypoints.md`. Environment coverage, paired performance verdicts, final
acceptance and remote CI/push remain outstanding.

## Complete inference invocation timing (2026-10-05)

`measure-inference SPECIFICATION DESTINATION` snapshots pinned controls and
measures a child `infer` command from before spawn through observed exit. It
binds successful records to accepted model archives; `collect-timings` accepts
E2E only with corresponding pinned invocation records. Cancellation/deadlines
remain cooperative so the child supervisor can clean up separate worker groups.
Two sequential Protenix Mini-default N1000 FP32/C4/S1/seed101 published-operator
runs on HIP 6 / PCI 8e / SMI 4 collect together and pass existing raw-array/CIF
gates against the native reference. E2E times 42.219/41.491 seconds establish
workflow integration, not a new speedup. See `docs/invocation-timing.md` and
`provenance/rust_invocation_timing_integration.json`. Paired performance verdicts
and complete cross-model E2E acceptance remain unfinished.

## Paired numerical and performance comparison (2026-10-05)

`compare-runs SPECIFICATION` authenticates both collections, rejects changes to
scientific work/software/device/cache conditions, requires exact declarations
of permitted backend/operator changes, and compares every requested sample.
Failed numerical gates produce `performance: null` and a nonzero CLI status.
Three sequential N1000 Mini-default FP32/C4/step1/sample1/seed101 pairs on HIP 6 /
PCI 8e / SMI 4 pass the existing gates. Native/FireProt medians are 50.604/23.460 s
for model time (2.157x) and 87.618/41.309 s for complete invocation time (2.121x).
This uses existing published operators and fresh model processes, not new kernel
optimizations or a warm resident-model measurement. See
`docs/paired-comparisons.md` and `provenance/rust_paired_comparison_integration.json`.
Environment coverage/overhead, AtlasFold CIF mapping, final installation/model
acceptance, CI and push remain outstanding.

## Rust paired execution (2026-10-05)

`benchmark-pairs SPECIFICATION DESTINATION` validates and snapshots every control
before launch, executes explicit pair order through the existing Rust batch
lifecycle, checks every later run against the first baseline anchor, checks each
pair, and retains an aggregate report and hashed order journal. Failure or
cancellation stops later jobs. Two N1000 Mini-default native/FireProt pairs run
entirely through this command on HIP 6 / PCI 8e / SMI 4 and pass all checks.
The 263.419-second batch interval includes comparisons and is not inference E2E;
two pairs correctly produce an insufficient-repeat timing trend. See
`docs/paired-execution.md` and `provenance/rust_paired_execution_integration.json`.
Full Rust tests pass 157; relevant Rust 1.85 tests pass 10. Avoid concurrent
toolchain builds into the same target while identity-pinned CLI tests are active.

## AtlasFold input-bound structure acceptance (2026-10-05)

AtlasFold now requires bundle schema 3: the worker retains the exact input bytes
it consumes; Rust checks their request-pinned hash, parses JSON/FASTA and validates
every sample's full atom14/CIF correspondence. Old schemas retain historical
replay guarantees. N1000 native, N1536 fused HIP and N1000 two-sample chunked/fused
HIP pass current acceptance and replay. Single-sample references and both samples
of the two-sample pair match exactly. Relocation without the original input passes;
a separately resealed wrong-input snapshot is rejected. N1536 retains 100 steps;
a mistaken 30-step comparison is explicitly excluded. Unchunked native N1000 with
two samples ran out of memory; the existing bounded chunked reference passes with
unchanged sample/step counts. See `docs/cif-validation.md` and
`provenance/rust_atlasfold_input_binding.json`. Tests: 165 Rust workspace and 272 Python passed / 110 skipped; after a
FASTA whitespace compatibility correction, final validation-crate runs pass 36
on stable and Rust 1.85, and the additional Python regression passes. Final
archive replay covers all four runs; Clippy/format pass. Remaining
rebuild work: environment coverage/overhead, final packaging and cross-model
acceptance, coverage audit, remote CI and push. Do not resume kernel tuning yet.

## Environment scan reuse (2026-10-05)

Rust environment capture now reuses a canonical file's digest only within that
capture, while preserving every alias and rehashing at each later checkpoint.
Unix inode/size/mode/mtime/ctime guards reject changes during the scan. Existing
build-tree callers keep ordinary per-entry hashing. All three pinned environment
manifests remain byte-identical. AtlasFold capture medians improve 15.855 to
12.202 seconds across three runs per arm (1.299x); this is not inference E2E.
84 runtime tests and ten Rust 1.85 tests pass; Clippy/format pass. See
`provenance/rust_environment_scan_dedup.json`. A point-in-time AtlasFold loaded-
library audit identifies system dependencies outside the current roots; the
full environment coverage audit remains open. Do not equate scan deduplication
with complete environment provenance or resume kernel tuning yet.

## Active conditioning verification (2026-10-05)

`experiments/conditioning_reuse/verified_reuse.py` implements experimental final-
pair reuse scoped to `_sample_rollout`, with mandatory native recomputation on
EVERY call and consumer-mutation/RNG checks. There is no unchecked reuse mode.
Twelve CPU diagnostic/lifecycle tests pass. The pinned upstream GPU module with
randomized nonzero parameters passes 12 N1000/N1536 scenarios (48 calls), covering
all-valid/partial/zero masks and conditioning enabled/disabled. Evidence:
`provenance/openfold3_conditioning_reuse_screen.json`; timings are not performance.
Two full-model diagnostics now run from fresh immutable copied workers under
`/tmp/fireprot-conditioning-reuse-model-v1`:
- N1000: HIP5, `conditioning-reuse-v1-n1000`, Rust PID 1695703, session 68574.
- N1536: HIP6, `conditioning-reuse-v1-n1536`, Rust PID 1695808, session 65044.
Keep CLI/bridge, inventory and copied worker tree fixed while running. Poll these
owned jobs and compare every model array before considering an unchecked cache.
The source copy predates a comment-only Apache attribution addition to the local
helper; source bytes used by the screen are pinned in its durable archive.

SSH access to DiamondHill began timing out during result collection. Do not
restart existing model tasks based on this observation failure. Formal N1536
session 92978 has returned exit 0, but its native array comparison is pending.
Read-only collector `/tmp/fireprot-record-formal-hip.py` saved the verified N1000
record in `provenance/openfold3_hip_fused_requests.json` and exited 0 (session
20994) after its stalled read-only SSH child PID 1862479 was terminated. No model
process was signalled. Read-only status session 28159 still awaits SSH.

## Cache preparation and intermittent SSH (2026-10-05)

Commit b89eff3's CI passed in run 37334622947. The duplicate run 37334623276
was cancelled by concurrency scheduling, not by a failing test.
SSH briefly recovered: formal N1536 supervisor phase is succeeded, physical
HIP7/PCI93/SMI5, and Rust compare-npz against native baseline passed (comparison
saved at /tmp/fireprot-formal-hip-n1536-comparison.json). N1000 verified-reuse
supervisor also succeeded, HIP5/PCIb3/SMI7, and its audit shows 200 calls/199 reuse
comparisons, all checks true. Rust final 11-array comparison passed and is saved
at /tmp/fireprot-conditioning-reuse-model-v1/n1000-comparison.json.
N1536 verified-reuse PID 1695808 was confirmed live at elapsed 501 seconds;
keep tool session 65044 and poll it. Subsequent SSH connections timed out again.
The compact evidence files still require fresh collection/pins; the checked-in
formal report conservatively retains its prior pending-N1536 status.

`pair_cache.py` and `worker-cache-overlay.patch` prepare actual compute skipping,
with first-native-call/tuner preservation, per-rollout teardown, dependency
replacement invalidation and source/strict-determinism guards. Six CPU tests
pass. No GPU trial or performance claim yet; do not confuse this candidate with
the earlier mandatory-recomputation diagnostic. Do not deploy over active trees.
The next step is collect existing results, then fresh copied-worker N1000/N1536
cache validation, then sequential paired timing only after exact output gates.

The stalled multi-SSH read collectors were terminated locally without signalling
remote model processes. Sessions 8175 and 89965 are terminal (8175 connection
failure; 89965 owned read-only collector stopped). A one-connection remote
collector is prepared locally at /tmp/fireprot-collect-current.py; invoke it with
ssh python3 on stdin and accept its JSON only after successful exit. Last output
/tmp/fireprot-collected-current.json is empty due to connection failure.

SSH subsequently recovered through a persistent multiplexed connection:
`ssh -o ControlPath=/tmp/fireprot-diamondhill-control pc@10.120.16.9 ...`.
For a new master use ConnectTimeout=8, ControlMaster=auto, ControlPersist=120.
The one-connection collector succeeded. Both formal HIP requests now pass all
11 arrays, pinned in `provenance/openfold3_hip_fused_requests.json`.
N1000 mandatory-recomputation validation is pinned in
`provenance/openfold3_conditioning_verified_models.json`; N1536 remains live.
Actual compute-skipping cache is deployed only in a fresh copied worker at
`/tmp/fireprot-conditioning-cache-v1`. N1000 HIP5 run
`conditioning-cache-v1-n1000` is active in tool session 66219. N1536 request is
prepared there but NOT launched; first finish its preceding diagnostic.
The original helper/CPU tests are unchanged after copying. No GPU cache
correctness or speedup is established yet. Keep all active source/binary trees fixed.

Both mandatory-recomputation full-model validations now pass all 11 arrays;
each records 200 calls, 199 reused pairs, unchanged RNG and no consumer mutation.
Actual cache N1000 also passes all 11 arrays: one miss, 199 hits, no fallbacks,
chunk 1024, 512,000,000 retained bytes. Evidence is now in
`provenance/openfold3_conditioning_cache_models.json`. N1536 actual cache is active
on HIP7, Rust PID 1716475, tool session 97740. N1000 session 66219 exited 0.

After this optimization change, the user requests GPL-2.0-only with an exception
LIMITED TO OPENFOLD 3. OpenFold 3-derived files stay Apache-2.0; the distribution
is explicitly multi-license. AMD (Advanced Micro Devices, Inc.) and its directly/
indirectly controlled subsidiaries receive an additional Apache-2.0 option for
original FireProt material only. Both scope choices were explicitly confirmed.
Do not introduce a generic Apache-2.0 combination exception. License preparation
is at /tmp/fireprot-license-draft; no effective license change has yet been made.
License work must preserve immutable remote model workers and binaries.

## License policy (2026-10-06)

At the user's explicit direction, original FireProt material is GPL-2.0-only
with a combination exception LIMITED TO OPENFOLD 3. The root LICENSE contains
the precise exception, AMD alternative and complete unchanged GPLv2/Apache2 texts.
AMD means Advanced Micro Devices, Inc. plus directly/indirectly controlled
subsidiaries; that group has an additional Apache-2.0 option only for original
FireProt material. Standard downstream Apache permissions and earlier releases'
existing grants are preserved. There is no blanket Apache-dependency exception.
OpenFold3-derived attention adapter, experimental attention overlay and verified
conditioning helper remain Apache-2.0 as whole files, with original copyrights
and modification notices. THIRD_PARTY.md is the component inventory.
Cargo now inherits license-file=LICENSE; no unrestricted dual-license expression
is used. Installer copies the license inventory/texts into share/fireprot.
Three installation tests, Cargo metadata for all five crates, exact LICENSE
inclusion in a packaged core crate, format and Python syntax checks pass.
Remote active model workers/binaries have not been modified by this license change.

## Completed cache gate and next trials (2026-10-06)

Actual N1000/N1536 conditioning cache now passes all 11 output arrays bitwise.
Both scopes have 200 calls, one native miss, 199 hits, no fallbacks, chunk 1024.
N1536 supervisor succeeded; session 97740 may still need its final transport poll.
The evidence and durable control archive are pinned in
`provenance/openfold3_conditioning_cache_models.json`. No paired speedup yet.
Fresh combined controls `/tmp/fireprot-conditioning-combined-v1` use the same
immutable `/tmp/fireprot-conditioning-cache-v1/python` worker and switch the
request backend to hip_fused. Active combined numerical checks:
- N1000 HIP5: conditioning-combined-v1-n1000, tool session 59510.
- N1536 HIP7: conditioning-combined-v1-n1536, tool session 17465.
Do not replace their worker tree, remote CLI/bridge, inventory or published bundle.
Remote CLI remains dfbbe3038421fc5f68dcaba80f4ad95aa8ad93960f29580ee990f5e68b69c817;
license/installer changes are local only so far. After exact combined checks,
prepare sequential same-source baseline/candidate paired E2E, with all scientific
settings unchanged. License commit e6ac271 follows optimization commit d3e0a86.

## Incremental cache E2E preparation (2026-10-06)

N1000 combined HIP+cache passes all 11 arrays; N1536 combined remains active
(PID 1727353, session 17465). Session 59510 completed successfully.
CI b7ece06 passed (run 37338832565). A fresh same-source benchmark worker is
prepared at /tmp/fireprot-conditioning-paired-v1, using current license notices.
worker-paired-overlay.patch explicitly switches only FIREPROT_CONDITIONING_CACHE
(0 baseline / 1 candidate); both arms use hip_fused. Three N1536 AB/BA/AB pairs
are pinned in pairs-n1536.json, SHA-256
20e833b9de7e801f07aaa7699bbfd2007f08c0e0da2148f2fd579957d1bf1656.
Do not launch the benchmark before N1536 combined output comparison passes.
This experiment measures the cache increment over HIP softmax, not a new direct
comparison to original Torch. Do not multiply historical ratios and label them
as directly measured total E2E. During the sequential benchmark, launch no
other FireProt GPU workloads; perform only CPU/source work and monitoring.

Both combined checks have now passed all 11 native arrays, with 199 cache hits
per model and 1,702/5,592 submitted HIP calls. See
`provenance/openfold3_conditioning_combined_models.json`. Both supervisor PIDs
exited and all eight GCDs had no running processes at the pre-benchmark snapshot.
The N1536 AB/BA/AB benchmark has now been LAUNCHED, tool session 29158:
/tmp/fireprot-conditioning-paired-v1/paired-n1536. Preparation/source pins and
its persistent archive are in
`provenance/openfold3_conditioning_paired_preparation.json`.
Keep that worker/source tree, remote CLI/bridge, bundle and environment fixed.
Launch no other FireProt GPU tasks until this sequence terminates. Poll its
specific live controller/journal; do not restart on transport observation timeout.
Results and maintained-route incremental E2E are still pending.

The live paired controller is Rust PID 1739049. The first journal entry confirms
baseline pair 0 started (conditioning-e2e-n1536-baseline-0).

## Read-only MFMA follow-up (2026-10-06)

The authorized GPT-6-luna xhigh ISA reviewer completed its follow-up without GPU
work or compilation. `docs/openfold3-mfma-investigation.md` records verified PDF
pages, limitations on accumulation/lane-layout inference, and the next evidence
gates. A SHA-verified read of the historical native N1536 trunk profiler yielded
GEMM symbol names containing MI16x16x4x1 and MI32x32x2x1; extracted records are in
`provenance/openfold3_historical_gemm_symbols.json`. Names suggest, but do not
prove, actual MFMA instructions. Do not claim native lacks matrix instructions,
map these aggregate symbols directly to QK/PV, or treat old diagnostic timings
as current optimized bottlenecks. Heavy disassembly/library scans and all new
GPU probes wait until the active paired benchmark terminates.
After that sequence, also compare its first HIP baseline against the retained
strict native N1536 reference before publishing the incremental result.

## Bounded attention diagnostic preparation (2026-10-06)

The first incremental cache pair passed all 11 arrays and CIF identity checks:
HIP baseline 690.574494673 s, HIP+cache 656.044704742 s, one-pair ratio
1.0526332880692626. This is explicitly insufficient repeats, not the final
three-pair result or a direct comparison to original Torch. The controller
1739049 remains live; pair 1 candidate started at journal offset 1352.604365844.
CI for 68f5cba completed successfully.

`experiments/attention_trace/` prepares a one-call QK/PV layout/correlation
observer for the next diagnostic. Its three CPU tests pass exact output/RNG,
labels/BMM metadata, scope, exception and unmatched-call checks. It is not
deployed and has no GPU or model validation. Follow its README only after the
paired benchmark terminates; use a fresh worker/profile, retain the cache/HIP
path, seal both diagnostics, and require the full native output comparison.
The observed N1536 dispatcher signatures include 32/64/128/256/512/1024 rows,
H4/D32 with strides [196608,32,128,1]. Selection includes tuning attempts and
does not prove submission or steady-state frequency. The initial trace selector
uses 32 rows conservatively and must not be extrapolated to other chunks.

## Further diffusion reuse preparation (2026-10-06)

Read-only upstream inspection identifies static atom conditioning as another
candidate: `get_atom_reps` returns cl/plm independent of noisy rl in the pinned
source, while ql adds `noisy_position_embedder.linear_r(rl)`. Actual rollout
invariance and consumer mutation still require measurement.
`experiments/conditioning_reuse/verified_atom_reuse.py` prepares mandatory native
recomputation, exact ql/cl/plm checks, RNG and post-diffusion consumer checks.
Four CPU tests pass, including injected signed-zero/RNG/mutation failures,
restoration, source rejection, inference gating and distinct-rollout reset.
The helper retains Apache-2.0 because it extracts the upstream noisy-coordinate
expression; THIRD_PARTY.md records it. No remote deployment or GPU validation.
Run this diagnostic only after the current paired sequence, in a fresh combined
HIP/cache worker, and require 200 checked calls plus all 11 final native arrays.

The second candidate (measurement 002) has now completed and passed the first
baseline reference. Controller 1739049 is running measurement 003, pair 1's
baseline. Final three-pair performance is still pending.

## Completed incremental E2E and live follow-ups (2026-10-06)

The three-pair sequence completed successfully; session 29158 exited 0,
controller 1739049 and all six workers exited. All eight GCDs were observed idle
after completion. Final E2E medians: HIP 692.372833244 s, HIP+cache
656.659018284 s, ratio 1.0543871537062393. Model medians: 597.6835733929183 s
and 562.3623078720411 s, ratio 1.0628087356254932. Every pair and reference check
passes all 11 arrays and CIF identity; a separate Rust compare-npz also confirms
the first HIP anchor matches the original strict native N1536 output exactly.
All candidates record 199 hits, one miss, no fallback and 5,592 HIP submissions.
See `provenance/openfold3_conditioning_cache_e2e_n1536.json`. Do not multiply
this incremental ratio by the earlier softmax ratio and call it measured E2E.

Durable paired controls:
/media/IntelSSD/fireprot/outputs/experiments/conditioning-cache-20261006/paired-controls.tar.gz
SHA-256 cdf4dc86e18cf3ef98b2a3e59a53d61cebcc236755b38d64d2c2240449a38536.
Local operational collector: /tmp/fireprot-collect-cache-pairs.py; collected
JSON: /tmp/fireprot-cache-paired-results.json. Both follow-up commits 6b13b04
and e454d96 have successful CI.

Three independent diagnostics were prepared, archived, inspected by Rust and
launched only AFTER the timing sequence and original-anchor comparison passed:
- attention-trace-v1-n1536, HIP7, supervisor 1782895, tool session 22381;
  /tmp/fireprot-attention-trace-v1/n1536.json and profile.json.
- atom-reuse-check-v1-n1000, HIP5, supervisor 1782902, tool session 65529;
  /tmp/fireprot-atom-reuse-v1/n1000.json and profile.json.
- atom-reuse-check-v1-n1536, HIP6, supervisor 1782896, tool session 64225;
  /tmp/fireprot-atom-reuse-v1/n1536.json and profile.json.

Their root n1000.log/n1536.log files capture supervisor output. Model outputs
remain under /media/IntelSSD/fireprot/outputs/<run_id>. Freeze these copied
worker trees, published bundle, environment and remote CLI/bridge until terminal.
These concurrent instrumented runs are NOT performance evidence. Require the
full original-native array comparison, all 200 atom checks and consumer/RNG
checks, or valid raw QK/PV trace correlations as applicable. No result claimed yet.
Requests/source files/launch handles are in
`provenance/openfold3_followup_diagnostics_preparation.json`. Its archive is
/media/IntelSSD/fireprot/outputs/experiments/conditioning-cache-20261006/followup-preparation.tar.gz
SHA-256 dc0a2593ffbc955d0885dd2e28f970392ec0004b0605098b9c91c710c912073a.
The local preparation capture also remains at /tmp/fireprot-followup-prepared.json.

## Atom reuse accepted; attention observer corrected (2026-10-06)

Both atom-reuse-check-v1 requests completed successfully (sessions 65529/64225
exit 0). Every one of 200 ql/cl/plm, RNG and consumer checks passes; both final
NPZs match all 11 original-native arrays. Retained cl/plm logical sizes are
69,620,736 bytes (N1000) and 105,829,376 bytes (N1536). Evidence:
`provenance/openfold3_atom_conditioning_verified_models.json`.
Next: implement compute-skipping atom reuse with explicit rollout scope and
dependency/immutability guards, then repeat both long-input exact-model checks.
The current verifier still executes native atom computations every step.

Attention trace v1 failed its required observation guard; session 22381 exited 1
and supervisor 1782895 exited. Upstream logged "Incomplete QK/PV attention
observation" and the outer supervisor reported "Upstream returned no prediction".
Nested TorchDispatchMode gives no ATen observations under inference_mode; this
was reproduced locally. The raw Kineto trace does contain input shapes/strides,
external IDs and launch correlations. The corrected probe extracts those records
without changing inference mode, rejecting ambiguous/duplicate correlations.
Four CPU tests pass, now including inference tensors and duplicate correlations.
The original raw trace also exercises extraction, but v1 has no accepted model
result and must not be reported as passing or as performance evidence.

Only the new attention-trace-v2-n1536 diagnostic is currently active:
HIP7, supervisor 1801868, tool session 40942,
/tmp/fireprot-attention-trace-v2/n1536.json and profile.json, root n1536.log.
Its copied worker is immutable. Evidence/preparation:
`provenance/openfold3_attention_trace_v2_preparation.json`.
Require complete correlated QK/PV records and all 11 native arrays before accepting.

Completed follow-up archive:
/media/IntelSSD/fireprot/outputs/experiments/conditioning-cache-20261006/followup-v1-results.tar.gz
SHA-256 5f16797930de0f4f7362512775226f9bc71f54a30904952d6904a567242a0b7e.
V2 preparation archive in the same directory: attention-trace-v2-preparation.tar.gz,
SHA-256 adc3797bef6a9dca75bf5cd33245956d902b7c0e8b61511f748bcf68dbfbd280.
Local result/preparation captures: /tmp/fireprot-atom-verified-models.json,
/tmp/fireprot-attention-trace-v2-prepared.json and
/tmp/fireprot-attention-v1-exclusion.json.

## Accepted QK/PV trace and active atom-cache trials (2026-10-06)

attention-trace-v2-n1536 completed, session 40942 exit 0. All 11 original-native
arrays match. Both QK/PV contain real BMM shapes/strides and one unique runtime
launch/GPU kernel correlation. Evidence: `provenance/openfold3_attention_trace_n1536.json`.
This is one B32/H4/N1536/D32 call, potentially tuning; no clean latency/ranking
or disassembled-ISA claim. The previously user-authorized cdna2_isa_review agent
is following up on the actual symbols, read-only on library copies, no GPU or
repository modifications. Never let objcopy rewrite an installed/published ELF.
Trace result archive: attention-trace-v2-results.tar.gz in the existing durable
conditioning-cache-20261006 directory, SHA-256
546ef9ddc32f36f21157fc1bc27863ac42036ad89bedd0abf13194d812d91142.

`experiments/conditioning_reuse/atom_cache.py` now implements actual static atom
reuse, preserving the first native call and recalculating only the rl-dependent
projection. Four CPU lifecycle tests pass. This helper is Apache-2.0 and listed
in THIRD_PARTY.md. Immutable source-pinned inputs/parameters/consumers remain a
prerequisite; there is no in-place mutation detection. Dependency owners are
retained through the rollout. Whole-model tests use the same HIP plus pair-cache
worker in a new frozen copy at /tmp/fireprot-atom-cache-v1.

- atom-cache-v1-n1000, HIP5, supervisor 1809047, session 88452: terminal success,
  199 hits/one miss/no fallback and all 11 original-native arrays exact.
- atom-cache-v1-n1536, HIP6, supervisor 1809044, session 19548: terminal success,
  199 hits/one miss/no fallback and all 11 original-native arrays exact.

Requests n1000.json/n1536.json and profile.json are under that control root.
Preparation/source pins: `provenance/openfold3_atom_cache_preparation.json`.
Archive atom-cache-v1-preparation.tar.gz SHA-256
834821a454fd3530c5688248cdcfa8ca220f40d15b1385d46dafe88a065ac309.
Local partial collection: /tmp/fireprot-atom-cache-models.json. Refresh N1536
when terminal, archive both completed trials and record acceptance before timing.
Current timings are concurrent numerical trials, not E2E performance evidence.

Another evidence-based follow-up: the accepted trace records empty_like/fill_
on the full softmax output. The old screen_inplace.py does not explicitly enable
strict deterministic algorithms/fill policy, so its tiny timing difference does
not close this strict-mode question. See experiments/hip_softmax/INPLACE.md.
Repeat under actual settings before deciding; never disable global determinism
or memory filling merely to improve reported performance.

Both actual atom-cache trials are now archived and accepted in
`provenance/openfold3_atom_cache_models.json`. Archive atom-cache-v1-results.tar.gz
SHA-256 8e3f3758d8d0b977f4e226756257eaf5fd5adfaf3e4e131ba818fd21e56ecde7.
No controlled atom-cache speedup is claimed.

## Strict in-place softmax screen and live complete-model checks (2026-10-06)

Atom-cache and accepted attention-trace work committed/pushed as 2acd294.
The strict operator follow-up shows a real difference from the old no-grad-only
screen: with inference_mode, strict deterministic algorithms and memory filling,
N1536/B128 out-of-place medians 13.418/13.793 ms become 9.918/10.291 ms.
All 32 checks pass bitwise; the shared owned_scores.py helper repeats the gain.
Evidence: provenance/openfold3_hip_softmax_inplace_strict_screen.json.
These are operator observations, not E2E/whole-model memory results.

Full-model v1 failed before worker creation (missing root import shim/native
source tree). Sessions 89239/26918 exited 1; no model/GPU trial occurred. Preserve
these failed output directories. Fresh v2 includes complete support trees and
current notices. Both supervisors are confirmed live:
- inplace-model-v2-n1000, HIP5, PID1823626, session38895, worker1823643.
- inplace-model-v2-n1536, HIP6, PID1823629, session58513, worker1823644.
Controls: /tmp/fireprot-inplace-model-v2/{n1000,n1536,profile}.json.
Logs: that root's n1000.log/n1536.log and each output's stdout.log/stderr.log.
Do not mutate copied worker trees, CLI/bridge, environment or published bundle.
These are concurrent numerical checks, not E2E timing. Poll known handles;
require all 11 original-native arrays and inspect final chunk_tuners and
experimental_owned_scores in operator-runtime.json before any benchmark.
Preparation/exclusion pins: provenance/openfold3_inplace_model_preparation.json.
No paired atom-cache or combined-original-Torch performance comparison yet.

Authorized ISA agent is finishing a read-only report with exact native symbols,
full hashes and descriptor evidence. Actual selected QK/PV use 16x16x4 FP32 MFMA;
A/B runtime pointer identity remains inferred, not traced. No new GPU kernel
implementation or speedup is established from this disassembly.

N1000 inplace-model-v2 completed; session38895 exited0. Supervisor succeeded,
all 11 original-native arrays are bitwise equal, 1,702 submitted in-place calls,
zero storage fallbacks; final module chunk tuners all1024. N1536 PID1823629 and
worker1823644 remain live; session58513. Current evidence is
provenance/openfold3_inplace_models.json; collector is
/tmp/fireprot-collect-inplace-models.py (run over SSH stdin), local capture
/tmp/fireprot-inplace-models.json. It compares a terminal successful N1536 run
against the original strict native NPZ before declaring acceptance.

The direct original-Torch versus combined optimization AB/BA/AB N1536 sequence
is PREPARED, NOT LAUNCHED. Control /tmp/fireprot-combined-native-paired-v1,
spec pairs-n1536.json SHA e1793fc125ba6841d9a018ab6c0b84d013fefcf7cae770ff5e258b0d2b72d826.
Worker has explicit pair/atom cache flags; baseline native/0/0, candidate
hip_fused/1/1 with the same source tree and owned score helper. Expected changes
include backend, both flags, profile hash and candidate bridge input. All run
IDs/outputs are new; HIP7. Once BOTH inplace model gates pass, all diagnostics
terminate and all GCDs are observed idle, run Rust benchmark-pairs with this spec
and a NEW destination under that control. Launch no other FireProt GPU work
throughout the sequence. Compare its native anchor to the historical native NPZ
as well before publishing. Current CLI/bridge/bundle/environment must stay fixed.
Preparation: provenance/openfold3_combined_native_paired_preparation.json.
Archive combined-native-paired-v1-preparation.tar.gz SHA
1b37119613a1ffc164491819d7ee53d37a771e3605f918f74084b18677c7e5ba.

Authorized ISA reviewer is finished/idle. Primary review corrected its shortened
symbol strings and mistaken PV reduction dimension: QK K=32, PV K=1536. Actual
ELF64 descriptor entry offsets and LDS sizes independently match both selected
preloaded bodies. Full corrected report: provenance/openfold3_native_mfma_review.json.
Archive native-mfma-review-v2.tar.gz SHA
ecffe69c8447537ed43ddcb875dadf621454d730657b8ac3f3f8df8948fdbfd2.
A/B runtime pointers and full K accumulation sequence remain unproven. Both
2acd294 and163d007 CI runs completed successfully. No plateau established.

## N1536 confidence mismatch; E2E candidate rejected (2026-10-06)

IMPORTANT: inplace-model-v2 N1536 is now terminal, session58513 exit0, but the
separate exact-array comparison fails. Coordinates match bitwise; five confidence
arrays differ (max1.90735e-5). Keep the bitwise gate unchanged. Evidence now records
numerical_rejected and the durable results archive. Runtime succeeded/accepted
only establishes finite output, not parity. Final tuner sizes512 except diffusion
conditioning1024; 5502 in-place submissions vs5592 earlier baseline. This difference
includes tuning and does not yet explain the confidence changes.
DO NOT LAUNCH /tmp/fireprot-combined-native-paired-v1: its candidate is rejected.
Its preparation evidence now marks not_launched_candidate_rejected.

New bounded diagnostic scope_softmax.py observes native aux_heads.forward and
can preserve out-of-place HIP there. Mode none = out-of-place everywhere; mode
trunk = in-place outside confidence. No arithmetic/cycle changes; still pair+atom
cache and strict inference. Scope return/exception restoration checked locally.
Fresh copied controls /tmp/fireprot-inplace-scope-v1, immutable while live:
- none-n1536.json + none-profile.json, HIP5, session79673.
- trunk-n1536.json + trunk-profile.json, HIP6, session9644.
- trunk-n1000.json + trunk-profile.json, HIP7, session7459.
Root logs have the same mode-nNNNN stem. Run IDs inplace-scope-v1-MODE-nNNNN.
All GCDs were observed idle before launch. No E2E benchmark is active. Require all
11 original-native arrays, final chunk_tuners and per-scope counters. If a guarded
candidate passes, investigate the confidence allocation/tuning difference and
prepare a NEW same-source E2E control; never reuse the rejected candidate spec.
Scope preparation: provenance/openfold3_inplace_scope_preparation.json, archive
inplace-scope-v1-preparation.tar.gz SHA
8e97a3dc5fa42148ec86ccccbf25ef403ab5b20ad5aac7becdebbfbcad56555d.

Scope supervisors confirmed live: none-N1536 PID1835931, trunk-N1536
PID1835934, trunk-N1000 PID1835940. All three have execution.json and live
processes. Five CPU storage/scope tests pass, including return/exception cleanup
and preexisting forward override restoration.

## Scope localization complete; preserve measured confidence chunk (2026-10-06)

All three scope trials now terminate successfully AND pass every original-native
array: none-N1536, trunk-N1536, trunk-N1000. Sessions79673/9644/7459 exited0.
provenance/openfold3_inplace_scope_models.json includes request/control pins and
its completed archive. N1536 none/trunk confidence chunk is256, unlike the rejected
all-in-place candidate's512. Other N1536 chunks match512 (conditioning1024).
N1000 chunks remain1024. This supports the allocation/tuning hypothesis; not yet
a controlled proof that chunk size alone explains the arithmetic difference.

Additional strict HIP0 operator screen covered actual observed N1536 B256/B512
calls: all16 synthetic checks pass. Session26417 exit0. It ran concurrently with
model diagnostics, so its timings are NOT speedup evidence. Source/results pins
are in provenance/openfold3_inplace_large_screen.json. The ordinary strict screen
now has an explicit --large option. No source changes to maintained HIP/C ABI.

New confidence_chunk.py keeps all eligible softmax calls in-place but caps the
fresh native confidence tuner at the measured256 ceiling only for N1536 FP32.
Native tuning still executes, smaller requests remain smaller, N1000 is unchanged.
It verifies chunk_utils/pairformer source hashes and rejects already-used tuners.
Three CPU tests pass shape/precision bounds, original callback execution,
return/exception restoration, reused-tuner and source rejection.

New frozen numerical controls /tmp/fireprot-inplace-chunk-v1:
- n1000.json + profile.json, HIP5, supervisor1855501, session35017.
- n1536.json + profile.json, HIP6, supervisor1855500, session10147.
Both confirmed live with execution.json. Root n1000.log/n1536.log and run IDs
inplace-chunk-v1-nNNNN. Do not mutate these workers or CLI/bridge/artifact/env.
All GCDs were observed idle before these launches. Only these two model trials
are now active; no E2E benchmark is running. Require all11 arrays against original
native references and inspect confidence_chunk_policy in operator-runtime.json.
Preparation: provenance/openfold3_inplace_chunk_preparation.json; archive SHA
103a68a08a899c680044c5f463d0cc077776cd774445f85729ceeebd30d7d4f0.

NEW original-Torch versus combined AB/BA/AB E2E control is PREPARED, NOT LAUNCHED:
/tmp/fireprot-combined-native-paired-v2/pairs-n1536.json. It uses the new chunk-cap
worker and NEW combined-native-v2-e2e-n1536-* output IDs on HIP7. Gate on BOTH
inplace-chunk-v1 numerical trials passing and all diagnostic GPU jobs terminal /
all GCDs idle. Then use Rust benchmark-pairs SPEC NEW_DESTINATION and launch no
other FireProt GPU work during the sequence. Before publication compare its first
native anchor against the historical native output too. V1 remains rejected;
do not accidentally execute that old spec. V2 source/archive/spec pins are in
provenance/openfold3_combined_native_paired_v2_preparation.json. No atom-cache /
in-place / direct combined E2E speedup is established yet.

Operational scope collector: /tmp/fireprot-collect-inplace-scope.py over SSH stdin;
local results /tmp/fireprot-inplace-scope-models.json. For new chunk trials adapt
/tmp/fireprot-collect-inplace-models.py to the new root/run IDs and include the
confidence_chunk_policy record; do not lose numerical_rejected status if any
array differs. If a model fails, retain its files and diagnose; no gate relaxation.
GitNexus debugging was used to inspect the local scope boundary and source; the
upstream tuner lives outside the graph and was read directly. Typed status showed
605e3ff up to date despite stale MCP cache messages.

## Exact all-scope reuse accepted; direct original-Torch E2E ACTIVE (2026-10-06)

Both inplace-chunk-v1 models now pass all11 original-native arrays. N1536 records
requested1024/bounded256/selected256 for confidence; N1000 remains1024 throughout.
All softmax calls remain eligible for in-place execution; zero storage fallbacks,
1702/5472 submissions. Pair and atom caches each record200 calls/199 hits/one miss.
Sessions35017/10147 exited0; both model supervisors exited. Archived evidence:
provenance/openfold3_inplace_chunk_models.json. Result archive SHA
abaa06cc8331cc2ff7bd3f1890472c1a75c1420cb57038dab29f45d55803cde7.
Collector /tmp/fireprot-collect-inplace-chunk.py and local capture
/tmp/fireprot-inplace-chunk-models.json remain available.

All74 paired preparation pins were verified and all eight GCDs were observed
idle. The direct original-Torch versus combined optimization sequence is now
RUNNING, Rust controller1868435, tool session2278:
/tmp/fireprot-combined-native-paired-v2/paired-n1536.
Its spec is /tmp/fireprot-combined-native-paired-v2/pairs-n1536.json;
stdout/stderr go to that root's paired.log. order.jsonl records baseline pair0
started (combined-native-v2-e2e-n1536-baseline-0). Poll the known controller and
journal; never restart just because SSH observation times out. Current run is
HIP7; do not launch ANY other FireProt GPU work, heavy DiamondHill builds or
library scans until the sequence terminates. Freeze CLI/bridge, copied worker
sources, published bundle and environment inventory. Source/launch pins updated
in provenance/openfold3_combined_native_paired_v2_preparation.json.
After completion require all three pairs/references/CIF identities, compare first
native anchor to historical native NPZ, archive controls and only then report a
direct combined E2E gain. No result/ratio is available yet.

Local source-only next experiment: experiments/mfma_qk/ contains HIP packed QK
probe, Rust-build boundary, pinned-bundle tensor probe and README. No HIP compile,
GPU test, numerical equivalence or performance claim. Python syntax passes.
Do not compile/run it on DiamondHill while E2E is active. The authorized ISA agent
completed small read-only follow-up and is idle. Manual/ASM cannot establish the
candidate lane mapping or full K order; single-instruction integer calibration
must precede K32 random/cancellation/subnormal/signedzero/nonfinite comparisons.
Repository dual_projection_kernel/project_qkv code suggests output query=lane%16,
key=4*(lane/16)+register for A=key/B=query, replacing the initially proposed
transposed mapping. This is still a hypothesis until the new calibration passes.
Native QK K32 is one DepthU32 tile; PV K1536 is48 tiles. The new probe touches no
maintained backend and does not fuse softmax or reduce global score traffic yet.
