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

Paired follow-up: controller1868435 remains live at elapsed43:09. Baseline0,
candidate0 and candidate1 are measured/verified; baseline1 is running. Both
completed candidates have5472 in-place submissions, zero storage fallbacks,
confidence256, and pair/atom caches each200calls/199hits/1miss/0fallback.
Do not publish a three-pair aggregate before the remaining runs finish.
Terminal-only collector prepared locally at
/tmp/fireprot-collect-combined-pairs-v2.py; syntax checked, not yet executed.
It checks all pairs, original-native anchor, caches and chunk policy before
emitting direct original-versus-combined evidence. Preserve the no-other-GPU /
no-heavy-remote-build rule while this controller is active.

The next MFMA probe now records the first native packed QK BMM with Kineto and
requires exact operand layouts plus one uniquely correlated GPU kernel. It
reuses and fingerprints experiments/attention_trace/probe.py. Four existing
CPU observer tests pass and Python syntax passes; HIP compilation, lane mapping,
GPU correlation and score equality remain untested. No new performance claim.

Local HIP compiler discovered and used without any GPU launch or DiamondHill
build. qk_probe.cpp compiles with the intended O3/C++17/shared/gfx90a flags.
Extracted code object has8 static16x16x4FP32 MFMA instructions,26VGPR/24SGPR,
wave64, zero reported spills/LDS/private segment. Evidence and local archive:
provenance/openfold3_mfma_qk_local_compile.json. This updates the earlier
"uncompiled" status only; lane mapping, GPU score equality and performance are
still untested. The direct local compiler diagnostic is not the Rust-published
DiamondHill build, and must not be deployed as such. Active E2E remains isolated.

Next independent candidate: experiments/hip_softmax/head16-adapter.patch widens
only an isolated adapter copy to head dimensions16/32 and records actual launches
by dimension. Runtime decisions show D16 long-input native fallbacks; the current
adapter explicitly accepts only32 while the softmax C ABI is score-only. Remaining
mask/layout guards may still reject calls; no frequency/performance claim yet.
Patch base SHA and gates are in HEAD16.md; git apply --check and patched-source
compilation pass locally. No maintained source or remote worker changed. Run both
N1000/N1536 strict exact-model gates only after current E2E finishes. Inspect tuner
changes if parity fails. The patch is Apache-2.0 and listed in THIRD_PARTY.md.

MFMA next-run archive prepared LOCALLY only:
/tmp/fireprot-mfma-qk-v1-prepared.tar.gz, SHA
13d46dc20bf6729e23b5f68dd81868b036cdda53eb9bc64da2388848bb456744.
It contains build/probe/kernel/trace-helper sources, notices, source hashes and
EXECUTION.md. Not deployed. Target future root /tmp/fireprot-mfma-qk-v1, fresh
cache/build/bundle/probe subdirectories; finish paired collection/archive first.

## Direct combined E2E complete; D16/MFMA numerical work active (2026-10-06)

combined-native-paired-v2 completed all six invocations and three comparisons.
E2E medians949.998639->633.711698 s,1.499102x (33.2934% less time); model medians
854.615744->538.416213 s,1.587277x. All11 arrays, repeated-anchor comparisons,
CIF identities and the historical-native anchor check pass. Every candidate has
5472 in-place launches, zero storage fallback, confidence256 and both caches
200calls/199hits/1miss/0fallback. Evidence:
provenance/openfold3_combined_native_e2e_n1536.json. This is a direct measurement,
not a product of previous ratios and not attribution to individual optimizations.
Archive combined-native-paired-v2-results.tar.gz in the durable conditioning-cache
directory includes controls and all six output directories; SHA
96f3d23db99c175f1eefa4a75175d0c3e4468a9f27982181d5b665fb5cd0bd77.
All six workers and controller1868435 exited; all eight GCDs observed idle.
The local SSH transport lingered after remote completion and was explicitly
closed (session2278 exit255). This is transport cleanup, not a model failure;
benchmark.json authoritatively records succeeded. Collector/results/archive:
/tmp/fireprot-collect-combined-pairs-v2.py,
/tmp/fireprot-combined-native-paired-v2-results.json,
/tmp/fireprot-combined-native-paired-v2-archive.json.

New D16 numerical controls /tmp/fireprot-head16-model-v1, fresh copied worker,
69 source/control pins. N1000 HIP5 PID1922873/session19996; N1536 HIP6
PID1922872/session44303. Both supervisors confirmed live. Root n1000.log/n1536.log,
request n1000.json/n1536.json, profile.json; run IDs head16-model-v1-nNNNN.
Preparation: provenance/openfold3_head16_model_preparation.json, durable archive
head16-model-v1-preparation.tar.gz SHA
b3ada9b202e1f11146ea6a6112ebec6017eda16b6ef1ff5e048d9c2981c54159.
Collector /tmp/fireprot-collect-head16-models.py; require D16 submissions and all11
original-native arrays at both lengths. All production artifacts stay unchanged.

MFMA archive deployed with all source hashes verified to /tmp/fireprot-mfma-qk-v1.
Rust build and publication succeed; build session4655 and publish53375 exited0.
Bundle /tmp/fireprot-mfma-qk-v1/bundle, manifest
e2c86aa1434b7371070e528a55d7a601151fc3064e31cbbb93477d1c9e102f38;
library SHA e68e3a765f8d1bb31965a1e3261d77498dd0052b5664bfdc1c2ffb23a64ae27d.
Actual remote code-object inspection matches local counts (8 static MFMA,
26VGPR/24SGPR, wave64, zero reported spills/LDS/private segment). Numeric probe
launched on HIP7, session31330; root probe.log and probe/probe.json. Do not rerun
into existing directories. Integer lane calibration must precede full K32 checks.
These parallel jobs are numerical diagnostics, not E2E/performance measurements.

D16 N1000 completes, session19996 exit0; all11 original-native arrays exact.
264 D16 plus1702 D32 submitted launches, zero storage fallback, all chunks1024.
N1536 PID1922872/session44303 remains live at elapsed7:03. Partial evidence:
provenance/openfold3_head16_models.json; local /tmp/fireprot-head16-models.json.
Refresh using /tmp/fireprot-collect-head16-models.py over SSH stdin once terminal,
then archive both outcomes. No D16 E2E result yet.

MFMA initial probe31330 exits1: integer lane calibration passes, but standalone
Torch default hipBLASLt differs from OpenFold3's explicit cublas/hipBLAS preference.
Its first BMM also includes a uniquely correlated runtime initialization memset,
so the initial one-total-kernel observation gate fails before score comparison.
Upstream policy is openfold3/entry_points/import_utils.py::_configure_torch_backend.
Corrected local probe.py sets that same BLAS preference and separately retains
the observed initialization memset; unknown/ambiguous kernels still fail. Deployed
as NEW /tmp/fireprot-mfma-qk-v1/probe-v2.py, output probe-v2/, log probe-v2.log.
Original source/results remain immutable. Session32847 exits1: lane calibration
passes and actual native QK name exactly matches the accepted model trace, but
random K32 scores differ at193,776,224 of301,989,888 elements. Later synthetic
cases are unexecuted; this candidate is numerically rejected, never integrated.
Evidence: provenance/openfold3_mfma_qk_probe.json, archive mfma-qk-v1-results.tar.gz
SHA f7012b71961f6e8ec664ca59d12e4b69f91c7010d4362675c766284928b1ed36.
Local full captures /tmp/fireprot-mfma-qk-results.json and
/tmp/fireprot-mfma-qk-probe-v2.json. Authorized cdna2_isa_review agent is now
analyzing native K32 fragment/input order read-only using local ASM and these
new findings; no GPU work or source changes delegated. Next possible diagnostic:
test K4-fragment permutations against a small tile sliced from the full native
reference, then require full K32 equality before any fusion. Cause not yet proven.

D16 N1536 now completes too, session44303 exit0, all11 arrays exact. It records
400 D16/2708 D32 submissions, zero storage fallback, main template/MSA/pairformer
chunks1024 (previous32-only512), conditioning1024 and confidence256. Both caches
still200calls/199hits/1miss/0fallback. Completed evidence/archived outputs are now
in provenance/openfold3_head16_models.json; preparation status updated terminal.
All launched FireProt model/probe jobs are terminal; no paired timing is active.
Next: prepare a fresh same-source
incremental D16 on/off N1536 comparison over the accepted combined backend.
Do not multiply that future ratio into the independently measured1.499102x.
Before starting another long isolated timing window, a short MFMA diagnostic
could compare sparse-K and dense-random16x16 candidate tiles to slices of the
full native B128/N1536 output, retaining native GEMM shape and backend. Sparse
exact inputs can check K indexing, while random ULP/absolute-error summaries
can distinguish indexing-sized errors from rounding-sized ones without accepting
either. The authorized ISA agent is still tracing LDS/global K mapping.

## MFMA K grouping corrected; D16 incremental E2E ACTIVE (2026-10-06)

Diagnostic scripts experiments/mfma_qk/diagnose.py and reorder_screen.py run on
HIP7, sessions1288/99096 both exit0. Full native GEMM stays B128/N1536/K32; small
candidate tiles use retained reference slices. Single-K covers all32 positions
and passes. Dense tile333/512 differs (max abs2.98e-8); first-K4 tile128/512;
adjacent cancellation native2/candidate1, absolute error1.0 (not1ULP). Testing
four K-order hypotheses on all five tile cases finds only even_odd_k8 exact.
The authorized ISA agent independently narrowed register/LDS issue order and
suggested this grouping, while explicitly not proving every lane/wave address.
Agent is finished/idle. Avoid the corrected earlier mistaken "1 ULP" wording.

qk_probe.cpp now uses k=(fragment/2)*8+2*k_lane+(fragment%2) for full K32;
raw integer calibration retains contiguous k_lane. Fresh source root
/tmp/fireprot-mfma-qk-v2, no modifications to v1. Rust build84096 and publish65686
exit0. New manifest a42b8b1150ccde3c3041e190b82baac6ecf071c1df8a9ee991b0bc0179dc3bf6,
bundle /tmp/fireprot-mfma-qk-v2/bundle; library
882621cd8a9d3ec004520f11b454eab279e6c060d15e8d8a6c6c182cc84f76d1.
Full probe1420 exits0: lane calibration plus ALL five packed B128/N1536/K32
cases pass bitwise (301,989,888 scores each). Actual native GEMM name exactly
matches the model trace. Evidence provenance/openfold3_mfma_qk_v2_probe.json;
local capture /tmp/fireprot-mfma-qk-v2-results.json. Durable archive
mfma-qk-v2-results.tar.gz SHA
087f3e3d3afb38856b797fba53194dd587ff68d24f8df382f9e14e795ad1fb48.
Archive includes diagnosis/order screen, build/bundle, sources and full probe.
No performance/attention/model acceptance; original strides and other batch
sizes need gates. Next local work may implement QK/softmax fusion, but no remote
build or GPU diagnostic while the new benchmark below is active.

Head16 same-source control prepared using experiments/hip_softmax/head16-control.patch
after the accepted adapter patch. FIREPROT_SOFTMAX_HEAD16=0/1 selects32-only vs16/32;
both arms retain all caches, in-place softmax, confidence ceiling and native tuners.
All76 preparation pins checked and all eight GCDs idle before launch.
ACTIVE /tmp/fireprot-head16-paired-v1/paired-n1536, controller1940816, session4919.
Spec root pairs-n1536.json, root paired.log; first baseline started. Requests use
head16-e2e-n1536-{baseline,candidate}-{0,1,2}, HIP7/PCI93, AB/BA/AB. Freeze worker,
CLI/bridge, bundle and environment. No other FireProt GPU work or heavy remote
build/library scan until terminal. Evidence:
provenance/openfold3_head16_paired_preparation.json; preparation archive SHA
215494dd2749f7e5332289f9d456fe61adda910eb5967c0926f81bf54d797b71.
Local preparation script /tmp/fireprot-prepare-head16-pairs.py and capture
/tmp/fireprot-head16-paired-prepared.json. After completion collect all paired /
reference / CIF gates and original-native anchor, archive, then report incremental
gain over the accepted combination. Do not multiply it into the1.499102x direct
original-Torch result. No plateau or serious blocker established.

## Local fusion prototype while D16 timing runs (2026-10-06)

experiments/mfma_qk/fused_qk_softmax.cpp and build_fusion.py implement a separate
packed N1536/K32 QK-plus-softmax prototype. No remote build or GPU run. Mode0
writes raw scores after a block/LDS transpose; mode1 includes native-order
mask/bias additions and the maintained32-slot wave softmax. Six256-column bands,
16 query rows,16 wave64 waves/1024 threads,16KiB LDS with a column XOR swizzle.
Compute/transpose each band immediately instead of retaining all six old tiles
alongside row values. QK uses the validated even/odd K8 order; PV remains native.
Pointer storage/nonalias conditions need a future tensor boundary and test harness.
FUSION.md records exact API, outstanding operator/model gates and layout limits.

Local HIP7.14 compilation passes. Softmax mode:70VGPR/28SGPR; raw score mode:
56VGPR/16SGPR. Both have zero reported spills/scratch,48 static MFMA instructions,
11 static barriers,16KiB LDS and max workgroup1024. Compiler occupancy remarks
estimate7/8waves per SIMD respectively; this is NOT measured residency or proof
of launch feasibility. Evidence provenance/openfold3_qk_softmax_local_compile.json;
local archive outputs/experiments/qk-softmax-local-20261006/compile-v1.tar.gz SHA
d208bfceaae9d1f0f7e0444e76365e1f6882c7bcc905b46e04c716a5d3f879b5.
No fusion numerical/performance acceptance or model integration. Next local work
can prepare the operator harness; do not run it remotely until timing finishes.

D16 controller1940816/session4919 is still live at elapsed17:04. Baseline0
measured/verified; candidate0 running. All earlier source/artifact freeze and
no-other-GPU/heavy-remote-build restrictions remain in force. Do not infer a
speedup from journal offsets or publish an aggregate before three pairs complete.

Fusion numerical harness prepared locally: experiments/mfma_qk/probe_fusion.py.
Default outerB32/BH128/N1536/K32, optional independentB128/256/512 screens.
Raw QK/LDS then ordered-add/softmax checks, all output bits, both bias layouts,
broadcast/per-outer/fully-masked/additive masks and adversarial inputs. Captures
actual native full-shape QK correlation. Fresh output storage and dense-input
restrictions guard experimental pointer calls. Syntax, wrong-environment report
and four invalid/non-GPU input rejection checks pass locally. No GPU run or
remote deployment; current D16 paired controller1940816 remains live (24:49
at latest observation, third invocation candidate1). No new performance claim.

Pending fusion v1 package is now prepared locally (NOT deployed):
/tmp/fireprot-qk-softmax-v1-prepared.tar.gz, SHA
b378a0fb730d75c0b8ae3274a78865c4727a818021a6e0f138908887b8737cd7.
Preparation receipt /tmp/fireprot-qk-softmax-v1-preparation.json; archive has
sources.json and EXECUTION.md. Future remote root /tmp/fireprot-qk-softmax-v1.
Require all21 numerical records (5 raw,16 softmax) at defaultouterB32; other
batches independently gated. Finish/collect/archive D16 pairs first.
Terminal-only collector prepared /tmp/fireprot-collect-head16-pairs.py, syntax
checked but unexecuted; both arms hip_fused, expected D16counts0/400 and main
chunks512/1024, same caches/owned storage. Never multiply its incremental ratio
into the direct original-Torch1.499102x result.

Separate local LDS alternative in experiments/mfma_qk/lds-swizzle.patch; base
fusion source unchanged. Base ASM uses scalar ds_write_b32. Full address
bijection and producer/consumer coordinates pass CPU enumeration. Modulo32
address model lowers max producer addresses per bank for contiguous32-lane
groups4->1, consumer remains1; no hardware/performance proof. Local compile
passes (full70VGPR,raw57VGPR,zero spills/scratch,16KiB LDS).
provenance/openfold3_qk_softmax_lds_candidate.json pins patch/base/localarchive.
Validate base fusion first, then independent patched copy. Authorized ISA agent
resumed read-only bank/address review; no remote/GPU work delegated.
Latest D16 observation: controller1940816 live29:28; first3 invocations
measured/verified; baseline1 (fourth invocation) running.

ISA agent finished read-only fusion LDS review. Confirms scalar B32 base stores,
barriers/waits and address-model reasoning; manual PDF p16/p94 gives32x4B banks,
p188 DS_WRITE_B128 four consecutive dwords. Precise lane/cycle arbitration not
established, so do not claim measured bank-conflict or throughput improvement.

New separate lds-vector-store.patch applies AFTER lds-swizzle.patch, never to
active source. Four-way switch prototype v3 compiles full100VGPR/28SGPR and12
static EXEC-zero branches; two conditional permutations v4 compiles full68VGPR/
32SGPR, raw53VGPR/16SGPR and zero such branches. Both have six static B128 stores,
zero scalar stage writes,24 scalar reads,48MFMA,11barriers,16KiB LDS,zero spills.
All4096 destination/value associations and16-byte alignments checked on CPU;
sequential patch application exactly reproduces v4 compiled source.
provenance/openfold3_qk_softmax_vector_candidate.json retains BOTH variants and
localarchive. No GPU acceptance/performance; static-resource choice only.
No modifications to frozen v1 package or D16 worker; remote GPU isolation still
required until paired controller1940816 terminates and collection/archive finish.

Fusion harness now additionally acceptsouterB64 and explicit B1024 host reference.
--reference-batch32 preserves full nativeQK shape, moves native scores to CPU,
launches full candidate, then compares all bits with bounded complete-row Torch
softmax references. Report distinguishes this from full-shape native softmax;
require B32 cross-check first and full-model gates later. SingleB1024 score36GiB;
default multi-full-GPU mode rejectsB>512. Host available memory preflight added.
CPU checks pass tail chunks/mask/transpose/NaNs/signedzero/one-bit mismatch counts;
no GPU validation. Earlier preparedv1 archive remains immutable and contains old
full-GPU-only probe; updated host mode needs a freshly pinned package.
Latest paired controller1940816 live41:46, fifth invocationbaseline2 running;
firstfour measured/verified. ISA agent finished/idle. No other GPU work launched.

Seven CPU staged-reference regression tests pass in tests/operators/test_openfold3_fusion_reference.py.
UPDATED local package (NOT deployed): /tmp/fireprot-qk-softmax-v1-prepared-v2.tar.gz
SHA faa481c0c2d69c6184d2df707255d8ecdf4eee1ba9369f94fe72e89e7d54c37a; receipt /tmp/fireprot-qk-softmax-v1-preparation-v2.json.
Supersedes earlier prepared archive for future execution; fresh remote root stays
/tmp/fireprot-qk-softmax-v1. Require ordinary B32 screen then stagedB32 cross-check
before large staged screens. All21 records each; full-model gates still required.

## D16 incremental E2E complete and archived (2026-10-06)

All six invocations/three pairs succeed. E2E medians628.688752332->540.848062898s,
1.162412876x,13.972% less time; model534.757986001->446.844362775s,1.196743275x.
All11 arrays and CIF identities/repeats pass, plus firstbaseline historical
original-native anchor. Every baselineD16=0/D32=5472/mainchunks512; candidate
D16=400/D32=2708/mainchunks1024. Confidence256, conditioning1024, zero storage
fallbacks, both caches200calls/199hits/1miss/0fallback. Incremental over combined
backend; NOT direct original-Torch speedup and do not multiply into1.499102x.
provenance/openfold3_head16_e2e_n1536.json records full results/limits.
Durable archive head16-paired-v1-results.tar.gz SHA
b079ac5dfc8b48548bd0e7d8b18f151246fa14a182e46762bfda846111642915,
444200847bytes, contains controls and all6 outputs. Rootcontrolcollected.json
also retains collector output. Local /tmp/fireprot-head16-paired-results.json,
/tmp/fireprot-head16-paired-archive.json and collector/archive scripts retained.

Controller1940816 and workers1940927,1948184,1955148,1962388,1970020,1977908
confirmed absent; all8GCDs observed idle. Lingering localSSH2239518 closed only
after independent terminal checks; session4919 exit255 is transport cleanup,
not a benchmark failure (benchmark.json=succeeded). No timing window is active.
Next: deploy updated fusion package /tmp/fireprot-qk-softmax-v1-prepared-v2.tar.gz
SHAfaa481c0c2d69c6184d2df707255d8ecdf4eee1ba9369f94fe72e89e7d54c37a
to NEW remote /tmp/fireprot-qk-softmax-v1, verify pins, Rustbuild/publish, and
ordinary B32 GPU numerical screen before any candidate or staged comparisons.

## Fusion base GPU validation active (2026-10-06)

Updated package deployed to NEW /tmp/fireprot-qk-softmax-v1,10 source/plan pins
verified. Rustbuild76335 andpublish40881 exited0. Bundle manifest
122d70650b679d09db2b61a8d815985f26a5c59fcf817f2edb04a5c3584cbc09;
library219e6abfef41252901c77c39830604cb7a288d0c6175496ebff9a6eba788aaaf.
Actual generated code-object inspection matches24scalar LDS writes/24reads/
48MFMA/11barriers each mode, rootinspection/ contains artifacts. No production
bundle changed. B32 full session9761 andstaged session30089 exit0,21/21 bitwise
records each301989888 elements. Full/staged record coverage equal, nativeQK
symbol matches earlier acceptedMFMA v2. Local captures
/tmp/fireprot-qk-softmax-v1-probe-b32.json and*-probe-b32-staged.json.

Larger sequential screen44862: B64/B128/B256 all21 exact. B512 fails with CUDA
OOM at count_nonzero in raw comparison, requesting36GiB temp with22.04GiB free;
zero comparison records, NOT numerical mismatch. Session44862 exited1; B1024
not attempted. Preserve originalrootprobe-b512/ andlog. Fixed local probe now
counts all bits in <=16 outer-batch slices, leaving native arithmetic/shapes
unchanged. Eight CPU tests pass including actual reduction-size/tail assertions.
Newremoteprobe /tmp/fireprot-qk-softmax-v1/probe_fusion_v2.py SHA
fa104615e1cdf81dc9246209574d6b81746f592f95138e2b9ac871d83e2290c4.
ACTIVE session23572 onHIP7: sequentialprobe-v2-b32,probe-v2-b32-staged,
probe-v2-b512,probe-v2-b1024. Each gated on21 exactrecords; last usesreference32.
Do not restart or overwrite. No performance test active, these are diagnostics.

Local-only candidate archives (not deployed) /tmp/fireprot-qk-softmax-lds-v2-prepared.tar.gz
SHAdfb319b5045367e668a90b20f26181a79152cb84ee8a0312c71e41076147f358 and
/tmp/fireprot-qk-softmax-lds-v4-prepared.tar.gz
SHA9b28694f239f88a3d158f8fd67064443bcb5d220c09b907e584dfff15261a573.
These contain pre-count-fix harness; replace by freshly pinned preparation or
add the corrected probe under a new verified path before testinglarge batches.
Each candidate source was patched in temporarycopy and matches localcompiled
v2/v4 source exactly. First finish/collect/archive basefusion results, then
build/test swizzle andvector independently. Fullmodel/performance stillunproven.

Fusion base sequence23572 now exits0: correctedB32/full+staged, B512/full and
B1024/staged32 each21 exactrecords. All sixouterbatch sizes32,64,128,256,512,1024
are covered;9 successful invocations including B32 repeats. No active GPU jobs.
Complete evidence provenance/openfold3_qk_softmax_probe.json; collector
/tmp/fireprot-collect-qk-softmax-v1.py andcapture/tmp/fireprot-qk-softmax-v1-results.json.
Durable archive qk-softmax-v1-results.tar.gz SHA
63aa4db76ccc423f828fb890240400ed555d8b9c83fcc35f2413a9829c79e417, size1567299 bytes.
Includes both probes, retainedB512countOOM, allsuccesses, native traces, sources,
Rustcache/build/bundle/inspection. Only packedsyntheticbasekernel accepted; no
PV/originalstrides/fullmodel/performance or LDSvariant GPU acceptance yet.
Next build/test independent LDSv2/v4 with corrected probe; earlier localprepared
variant archives need updated probe addition/pinning before large screens.

## LDS variants accepted; complete attention timing ACTIVE (2026-10-06)

Fresh corrected-probe preparations forlds-v2/v4 deployed from *prepared-v2.tar.gz.
Hashesac54c1d10fc3db0d203570559294ab42d160c1975a7a5ae9ddd0f5fcdcccf4a1 /
278a8d7d9e872be6d0e649937a283bcf82388f5110daa9fb44bf050382201593.
Initialdriver84670 stopped beforeGPU at local-vs-remoteVGPR assertion. Source
hashes match; remoteactual swizzlefull66VGPR(notlocal70),vectorfull66(not68),
raw57/53; zero spills/scratch16KiB LDS. Resumeddriver25047 exits0: BOTHvariants
all sixbatch sizes32,64,128,256,512,1024 each21 bitwise records. No arithmetic
or tolerance relaxation; local register estimates are not numerical contracts.
Actualcodeobjects retain48MFMA/11barriers,24B32vs6B128 stage stores.
Bundles /tmp/fireprot-qk-softmax-lds-v2/bundle manifest
3be6b6256d852f0ac64e81e7bc3aff704281ccc4215ce7b39091b82999b4edea;
/tmp/fireprot-qk-softmax-lds-v4/bundle manifest
44443e24b183e48b86b9fa9daf043fdd00df8249264efdd8826dbb085788d492.
Evidence provenance/openfold3_qk_softmax_lds_probes.json; full localcapture
/tmp/fireprot-qk-softmax-lds-results.json includes durablearchivepin. All diagnostics
terminated before new timing; archive completed; all8GCDs observedidle.

NEW benchmark root /tmp/fireprot-qk-attention-bench-v1, sourcearchive SHA
75c40e86e4304b6a4d59f2747f1326b4efc9c05f527cd3719099cafeac9072da.
Controls controls-b32.json/controls-b256.json pin all3 fusion bundles and exact
batch screens plus maintainedsoftmax. Sequential driver session30943 is ACTIVE
onHIP7: b32/ then b256/, logs b32.log/b256.log; localdriverlog
/tmp/fireprot-qk-attention-bench-v1-driver.log. NO other GPU jobs/heavy remote
builds/libraryscans until it terminates. Script experiments/mfma_qk/benchmark_attention.py
includes real head-lastQ/KTpacking, allocations/filling, bias conversion where
used, andnativePV. Fivearms native/splitHIP/base/swizzle/vector;2biaslayouts;
10rotated/reversedrounds,100 measuredcalls perbatch; everyoutput exact aftertiming.
B32/B256 goldscreen is mandatory; no performance/model/E2E acceptance yet.
CPUsyntax/orderbalance/smallhead-lastpacking correspondence checks pass.

## Complete attention fusion rejected for promotion; GPU trace complete (2026-10-06)

Benchmark30943 exited0; all200 measured outputs exact across B32/B256, two bias
layouts, five arms,10 samples per arm/layout. B256 contiguous split43.482ms,
fusion base72.036/swizzle71.852/vector71.989ms; transposed split43.785ms,
fusion86.363/86.107/87.079ms. No meaningful LDS gain established. Fusion is NOT
promoted and no model/E2E claim follows. Evidence openfold3_qk_attention_benchmark.json.
Archive qk-attention-bench-v1-results.tar.gz SHA
86a6c3be25ec0c55f4de7521be62662f4d534dc84cd62a284ab7a20f8737d428.

New diagnostic root /tmp/fireprot-qk-attention-profile-v1, session24466 exits0;
all8 profiled outputs exact. B256 fused core50.47–50.70ms contiguous vs nativeQK
9.95ms+splitsoftmax19.17ms; nativePV11.80–11.96ms. Transposed fused64.85–65.48ms.
Torch peak increments split9.375GiB/fusion9.750GiB. Single warmed profiled calls,
not statistical latency/E2E; Torch memory only. Evidence openfold3_qk_attention_profile.json
pins full traces/archive. No GPU job remains active; no benchmark lock.

Authorized ISA agent finished read-only review. Actualremotev4 has66VGPR and
wave index/address bases remain VGPR; no readfirstlane in ASM. Candidate explicit
__builtin_amdgcn_readfirstlane(threadIdx.x >> 6) is valid only for current fixed
1024x1x1/wave64 launch; resource/performance benefit UNPROVEN. Next local compile
can test this address-only change without reordering MFMA arithmetic. Partial
key-load overlap exists; future fragment prefetch is another untested option.
All new variants require independent full numerical gates before timing.

## Wave scalar candidate: exact and faster, still below split HIP (2026-10-06)

Previous goal turn was progress: complete attention and GPU traces archived,
committed/pushed34b992d. New wave-scalar.patch applies AFTER lds-swizzle.patch
and lds-vector-store.patch. Explicit readfirstlane(threadIdx.x>>6) under fixed
1024x1x1/wave64 retains all arithmetic order. Local default-flags62VGPR/35SGPR,
raw51/17; actual DiamondHill60/35,raw51/17. No spills,16KiB LDS,48MFMA,
11barriers,6B128 stores per mode. Local accidental -ffp-contract=off diagnostic
was superseded by retained default-flags compilation; remote uses Rust standard
recipe. Local evidence openfold3_qk_softmax_wave_candidate.json.

Root /tmp/fireprot-qk-softmax-wave-v5; driver83047 exits0. B32/B256 each21 exact
packed records. Bundle manifest f4cca2c8fde0a458b9bc30c6c889cee60da774f3b1cb727ced83baa2567995af.
Numerical archive qk-softmax-wave-v5-results.tar.gz SHA
a1b67804b9d142bb86278331819d47a76b24b062f8d1d1b9c99e7c7f21826cf0.
Evidence openfold3_qk_softmax_wave_probe.json. No other batches/fullmodel gate.

Fresh root /tmp/fireprot-qk-attention-wave-v5; driver72942 exits0. All200calls
bitwise acrossB32/B256/two biaslayouts/fivearms. B256contiguous vector71.970ms
->wave54.934ms (23.67%less), split43.491ms; transposed87.037->75.259ms,
split43.770ms. B32contiguous9.140->6.899ms, split5.528ms. No model promotion.
Evidence openfold3_qk_attention_wave_benchmark.json; archive
qk-attention-wave-v5-results.tar.gz SHA
27fca07553628c58771a2e46e07c9e46f8b359c84cea394c643784076a2a02ac.
All diagnostics/timing terminal; no GPU job active. Next hypothesis: defer
mask/bias loads until after the six QK/LDS bands, keeping per-element ordered
adds and every MFMA/softmax operation unchanged; reduce load waits between
barriers. Not implemented or tested yet. Wave scalar alone does not establish
actual occupancy as cause, and no new E2E result follows.

## Deferred additions trial complete: layout-specific regression (2026-10-06)

Previous turn made progress and pushed53d32eb (wave scalar improvement).
New deferred-adds.patch applies after swizzle/vector/wave patches, moves
score+mask then +bias from each QK/LDS band to a24-slot loop after all bands.
Per-element math/padding/reduction order unchanged. Patch chain exactly verified.
Local/remote60VGPR34SGPR softmax,raw51/17,16KiB LDS/no spills/48MFMA/11barriers.
Root /tmp/fireprot-qk-softmax-deferred-v6, driver89370 exits0, B32/B256 each21
exact records. Bundle manifest06db6244adf915813bba79bec85b98ff03cfd42c4924231f8f2bb6583d717ebb.
Numerical archive qk-softmax-deferred-v6-results.tar.gz SHA
c73edc9e17d28b0a03d4f20711229c7d7cd4a6b930db81eb7f49b226a2ee5977.
Evidence openfold3_qk_softmax_deferred_candidate.json / *_deferred_probe.json.

Complete-attention root /tmp/fireprot-qk-attention-deferred-v6, session1264
exits0. All200 measured calls exact. B256contiguous wave55.057->deferred52.187ms,
split43.498ms; transposed75.260->117.302ms,split43.791ms. B32samepattern.
Not promoted; no fullmodel/E2E result. Evidence
openfold3_qk_attention_deferred_benchmark.json pins fullarchive. Alljobs terminal.

Next hypothesis: bias-cache reuse through block scheduling on wave-v5 (without
deferred adds). Original x=querytile96,y=outer*4+head. Flatten block index and
interleave8outer batches for each head/querytile, mirroring maintainedsoftmax
reuse policy. For linear index within group8*4*96, outer=group*8+linear%8,
head_query=(linear%3072)/8, head=head_query/96, querytile=head_query%96.
Only when batch divisible8; otherwise preserve original mapping. Prove bijection
and complete coverage on CPU, then independent Rustbuild/numerical/timing gates.
Not implemented yet; no cache-hit or speedup claim. This may address the bias
layout sensitivity while keeping Q/K working sets of8matrices within nearby
blocks. Do not assume scheduling order guarantees actual cache residency.

## Bias reuse scheduling trial complete (2026-10-06)

Previous turn made progress and pushedbedbf29 (deferred layout regression).
New bias-reuse.patch applies to wave-v5 afterswizzle/vector/wave patches,
WITHOUT deferred-adds. Interleaves8outer matrices perhead/querytile whenbatch
is divisible8, originalmapping otherwise. CPUbijection verifiedB1,7,8,9,32,64,
128,256,512,1024,1536; patchchain matches testedsource. Local62VGPR33SGPR,
actualremote60/33,raw51/20,16KiB LDS/no spills/48MFMA/11barriers/6B128 stores.
Root /tmp/fireprot-qk-softmax-reuse-v7, driver54506 exits0; B32/B256each21exact.
Bundle manifestd1018508012824a7c8f7b7f325408e05401b9c05425e0dc68308df370085838a.
Numericalarchive qk-softmax-reuse-v7-results.tar.gz SHA
0ea7b2ee4d2467a352da4dd8dad0ee5d88c185a9bd6cb946bb9c42eaee0e6db1.
Evidence openfold3_qk_softmax_reuse_candidate.json / *_reuse_probe.json.

Fresh completeattention root /tmp/fireprot-qk-attention-reuse-v7, session56730
exits0. All200calls exact. B256contiguous wave55.025->reuse52.957ms,
split43.509ms; transposed75.276->73.217ms,split43.782ms. B32samepositivepattern.
Actual cachehit rate unmeasured; not proof of causalcachebenefit. Not promoted;
no fullmodel/E2E gain. Evidence openfold3_qk_attention_reuse_benchmark.json pins
archive. No GPU job active. Next hypothesis: prefetch one/two future Kfragments
while preserving exact MFMA K-order, on bestappropriate source. Compareactual
registerresources (currently60VGPR,close64threshold), allnumericgates then
completeattention against current splitHIP. Also possible later combine reuse
with deferredadds ONLY after accepting layout-specific scheduling or explicit
biascontiguous conversion; that conversion cost must stay inside timing.
No plateau yet demonstrated; existing kernel alternatives remain below split.

## Natural grouped key prefetch accepted at operator scope (2026-10-06)

Previous turn made progress and pushed1ee7c68 (bias reuse scheduling).
Local variants8/9 force four key operands through empty inlineasm register
barrier (without/with memory clobber),120VGPR; variant10 forces two,112VGPR.
All three rejected on static resource grounds beforeGPU, no numeric/performance
claim. Their source/libs/ASM are archived with selectedvariant11: four local
key operands followed by four MFMAs, NOcompilerbarrier. K-order unchanged.
key-prefetch.patch applies AFTERswizzle/vector/wave/bias-reuse, NOdeferredadds.
Patch chain exactlyverified. Local/remote softmax58VGPR33SGPR, raw51/20;
16KiB LDS/no spills/48MFMA/11barriers. LocalfirstfourMFMAs show aheadloads and
vmcnt3/2/1/0. Actual scheduler remainscompiler-owned. Evidence
openfold3_qk_softmax_prefetch_candidate.json includes all4localvariants.

Root /tmp/fireprot-qk-softmax-prefetch-v11, session69788 exits0, B32/B256each21
exactrecords. Bundlemanifestb682ffbc845920ae2d3f675f793ab873b26a5d5b3ec80d50fcf076442d5372ce.
Numericalarchive qk-softmax-prefetch-v11-results.tar.gz SHA
ea59351e06102375e2c77720021a9b0eebbf26e6892b685543ccb326519e12fe.
Evidence openfold3_qk_softmax_prefetch_probe.json. Otherbatches/modelgates pending.

Root /tmp/fireprot-qk-attention-prefetch-v11, session73616 exits0, all200 timed
outputs exact. B256contiguous reuse52.962->prefetch48.892ms,split43.487ms;
transposed73.407->65.693ms,split43.800ms. B32contiguous6.621->6.209ms,
transposed9.225->8.331ms. No promotion/E2Egain. Evidence
openfold3_qk_attention_prefetch_benchmark.json pins fullarchive. No activeGPUjob.
Next: updatedkerneltrace forprefetch tosplitremainingcore/packing/fill/PV costs;
then test combining deferredadds with biascontiguous materialization inside
completeattentiontiming, or deeper originalQKV-layout support ifpacking matters.
Any bias conversion cost must be included, and all math/output gates retained.
No plateau: this step stillgave7.69%/10.51% operatorincrement, but the existing
splitHIP remainsfaster and modeldefaultmustnotchange.

Updated prefetchetrace COMPLETED thisturn (supersedes preceding next-trace plan):
root /tmp/fireprot-qk-attention-profile-prefetch-v11, session47358 exits0;
8outputs exact. Contiguousprefetch core27.522ms vs splitQK9.942+softmax19.141ms.
PV11.8–12.0ms. The gap is now largely extra allocationfilling: large
[256,4,1536,1536] score aten::fill_ withNaN maps to8GPUkernels totaling6.970744ms,
externalid1555 in contiguous-prefetch trace. Q/Kpacking not dominant; no need
to assume layoutkernel rewrite will close the gap. Evidence
openfold3_qk_attention_prefetch_profile.json retains fillcorrelationanalysis.
Archive qk-attention-profile-prefetch-v11-results.tar.gz SHA
aec16f406b054cec3b1ab1c7629846b92c05ce1938806253abeb81d5d0177ced.
Local rawtrace /tmp/fireprot-prefetch-contiguous-trace.json hash verified.
Nextpriority: bounded private initializedscore workspace reuse as an explicit
experimental treatment, strictdet/fill flags unchanged. Must prove fullwrite,
stream/lifetime safety, include firstallocation coldcost and retainedmemory,
never expose or mutate a previouspublicresult. Needs tensor boundary/CPUreject
and GPU repeated-distinct-input gates, then completeattentiontiming. Model
memory/tuner/workspace lifecycle acceptance required beforeintegration. No
workspace policy exists yet; do not claim equivalentcacheconditions or E2Egain.

## Initialized private workspace experiment passes (2026-10-06)

Previous turn made progress and pushed3520c63 (prefetch plus fill trace).
New workspace_attention.py subclasses unchangedPackedFusion only for artifact
loading; owns bounded B32/B256 fixedshape scores (1.125/9GiB), explicitbytebudget,
constructor-thread/nonblockinglock/firststream+device binding, no graphcapture,
strictinfer/determinism/fill enforced. Ordinarytorch.empty/NaNfill on firstuse;
full HIPoverwrite thereafter, nativePV resultcannotaliasscores. QueuedPV and
nextoverwrite orderedonsamestream. Submissionerrors closeinstance; close
synchronizes and releases tensor. No framework/model integration. Biascontiguous
option materializesinsidecall. Six CPUrejectiontests pass.

New benchmark_workspace.py gates3seeds101/102/103×2layouts×2cachedarms perbatch,
includingfullymaskedouter0 andprior-scoreNaN/finitepoison; compares ALLprobability
bits plusPVoutput.10priorpublicresultimmutabilitychecks perbatch. Cross-thread/
stream attempts rejectwithoutsubmission/reallocation. Firstallocationcalls
recorded separately; two poolsremainresidentduringALLtimingarms. Fivearmsnative/
split/freshfusion/workspace/workspace+biascontiguous;100timedcalls eachbatch,
allbits exact. Everypool1allocation30calls andexactreleasedbytes. Flagsunchanged.

Root /tmp/fireprot-qk-workspace-v1, session66086 exits0; sourcearchiveSHA
680e79e634fe5325cd69e9320682cef421d1a4ff72c8c6721fdc552fa859197b.
B256 split43.469/43.774ms, fresh48.942/65.783ms, workspace41.914/58.702ms,
normalizedworkspace42.093/42.384ms (contiguous/transposed). About3.2%warmgain
versussplit withnormalizedbias. B32normalized5.360/5.663 vs split5.528/5.827ms.
FirstB256workspacecalls49.13/49.19ms (singleobservations,otherTorchalreadywarm).
Two poolsretain18GiB duringbenchmark; deployedonewouldretain9GiB. This is an
explicitcache/allocationtreatment, NOTidenticalcachepolicy or modelmemorybenefit.
Evidence openfold3_qk_workspace_benchmark.json; fullarchive
qk-workspace-v1-results.tar.gz SHA
1ee16f1fb30c8afb315d2920b988dc1ea3b9e32b6299b379c7f4b22034100d54.
No activeGPUjob, no model/E2Eacceptance. Next: combineacceptedprefetch/reuse with
deferredadds under normalizedbias andsameprivateworkspace, withfreshnumeric
screens beforetiming, toseeklargeroperatorgain. Thenmodelmemory/lifecycle/chunk
policy needsexplicitdesignandverification (Rust ownsframeworkresourcepolicy).
Do not blindlyretain36GiB B1024buffers or forceB256chunkswithoutcomparison.

## Combined kernel with normalized workspace: 11.4% operator time reduction (2026-10-06)

Previous turn made progress and pushed8d61243 (initialized workspace).
Existingdeferred-adds.patch appliescleanly AFTERprefetch-v11, fullchain
swizzle/vector/wave/reuse/prefetch/deferred exactlymatchescombined-v12.
Local/remote62VGPR34SGPR softmax,raw51/20,16KiB LDS/nospills/48MFMA/11barriers.
Root /tmp/fireprot-qk-softmax-combined-v12, session71986 exits0, B32/B256each21exact.
Bundlemanifest1f04b110c79520749dc2a9a483b77f35330b559f5c043dfcdd13c72947ec5122.
Numericalarchive qk-softmax-combined-v12-results.tar.gz SHA
1e26d539271aa80de04a95030053ee9a000fb9ee0c78a9f021acf326e72c2424.
Evidence openfold3_qk_softmax_combined_candidate.json / *_combined_probe.json.

benchmark_workspace.py nowaccepts optionalworkspace_reference. Bothretainedarms
normalizebias withseparatepinnedkernels; receiptsrecordactualnormalizationflags.
Nochange toworkspace_attention.py.6CPUtests pass. Newroot
/tmp/fireprot-qk-workspace-combined-v12, session87864 exits0; B32/B256each12full
score+PVqualifications,10priorresultimmutabilitychecks,thread/streamguards,
100timedoutputs exact,1allocation30calls/exactreleasedbytes perpool.
B256prefetch41.863/42.173->combined38.500/38.792ms; split43.481/43.793ms.
About8%incrementvspriorworkspace,11.46%/11.42%less timevssplit. B32combined
4.918/5.229ms vs split5.529/5.830ms. Cold/retainedmemorytreatmentunchanged;
oneB256pool9GiB,twoduringbenchmark18GiB. No model/E2Eclaim.
Evidence openfold3_qk_workspace_combined_benchmark.json; fullarchive
qk-workspace-combined-v12-results.tar.gz SHA
fcc56b9e5efd0b3683766fb46a129035e597709b5c82c61cc8d336dd23a67a5b.
Alljobs terminal; noGPUjobactive. Benchmarksourcepackage SHA
353eda484902dc35da3fb57db22249198b90f953d22322b025095c30788f8734.

Nextpriority: B512/B1024 full-batch nativeQK + acceptedOwnedScoreSoftmax + nativePV
reference, compared to B256workspace microchunks (includingassembly andbiascopy),
withfullprobabilitybitchecks. FullTorchout-of-place referenceB1024exceeds64GiB;
acceptedin-placeHIPbaseline is already11-arrayexact inmodel. Fullscore36GiB plus
9GiBworkspace plusQKV shouldbe budgeted; checkactualfree memorybeforelaunch.
Do NOTassumemicrochunkPV selectsidenticalmath: thisisthegate. Start512then1024.
Keepmodelouterbatchcapabilitieslimitedtomeasuredsizes; B1536referenceworkspaceneeds
separatehoststagingorcarefulallocation. Currentworkspaceclassonly32/256; usean
explicitexperimentalwrapper forlargerbatch, notanunverifiedbroadening.
Adaptereligiblecurrentlyrejects is_neg/is_conj, nonFP32, mismatcheddevices, etc;
retainthese whenextending (standaloneworkspacequalificationusesplainrandnviews).
Thenmodelmemory/lifecycle/chunktunerpolicy plan: existingD16comparisonmainchunks
1024 andconfidence256. Avoidforcing256upstreamchunks withoutcomparison; internal
microbatchingmightpreservetheouterAPI butmustbeproven. Actualmodelmemoryheadroom
notyetread. Pairedhead16report model_residency=fresh_process_per_job, notmemory.

## Large-batch microbatch gates and timing passed (2026-10-06)

Previous turn made progress and pushed3453671 (combined workspace).
New microbatch_attention.py owns one unchangedB256WorkspaceFusion, onlyouterB512/
1024, normalizesbiasonce then slicesQKV/mask andconcatenatesPV. ChecksplainFP32,
negative/conjugateviews, metadata/thread/stream/currentdevice/capture/strictflags.
Referenceprobabilityargument is diagnostic: poisonsworkspacescoresperchunk and
comparesallbits. Newbenchmark_microbatch.py usesfullbatchnativeQK/PV withaccepted
OwnedScoreSoftmax reference. No freshout-of-placeTorchsoftmaxclaim.13CPUtests pass.

Firstroot /tmp/fireprot-qk-microbatch-v1, session45106 exits1 BEFOREcandidate
execution: trace_operations rejectedmissingfireprot.attention.full_reference
annotation. Qualification/records empty;B1024neverstarted. Retainedfailurearchive
qk-microbatch-v1-failed-trace.tar.gz pininsidefinalprovenance. Correctedannotation,
newsourcepackageSHAa6d9cfa79b4ce1494873473c9aa1a21f7853369b4e6c93d2e8eb3e0d4ff960ea.
No restartontransporttimeout; thiswasverifiedterminalharnesserror.

Correctedroot /tmp/fireprot-qk-microbatch-v2, session36586 exits0; B512/B1024each
6qualificationsseeds101/102/103×2layouts,broadcast/per-outermask,fullymaskedouter,
ALLprobabilitybits/PVbits/outputstrides/priorresultimmutability pass. Fullshape
QK/PVtracecaptured;bothkernelsymbolsmatchpriorB256splittrace.40timedcallseachpass.
Microbatchfullcalls31,kernelcalls62/124,1allocation,9GiBretainedandfullyreleased.
Reference32submittedcalls,0fallbacks. Memorypreflightpassed,free~63.86GiB vs
estimated32/53GiB beforeinputs. NoactiveGPUjobs.

B512contiguous87.054->77.389ms,transposed87.307->77.650ms; B1024contiguous
173.873->154.510ms,transposed174.167->154.751ms. About11.1%warmoperatorsavings,
biasconversion+concatincluded. Firstworkspacecalls84.58/161.59ms. Botharmsretain
9GiBpool. B1024Torchpeak49.57GiBsplit vs13.57–13.61GiBmicrobatchunderthisbenchmark;
notmodelmemory. Evidence openfold3_qk_microbatch_benchmark.json includesfailed
attempt+kernelcomparison. Successfularchive qk-microbatch-v2-results.tar.gz SHA
916accf6a4e6eaa34f4258671a267092753bfe1d188da07df2003343e31cd502.

Important model evidence reread: provenance/openfold3_head16_e2e_n1536.json
candidate aggregate ownedshapehist B2562892,B64192,B51212,B102412; D16400,D322708.
Countsincludeautotuning, notsteadyfrequency. Maincachedchunks1024/confidence256.
So B256dominatesactualattentioncalls; largebatchesincludetuning. Existing
OwnedScoreSoftmax alreadyenforcesinference_mode, soacceptedmodelrunsprove that
context, thoughbaseworker delegatesLightning. _eligible rejectsneg/conj and
nonFP32etc. Upstreamattention path core/model/primitives/attention.py, pinned
SOURCE_SHA25630fe5433982ee73e3d7ce5341fa6b510b1ea332dbde072d56129632cc53f0746.
Next: modelintegrationplanwith bounded9GiBper-forwardworkspace; keepD16/N1000/
B64andunmatchedfallback, coverB256directandB512/B1024microbatch. NeedRust-owned
resourcepolicy/registryboundary andcallbacklifecycleverification, modelmemory/
chunk-tunerobservation, all11arrays+CIFgates thenisolatedpairedE2E. Do notclaim
modelgainfromoperators. Frozenacceptedhead16worker/profile at
/tmp/fireprot-head16-paired-v1; actualmodelmemoryheadroomstillunmeasured.

## Real head-last bias layout accepted (2026-10-06)

Previous turn made progress and pushed18819bb (microbatch). Historicalactual
modeltrace biasstride[4,1,6144,4] wasoutsidepriorcontiguous/headfirst-transposed
fixtures. Addedbias_layout_supported predicate: denseheadfirstalways; dense
headlast andheadlasttranspose ONLYwhenexplicitnormalization enabled. Reject
unresolvedneg/conj/overlapping/gappedlayouts; directWorkspace nowalsoplainTensor
andlazyviewguards. MicrobatchnormalizesbeforeB256pool. NoHIParithmeticchange.
BothbenchmarkCLIs add --bias-layouts head-last;freshfusionconversionincluded;
directheadlastbenchmarkrequiresnormalizedreferencearm.19CPUtests pass.

Root /tmp/fireprot-qk-headlast-v1, session68912 exits0. B32/B256each12fullscore+PV
qualifications and100timedoutputs; B512/B1024each6qualifications and40timedoutputs.
Allbits/previousresult/stride/releaseaccounting pass; directarmsrepeatthread/
streamrejections. SourcearchiveSHA923596064094fd2cb30fe84f9126b5da1feba4c58cd792eca5e4b013f031c4d6.
B256headlastsplit43.663->combined38.691ms,transposed43.998->39.016ms;
B1024headlast174.044->154.595ms,transposed174.416->155.119ms. About11%warmgain.
Evidence openfold3_qk_headlast_benchmark.json; archive qk-headlast-v1-results.tar.gz
SHA2722f1a18d94292209c52f91a5bf5565b2fe64b0558c2a554798fa36b3c9210b.
No activeGPUjobs. Stillno model/E2Eacceptance. Nextmodelintegration mustshareONE
9GiBpool acrossB256/B512/B1024; do notinstantiatefixedbatchwrapperperoutershape
andaccidentallyretain27GiB. Rust shouldownbudget/singlelease/lifecyclepolicy;
Torch boundaryownstensorallocationandGPUcalls. PreserveD16/N1000/B64/native
fallback andfrozenacceptedhead16baseline. Needoperatorcoverage/peak/tunercounters,
11arrays+CIFmodelchecks, thenisolatedpairedE2E. GitNexusplan skillread; planning
willpinthisacceptedlayoutstate andexistingRustregistry/bridgeextension patterns.

## Shared model workspace policy implemented (2026-10-06)

Previous turn pushed a2bfe6b, the complete model-integration plan at
docs/plans/2026-10-06-gitnexus-plan-openfold3-workspace-model-integration.md.
Current execution branch is openfold3-workspace-model-integration, forked from
a2bfe6b. gitnexus-work skill active; safe read-plan receipt and provenance drift
comparison in /tmp/fireprot-workspace-execution.json. All12 cited paths match;
only plan commit moved HEAD. Untracked docs/isa/CDNA2.pdf remains excluded from Git.

Step1 committed ee406a0: core WorkspaceLease checked shape/budget, actual Rust
thread ownership, bound device/stream, New/Reserved/Ready/Busy/Failed/Draining/
Released states. Reservation rollback only before allocation commitment.
charged_bytes conservatively tracks reserved/potential storage; committed_bytes
requires allocation acknowledgment. Failed drain cannot release. No GPU calls.
Step2 committed7d5ec05: additive bridge.WorkspaceLease; existing ABI_VERSION1 and
OperatorRegistry semantics preserved. Rebuilt release CLI/bridge locally:
fireprot SHA6e81349891a472b92bc29ff43825628a3d0352e56996784a20594e6c7c2b4fce;
bridge SHAe021ec5d8e1089151abda953812a7e4f6a94a5adef428f413c6c8716cf06dc7a.
Remote accepted binaries have NOT been overwritten; both future comparison arms
need the newly pinned build. Exact current source includes subsequent Python-only
model owner (step3), with no additional Rust binary change.

ModelWorkspaceFusion in experiments/mfma_qk/model_workspace.py creates one
WorkspaceFusion(B256) and one Rust lease; supports outer256/512/1024, N1536/D32.
Explicit ordinary strict torch.empty preinitializes the existing pool's private
scores before first launch. One bias conversion, B256 kernel/nativePV per slice,
independent outputs and one shared lifetime. Does not modify old wrappers.
Every invocation rechecks strict settings/current device/stream/capture/owner;
eligibility is immutable tensor metadata only. BaseException poisons after work
starts; close drains/drops before release acknowledgment; context manager preserves
original errors if cleanup also fails. manifest separates successful kernel calls,
complete outer-call histogram, allocations, charge/commit bytes and release.
No model hook/worker overlay is installed yet. No mixed-batch GPU run yet.

All180 cargo test --workspace --locked tests pass;47 targeted Python tests pass
(13 new model owner,5 bridge,10 registry,19 old workspace/microbatch/layout).
Seven new Rust core tests included in180. Core/bridge Clippy with -D warnings
passed. Initial test harness failures were invalid SimpleNamespace __class__
assignment and patching Torch's fill property via import strings; corrected
without changing implementation or weakening assertions. CPU fault injection
uses allocation/device doubles, never claims GPU parity.

Remote read-only revalidation: accepted /tmp/fireprot-head16-paired-v1 exists.
Combined bundle /tmp/fireprot-qk-softmax-combined-v12/bundle/kernel.json matches
1f04b110c79520749dc2a9a483b77f35330b559f5c043dfcdd13c72947ec5122.
Usable controls /tmp/fireprot-qk-headlast-v1/controls-b256.json fusion section,
numerical_screen /tmp/fireprot-qk-softmax-combined-v12/probe-b256/probe.json SHA
e139e51ce1e7c14646ac54c1bcd415b0b0526aec422296eb87c8316d4e76ae39.
No new remote process launched this turn. Next: qualify mixed B256/512/1024 on
HIP7 with one owner, full nativeQK+accepted in-place softmax+nativePV reference,
full probability/output bits and old-output independence,1allocation9GiB and
release receipts. Stage a new source root and new bridge, preserve existing
reference artifacts; then overlay frozen accepted D16 worker, model gates and
isolated three-pair timing. Keep goal active; no workspace model/E2E gain yet.

## Model workspace GPU/model gates passed; revise coverage before timing (2026-10-06)

Previous turn pushed a379258. This turn committed e22426f (mixed-batch GPU gate)
and bac2074 (opt-in model dispatcher/worker overlay). Branch remains
openfold3-workspace-model-integration. No active GPU jobs after final collection.

Mixed-batch root /tmp/fireprot-model-workspace-v1, driver2018184/child2018191
both terminal, returncode0.18 calls (3seeds×2head-last layouts×3batches), allscore/
PVbits, outputstrides and old-output independence pass.42 kernel calls,1allocation,
exact9GiB physical tensor release agrees with Rust receipt. Peak Torch55,429,834,752
bytes with full B1024 reference retained; notmodelmemory. Archive
model-workspace-v1-results.tar.gz SHA678f049127c3502a0b04807b93b2fdc38e39e07de29ca86c85e358fc840dd4d9,
includes prepared source+rebuilt binaries. Sourcepackage SHA1a8448b63d6ca4fcb26fa7ab47138d41d15a54816a7d751f3f5ca03f6891ca95.
Evidence provenance/openfold3_model_workspace_qualification.json.

New workspace_dispatch.py composes its own Rust OperationRegistry around the
frozen accepted AttentionDispatcher. Enabled only via FIREPROT_QK_WORKSPACE=1;
mode0 directly invokes accepted dispatcher. Controls require path+SHA variables.
Dynamic lease guards still run after registry cache hits. All51 CPU tests pass;
initial missing required inplace=False argument was caught and corrected BEFORE
deployment. Original worker source SHA2ddc92f55f70e3b25b38e721fa71eb3f91a72720b0bf386f36fbfef6a357216e;
model-workspace.patch applies without fuzz. Shared pool context exits before
operator-runtime.json, which includes peak allocated/reserved bytes and both
registries. No maintained worker defaults changed.

Model root /tmp/fireprot-workspace-model-e2e-v1 copies frozen head16 baseline,
then overlays new code/binaries under python/experiments plus worker patch.
Overlayarchive /tmp/fireprot-workspace-model-e2e-v1-overlay.tar.gz SHA
04eb211410134fc8c163d6e18f27dff0ad09394b086c1a3715e06f4e0ab7b99a.
Remote preparation /tmp/fireprot-prepare-workspace-model-e2e-v1.py; root
preparation.json holds all base/overlay source pins. Bridge and CLI hashes remain
the prior turn's rebuilt values. Candidate/baseline profiles differ ONLY by
FIREPROT_QK_WORKSPACE flag; both retain D16, conditioning/atom caches, confidence
cap and original scientific workload. Both profile paths point at this root.

N1536 driver2020100/CLI2020107 exited0; N1000 driver2024791/CLI2024801 exited0.
They overlapped onHIP7/HIP6 for CORRECTNESS only. Do not treat551.189/240.894s
driver durations as isolated E2E observations. Outputs:
/media/IntelSSD/fireprot/outputs/workspace-model-n1536-candidate-0
/media/IntelSSD/fireprot/outputs/workspace-model-n1000-candidate-0.
root/collect.py invokes Rust compare-npz and requires11 bitwise arrays, complete
CIF byte equality, released/zero-charge Rust lease, and exact call/allocation
accounting. gate-n1536.json passes against BOTH accepted head16 candidate0 and
originalnative hip-softmax-e2e-n1536-baseline-0. gate-n1000.json passes against
zero-mask-e2e-n1000-baseline-0; no workspace allocation or call. All8GCD idle.

CRITICAL next action: DO NOT launch the v1 paired timing script. Actual N1536
MSA/pairformer chunks changed1024->512; template andconditioning remain1024,
confidence remains256. Workspace hits only28 complete calls (B25618/B5128/B10242),
42 microchunks. Accepted fallback nowB1285040, B256402, B64192, B32192, B51214,
B10248; D16400 andD325448. Fallback reasons: outerbatch5424, qkvshape400,
bias/masklayout26, baseadapterineligible6204. The old B256-dominant histogram
was real outerbatch (owned_scores.py keys scores.shape[0], NOT flattenedheads),
but the new retained pool changes the model's tuning regime. Do not repeat the
incorrect hypothesis that oldB256 meantB64 after flattening; source disproves it.

Model peak Torch51,757,716,992 bytes; pool charged9GiB and correctlyreleased.
Upstream core/utils/chunk_utils.py SHA4c9ab484854868e3c3d9a28a0d2e8fe0f9b1c75a35dca4984e0631591489fac5:
ChunkSizeTuner._determine_favorable_chunk_size binary searches powers-of-two plus
maxsize; test_chunk_size calls fn under no_grad and catches ALL RuntimeError.
Thus reducedtuning suggests capacity pressure but no OOM cause has been captured.
Need instrument representative_fn failures while preserving rethrow/upstream
selection, then consider4.5GiB B128 pool, microchunk larger tested sizes, and
support dominantB128. Requalify B128 raw/softmax21cases on combined-v12 first
(currently its saved numerical screens onlyB32/B256), then fullscore/PV and
mixedbatches. Existing WorkspaceFusion onlyaccepts32/256; plan/coreModel cap9GiB
must be revised coherently. No needchangeHIPmath. Route plan to Deepen for the
capacity/capability decision; user alreadyauthorizedDeep/autonomous optimization.

Full model archive workspace-model-e2e-v1-gates.tar.gz SHA
f27f5e4881cc5a63b2959e30dec3ba5c0a025ce47337d00d5a18742d67519393 (74,196,893bytes)
contains source/deployment, binaries, both complete modeloutputs, pins/collector.
Evidence provenance/openfold3_model_workspace_model_gates.json records the
coverage issue and no-speedup limitation. Baseline runtime pin52d809f646a98a18751daea539e9024cf30f3a71d527c987353cf222513ff57f
is unchanged; oldchunkmain1024 confirmed. /tmp/fireprot-prepare-workspace-pairs-v1.py
was uploaded but NEVERexecuted; /tmp/fireprot-workspace-paired-v1 doesnotexist.
Keepgoalactive. Correctness gatespassed; acceptablemodelcoverage and E2Egain
remainunproven. Nextmustfixthisrealcoverage/memory issue, not waste3pairs onv1.

## Smaller workspace qualified; model trial active (2026-10-06)

Previous license-only reply verified an already-landed change; the benchmark
poll did establish terminal successful completion. This turn archives and
records the actual pending evidence and launches the next model trial.
Commits149b011/d022490/b6cf2f3 revise plan, observe tuner exceptions and qualify
explicit capacities32/128/256. Rust CLI/bridge unchanged. Seventy-three targeted
CPU tests passed in their implementation turn. Current plan digest6177e9b38a9535dde6ecc6a159e630986ed7a69760e2fe180ab38d76d8a9782f;
receipt /tmp/fireprot-workspace-v2-execution.json, current reanchor snapshot
/tmp/fireprot-capacity-current-provenance.json. GitNexus work remains active.

B128 raw21case screen passes; provenance/openfold3_qk_combined_b128_probe.json.
/tmp/fireprot-workspace-capacity-v1: both capacities128/32 complete24 calls across
B128/256/512/1024, allprobability/outputbits andguards/releasereceipts pass.
128pool4.5GiB90launches;32pool1.125GiB360launches. Complete-operator screen finished
exit0 (oldsession67404),144timedoutputs exact. B256split43.728/44.106ms versus
cap12838.867/39.242ms, cap3239.540/39.934ms. Bothpoolsresident5.625GiB duringallarms.
Evidence openfold3_workspace_capacity_screen.json. No model/E2Egainclaim.

Tuner-v1 terminalfailure dueobserver installedbeforeconfidencehooksourcecheck;
correctedv2entersconfidencehookbeforeobserver, preservingguard. Isolatedv2model
exit0,551.744s diagnosticduration, all11+CIFbytesexactvsnativeandD16. Twoactual
OutOfMemoryErrors atMSA/Pairformerchunk1024 trying36GiB; bothselect512. Pool9GiB,
28calls unchanged. Evidence openfold3_workspace_tuner_diagnosis.json. Bothfailed/
successfuldeployments+outputsarchived. Syntheticallocatorrepro shows9/4.5GiB
poolcanpin36GiBcachedsegment;purgebeforepoolfixessyntheticrequest. Actualmodel
segmentnotobserved;noallocationpolicychange. Evidence
openfold3_workspace_allocator_reproduction.json. Allarchivesunderexistingdurable
conditioning-cache-20261006root;JSONfilespinsha256andbytes.

ACTIVE MODEL TRIAL: /tmp/fireprot-model-cap128-v1 onDiamondHill HIP7.
Driver2050018,CLI2050019, localshellsession63470. Single4.5GiBpool/cap128 supports
B128; ordinarystrictallocatorunchanged. Scopedchunkdiagnosticsremainenabled.
Preparedfromexacttuner-v2worker plus3qualifiedcapacityfiles; allhashesin
preparation.json. Local/remote /tmp/fireprot-prepare-model-cap128-v1.py records
setup. Request /media/IntelSSD/fireprot/outputs/model-cap128-n1536-0, R3/D200/S1/
seed101 strictFP32. All8GCDidleconfirmedbeforelaunch. Rootdriver/start/exit/log;
collect.py1536 checksall11andfullCIFagainstbothnativeandD16 pluscapacity-aware
leaseaccounting. Nextpollknownprocess/exit;NEVERrestartonobservationtimeout.
RuncollectorONLYafterrc0. NeedN1000fallbackgateforselectedimplementation then
isolated3pairs withdiagnosticsoff;doNOTlauncholdv1paired28-calltreatment.
IfsmallercapacitystillcausesOOM, consideractualsegmentdiagnostic orcontrolled
allocationisolationtrial;syntheticpurgeisnotmodelproof. AcceptedE2Eunchanged:
directnative1.499x;D16incrementalseparate1.162x. Keepgoalactive.

## Capacity128 model complete; fix allocation and mask layout next (2026-10-06)

Previous turn pushed7b6f0c5, evidence forcapacityGPUgates/tunerOOM/syntheticallocator.
This turn polledsamePID untilterminal; driver2050018/CLI2050019 exit0 at548.7739s.
All8GCDidleconfirmed. /tmp/fireprot-model-cap128-v1/collect.py1536 succeeds:
all11arrays+completeCIFbytesexact againstoriginalnativeandacceptedD16. One4.5GiB
lease released,2548fullcalls,2604kernelcalls. B1282520/B25618/B5128/B10242.
MainMSA/Pairformerstill512;bothactualOOMtrying36GiB at1024 with15.77GiBallocated,
33.53/33.52GiBreservedunallocated,and14.24GiBfree. Peakallocated51757716992bytes.
Fallbackbias/masklayout2546,outerbatch384,qkvshape400,baseineligible6204.
AcceptedOwnedScorefallbackB1282520 (halfremaining),B256402,B51214,B10248,B32192,
B64192. No pairedE2Eclaim;singlediagnosticdurationisnotisolatedpairedspeedup.
Evidence provenance/openfold3_workspace_cap128_model_gate.json. Archive
model-cap128-v1-results.tar.gz SHAa88956b431289ceb47ccdb651fe4e67665a4b11df0c8bca9b0210fccff9f31b8,
47732370bytesunderusualdurableroot;containsdeployment+completeoutputs+preparer.

NEXT: resolveallocationandlayoutbeforepairedmeasurement. Upstream
core/model/layers/triangular_attention.py transposesmaskforendnode,thencomputes
(self.inf*(mask-1))[..., :,None,None,:]. CPUreproN1536/B128 yieldsstride
(1,2359296,2359296,1536),noncontiguous;startnodecontiguous(1536,1536,1536,1).
ModelWorkspaceFusion currentlyrejectsnoncontiguousmask,explainingaPLAUSIBLE
remaininghalf,notyetprovenbyactualrejectedtensortrace. Needboundedmetadata
observationorqualifiedupstream-equivalentfixture;normalizeonlyqualifiedmask
layouts,checkallprobability/outputbits,thenmodelgate. Avoidblindwidening.
Allocationnextoption:one-timeemptycachebeforefirstpool orisolatedpool;synthetic
reproexistsbutmodelplacementnotobserved. Ordinarystrictfillmustremain. Plan
scopeacceptsqualifiedcapacity/layoutandallocationrefinement;noarithmeticchange.

PREPARED BUT NOT RUN: /tmp/fireprot-model-cap128-n1000-v1, requestoutput
/media/IntelSSD/fireprot/outputs/model-cap128-n1000-0;local/remotepreparer
/tmp/fireprot-prepare-cap128-fallback-v1.py. Holdsoldcapacity128implementation;
ifimplementationchanges,preparefreshroot. NoactiveGPUjobnow.
LocalONLY /tmp/fireprot-prepare-cap128-pairs-v1.py requiresbothgates+>1000calls;
neveruploaded/executed. DoNOTlaunchyetmerelybecausecoveragegatepasses: resolve
remainingOOM/layoutfirst. /tmp/fireprot-cap128-paired-v1 doesnotexist.
AcceptedE2Eclaimsremainunchanged;keepgoalactive,substantialprogressnotplateau.

## One-time cache release implemented and GPU-qualified (2026-10-06)

Previous turn pushed8d616fb (capacity128 exact model gate and remaining issues).
This turn is real progress: implemented an opt-in allocation policy, tested it,
and launched a fresh model trial. No plateau or external blocker established.

ModelWorkspaceFusion accepts allocation_policy='ordinary' (default) or
'release_cached_once'. Hashed controls pass the option through load_workspace;
qualifier honors the same field. Under the existing owner/stream/strict checks,
_allocate reserves the Rust lease, binds device/stream, then optionally calls
Torch empty_cache exactly once before ordinary strict torch.empty. Live tensors
and memory filling remain intact. Counters distinguish attempts/completions.
Failure propagates and poisons the owner without submission/fallback; close
retains the existing conservative charge/drain rules. This is tensor-allocation
policy within the current plan, not a new framework allocator service. No Rust
or HIP math changes. Default and unused workspaces never release cache.

81 related CPU tests pass. The new ordering test first failed against the old
constructor (missing option), then passed. New tests cover no repeat release,
reservation/binding order, unused/default behavior, invalid options and exact
failure identity with no allocation. Current plan unchanged; reanchor snapshot
/tmp/fireprot-allocation-policy-provenance.json. Impact class LOW (benchmark,
qualifier, dispatcher), loader UNKNOWN/no indexed callers cross-checked by rg:
worker overlay and dispatch tests. Existing benchmark keeps default ordinary.

GPU root /tmp/fireprot-workspace-release-cache-v1, session66479 terminal0.
Capacity128,24 complete calls,90 launches, all probability/output bits and guards
pass; one cache release, one4.5GiB allocation, complete physical/Rust release.
Source overlay SHA8d80f4e8983ca88ff903eb5803abe12801417248b9b4597295174c94caf30f4a.
Evidence provenance/openfold3_workspace_release_cache_qualification.json.
Archive workspace-release-cache-v1-qualification.tar.gz SHA
9d0437493e8b6e8c9734a834361ba5ca9151c49e5316dcdee1575517565e739d,
2585734bytes under usual durable archive root. No model fix or speedup claim yet.

ACTIVE: /tmp/fireprot-model-release-cache-v1 on DiamondHill HIP7.
Driver2060234, Rust child2060243, detached with start_new_session; poll actual
PID/start/exit/log. Never restart for a transport/observation timeout.
Output /media/IntelSSD/fireprot/outputs/model-release-cache-n1536-0.
Prepared from exact cap128 model plus qualified model_workspace/dispatcher;
all files bound in preparation.json. Local/remote preparer
/tmp/fireprot-prepare-release-cache-model-v1.py. All8GCD idle before launch.
StrictFP32 R3/D200/S1 seed101; same capacity128/layouts, only allocation policy
changes. Scoped tuner observer remains enabled. collect.py1536 requires all11
and complete CIF against native+D16, lease release and exactly one cache release.
Run collector only after returncode0. Keep this diagnostic isolated; no other
GPU job until terminal. Archive model before changing any deployment files.

Next: inspect whether MSA/Pairformer recover1024 and whether OOM disappears.
Then address ending-node transposed mask with explicitly qualified fixtures and
normalization (or capture rejected metadata if needed). Current model still
rejects noncontiguous masks; do not silently widen capabilities. N1000/pair
preparers from preceding turn remain unrun and target the old implementation;
prepare fresh roots after selecting the improved implementation. Accepted E2E
claims unchanged. Keep goal active.

## Cache release fixes model tuning; transposed masks qualify (2026-10-06)

Previous turn pushedf956b00. This turn completed its model gate and implemented
an opt-in exact transposed-mask path. Goal remains active; substantial progress.

Release-cache model root /tmp/fireprot-model-release-cache-v1 finished rc0,
536.879s diagnostic duration (not paired E2E). Driver2060234/CLI2060243 terminal.
Collector1536 passes all11 arrays and full CIF bytes vs native+D16. MSA and
Pairformer recover1024; observer records ZERO rejected tuning attempts. Other
main chunks1024, confidence256. Pool one4.5GiB, one cache release,1258 fullcalls
(B2561254/B5122/B10242),2532 launches, clean lease release. Peak Torch allocated
55585914368. Evidence openfold3_workspace_release_cache_model_gate.json.
Archive model-release-cache-v1-results.tar.gz SHA
c08cedf351772b215b31839939ddae5edafcdc1a42d19a804b886b98b88a673f,
47731676bytes under usual durable root. Actual pool segment not observed;
restored tuning is direct evidence of effectiveness, not a claimed E2E gain.

New transposed_pair_mask_supported accepts ONLY plain FP32 [B,1,1,1536],
B1/128/256/512/1024, strides0=1/final=1536, no neg/conj. Model constructor
normalize_transposed_mask plainbool defaultFalse; hashed controls forward it.
When enabled, known transposed masks become contiguous once before slicing;
mask_copies recorded. Existing matched contiguous path stays available. Invalid
metadata still falls back before submission. Copy interruption poisons owner
before reserve; no fallback, lease zero-charge cleanup proven. No HIP/Rust edits.
98 CPU tests pass. New normalization test first failed against previous
constructor; metadata tests cover offsets0/17, broadcasts, overlapping and
unmeasured pitches, invalidoptions, input preservation and shared copiedstorage.

GPU root /tmp/fireprot-workspace-mask-v2, session15691 terminal0. Ordinary
allocation, capacity128. Qualifier control qualification_mask_layout=
transposed_pair builds real strided column slices including nonzero offsets;
24 complete calls over B128/256/512/1024, full probabilities+outputs exact,
old-output/stride/guard/release checks pass,24copies. benchmark_mask.py:
96 timed exact outputs; B128 split53.533/53.926 ->19.398/19.798ms; B256
106.723/107.135 ->38.862/39.233ms. One4.5GiB pool resident in botharms; normalization,
packing and concat included. ~2.7x is COMBINED layout+fusion gain, not pureHIPgain.
Need normalization-only ablation for attribution; current screen has twoarms.
Evidence openfold3_workspace_transposed_mask_screen.json. Archive
workspace-mask-v2-results.tar.gz SHA
0d9534a9ac90b9e0c4e5d2a5d2bb4be5971f4d81f80b284cbb72b7100da364f6,
2594817bytes. Overlay5791b30ddc92d6af0cf46ccf672a1f76c831ad9346c0d3a76a648ccba374879e.
/tmp/fireprot-workspace-mask-v1 was PREPARED ONLY with release_cached_once,
never executed. v2 explicitly ordinary allocation, tests one new treatment.

ACTIVE MODEL: /tmp/fireprot-model-mask-v1, driver2069120, CLI2069128 onHIP7.
Output /media/IntelSSD/fireprot/outputs/model-mask-n1536-0. Same strictFP32
R3/D200/S1 seed101, one4.5GiB capacity128, normalize_transposed_mask=True,
allocation_policy=ordinary (NO empty_cache), observerenabled. Prepared from
exact cap128 model plus GPU-qualified model_workspace/dispatcher; allsourcepins
in preparation.json. Local/remote /tmp/fireprot-prepare-mask-model-v1.py.
Driver detached with start_new_session; poll PID/start/exit/log, neverrestart
on timeout. Collector1536 requires all11+fullCIF vsnative+D16, copies>0,
zero cache releases and full lease accounting. Run onlyafterrc0.

Reason for ordinary allocation in this trial: unsupported ending-node masks
previously fell back to full nativeQK, potentially causing the36GiB request.
Covering both directions MAY remove that request as well as eliminate expensive
strided mask access. This is a hypothesis until model counters/tuning are read;
not a reason to force chunks. If ordinary+mask stillOOM, combine the separately
verified cache-release option in a fresh exact model trial. Do not mutate the
running root. No other GPU jobs while this diagnostic runs. Model gate, N1000
fallback and isolated3pairE2E remain before promotion. Prior N1000 andpair
preparers targetoldercode and remain unrun; use fresh deployment roots.
Plan unchanged; reanchor /tmp/fireprot-mask-provenance.json. GitNexus classLOW,
loaderUNKNOWN/no graphcalls textverified workerpatch+tests; qualifierLOW.
Accepted E2E claims unchanged: directnative1.499x, separateD16increment1.162x.

## Both mask model gates pass; three E2E pairs running (2026-10-06)

Previous turn pushed39f2fed (qualified transposed-mask normalization). This turn
completed its N1536 model, measured a three-arm attribution, passed N1000 and
started actual E2E. This is progress; keep the goal active, no plateau/blocker.

N1536 root /tmp/fireprot-model-mask-v1 driver2069120/CLI2069128 terminal rc0,
449.25774s diagnostic duration. Collector1536 passes all11 and fullCIF vs both
native and acceptedD16. Ordinary allocation, NO cache release; mainchunks1024,
confidence256, NO rejected tuning attempts.2516 fullcalls B2562508/B5124/B10244,
5064 kernelcalls,1258 maskcopies, one4.5GiB lease fullyreleased. Peak51757716992.
Thus mask coverage alone removes the observed tuning failure in this run.
Evidence provenance/openfold3_workspace_mask_model_gate.json. Archive
model-mask-v1-results.tar.gz SHA7849dcab7a7fb21083d2099ae3b03ba9a0f5e104193d3a4d5e76c36e9062e9e7,
47731770bytes, usual durable root. No paired speedup inferred from449s.

benchmark_mask.py now adds split_normalized_mask, with contiguous copy INSIDE
its timed call. Three arms, balanced12samples eachcase,144 timed outputs exact.
Root /tmp/fireprot-mask-ablation-v1, session55189 terminal0. B256 headlast:
original106.740ms, normalization-only43.718ms, workspace38.872ms. Transposedbias:
107.143/44.102/39.247ms. B128 original53.513/53.930, normalized21.918/22.315,
workspace19.413/19.799ms. ~93% of the saved time in these cases is mask layout;
fusion adds ~11% time reduction over normalized split HIP. Do NOT attribute
combined2.7x to fusion alone. One4.5GiB pool retained throughout all arms.
Evidence openfold3_workspace_mask_ablation.json. Archive mask-ablation-v1-results
SHA d206e898ec489aeac509e22be9b8ecd07f617ebf7b44e69a77239c48f788997e,
2585321bytes. Model code unchanged; benchmark-only edit, GPU-tested.

N1000 root /tmp/fireprot-model-mask-n1000-v1 driver2077632/CLI2077639 terminal0,
238.9724s diagnostic duration. Collector1000 passes all11+fullCIF againstnative;
zero allocations/calls/maskcopies/cache releases, clean lease. Same code asN1536,
configured normalize=True, but unsupported length retains accepted fallback.
Gate appended to same model-gate provenance. Archive model-mask-n1000-v1-results
SHA57832cb364ba57620ec31e6bd3ed16bc812f8bd2cca040d0590c977eb982aa41,
26579869bytes. Collector scope text corrected before launch and prep pin updated.
Local/remote /tmp/fireprot-prepare-mask-fallback-v1.py reproduces that correction.

ACTIVE PAIRED BENCHMARK, no other GPU work/heavy remote builds until terminal:
/tmp/fireprot-mask-paired-v1 onDiamondHill, driver2083396, Rust scheduler2083407.
Detached start_new_session. Driver/source /tmp/fireprot-mask-paired-v1/driver.py;
local /tmp/fireprot-mask-paired-driver-v1.py. Root start.json/exit.json/paired.log;
root paired-n1536/order.jsonl and benchmark.json/comparison.json when complete.
Spec pairs-n1536.json SHA3282f9e6db06fcabe276dd8e98ab023e8ca86d2fadeeff7952c3d802fd57d77b.
AB/BA/AB: mask-e2e-n1536-baseline-0,candidate-0,candidate-1,baseline-1,baseline-2,
candidate-2. Outputs /media/IntelSSD/fireprot/outputs/<runid>. Firstbaseline
started; inspect journal+actual PIDs, neverrestart for observation/networktimeout.
Parent can be alive while child jobs change; scheduler authoritative outcome.
All8GCD idle preflight saved. No overlap with diagnostic/ablation/archive jobs.

Both profiles use SAME /tmp/fireprot-model-mask-v1/python and rebuiltCLI6e8134/
bridgee021ec; diagnostics0 botharms, samehashedcontrols. Onlyworkspaceflag0->1
anddeclaredprofilehash differ. Candidateordinaryallocation+capacity128+masknorm;
baseline acceptedD16/pair+atomcaches/ownedsoftmax/confidencecap. BothstrictFP32
R3/fourtrunkpasses,D200,S1,seed101,HIP7/PCI93. This is incremental vsacceptedD16,
NOTdirectoriginalTorch. Do not multiply into a directratio. Runtime limits7200s
wholebatch,1800s/invocation. Expected~50min from prior individual observations.
Preparation local/remote /tmp/fireprot-prepare-mask-pairs-v1.py; sourcegatespinned.
Launch evidence provenance/openfold3_workspace_mask_paired_launch.json is only a
historical running-state record, not acceptance. Record final complete outcome
separately. Rust compare_prediction uses Policy::Exact for ALL OpenFold3 arrays
(bytecomparison), and schedulerchecksCIFatomidentities aftereachrun. Still verify
fullCIFbytes and nativeanchor in final collector, as stronger existing evidence.

Next: poll known scheduler/journal, do not launch another GPU experiment.
Aftercomplete, collect all6 runtimes/chunks/copycounts and model/invocationtimes,
require11arrays/completeCIF/nativeanchors, archive entire pairedroot+6outputs+
source/binaries, then publish measuredratio or rejection. Do not promote on
launch or partialpairs. Complete plan's final gates before promotion; no need
rerun unchanged Rust tests repeatedly while timing. Possible future CPU-only
analysis: normalization could benefit remaining D16/N1000 fallback paths, but
that is beyond this plan's retained-fallback scope and needs a new/deepened plan
plus exact gates after currentpairs. Acceptedpublished E2E remains1.499xdirect
andseparate1.162xD16increment until the new result is fullycollected.
