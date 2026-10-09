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

## Paired benchmark in progress; final CPU gates and ISA review (2026-10-06)

No model or kernel changes this turn. Same detached benchmark remains active:
/tmp/fireprot-mask-paired-v1, driver2083396 and scheduler2083407. Latest poll at
33:25 scheduler elapsed: first four runs measured and verified, baseline2
started at journal offset1991.497448605. No restart and no overlapping GPU work
or heavy remote build. Continue polling the SAME handles; a network timeout is
not a failed run. Full benchmark/collector acceptance remains pending.

First pair only: invocation537.407814490 ->449.722786494 seconds,1.19497572867x,
all11 arrays exact and CIF atom identities checked. This is preliminary and
incremental versus D16, not a published three-pair result or direct Torch ratio.
Do not update accepted speedup claims from partial measurements.

Final local cargo test --workspace --locked:180 passed,0 failed,1 ignored across
44 test summaries. Release CLI/bridge/runtime build passed; CLI6e81349891a472b92bc29ff43825628a3d0352e56996784a20594e6c7c2b4fce
and bridgee021ec5d8e1089151abda953812a7e4f6a94a5adef428f413c6c8716cf06dc7a
match timed deployment. Logs /tmp/fireprot-workspace-final-cargo-tests.log and
/tmp/fireprot-workspace-final-build.log. Plan Python regression command plus
model_workspace/workspace_dispatch/tuner_observer:74 passed. Additional mask
layout and Rust workspace tests:21 passed. This turn's95 tests do not replace
the earlier98-test implementation record. No behavioral edit after these gates.

Terminal-only collector is local /tmp/fireprot-collect-mask-pairs-v1.py and
remote /tmp/fireprot-mask-paired-v1/collect.py, SHA
d3fa15d870784cb641e5dd49b787dd12c6407b2ce84f17b6603e86efc4897392.
It refuses a live benchmark (guard exercised successfully), requires all6 jobs,
18 journal events and3 correct pairs, replays Rust compare-runs from pinned
receipts, and compares EACH output to BOTH native and D16 anchors with all11
bits and full CIF bytes. It checks ordinary4.5GiB lease/call/copy accounting,
diagnostics disabled, caches and policy. Creates collection-v1 exclusively;
if post-terminal collection fails, investigate before creating a new versioned
collection root. Writes collected.json exclusively only after all gates pass.
No collection root or collected.json was created during the live-guard check.

Terminal-only archiver local /tmp/fireprot-archive-mask-pairs-v1.py and remote
/tmp/fireprot-mask-paired-v1/archive.py, SHA
4eebd88f932f5a9f54ab50c49b030adb0401ae3d664e3af56c890792b257e2de.
After collector success, creates usual durable-root/mask-paired-v1-results.tar.gz
exclusively, including controller,6outputs,frozenworker/binaries,2reference
outputs,kernelbundles,B128screen,input; checkpoint bytes remain external pins.
Both helper scripts syntax checked. Neither terminal action has run yet.

LOCAL ONLY guarded direct-native preparer:
/tmp/fireprot-prepare-mask-native-pairs-v1.py SHA
188171436bebbc90c07dbd1d48d033b1daf13c783ff44f26758f62162b16d156.
Not uploaded/executed; /tmp/fireprot-mask-native-paired-v1 does not yet exist.
Requires complete successful incremental collection with faster median and p95.
Then creates3 AB/BA/AB native-vs-full-candidate pairs, same frozen worker/CLI,
strict science unchanged. Native backend returns upstream forward before HIP
hooks; pair/atom caches separately disabled. Declared treatment: backend and
conditioning/atom/head16/workspace flags plus profile hashes. Timeout10800s,
1800/invocation. No driver prepared for this future sequence yet. Finish current
benchmark and archive before deciding/launching direct comparison.

Existing user-authorized GPT6-luna xhigh ISA agent resumed read-only and finished;
no GPU/compilation/source edits. It verified actual localv12 asm SHA
97f2bcd6cae933131d7373b040f8145c98f2c0f42ab16ce813429bc5cbcb32ef
and metadata SHA07886b5053d81e9cf578f69e708ab26b86bac77f769bcdeb4cf53d66537a38a4
at /tmp/fireprot-fusion-combined-v12.asm and -metadata.txt. Actual true kernel
has11 barriers,16KiB LDS,62VGPR34SGPR,no spills. Six producer barriers and five
consumer-to-reuse barriers are distinct. All DS accesses currently useVGPR
addresses with offset0; no explicit M0 reference in assembly.

Two FUTURE experiments, not implemented or authorized by this plan's arithmetic
scope: (1) native torch.bmm(out=retained_scores) plus same in-place HIPsoftmax to
separate allocation reuse from handwrittenQK gains; validate selected kernel,
hidden allocations/fill and full probability/PV bits. (2) alternate two16KiB LDS
buffers, potentially remove5 reuse barriers while retaining6 producer barriers
and exact arithmetic order. ISA offset encoding covers32767B; new compilation
must establish32KiB segment/addressing,barrier count,resources,no spills and
runtime LDS range. Actual occupancy and performance cannot be inferred from
LDS size alone. No speedup claim. Plan currently forbids HIP arithmetic edits;
complete current E2E/promotion decision before a new/deepened implementation plan.
User PDF remains untracked and must never be committed.

Reanchor /tmp/fireprot-mask-paired-current-provenance.json: HEAD1203df1;
cited manifest unchanged since prior reanchor, dirtydigest2c6402b0053bf1b1d85ccb785f269c24e0d3531ff1b0a288fad3cf5b9ec36643
(only untrackedISA). Typed GitNexus status/context agree: sameHEAD,current
schema4runner,incompleteReasons[],up-to-date. Plan digest remains6177e9b38a9535dde6ecc6a159e630986ed7a69760e2fe180ab38d76d8a9782f.
Step5 still active; step6 pending; goal remains active, no plateau or blocker.

## Guarded completion controller launched; SSH observation timeout (2026-10-06)

Previous turn was progress (final CPU gates and verified live benchmark), pushed
ea6f3e9. This turn first confirmed2083396/2083407 alive at34:52, baseline2 running
after four measured+verified invocations. No concurrent GPU work or source edit.
Read the frozen worker directly: native returns upstream forward before allHIP
hooks; pair/atom cache scopes are separately conditional on their environment
flags. Native direct-comparison profiles correctly disable both caches.

Prepared and LAUNCHED detached completion controller2117145 in
/tmp/fireprot-mask-completion-v1 onDiamondHill; local source
/tmp/fireprot-mask-completion-v1.py. SSH returned the PID after verified upload
and syntax checks. Subsequent status SSH stalled (localsession55965) and an
independent no-ControlPath connection timed out (session9114 terminal255).
This is an OBSERVATION failure, not benchmark/controller failure. Do not restart
either. Re-poll2117145,2083396,2083407 and both roots' status/start/exit/journal.
Controller status.json/controller.log are authoritative for follow-up stages.

Controller waits on the existing driver and exit.json in30s intervals, with a
7200s observation deadline that stops controller only, never kills/restarts the
benchmark. After successful terminal receipt it runs existing collect.py and
archive.py, writes archive-receipt.json, and only if exact acceptance plus faster
median AND p95 are proven prepares /tmp/fireprot-mask-native-paired-v1. It then
installs native driver/collector/archiver, validates all six requests via Rust
inspect, requires all8GCDidle, and runs the new3pairdirect benchmark. After that
it collects every native/candidate output against both historical native+D16
anchors with all11bits/fullCIF and archives. Any failure stops with stage=failed;
no automatic retries. Do not launch direct pairs manually while controller owns
the sequence. Do not mutate scripts/deployment while controller is running.

All exact local and remote helper pins plus full local source archive recorded
in provenance/openfold3_workspace_mask_completion_launch.json. Archive
outputs/experiments/qk-softmax-local-20261006/mask-completion-controller-v1.tar.gz
is intentionally ignored. New native collector requires absence of HIPruntime
and cache reports for originalnative, verifies fullworkspace/cacheaccounting only
forcandidate. It explicitly reports no native instrumentation/peak data rather
than inventing it. Native driver requires successful incremental receipt hash,
terminal0,archive receipt and fresh output root before GPU preflight.

Latest remote phase is not yet observed after spawn; no new E2E acceptance claim.
Continue current plan steps5/6 and goal; no plateau, no blocking audit threshold.

## Network recovered; incremental result accepted; whole-model priority (2026-10-06)

User resumed optimization and explicitly prioritizes OpenFold3's largest remaining
whole-model costs, without fixating on Protenix issue354 or the current QK kernel.
Use NEW latest-candidate stage shares before choosing the next implementation.
No plateau established. Old native profile is not current hotspot attribution.

Incremental /tmp/fireprot-mask-paired-v1 completed rc0 in2997.233s. All6 runs
verified; collector and archive succeeded while SSH was unavailable. Three-pair
E2E540.302281723 ->449.722786494s,1.201411843x; model445.333388089 ->355.204169396s,
1.253739191x. All11 arrays/fullCIF match native+D16 for every output. Both arms'
peak Torch allocation51757716992B. Evidence now tracked in
provenance/openfold3_workspace_mask_e2e_n1536.json and docs/openfold3-performance.md.
Remote collected SHA4e42adfa5a882f1f35220d4e26a2f74ac4e5f7dd8d3ac1da855dd3114edec5af;
archive usualroot/mask-paired-v1-results.tar.gz SHA251343a481a26277d69674b767f08629cb696abee78f7772020a6dcd6d058d95,
585228627B. This remains incremental, not direct originalTorch.

Old controller2117145 and native-v1 driver are TERMINAL. Native-v1 completed
two invocations then stopped at first pair validation: treatment omitted
launch.inputs.rust-bridge (native null; HIP pinned bridge). Not a numerical
failure. /tmp/fireprot-native-declaration-audit-v1 independently revalidates
the two retained receipts with ONLY that declaration added: Rust comparison,
all11 arrays pass. Original failed benchmark unchanged. This retrospective
single pair is diagnostic, not accepted3pairE2E. Failed run+twooutputs+controller
and audit archived to usualroot/mask-native-v1-declaration-failure.tar.gz SHA
9cfd90a2d5d1c78f6406838308496b1e1406235d00359e397bf6337249db053c,106039006B.

ACTIVE QUEUE2149322: /tmp/fireprot-mask-native-paired-v2 onDiamondHill.
queue-status.json,queue.log,queue-start.json are authoritative. Verified live;
idle wait completed and stage=benchmark, child driver2149830. It did not kill
other jobs. The six-hour idle gate precedes fresh3 AB/BA/AB pairs with
declared bridge addition; same frozen model worker, unchanged science/source.
SpecSHAada748f0de84859ab4c37505f079ae6457ca1e26feeaf05fa4e15e4b23f0d70f.
RunIDs mask-native-v2-e2e-n1536-{baseline|candidate}-{0|1|2}.
Once started driver/start.json identifies actual driver/scheduler. Queue runs
driver,collector,archive sequentially; any error is terminal, no automaticretry.
Do not launch competing GPU work or change timed deployment. Allsix requests
are inspected before waiting. Follow original-PID/receipt policy on timeouts.

ACTIVE FOLLOW-UP2150203: /tmp/fireprot-latest-stage-profile-v1. Verified live;
status.json currently waiting_direct_benchmark. This is a separately frozen copy
of latest mask candidate, with existing model_probe adapted to GPU stage events
only; dense Torch profiler disabled. Wraps run_trunk,_rollout,input/template/MSA/
pairformer, sample_diffusion and aux_heads; timings inclusive/nested, never sum
parents+children. Transport includes operator-profile.json in accepted artifacts.
Request latest-stage-profile-n1536-0 inspected successfully, strictFP32R3D200S1
seed101HIP7. No arithmetic or requested-work change. After native-v2 succeeds and
GCDs idle, it runs one Rust-supervised diagnostic, checks all11/fullCIF against
native,D16 AND latest candidate, then writes collected.json with stage totals.
Never count instrumented duration as new E2E performance. If prior queue fails,
follow-up also stops for investigation. No diagnostic GPU work has started yet.

All new helper sources and exact local archive pins recorded in
provenance/openfold3_mask_native_v2_and_profile_launch.json. Source archive under
ignored outputs/experiments/qk-softmax-local-20261006. Local helpers:
/tmp/fireprot-prepare-mask-native-pairs-v2.py, /tmp/fireprot-mask-native-v2-queue.py,
/tmp/fireprot-mask-native-v2-{driver,collect,archive}.py,
/tmp/fireprot-prepare-latest-stage-profile-v1.py and
/tmp/fireprot-latest-stage-profile-driver-v1.py. No maintained model/kernel edits.

Issue354 re-read publicly: body claims3.5x onMI300 from HIPfusedLayerNorm+ROCm
Triton, lacks reproducible workload/timing details; GitHub API returned no
comments. No author code used. Our LayerNorm6.3xoperator/1.079xmodel and
OpenFold3softmax1.376xE2E support the general direction, not that specific claim.
User explicitly says optimize the large remaining costs rather than this issue.


## Native-v2 rejected for system drift; stage profiling now runs first (2026-10-06)

User requested status. Authoritative remote results: native-v2 scheduler stopped
at index3 baseline1; first3 runs succeeded. Supervisor for
mask-native-v2-e2e-n1536-baseline-1 reports environment files or topology differ
from pinned inventory after931.949s. No accepted result for that run. Existing
firstpair950.335467275 ->465.143082332s (2.043103517x) remains preliminary only;
three-pair direct result is NOT complete. Queue2149322 and follow-up2150203 are
terminal, no active old scheduler. Profile-v1 never ran model inference.

Independent Rust capture-environment at /tmp/fireprot-env-drift-audit-v1 records
EXACTLY9 changed identities: /etc/ld.so.cache changed; libiperf.so.0/.0.0.0 and
libsctp.so.1/.1.0.19 added through system-libraries and system-libraries-alias.
No inventoried Torch/model files differ. This does not retroactively accept
failed outputs or prove which process installed those libraries. Preserve old
inventory; no guard relaxed. current.json and diff.json retain the audit.

Additional controller-reporting bug: the script named queue.py shadows stdlib
queue when Ubuntu apport handles its exception, re-importing the controller and
overwriting status with AssertionError. queue.log retains original benchmark
failure, and supervisor/journal establish the actual environment rejection.
Future controllers must avoid stdlib module names and use main guards. No old
controller was manually restarted; exception-hook import caused that trace.

Prioritize actual remaining model hotspots, per user. Created fresh
/tmp/fireprot-latest-stage-profile-v2 from prepared stage-v1 source, new request
latest-stage-profile-n1536-v2, same strictFP32R3D200S1seed101HIP7 and arithmetic.
Pinned fresh full inventory SHA1c0fd5dc9e729cccc2ae6273663f79c59b359c3bda7f5b1d7d78296a7a09bb2f
at its environment.json; old inventory unchanged. Removed only prerequisite
waiting on failed native comparison; all8GCD idle confirmed before inference.
ACTIVE driver2186540, Rust child2186549; stage=infer and PID confirmed live.
Root status.json,driver.log,infer.stdout,infer.stderr. Do not restart on timeout.
Local preparer /tmp/fireprot-prepare-stage-profile-v2.py and launch receipt
/tmp/fireprot-stage-profile-v2-launch.json. Existing driver retains all11 bits
and fullCIF checks versus native,D16,latest anchors after inference, and writes
collected.json stage totals only after success. Stage intervals are inclusive;
no dense profiler or E2E gain claim. Next inspect this same run, rank measured
latest-model stage costs, then pursue dominant operators. Direct paired E2E
must eventually rerun under one stable pinned environment; do not delay hotspot
analysis behind another long comparison first. No GPU work overlaps this profile.


## Latest stage profile succeeds; Pairformer dominates (2026-10-06)

User requested retry. Existing stage-v2 was already TERMINAL SUCCESS; do not
rerun it. All11 arrays and fullCIF match native,D16,latest anchors. Coarse GPU
inclusive totals: input3.68776s; template49.19386s/4calls; MSA20.92875s/4calls;
Pairformer221.90618s/4calls; trunk295.92225s; diffusion46.51445s; auxheads7.65773s;
rollout54.17223s includes diffusion+aux. Trunk ~85% of trunk+rollout, Pairformer
~63%. Instrumented attribution only, NOT E2E. Collected record SHA
dc87f812eab418a0cbc2fb42b0c58020cdc937a97782eabca38fe62660edc135 copied to
provenance/openfold3_latest_stage_profile.json with fresh inventory and9 system
identity changes. This does not accept the failed native-v2 E2E run.

ACTIVE fine stage profile /tmp/fireprot-latest-stage-profile-v3, driver2194758,
Rust child2194767; status=infer and actual driver confirmed live. Request
latest-stage-profile-n1536-v3, same fixed science/source/environment. Frozen
copy adds inclusive forward timers for tri_mul_out/in,tri_att_start/end,
pair_transition,attn_pair_bias,single_transition plus MSA operations. Labels
separate trunk_pairformer/template/MSA/confidence. Template pair embedder/stack
also timed separately. First-call tensor metadata records shapes,strides,dtypes
without tensor arithmetic, copies or scalar checks. No dense Torch profiler.
Wrapper cleanup and full11/CIF gates unchanged. Outputs status/driverlog and
collected.json in root. Local preparer /tmp/fireprot-prepare-stage-profile-v3.py;
launch /tmp/fireprot-stage-profile-v3-launch.json. No concurrent GPU experiments.

Source audit: external pairformer is core/model/latent/pairformer.py, not layers.
PairFormerBlock contains pair_stack,attn_pair_bias,single_transition; PairBlock
contains the five main triangle/transition targets. TemplateEmbedderAllAtom
(core/model/latent/template_module.py) always computes template_pair_embedder
using current z, runs template_pair_stack, sums/divides,ReLU,linear_t; there is
no final template-absence zeroing in this function. Therefore no-template search
alone does NOT justify skipping its49s or caching its whole output across cycles.
Do not implement that shortcut without separate scientific/source/bitwise proof.
Next use actual fine-stage totals to choose largest remaining operator, then
profile that operator's GEMM/copy/norm/attention components. No new HIP kernel
chosen yet; user explicitly prioritizes measured major costs.


## Fine stage profile accepted; target native triangle multiplication (2026-10-06)

User authorizes continuation. Polled existing v3 PIDs until terminal success;
all11 arrays and fullCIF remain exact versus native,D16,latest. Evidence
provenance/openfold3_latest_pairformer_profile.json; remote collected SHA
331496414114ee840ab115f66d828cadff696e1eab2de8ea8e3e32c5ea67ca2e.
Trunk Pairformer228.209s inclusive: tri_att_start54.0195,end56.7705 (110.790total);
tri_mul_out48.8124,in48.3926 (97.205total); pair_transition13.7981;
attn_pair_bias3.5557;single_transition0.0586. Each leaf193calls, including tuning.
Other major observed template stack48.808s: attention11.5674+25.4130 and
multiplication5.2013+5.2166. Trunk303.444, diffusion46.7685,confidence7.6650.
Nested diagnostic events and profiler overhead are not new E2E claims.

First next target is native triangle multiplication, which remains97s after
multiple attention optimizations. Do not automatically implement moreQK tuning.
Attention remains110.8s and will stay a major alternative based on actual
remaining opportunities. Native source audit in
core/model/layers/triangular_multiplicative_update.py found standard inference
uses chunked projections/repeated norms and overwritten-z cache; also a distinct
FusedTriangleMultiplicativeUpdate exists. Need actual runtime module type and
operator attribution before selecting a rewrite. Do not assume standard versus
fused from class names in docs. Preserve aliasing, GEMM shape/order and all bits.

ACTIVE targeted diagnostic /tmp/fireprot-trimul-profile-v1 onDiamondHill,
PID2202273, Rust child2202285. status=infer; actual PID confirmed live. Request
trimul-profile-n1536-v1; same science/environment/workspacecandidate. Captures
exactly ONE outgoing and ONE incoming trunk multiplication using Torch profiler,
records stage-labelled operation shapes/self CPU/device attribution. First
actual inputs/masks and module state_dict are saved to two CPU .pt files in
requestoutput, with class_name stored as module_type, outgoing flag and actual
forward flags. CPU copies preserve original tensors before inplace execution;
snapshot overhead makes outer stage timing unsuitable for comparisons. Full
model proceeds unchanged and all11/fullCIF checks still required versus three
anchors. Driver pins both input artifacts only after all gates. No dense whole-
trunk capture. Root status.json/driver.log/collected.json; no automaticretry.
Local preparer /tmp/fireprot-prepare-trimul-profile-v1.py; launch receipt
/tmp/fireprot-trimul-profile-v1-launch.json. No concurrent GPU workload launched.
Next inspect bounded operation tables and runtime class, replay those captured
trained operators, then plan/implement a qualified HIP/CK change for largest
cost. Existing workspace plan does not authorize unplanned arithmetic rewrites;
use a new focused plan after this operator evidence resolves the seam.

Early read-only snapshot inspection (no GPU replay) confirms runtime classes
TriangleMultiplicationIncoming and TriangleMultiplicationOutgoing, NOT fused
variants. Both actual inputs are FP32 [1,1536,1536,128], contiguous strides
[301989888,196608,128,1]. Flags inplace_safe=True,_add_with_inplace=True;
cueq/tritonFalse. Six projection matrices128x128 and input/output norm vectors128.
Saved files exist and Torch weights_only/mmap CPU read succeeds. Do not use their
existence as completed diagnostic acceptance; fullmodel check remains pending.
Possible follow-up seams are exact reuse of repeated input normalization and
fusion of native-order pointwise gates with packing; preserve actual GEMM shapes
and residual aliasing. Choose only after bounded operator table confirms cost.

## Triangle memory work: real operator wins, model validation active (2026-10-06)

User explicitly requests continued optimization of BOTH triangle multiplication
and attention. New committed plan8bb3e13 on branch
openfold3-triangle-memory-optimization:
docs/plans/2026-10-06-gitnexus-plan-openfold3-triangle-memory-optimization.md.
Scope is exact memory/layout improvements; arithmetic/recycle changes excluded.
Prior user Deep preference retained. GitNexus current schema4runner and index
proved; new experiment symbols have no previous consumers. Loader reused unchanged.

Both bounded diagnostics succeeded, all11/fullCIF against native,D16,latest.
Compact provenance/openfold3_triangle_{trimul,triatt}_profile.json retains pins
and aten tables. Attention current per-call QKsoftmax144ms,PV71ms; K packing8ms.
Do not sum aten parents with kernel events. Attention memory work is only a small
component; future compute changes need their own qualified scope.

New experiments/triangle_memory bit-copy C++ HIP, guarded ctypes glue and
source-pinned upstream method substitution. Native sourceSHA
612caf7abd07a86a7fdf4eda7e0a2170c04b407f9121fc286ab864a648db1ba7.
Copies only: two projection assignments, BMM input contiguous, output LN input
contiguous. Existing GEMM shape,norm,sigmoid,residual unchanged. Derived replay
Apache attribution retained in THIRD_PARTY. Rust native HIP builder reused.
All8synthetic cases exact inclNaN/zero,offsetcanaries,nondefaultstream;2fallbacks.
Real trained fulloperator incoming238.182->160.799ms1.481x;outgoing239.371->167.584
1.428x,all14outputs per direction exact (firstcalls excluded from medians).
First replay attempt failed source-substitution count before GPU; fixed to target
only nonTritonbranch. Successful final source is retained. No E2E claim.
Remote /tmp/fireprot-triangle-memory-v1 has build/build.json,copy-screen.json,
trimul-replay.json and source. Provenance/openfold3_triangle_copy_operator_v1.json.

ACTIVE fullmodel driver2219704 Rust2219713 /tmp/fireprot-triangle-model-v1.
Frozen lateststagev3 + experimental scope, strictR3D200S1seed101N1536HIP7; ordinary
memoryfill remains. Outputs triangle-copy-model-n1536-v1. Driver verifies all11
and fullCIF against3anchors, retains fine stage events and triangle-copy.json.
Unsupported calls native. Model success still pending; no E2E measurement.

ACTIVE guarded attentioncontroller2226479 /tmp/fireprot-triangle-attention-v1.
Waits for model collected correctness and idle8GCD; then Rustbuild+synthetic
Kpacking screen and real trained attention replay. No concurrent GPU work.
status.json/controller.log,screen.stdout/stderr,replay.stdout/stderr authoritative.
No automaticretry. Source attention_copy.cpp packs exact B128H4N1536D32
headlastK withstrides196608,32,128,1. Replay source-pins existing WorkspaceFusion
and changes only packed_k; native arithmetic/workspace lifecycle retained.
Pending actual performance and exact acceptance. Old model roots not mutated.
Launch/sourcearchive pins provenance/openfold3_triangle_memory_launch.json.
Local archive outputs/experiments/triangle-memory-local-20261006/launch-sources-v1.tar.gz
contains preparer,controller,fullbuildreceipt and rawboundedprofilereports.
Next poll originalPIDs, resolve any failures without restart, then N1000fallback,
combinedmodel ifattentionwins, and fresh pairedE2E stableenvironment.

## Both operator tracks pass; guarded combined E2E sequence launched (2026-10-06)

Committed/pushed4cb966e with experimental source and operator evidence. Newbranch
openfold3-triangle-memory-optimization. ISA PDF remains untracked. Force-index
refresh fixed malformed BasicBlock IDs in an incremental detect_changes result;
subsequent full staged gate was complete,33expected symbols,2expected processes.
Typed schema4runner/indexcurrent proved. Never trust old MCP behind-count prose.

Multiplication model2219704/2219713 TERMINAL SUCCESS. All11/fullCIF vs native,D16,
latest exact. Selected430modulecalls,72fallbacks,7740HIPcopies,0copyfallbacks.
Trunk tri_mul_out32.634s,in31.767s;Pairformer192.279s,trunk263.587s diagnostic.
Recorded provenance/openfold3_triangle_copy_model_v1.json. No new E2E ratio.
Attentioncontroller2226479 TERMINAL SUCCESS. Synthetic Kpack0.771->0.265ms,
complete trained attention end282.995->276.846ms1.02221x,start272.867->266.748
1.02294x. All14outputs per directionexact;168packcalls,actualshape128,4,1536,32
stride196608,32,128,1. provenance/openfold3_triangle_attention_copy_v1.json.

ACTIVE combined diagnostic2229872/Rust2229881 at
/tmp/fireprot-triangle-combined-v1; status.json/driver.log authoritative.
Frozen multiplicationmodel+newkey_scope replaces only WorkspaceFusion Kpacking,
sourceSHA47d2efa63c4402367af9282f05411b2caa123dc9f8826b5314f82afae8cc966d.
All11/CIF vs3anchors and >0mul/keydispatch gates retained. No arithmetic change.

ACTIVE controller2235527 /tmp/fireprot-triangle-e2e-v1,waiting_combined_model.
Fresh separately frozen worker shares both arms; removes diagnostic model_probe;
FIREPROT_TRIANGLE_MEMORY=0/1 controls bothscopes. Otherworkspace/cache/D16flags
identical. Expectedchanges onlythisflag+invocation.profileSHA. Aftercombinedgate:
1 inspect7requests;2 N1000fixedscience fallback againstzero-masknative all11/CIF,
   requiresmulaccepted0,copies0,keycalls0,nativefallback>0;
3 verifyallfrozenfiles,idle8GCD,then Rust3AB/BA/ABincrementalN1536pairs;
4 all6outputs all11/fullCIF versus native,D16,latest and nooperatorprofile;
5 reverify frozenfiles,collectrealperformance and archive to durable usualroot
   triangle-memory-e2e-v1.tar.gz. No automaticretry; anyfailure stopssequence.
status.json/controller.log,{fallback-infer,benchmark}.stdout/stderr,paired-n1536/
benchmark.json/order.jsonl and collected.json authoritative. Do not restart on
SSHtimeout or launch competingGPUwork/heavybuild while sequence runs.
Launchpins provenance/openfold3_triangle_combined_e2e_launch.json; localarchive
outputs/experiments/triangle-memory-local-20261006/combined-and-e2e-launch-v1.tar.gz.
Originalnative directratio remainsofficial1.499x; old2.043singlepair preliminary.
Latestworkspace1.201x incremental; do not multiplyratios. No plateau established.

User requested Protenix recap while workcontinues: docs/protenix-native-requests.md
historicalMini-defaultv0.5FP32C4S1seed101threepairs N1000single-request82.074->27.949s
2.937x/model3.312x;N1536177.345->45.596s3.889x/model4.432x. Explicitlyreported
historicalHIP/SMImapping isolationlimitation,worker-timeboundary excludesparent
preflight/startup,andsamecheckoutnativebaseline (notunmodifiedupstream). Do not
silentlypromotehistoricalmeasurements tonewRustisolatedE2E. Ongoingtaskstillboth
OpenFold3triangleoptimizations,notreplacedbyrecap.

## User sequencing update: finish this cycle, one more, then README (2026-10-06)

User explicitly requests: finish the current combined triangle-memory cycle;
perform ONE additional optimization cycle; then improve the main README with
clear entry-point usage and E2E speedup ratios. Preserve correctness checks each
trial. Do not jump to README before the second cycle's evidence exists. Final
README must distinguish direct original-implementation comparison from incremental
ratios, model timing from complete-request timing, model scientific contracts,
experimental versus maintained paths, and historical isolation limitations.
Repository prose remains English; responses Chinese. This narrows the formerly
open-ended optimization sequence to these two cycles followed by documentation.
Current cycle is controlled by2235527 /tmp/fireprot-triangle-e2e-v1; no competing
GPU work or heavy builds while its benchmark runs. Next cycle should target the
remaining major measured cost (attention QK+softmax/PV), not just small key packing;
new arithmetic/kernel scope requires a focused new/deepened implementation plan.

## Combined model and N1000 gates pass; six-run E2E now active (2026-10-06)

Combinedmodel2229872/Rust2229881 TERMINAL SUCCESS: all11/fullCIF against3anchors,
430multiplication calls,7740copies,5064keypacks. N1000 controllerfallback completed
all11/fullCIF against zero-masknative; acceptedmul0,copies0,keycalls0,nativefallback
positive. Evidence provenance/openfold3_triangle_combined_model_v1.json includes
both gates. No E2E acceptance until allpairs finish.

ACTIVE controller2235527 now stage=benchmark, Rustscheduler2242825 under
/tmp/fireprot-triangle-e2e-v1. paired-n1536/order.jsonl begins baseline0, same
frozen sharedworker and expectedmemoryflag/profilechanges. Do not mutate frozen
worker,run additionalGPUjobs or heavyremotebuilds untilterminal. Rootstatus,
benchmark.stdout/stderr,paired-n1536/benchmark.json,order.jsonl authoritative.

NEXT-cycle GitNexus plan PUBLISHED but execution not yet started:
docs/plans/2026-10-06-gitnexus-plan-openfold3-attention-final-cycle.md.
CurrentHEAD9aab0cd,typedcurrent schema4index,strictrefresh1. Planning receipts
/tmp/fireprot-attention-cycle2-{provenance,read-plan,status}.json;read-plan digest
must be retained on execution. Raw v12 source locally
/tmp/fireprot-fusion-combined-v12.cpp plus asm/metadata. Read all6patches and
FUSION.md; retainaccumulationorder. Alternatives: nativebmm(out=retainedscores)+
existingOwnedScoreSoftmax; or two16KiBLDS buffers removingonly5reusebarriers,
retaining6producerbarriers. QK+softmax144ms vsPV71ms/call actualprofile.
No new GPU experiment launched. Plan supports exactlyoneadditionalcycle before
README, notindefiniteoptimization. Source-aware queriescontextWorkspaceFusion,
PDGcontrols10of114 edges (explicitlypartial); owner/lock/drain constraintssource
verified. ExistingWorkspaceFusion remainsunchangedoutsideexperimentalscopes.

README preparation read-only findings: docs/rust-running.md still has obsolete
intro 'accelerated OpenFold3 pending' and migrationremaining-gates text despite
acceptedroutes. Repair relevantlinkedusage documentation when finalREADMEdone.
Actual target/release/fireprot --help confirms inspect/infer/infer-batch,
measure-inference/benchmark-pairs/compare-runs,build/install commands. Build all
CLI/runtime/bridge viaCargo--locked; scripts/infer.sh isexecRustinfer,smoke.sh is
Rustinfer-batch. Requests requirefresh run_id/output_dir; RustsetsGPUvisibility
fromrequest,so don'tsuggestbareHIP_VISIBLE_DEVICES overridesprofiles.
Atlasfoldfinal docs: N1536chunked-native553.359->203.910s2.714x,sequentialsets,
uncutnativeOOM; excludesimports/hash/preflight,seed102R4/fivepassesD100S1native
precision withAtlasLMpreserved. Earlierpaired2.671x isdifferentcandidate. Need
clearlylabel allmodels'timingboundaries andbench versions in READMEtable.

## Final-cycle CPU preparation and guarded split replay queued (2026-10-06)

CurrentHEADbb28827, newworkingbranch openfold3-attention-final-cycle. Plan isalready
committed,pushed onparentbranch;read-plan digest
sha256:a92e3b7cbc997193f7a4f0b84b7c0bea261eff211321658979409ff02746bdbf.
Workreanchor /tmp/fireprot-attention-cycle2-work-provenance.json proves all6cited
paths unchanged fromplan; HEAD/journal/docs onlychanged. IndependentCPUpreparation
advanced while priorE2E runs; newGPU work remains strictlydeferred toits terminal.

New experiments/attention_final/{split_workspace,screen_split,replay_split}.py:
source-pinned WorkspaceFusion callfactory preserves lock,owner,stream,private
storage/fail-closed/nativePV. Botharms useacceptedHIPkeypacking. Splitarm replaces
only fusedQKsoftmax launch withtorch.bmm(out=retainedscores) andOwnedScoreSoftmax.
21raw/probability cases withPVbits checks; then14trainedcalls perstart/end direction,
6timed perarm excludingfirstcall. Exactmode strings inreport distinguishsplit
fromfused ratherthan claiming fusedkernel launched. Syntaxchecked; dynamicmethod
substitution separatelycompiled fromactualsource viaAST. NoGPUresultyet.

ACTIVE QUEUED controller2250000 /tmp/fireprot-attention-final-split-v1.
status=waiting_previous_cycle. Waits original2235527 untilsucceeded, verifies
collectedcorrectness,median_speedup>1 andobserved_trend=faster_median_and_p95,
thenall8GCDidle; numericalscreen then trainedreplay, noautoretry. Priorfailure or
nonwinningE2E stops beforeGPU and requiresreview. controller.py/main guard avoids
stdlibshadow issue. status.json/controller.log,screen.stdout/stderr,screen.json,
replay.stdout/stderr,replay.json authoritative. It does NOT startfullmodel orE2E.
AlternativeB twoLDSbuffers notimplementedyet. README remainsafterthiscycle.
Launch/sourcearchive pins provenance/openfold3_attention_final_split_launch.json.

Latest priorE2Ejournal: baseline0 measured+verified atoffset465.448s; candidate0
started. No completepair or3pairratioyet. Scheduler2242825 anddriver2235527 still
active; do notlaunch competingGPUwork orheavybuild. Bothcontrollersalreadyown
sequence; poll originalPIDs/receipts,neverrestartonSSHtimeout.

Read-only next-scope evidence: actualcombinedv12 kernel hasstage16*256floats,
vectorwrite stage[destination],read stage[stage_index(wave,...)],6producerbarriers
and5reusebarriers. Potentialtwo-buffer transformation doublesstage, adds
(band&1)*4096 toBOTHwrite/read addresses andremovesonlyif(band!=5)barriers; keeps
MFMA/add/reductionorder. NeedcompilerISA/resourceproof andGPUstressbeforerelying.
Nativebmmout prototype stillneeds selectedkernel/hiddenallocation inspection
beforepromotion even ifbitwise/replaypass. UserISA PDF untracked; nevercommit.

## Next model supplied by user: Protenix Base-default v0.5.0 (2026-10-06)

User's NEXT target checkpoint on DiamondHill:
/media/IntelSSD/fireprot/runtime/checkpoint/protenix_base_default_v0.5.0.pt.
Remote stat/readlink verified file exists,1474265486bytes,resolvedpathidentical,
mtime2026-10-06 21:34:54+0800. No fullhash or torchload yet; deferweightI/O until
currentbenchmarkterminal. Inventory appended docs/diamondhill-checkpoints.md.
This extends the queue AFTER currentOpenFold3cycle +oneadditionalcycle +README,
not an instruction to interrupt those. Long1000/1536workloads remain priority.

Source read Protenix/configs/configs_model_type.py:70-76 Basev0.5 defaultN_cycle10,
N_step200; Mini:153-180 C4,D5 andsmallerblocks. Resolveactualpinnedconfig before
baseline; do notuseMini's work/speedupratios as Base evidence. Maintainedworker
python/fireprot_workers/protenix.py:18-24 explicitlyrejectsBase now; requires
explicitintegration,source/weightpins,capabilityaudit,strictload andnativeagreement
before optimization claims. No source/modelexecution change made inthisturn.
Current E2E2235527/2242825 and queuedfinalcycle2250000 unchanged,stillactive.

## Both requested cycles complete; README clarified (2026-10-06)

Triangle-memory controller2235527 TERMINAL SUCCESS, six runs and all18anchor
comparisons (native,D16,latest) all11bits/fullCIF exact. ThreeAB/BA/AB E2E medians
452.470689673 ->416.226066696s,1.087079176x,8.01038%less time; observedp95
464.711334616->428.427770304. Model356.594822704->321.698897176s,1.108473874x.
Allcandidate430mulcalls/7740copies/5064keypacks; N1000fallbackpasses. Frozenworker
integrity andRustinventoriespassed. Evidence
provenance/openfold3_triangle_memory_e2e_n1536.json. Archive durableusualroot
triangle-memory-e2e-v1.tar.gz SHA
cc3ec462b41d8ebb71271fbe03248cbd12d0c550bcadaef9050853d6ba9d555f.
This remainsincremental; latestcombination hasno accepted3pairdirectnative result.

Finaladditionalcycle alternativeA2250000 TERMINAL SUCCESS numerically, REJECTED
forperformance. Nativebmm(out=retainedscores)+ownedHIPsoftmax passes21raw/probability
casesplusPVbits andtrainedstart/end replay. End276.495->306.176ms;start266.272->
296.169ms,about11%slower. No furthermodel/E2Egate needed forrejectedarm.
provenance/openfold3_attention_final_split_result.json.

AlternativeB new qk_double_buffer.cpp reconstructs source-pinnedv12 with two16KiB
LDSbuffers; BOTHproducer/consumeraddresses use(band&1)*4096; removesonly5reuse
barriers. SameMFMA/add/softmaxorder. Prioruser-authorizedLunaISAagent reviewedactual
localasm:producer/consumerparity0/16384 alternates,rangeends32767,6barriers(mode
false independentlycounted6too),32KiBLDS,trueVGPR62SGPR34/no spills. Remote actual
metadata confirms32KiB,VGPR62/51,SGPR34/20,no spills. Noactualoccupancyclaim.

v1controller2284410 FAILED beforeGPU publishing: my llvm-objcopy --dump-section
command lacked a separateoutput andrewrote the unpublishedexperimentalELF layout
inits Rustcache. Rust correctly rejectedSHA mismatch. Rawfailedartifact/expected
andactualSHA preservedv1/inspection-failure.json; no measured/maintainedkernel
bundle changed. Explicit recoveryusesFRESHisolatedcache atv2 andhost-copy.so output;
checks librarySHAunchangedbeforepublish. Do notreusecorruptedv1cacheentry.

v2controller2286046 /tmp/fireprot-attention-final-double-v2 TERMINAL SUCCESS,
21/21numericalrecords andtrainedattention exact. End276.268->279.788ms;
start266.095->269.559ms (~1.3%slower). REJECTED, no modelpromotion. Evidence
provenance/openfold3_attention_final_double_result.json. Bothalternative roots,
v1failure andv2source/build/bundle/inspection/probe/replay archived atdurableusualroot
final-attention-cycle-v1.tar.gz SHA
b316bc75ae3de473ae0d363ad95a78337ecb6b00f25144a6599a82f56dc8d356,2043565bytes.
No activeexperiment/controller remains. StopOpenFold3tuning peruser'sone-more-cycle
instruction; retainedbest istriangle-memory combination. Notaglobaloptimumclaim.

README nowEnglish and reorganized: Rustbuild/probe,request-vs-profile,hip_device,
freshIDs,5real request/profileentryexamples,output/measure/pairedcommands, support
matrix includingBaseplanned, directnative versusincrementalperformance tables,
workload/cache/isolationlimits, maintainedversusfrozenexperiments. Keptlicense
sectionbyteidentical andChinesehistoricalbaselineuntouched. All5referencedrequests
passactualRustinspect;3profilesexist/parse;allediteddocslinks+anchorsvalidated.
FixedstaleRustmigration/acceleratedOpenFold3claims inrust-running/worker-protocol
andcheckpointinventory. UpdatedOpenFold3performance topwithfinalresults/rejections.

IMPORTANT verifiedfromprovenance/protenix_native_requests_fa7cc55.json protocol:
Protenixhistorical2.937x/3.889x usesC4/D1/S1,notdefaultMiniD5. Explicitlyinformeduser,
READMEanddocs/protenix-native-requests.md nowstateD1. EarlierchatrecapgaveC4/S1without
D; nofutureBase200-step speeduptransfer. AtlasN1536reported2.714xsequentialsets
vschunkednative;uncutnativeOOM. CurrentOpenFold3direct1.499x isearliercombination,
notlatest; nevermultiplyratios. NexttaskProtenixBasecheckpointperpriorentry.

## 2026-10-06 Protenix Base entry integration in progress

OpenFold3 final cycles and README committed/pushed as 04f6ca0. Base checkpoint
/media/IntelSSD/fireprot/runtime/checkpoint/protenix_base_default_v0.5.0.pt
SHA256 9ea20b0aba42f2256711da1d0cd081510a4b291e64375bff6b70ced70b87a5f1,
1474265486 bytes. CPU weights_only load and strict Protenix state-dict loading
passed: 4092 tensors, 368088191 FP32 parameters, no missing/unexpected keys.
Defaults C10/D200/S5, 48 Pairformer blocks. Entry now permits Base v0.5 FP32;
30 worker request tests pass. N1000/N1536 native and experimental FireProt
requests explicitly use C10/D200/S1 seed101 no MSA/templates. Base profile
keeps Mini-only fast initialization and template omission OFF. Existing shape
capability guards remain intact. User renewed restriction: HIP6/7 only.
First N1000 native complete invocation measurement launched PID2289487 at
outputs/experiments/protenix-base-first-20261006 on DiamondHill, output
outputs/protenix-base-n1000-native. Check terminal result before any next GPU
experiment. No Base GPU speedup or numerical agreement claimed yet.

## 2026-10-06 Base first migration results and strict initialization qualification

Base entry/config commit b7550f1 pushed on protenix-base-integration. GPU restriction
is HIP6/7 only; every GPU run in this round used HIP7. User narrowed experiments:
N1000 is sufficient; approximately N600 allowed; do not launch N1536 this round.
All five GPU invocations and CPU initialization probe are now TERMINAL, no active
experiment remains.

Initial nondeterministic Base FP32 C10/D200/S1 seed101 N1000 pair completed.
Native E2E419.05271204s/model349.312415634s, broad Mini HIP/CK transplant
E2E185.191887449s/model115.086834583s. CK attention and packed multiplication each
selected1048times. REJECTED: coordinate alignedRMSD0.0467942884A >1e-3 gate;
confidence maxdelta0.000518799. Rust withholds performance verdict. Identical
native repeat ALSO REJECTED: alignedRMSD0.00828346279A, so do NOT assign all error
to CK. Initial native and candidate used upstream deterministic=False.
Evidence provenance/protenix_base_first_transplant.json retains both failures.

Base worker now MANDATES strict deterministic algorithms, warn_only=False and
fill_uninitialized_memory=True, configdeterministic=True, for bothbackends;
asserts contract at model boundary. Mini/Tiny behavior unchanged. Explicit input
benchmarks/protenix/inputs/umem_L01000_prefix600.json is a 600residue prefix ofthe
N1000 developmentcase; metadata records derivation, not scientificaccuracy set.

CPU experiment experiments/protenix_base/check_initialization.py compared actual
Base native vs deferredconstruction, strictcheckpointload, allstate tensors/all
buffers and Python/NumPy/TorchCPURNG: EXACT. Native50.22233s vs3.63491s construction,
1553deferredcalls/269132320uniformelements, no unsupportedinitializers. Extended
construction._reason only to add Basev0.5 FP32, retaining fullcoveragecheck,
sourcehashes/thread/RNGguards/native recovery. Tests now cover Base missing/full
checkpoints;42focusedtests pass. All6Base requests pass Rustinspect.

StrictN600 C10D200S1seed101 FP32 noMSA/templates, HIP7 serialnative->candidate pair:
E2E179.349930964->127.046555968s=1.41168668129x, ONEPAIR only(insufficient_repeats),
notsteady-state model speedup. All9arrays bitwise, rawcoords0/confidence0, CIFatom
identitiesexact. Runtimefastconstruction selected1/recovery0/errors0, all1048calls
per trianglefamily native. Profile configs/diamondhill/protenix-base-init-only.json
explicitlydisables GPU substitutions, enablesfastinit, retains templates.
Use configs/requests/protenix-base-n600-{native,fireprot}.json withfreshIDs/outputs.
Qualifiedscopeisinitialization-only N600. Broadoperator profile remains experimental,
strictN1000 operatorqualification OPEN; no accepted CK/GPUoperator speedup onBase.

Durablearchive(all5GPUoutputs/controls/comparisons/CPUreceipt):
/media/IntelSSD/fireprot/outputs/experiments/protenix-base-qualification-20261006.tar.gz
SHA25639a2bd4260e632529ee5b9088323c2f3a1282f697be2f3ad9286c4b92fc0d20d.
Evidence provenance/protenix_base_strict600_initialization.json and
provenance/protenix_base_cpu_initialization.json. README/docs nowreportvalidated
singlepair1.412x separatelyfromfailedbroadtransplant. Do nottransferthese
numbers toN1000/N1536 ordefaultS5. Subsequentoperatorisolation needsstrictnative
reference andpertrialcorrectnesschecks; no thresholdrelaxation authorized.

## 2026-10-06 active goal: strict N1000 attention isolation started

Goal remains active: consistently optimize Protenix Base while keeping correctness.
Previous turn was progress (ce54cbf exact N600 initialization qualification).
Rechecked local clean tracked tree and DiamondHill GPU idle before launch.
Current LIVE native measurement PID2308434 under
/media/IntelSSD/fireprot/outputs/experiments/protenix-base-strict1000-attention-20261006.
Output outputs/protenix-base-strict1000-attention-native. HIP7 only.
Fixed BaseFP32 N1000 C10D200S1 seed101 strictdeterministic/memoryfill.
Profile derived from base-init-only; sets DISABLE_CK_ATTENTION=0,
DISABLE_LONG_TRIANGLE_ATTENTION=0, DISABLE_LONG_DIRECT_ATTENTION=0.
Other GPU substitutions stay disabled. Native backend ignores these hooks;
candidate will select CK attention and fast initialization only.
Candidate request/spec PREPARED, NOT LAUNCHED. Poll PID2308434 and native invocation
receipt; do not launch another GPU run until native terminal. Then run the
prepared fireprot-spec.json with Rust measure-inference into fireprot-measurement,
same root/CWD/env HIP7. Compare via Rust compare-runs with unchanged 1e-3 gates.
Do not restart based on a timeout. No strict N1000 result yet. Local launch
receipt /tmp/fireprot-base-strict1000-attention-launch.json records control hashes.
Potential fallback direction if CK-only fails: existing exact OF3 HIP softmax
supports N1000, but requires two biases; Protenix often provides only one. Need
real tensor/bias/add-order qualification before reuse; no implementation change yet.

### Strict N1000 attention continuation: verified live wait and fallback preparation

Re-polled PID2308434: confirmed live at elapsed03:06; no native invocation receipt
or completion yet. Do NOT restart. Candidate still NOT launched. Continue same
root and compare original prepared control hashes from local launch receipt.
Prepared unpromoted experiments/protenix_base/bias_softmax.cpp and
screen_bias_softmax.py while native runs. Based on maintained HIP softmax,
adds N600 and optional mask (no artificial +0 for single-bias Protenix), retains
ordered adds/native wave reduction; screen checks private in-place scores at
N600/N1000 B1/2/8 random/zero/large/partial-mask/all-mask. Python syntax check passed.
NO GPU build/screen yet; do not describe as qualified. Run only after timed
experiments finish. Existing CK-only candidate takes priority before fallback.
These files are untracked work in progress, not deployed or integrated. Keep goal
active; no new accepted speedup in this continuation.

### Strict N1000 CK attention rejected; exact softmax screen passed

Native PID2308434 TERMINAL SUCCESS, E2E430.647801512s. InvocationSHA
88e7989b111c197996f31445aa7950451a1f65ef6d4c04314da3037e3be8a486.
CK-only+fastinit PID2311416 TERMINAL SUCCESS in content but pairedcorrectnessFAIL:
E2E216.681660333s, alignedRMSD0.07495934435A, confidence maxdelta0.00131607056;
performance=null. Runtime1048CKattention,1048NATIVEmultiplication,fastinitselected1.
Evidence provenance/protenix_base_strict1000_attention.json. No accepted GPU gain.

Prepared HIP fallback now locally compiled AND remotely Rust-built/screened onHIP7
with no overlappingtiming. experiments/protenix_base/bias_softmax.cpp optionalmask
avoidsinvented+0add; retainsnativebiasadd/reductionorder, privateinplacescores.
screen_bias_softmax.py passes30/30BITWISE cases: N600/N1000 x B1/2/8 x
random/zero/large/partial-mask/all-mask. Evidence
provenance/protenix_base_bias_softmax_screen.json; remote screenroot
outputs/experiments/protenix-base-softmax-screen-20261006 contains build/cache/source.
NO trained-attention replay or model/E2E qualification yet. No productionintegration.
N1536 inheritedkernelbranch is notqualifiedbythisscreen and notrequested.

Now LIVE initialization-only N1000 control PID2314374, SAME attentionexperimentroot.
Files init-control-{request,profile,spec}.json; measurement init-control-measurement;
log init-control-launch.log; output outputs/protenix-base-strict1000-init-control.
HIP7, sameC10D200S1seed101 strictBase. All3attentiondisableflags resetto1; otherwise
sameprofileasCK experiment. BeforeanynextGPUwork pollthisPID/receipt toterminal.
Compareagainststrictnative viaRustcompare-runs: declarebackend,bridge,
3FIREPROT_DISABLE_*attentionenv0->1 and invocation.profile_sha256 change explicitly.
Thiscontrol checks N1000 strictbaseline/fastinitrepeatability beforeassigningcause
toCKalone. Nextoptimization: trainedreplay/fullmodel of exactHIPbias-softmax while
keepingnativeQK/PV. Must keepnative computationalordering and all9arraygates;
neverpromoteon30syntheticcasesalone. Goalactive, previouscontinuationmadeprogress.

## 2026-10-07 USER CORRECTIONS: quality reference, tolerance, and rigid drift

User explicitly corrected evaluation: do NOT equate agreement with native inference
with experimental structural accuracy. User confirmed experimental PDB/mmCIF truth,
agent chooses samples, then clarified exactness is not so important and authorized
aligned coordinate RMSD tolerance0.1A. User also required compensating global
translation/rotation: relative deformation matters. Keep numerical-vs-truth scopes
separate; preserve historical failed1e-3 receipts, do not relabel them originalpasses.
New0.1A is alignedRMSD, NOT maxperatom. Confidence discrepancies must remain visible;
no hidden rewriting of existinggenericMini/OF3 gates. CK route remainspriority.

StrictN1000 initcontrol PID2314374 nowTERMINAL SUCCESS: E2E357.921254723s,
model333.300734983s, all9arrays BITWISE vsstrictnative. Newfull CK/HIP+fastinit
PID2320185 TERMINAL SUCCESS: E2E146.034532219s vsnative430.647801512s (~2.949x,
onepair), allatomalignedRMSD0.0546879705A <user0.1A; CAaligned0.0439675883A.
RelativeCApair-distance RMSdelta0.0337561904A,p95abs0.0341522266A,max0.838484831A;
CA-onlylocal lDDT to NATIVE output0.99965923 (nottruthaccuracy). Confidence
maxdelta0.00192642212, mean0.000142860413; oldfixedconfidence1e-3 gate alsofails.
CIFatomidentities and4metadataarrays exact; CKattention1048 andpackedtrimul1048,
fastinitselected1. Legacycompare full-comparison-old-policy.json remainsfailed;
separatefull-drift.json is newUSER coordinate-gate report. Do notclaimalllegacy
confidencegatespassed. CK-only drift0.0749593443A, CA0.0631558372A,relativeCA
pairRMS0.048096469A,CA-lDDTtonative0.999299713. NowpassesUSERcoordinategate.

ExperimentalRustdiagnostic crates/fireprot-validation/examples/base_drift.rs
uses existingKabsch, bindsNPZcoords toCIFidentities/order, computesCApairdistance
andlocalCA-lDDT withoutalignment; emits0.1Acoordinateverdict, nottruthquality.
2testsprove rigidrotation/translationinvariance andinternaldeformationdetection;
releasebuilt/copied to SAMEremoteexperimentroot/base-drift. Firstcompilefailed
missingread_npz cap argument; fixedwith512MiB cap; tests/buildpassed. Uncommitted.

CURRENT LIVE fullrepeat PID2320885 placeholder: use actualPID from
outputs/experiments/protenix-base-strict1000-attention-20261006/full-repeat-pid
(onDiamondHill). full-repeat-{request,spec}.json, full-repeat-measurement,
full-repeat-launch.log. HIP7only, samefullprofile C10D200S1seed101 strict. Before
newGPUwork inspectactualPID/receipt; repeat intendedtoverifyfullpathstability.

GT discovery: exactN1000 RCSBsearch(entity_poly.rcsb_sample_sequence_length=1000,
resolution<=3A,Protein) returns3DEC_1,5UM6_1,5ZMW_1,8FY3_1. Locallyavailable mmCIFs
under/media/PM982/onestepfold/Dataset/raw/pdb_mmcif/{id[1:3]}/{id}.cif.gz.
Gemmi0.7.5 onfold env. CAcoverage:3DEC983/1000,res2.80;5UM6962/1000,res2.794;
5ZMW1000/1000,res2.500;8FY3561/1000. AllcanonicalstandardAA. Prefer5ZMWchainA
forfirsttruthcase (membraneCaATPase,E309Q,ligandstatecontext mustbereported).
NoGTcasecreated/runslaunchedyet; noexternaldatasetwrites. Eightexistingcases
areonly124-164aaandcannotvalidateN1000CK. CurrentumemN1000hasNOexperimentalGT.

Prepared untracked experiments/protenix_base/model_softmax.py for isolated
trainedverification, but deferdeployment whileCKpriorityundernewtolerance.
NoHIPsoftmaxmodel run yet. Do notdiscardthiswork; goalstillactive.
Actual full-repeat PID is 2322358 (authoritative launch output and full-repeat-pid).
The preceding PID2320885 text was explicitly a placeholder, not a process to poll.


## 2026-10-07 full repeat complete; baseline clarification and branch policy

Full-repeat PID2322358 TERMINAL SUCCESS: E2E146.199362672s/model121.540581826s,
invocation SHA4048ffe132efa42eeb87c59d9c1ec9f30d262a7cc8d0dfa0753a78f7dac3ea82.
All9arrays bitwise equal to firstfull146.034532219s/model121.083635204s.
Rust full-repeat-drift.json vsinitcontrol passes user0.1A coordinate criterion,
alignedRMSD0.0546879705055A. NoGPU experiment remains live fromthiscycle.
Use357.921254723s nativeGPU+fastinit asincrementalbaseline: full~2.451x;
vsCK-only216.681660333s ~1.484x. Raw-native430.647801512 includesnativeinit;
its~2.949x isseparate andmustnot silentlyreplace357.921 baseline. Not3pairedruns.
User explicitly directs: afterthiscycle switchto master, setGitHubdefaultmaster,
andworkdirectlythere consistently. Remote currently hasmain, nomaster; create
masterfromcompletedcycle, retainhistory/otherbranches. This overrides routine
featurebranch guidance. docs/isa remainsuser-owned/untracked. OnlyHIP6/7 allowed.

## 2026-10-07 next cycle: profiling and user-requested HIP Graph

Working/defaultbranch now master at05be488 (GitHubverified). Previouscycle made
progress: twofullCKruns146.035/146.199s, all9arraysrepeatbits, usercoordinatepass.
User explicitlyrequested graphcapture exploration. MaintainedBase path contains
noCUDA/HIPGraph capture; onlyunusedupstreamtriattentionautotune exposesoption.
Newisolatedexperiments stage_probe.py and recycle_graph.py, no productionedits.
Rust-supervised HIP7 stageprobe PID2325607 TERMINAL SUCCESS, 146.412439305s
instrumented invocation (NOT speedup); all9arraysbits vsfullCK baseline.
Pairformer10calls64.6418s; diffusion200calls25.8461s; MSA10calls11.7120s.
Nestedtrunk attention30.5245s, multiplication21.8787s, pairtransition8.5426s;
GPUeventintervals includehostgaps andparent/childoverlap, do notsumallrows.
Evidence provenance/protenix_base_stage_profile.json; fullprofile onDiamondHill
outputs/protenix-base-stage-profile-v2/base-stage-profile.json. Experimentroot
outputs/experiments/protenix-base-stage-profile-20261007. InitialPID2325295failed
preflightmissingframeworksiblings beforeGPU; retainedfailedreceipt. v2 adds
frameworkfireprot/kernels/runtime/target symlinks toexistingroot, newrunID.

CURRENT graphON PID2328007, root outputs/experiments/protenix-base-recycle-graph-20261007,
on-{request,profile,spec}.json/on-measurement/on-launch.log/on-pid. Snapshotpython
workerinjectsgraph_install aroundactualmodelcall. Profileflag
FIREPROT_BASE_RECYCLE_GRAPH=on. Onlytrunk48blockstack: firstactualcycleeager on
sidestream, secondcapture+replay, remaining8replay, staticS/Zupdatedeverycycle.
RNGcapturecheck, signatureguards, expected10calls/1capture/9replays, abortonerror.
CompleteE2Eincludescapture, no reducedwork. OFFcontrolpreparedNOTlaunched;
waitONterminal+correctness beforecontrol. LatestpollONlive01:57. Modeloutputs
outputs/protenix-base-recycle-graph-on. Do notlaunchcompetingGPUwork/restart.

GraphON2328007 TERMINAL SUCCESS:1capture/9replays/10calls, capture7.675233594s,
E2E145.534203221s/model120.948504849s, peakallocated10069061632B,reserved15661531136B.
All9arraysBITS vspriorfullCK; nativealignedRMSDunchanged0.0546879705055A. GraphOFF
2330587 TERMINAL SUCCESS sameworkersnapshot147.097888197s/model122.691487418s;
all9arraysbits ONvsOFF. Onepair1.01074445x only, no robusttrend/promotion.
Evidence provenance/protenix_base_recycle_graph.json includesinvocationreceipts.

NOW LIVE diffusiongraphON PID2332799, root outputs/experiments/protenix-base-diffusion-graph-20261007;
on-{request,profile,spec}.json/on-measurement/on-launch.log/on-pid. D200S1C10,
HIP7 fullCKwithoutrecyclegraph, frozenworkerinstallsnewdiffusion_graph.py.
Onlydenoiser captured; sampler/randomaugmentation/noise/Eulerupdatesoutside.
Firstactualstepwarm,secondcapture+replay,199replays expected. Staticx/noiselevel
updated; fixedinputs identity/versionsguarded. RestrictedN1000/7979atomcase.
OFFcontrolpreparedNOTlaunched. PollsamePIDreceipt; nooverlap/restart. Graphsource
newfilesuncommitted; onfailurecapturediagnosticthenfixisolatednewroot, preservev1.
DiffusionON2332799 TERMINAL SUCCESS E2E148.687373836/model123.635886643,
1capture199replays200calls, capture0.095326059s, allocated9521970176B,
reserved12245270528B. All9arraysbits vsfullCK; nativealignedRMSDunchanged.
GraphONinvocationSHA148db206ce6cf3de70f7c0f7a28050264bbe30ca13a65a9588b0127d0c6e4d22.
CURRENT LIVE diffusionOFF2335524, sameexperimentroot/off-measurement/off-launch.log,
output outputs/protenix-base-diffusion-graph-off. Waitterminal, thencompare/preserve.
Pairformergraph Rustcompare-runs passedunchangedgates ONvsOFF, onepairinsufficient
repeats. Comparisonembeddedin provenance/protenix_base_recycle_graph.json.

DiffusionOFF2335524 TERMINAL SUCCESS:145.257897318s/model120.845672093s.
ON148.687373836s gives0.976934985x (2.36%slower), oneBApair. All9arraysbits
OFFvsON; Rustcompare-runs correctnesspassed unchangedgates. Evidence
provenance/protenix_base_diffusion_graph.json. Neithergraphcandidatepromoted:
trunk~1%singlepairnotrobust; diffusionnegative. Bothproof-of-conceptscomplete,
noactiveGPUprocessfromthiscycle. Nextpriority remainingtrunkoperators (attention,
multiplication,transition); goalnotplateauedoverall. Maintainmaster/HIP6or7,
357.921s incrementalnativefastinitbaseline separatefrom~146sfullCK baseline.

## 2026-10-07 CK tile rescreen and published artifact trial

Previousgoalturn progress: graphproofscomplete, no graphpromotion; master3161a03.
Two new synthetic CK vector-bias tile screens TERMINAL SUCCESS onHIP7:
root outputs/experiments/protenix-base-ck-tiles-20261007 PID2337937;
M128N32W4 20.0085ms; M64N32W4 26.6160; M64N64W4 22.1360; M128N64W4 20.9253.
Wide root outputs/experiments/protenix-base-ck-tiles-wide-20261007 PID2339141;
M128N32W4 20.0298ms; M256N32W8 21.8518; M128N128W4 23.5046; M256N64W8 32.6484.
Allrandom/large-biascases pass2e-5abs/rel againstmaintainedCK;N32variantsbitwise.
Alternatingvariantorder,4warmrounds16measured, strictFP32/memoryfill, seed101.
Syntheticonly, outputallocationincluded, biasalreadycontiguous; no trained/E2E
claim. Retaincurrenttile; no slowercandidate deservesmodeltrial. Rawbuildreceipts
screen.build.json ineachroot; compactprovenance/protenix_base_ck_tiles{,_wide}.json.
Newexperimentfiles ck_tiles.cpp,ck_tiles_wide.cpp,screen_ck_tiles.py uncommitted.

NEXT live fullBase published-artifact trial PID2341331 under
outputs/experiments/protenix-base-published-20261007. published-{request,profile,spec}.json,
published-measurement/published-launch.log/published-pid. Output
outputs/protenix-base-artifacts-published. HIP7 C10D200S1FP32seed101, sameworkerbinary,
fullCK+fastinit. Changesonly4FIREPROT_*_BUNDLE envvars; no graph, no templateomission.
SourcecontrolpreparedNOTlaunched. BeforemoreGPUwork poll2341331/receiptterminal.
Rustpublishedsamebinarylibs to runtime/kernels/protenix-base-amd_extension-20261007
manifest91193edfb1c1fbfb3ba95bf2c92ac360d7e04197de672e44d58298415d7a12f8,
librarya544ef7a75ec3d4ed223ff9ef1ab8fe75ea542d5274b2e4b7ef7a84b6db16029;
CK runtime/kernels/protenix-base-ck_attention-20261007
manifestc855ec8cf65236dc42428f93cadff31f79cfcd0934fd2ca3e0bee45afbf41cb8,
library6109cb8ef87d1549e7180ea7daf5322c3a06d10302c38d79b4182a3feb1d5628.
publication.json pluslogs retainRustreceipts. Intendedremovecompilerinputscans,
notalterGPUmath. Verifybothbuild.mode=published,1048CK+1048packedcalls,all9bits;
thenlaunchsource-spec.json control. Usercoordinate0.1A policyunchanged; native
comparisonlegacyconfidencefailstillseparate. No speedupclaim beforecomparison.
Publishedcandidate2341331 TERMINAL SUCCESS137.23949901s/model112.447313437,
all9arraysbits vsfullCK, bothbuild.mode=published,1048attention+1048mul unchanged.
CURRENTsourcecontrolPID2342588,sameartifactexperimentroot/source-measurement,
source-launch.log/source-pid; output outputs/protenix-base-artifacts-source.
TwoadditionalRust-ownedpairs prepared repeat-spec.json (ABthenBA), NOTlaunched
untilsourcecontrolterminalandfirstpairverified. CombinedwithinitialBA gives3pairs.
Newconfig configs/diamondhill/protenix-base-published.json pinsbothpublished
manifestsandfastinit1, templateomission0; firstcandidateexact, repetitionpending.
Source2342588 TERMINAL SUCCESS146.264570738s/model121.883916005s. Firstpair
published137.23949901s gives1.065761474x/6.17%lessE2E; all9bits exactbetweenarms,
Rustcompare-runs unchangedgatespass. Evidenceprovenance/protenix_base_published_first_pair.json.
CURRENT LIVE Rustbenchmark-pairs PID2344669, rootartifactexperiment/repeat-measurement,
repeat-spec.json/repeat-launch.log/repeat-pid. TwoadditionalpairsABthenBA,4fresh
requests source-r1/published-r1/source-r2/published-r2. WaitsamePID, neverrestart.
Onterminal verifybenchmark.jsonsuccess,compareeachrepeatall9bitsandpublishedmodes;
aggregateexistingfirstpair+2newpairs viaRustcompare-runs (3observationsperarm)
using retainedaccepted/invocationpins. AllstrictgatesapplybetweenidenticalCKroutes.
Do notreplace357.921nativefastinitbaselinewithoutlabeling. NootherGPUexperiment
shouldoverlap orheavyremoteCPUbuildduringtimedruns. Workdirectlymaster.

## 2026-10-07 active published repetition and next Lt screen preparation

Previousgoalturn progress: CKtile alternatives rejected; publishedloadingfirstpair
1.06576x exact, committed/pushedmaster4dce024. CurrentRustbenchmark2344669 verified
live05:06. Journalnewbaseline measured146.3407s andnewcandidate~133.0227s, gatespass;
index2published-r2 started, index3source-r2 followsautomatically. Do notrestart or
launch GPU/remoteheavybuild untilbenchmarkterminal. Exacttimesuseinvocationreceipts,
notjournaloffsetsubtraction infinalreport.

Prepared newisolated files experiments/protenix_base/lt_padded.cpp and
screen_lt_padded.py, copiedonly tooutputs/experiments/protenix-base-lt-padded-20261007.
NO remotebuild/GPUrun yet. Screen targets actualBaseN1000 packedinput
shape128x1000x1000, strides1000128/1000/1, outputcontiguous. CurrentLtadapteronly
acceptscontinuousN1536, cannotdirectlyreuse. NewRAII C ABI queriesup to16heuristics
perincoming/outgoing, FP32HIPBLAS_COMPUTE_32F,32MiB callerworkspace; compares
alloutputswithTorchhipBLAS beforeinterleavedtimings. Synthetic2e-5abs/rel,seed101,
4warmup12measuredrounds, outputallocation/fill included,reusedworkspaceexcluded.
LocalPythonASTpasses. Localhipcccompilefailsmissinghipblaslt/hipblaslt.h inlocal
ROCm layout; do notclaimcompiled. RemoteexistingLtbuildprovidesheader/library,
performRust-ownedremotecompile onlyaftercurrenttimedbenchmarkterminal.
No productionchanges; useronlyHIP6/7, workmaster, maintaincoordinate0.1A policy.

## 2026-10-07 published repetition aggregation correction

Rustbenchmark2344669 TERMINAL SUCCESS, bothadditionalpairs andall6reference/pair
checks pass. Independentlychecked all6outputs(initial+repeats) all9arraysbits vs
initialsource; all3publishedroutecounts1048/1048 andbothmodepublished.
Two-compatible-pairE2Emedians146.1837160225->133.183402945s, positive.
Attempttoaggregateinitialmanualpair+2repeats REJECTEDbyRust:
"schema error: incompatible timing conditions". Actualcauseinitialinvocation
 timeout7200seconds, repeatedjobs1800seconds. Do NOT editreceipts/weakencomparator
orclaimsuccessful3pairresultfromthatmix. Failedspec retainedthree-pair-compare.json;
provenance/protenix_base_published_aggregation_rejection.json recordsreason.

NOW LIVE thirdcompatiblepair RustPID2350448 atsameartifactroot:
third-pair-spec.json/third-pair-measurement/third-pair-launch.log/third-pair-pid.
ABorder, source-r3 thenpublished-r3, both1800s invocationtimeout. Initialmanualpair
nowindependentpilot; final3pairaggregate must ONLY use repeat-measurement jobs
0(source-r1),1(published-r1),2(published-r2),3(source-r2), plus third-pair-measurement
0(source-r3),1(published-r3). Allsameprofiles/scientificsettings. FinalorderAB/BA/AB.
Old /tmp/fireprot-aggregate-published.py includespilot; DO NOT rerununchanged.
Onthirdterminalsuccess regeneratecompare-spec using3accepted/invocationsperarm,
retain all9bitwise andpublishedselectionchecks; reportmedians/modeltime andtrend.
NewLt screen remainssourcepreparedonly; noGPU untilthisbenchmarkterminal.
Compatibleaggregation script prepared atDiamondHill
/tmp/fireprot-aggregate-published-compatible.py; requires bothbenchmark.jsonphase
succeeded, excludesinitialpilot, combinesAB/BA/AB usingrepeat jobs0/3vs1/2 and
thirdpairjobs0vs1, verifiesall9bits/allpublishedselectionthenRustcompare-runs.
Writes three-pair-summary.json and three-pair-comparison.json inartifactroot.
ThirdpairPID2350448 verifiedlive02:42, baselinesource-r3 measured145.4045s and
verified, finalpublished-r3 nowrunning. Lt screen notbuilt/launched yet.
Thirdcompatiblepair2350448 TERMINAL SUCCESS. Correctedcompatibleaggregationpassed:
provenance/protenix_base_published_three_pairs.json. Final3AB/BA/ABpairs ONLY
1800secondinvocations, initial7200pilotexcluded. E2Emedians146.146472885 ->
133.436657165s,1.095249806x,8.70%less; individual1.0857875-1.0999834x,
p95ratio1.09278194, faster_median_and_p95. Model121.802533924->109.062242168s,
includeslazyloading; do notcallsteady-stateGPUcomputeimprovement. All6outputs
all9arraysbits tosourceanchor; all3pub AMD+CKmodepublished/1048each, Rustgatespass.
README/docsupdatedto3pairresult. Changesstilluncommittedthisturn.

NOW LIVE paddedLt screen PID2354292, root outputs/experiments/protenix-base-lt-padded-20261007,
launch.log/pid/screen.json/screen.build.json. Rust-ownednativecompile +PythonGPU
microbench, HIP7only. All8GCDidlepreflight savedpreflight-processes.json. No E2E
modelbenchmarkactive. PollsamePIDbeforeanyotherGPUexperiment. Onscreencomplete
inspectnumerics/timings bothdirections; onlyfasteralgo deservesBasefullmodeltrial.
No productionLtN1000adapteryet. Existingreference remainsTorchhipBLAS.
PaddedLt screen2354292 TERMINAL SUCCESS, remotebuildworked (localheaderfailure
notremoteissue). 16algorithms*2directions allpass2e-5syntheticgate. Fastestindex0
incoming7.85658ms vsTorchhipBLAS9.24803 (bitwise); outgoing7.90506 vs10.55755ms
(maxabs3.2782554626e-7, notbits). Evidenceprovenance/protenix_base_lt_padded_screen.json;
rawbuildreceiptinexperimentroot. Publishedsame library134e817b2f642812d8af854469cfc6d156335209856240e5eaa0c418b3b4efac
at runtime/kernels/protenix-base-lt-padded-20261007, manifest
23e3e90adb333bc26f589b66db0cb6b22cf89baf932f0ceee3f796f3dc3f174b.

CURRENT LIVE fullBaseLt candidate PID2355487, root
outputs/experiments/protenix-base-lt-model-20261007;on-{request,profile,spec}.json,
on-measurement/on-launch.log/on-pid; output outputs/protenix-base-lt-on.
Frozenworkersnapshot wrapsmodelcallwithmodel_lt_padded.py context (newexperiment,
notmaintainedLtadapter). Selectalgorithm0, exactstridesguards, retained32MiBworkspace,
requires524incoming+524outgoing/no fallback. FullpublishedAMD/CK profile otherwise
unchanged, strictFP32N1000C10D200S1seed101 HIP7. BothON/OFFspecs timeout1800consistent.
OFFcontrolpreparedNOTlaunched. PollcandidatebeforeGPUwork; inspectbase-lt-padded.json,
Rustbase-drift vsstrictinitnative AND versuspreviouspublished/fullCK predictions,
all9arraydeltas/confidences/CIFmetadata. User0.1Aalignedgate; old1e-3legacygate
mayfail andmustnotbehidden. Onlyaftercandidateacceptedcoordinateassessmentstart
OFFcontrolsameworker; compareE2Ethenrepeatsifgain. No newLtmodelresultclaimedyet.
LtON2355487 TERMINAL SUCCESS E2E134.389951601s/model109.776045909s,
524incoming/524outgoing/fallback0. oninvocationSHAd5ab774efc3d2ee48e0706f47a803aebc06538b721fe8c4754d2fbf60677a425.
Rustnative-drift vsstrictinitnative aligned0.00609982394215A,CA0.00602146;
previous-drift vspublished-r1 aligned0.0505035007132A,CA0.03884285. Bothuser0.1gatepass,
CIFidentitiesmatch,all4integermetadataarraysbits. candidate-checks.json retains
all9arraydelta maxima; remainingconfidence summary/relativegeometryreports should
be included infinalassessment. NativeagreementNOTexperimentalaccuracy.
CURRENT LIVE OFFcontrolPID2356631 sameLtmodelroot/off-measurement/off-launch.log,
output outputs/protenix-base-lt-off. PollbeforeGPUwork. Nomodelspeedupclaimyet:
candidate134.39 vspriorpublishedmedian133.44 isnotmatchedpair andnotfastersofar.
Master714427e pushedcontains3pairpublishedqualification+Ltcode/screen; thislatest
stateappendisaftercommit anduncommitted. Keepgoalactive, controlinprogress.

## 2026-10-07 Lt E2E rejection, trained profile, Torch BLAS reuse

InitialLtOFF2356631 TERMINAL SUCCESS132.947387034s/model108.921124792.
DirectC ABI LtON134.389951601s =>1.09%slower. User0.1Acoordinatecriteria passes,
but unchangedRust1e-3coordinate/confidencegatefails (summarymax0.00187683105),
performancewithheld. Preservecomparison-old-policy.json; compactresult
provenance/protenix_base_lt_first_pair.json. NativeCApairRMS0.00465853A; relative
previousCKpairRMS0.0298676A. NativeagreementnotGTaccuracy. Notpromoted.

TrainedprofileON2357913/OFF2359033 TERMINAL SUCCESS at
outputs/experiments/protenix-base-lt-profile-20261007. Bothprofiles all9arraysbits
vsrespectiveuninstrumentedoutputs. Newpacked_probe.py wrapsactual1048contractions
withGPUeventsandCPUenqueue timers. Offtotalout5466.859ms,in4918.433ms;
OndirectLt out4498.394,in4401.296ms. Ltsetup2111.568ms vsnative0.0117ms;
Ltfirstevents272.776/213.279ms, medians7.87058/7.81258ms. Startupcostoffsetsgain;
do nottreatinstrumentedtotalsasE2E. provenance/protenix_base_lt_profile.json.

TorchBLASscreen2360039 TERMINAL SUCCESS at
outputs/experiments/protenix-base-torch-blas-20261007. UsesTorchpreferredhipblaslt
onlyaroundtorch.bmm thenrestorespreviouspreference; switchcostincludedtiming.
Incoming9.25634->7.51490ms(bits), outgoing10.53706->7.56682ms(max3.27825546e-7).
NoextraCABIextension/publishedbundleload. Syntheticonly, preferredroute doesnot
proveeveryinternalkernelidentity. Evidenceprovenance/protenix_base_torch_blas_screen.json.

CURRENT LIVE TorchLtfullmodel candidatePID2360590, root
outputs/experiments/protenix-base-torch-lt-model-20261007; on-{request,profile,spec}.json,
on-measurement/on-launch.log/on-pid. Output outputs/protenix-base-torch-lt-on.
Newmodel_torch_lt.pycontextinfrozenworker setsTorchBLASprefonlyaroundmatching
N1000paddedbmm, restoresinfinally; requires524callsperdirection/nofallback.
Bothspecs1800s, standardpublishedAMD/CK otherwiseunchanged, HIP7strictFP32C10D200S1.
OFFpreparedNOTlaunched. Pollcandidate, verifybase-torch-lt.json+Rustdrift vsnative
andpreviousCK (and comparetoearlierdirectLtoutput), thenOFFsameworkersnapshot.
Source/provenance/docschangesuncommittedthisturn. NoTorchLtmodelspeedupclaimed.
TorchLtON2360590 TERMINAL SUCCESS130.92651685s/model106.646226768,
all9arraysBITS todirectLtON (so0.00609982A nativeRMSD/0.0505035A previousCKsame),
524perdirection/nofallback. OFF2361595 TERMINAL SUCCESS132.95050763s/model108.905800575,
outputbits topreviouspublishedCK. Onepair1.015458983x/1.52%lessE2E. OldRust1e-3
coord/confidencegatesstillfailwithheldperformance, keptcomparison-old-policy.json;
newuser0.1Acriteria passes. Evidenceprovenance/protenix_base_torch_lt_first_pair.json.
No repeatedtrend/promotionyet. Newsourceprofilingand4experimentalPythonfiles
plusprovenance/docs remainuncommittedthisturn.

CURRENT LIVE globalTorchBLAS experimentPID2362740, root
outputs/experiments/protenix-base-global-blas-20261007; hipblaslt-{request,profile,spec}.json,
hipblaslt-measurement/hipblaslt-launch.log/hipblaslt-pid; output
outputs/protenix-base-global-hipblaslt. Fullforward preferredhipblaslt withrestore
infrozenworkercontextmodel_global_blas.py, keepallcustomHIP/CK/source/FP32TF32off
C10D200S1N1000seed101 HIP7 andpublishedkernels. hipblascontrolpreparedNOTlaunched.
This isseparatefromper-contractionTorchLt. PollsamePID; validateoutputs0.1A against
nativeandpreviouspublishedCK plusconfidence/relativegeometry beforecontrol.
Preparedglobalworker onlyrequestsTorchpreference; no guaranteeofeachinternalkernel.

## 2026-10-07 global Lt completed and experimental 5ZMW qualification started

Global BLAS candidate/control both terminal: 132.596815995 / 133.003585493 s
complete invocation, only 1.003067717x single-pair gain; not promoted. Candidate
native-aligned RMSD 0.04795159931865 A, previous-CK-aligned 0.00969020791282 A;
both user 0.1 A criteria pass. Control all nine arrays match published CK.
Evidence provenance/protenix_base_global_blas_first_pair.json remains uncommitted.

Prepared experimental development case benchmarks/protenix/experimental/5zmw_A
locally and on DiamondHill. Full 1000-residue chain A/entity 1/model 1, 1000 valid
CA positions. Source /media/PM982/onestepfold/Dataset/raw/pdb_mmcif/zm/5zmw.cif.gz,
gzip SHA b74b97cfda7a8fe4a3a30a224e439e6efcc33e9d217e915bff417503d2574272.
Input SHA 938324d41314ec19427baf6bef797cfb67488c03cf0cf1f21681a0e5d97c7003;
gt_mapping.npz SHA b8ab495c1bcea46afa9c4343557ec53b2c5c4f6493418320b99fc1e00b2f31cf.
Truth is CA-only coordinates[mask], no reference field; metadata records sequence,
residue mapping, occupancy/altloc selection and source hashes. Not an independent
or training-excluded test set. Sequence-only inference omits experimental ligands.

CURRENT native PID2366088 confirmed live at 03:20; root DiamondHill
outputs/experiments/protenix-base-5zmw-20261007/native-measurement, native-launch.log.
Output outputs/protenix-base-5zmw-native. Native GPU computation plus bitwise-qualified
fast initialization; HIP7 FP32 C10/D200/S1 seed101 timeout1800. Both arms use frozen
outputs/experiments/protenix-base-torch-lt-model-20261007/python worker. Optimized
request/profile/spec prepared at same root, NOT launched; wait native terminal.
Optimized includes published CK/HIP plus scoped Torch Lt. Do not overlap GPU jobs
or heavy remote CPU builds. Experimental CA scoring in Rust remains to implement;
also assess native-versus-candidate 0.1 A aligned RMSD on this new input. Do not
assume old umem numerical or performance results transfer to 5ZMW.

5ZMW native2366088 TERMINAL SUCCESS357.848379466s/model333.304561644s,
7711 atoms/1000 residues. CURRENT optimizedPID2368531, same root,
optimized-measurement/optimized-launch.log. Started after native terminal;
HIP7 only. New Rust example experimental_ca.rs compiles and test passes:
rigid-motion invariance, internal-deformation detection; manual wrong-chain and
modified-input-hash rejection also verified. Native experimental CA RMSD
30.3493519552 A, CA-lDDT0.31538224243: absolute sequence-only structure quality
is low. Local native NPZ/CIF/score at /tmp/fireprot-5zmw-native. Do not claim
optimized experimental result before terminal. Collection script prepared remotely
/tmp/fireprot-collect-5zmw.py (not run); it requires both receipts and computes
Rust native-vs-optimized drift. Experimental scoring still needs candidate.

5ZMW optimized2368531 TERMINAL SUCCESS130.295645407s/model106.450242092s.
Single native357.848379466s pair gives2.746433915x. Route524incoming/524outgoing,
no fallback. Rust native-vs-optimized alignedallatomRMSD0.0639613409462A passes
user0.1A; CA0.0417187393866A. Experimental CA nativeRMSD30.3493519552A,
optimized30.3501750962A; CA-lDDT0.31538224243->0.315348133069. Bothlowabsolute
sequence-only accuracy; no independenttestclaim. Summaryconfidence maxdelta
0.0001792907715<0.001. OldRustcompare-runs correctly FAILS0.001Acoordinategate
andwithholdsperformance; preservedcomparison-old-policy.json inremoteexperiment.
Compactprovenance/protenix_base_5zmw_first_pair.json includesallreceiptsandresults.
NewRustexperimental_ca example testpasses andwrong-chain/tampered-inputnegative
checks pass. NoGPUjobcurrentlylive. Candidateper-BMMLtremainsunpromoted; future
workrepeatE2Etrend underexplicituser0.1Apolicyandretainconfidence, thencontinue
profilingremainingattention/trunkhotspots. Nooverallplateauclaim.

## 2026-10-07 scoped Torch Lt repetition and next triangle probe

Master7ee9ec7 previously pushed; 5ZMW qualification complete. Only docs/isa was
untracked initially. Started two additional matched umem N1000 pairs, same frozen
TorchLt worker/profile and timeout1800 as original pair. Prepared request/specs
in outputs/experiments/protenix-base-torch-lt-model-20261007:
off-r1,on-r1,on-r2,off-r2. Originalpair was ON/OFF; next pairs OFF/ON,ON/OFF.
Each new output must match all9arrays BITWISE against its original same-arm
output (off or on); this preserves original user0.1A regression assessment.
Generic old cross-arm0.001 gate still fails; do not reinterpret as passed.
No scheduler: each Rust measure-inference is launched individually after previous
terminal receipt and exact check. OnlyHIP7; no overlapping GPU jobs/builds.

OFF-r1 PID2371101 TERMINAL SUCCESS133.92423798s. Rust compare-npz receipt
same root/off-r1-exact.json passed vs outputs/protenix-base-torch-lt-off.
CURRENT LIVE ON-r1 PID2372460, on-r1-measurement/on-r1-launch.log/on-r1-pid.
Poll that PID/receipt; then compare all9bits to outputs/protenix-base-torch-lt-on
before launching on-r2. After on-r2 exact check launch off-r2; aggregate only
three corresponding pairs with identical1800timeout. No robustspeedupclaimyet.

Prepared NEW uncommitted experiments/protenix_base/triangle_probe.py, ASTchecked,
no model runyet. Events for dual_projection_tiled, packed_norm_vector,
triangle_residual, packed_bmm plus triangle module layer_norm_in/linear_z/linear_g.
Restores wrappers/hooks in finally. Eagerextensionload/eventoverheads diagnostic,
not E2Espeedup. Must show all9arrays sameas scopedLt anchor before interpreting.
Remote frozenworker+request/profile/spec ready, NOTlaunched, at
outputs/experiments/protenix-base-triangle-profile-20261007. Copied sourceworker
TorchLt snapshot, nested triangle_measure inside lt_install. Run ONLY after
repetition jobs terminal, output outputs/protenix-base-triangle-profile.
Purpose split remaining triangle cost; earlierBMM roughly10.4sof21.9s inclusive
trianglecost, so optimizing onlyBMM overlooks projections/norm/gate/memory.
No production source change, no overallplateauclaim. Activegoal continues.


## 2026-10-07 scoped Torch Lt three pairs complete; triangle profile live

ON-r1 PID2372460 TERMINAL SUCCESS131.303280918s, ON-r2 PID2373489 TERMINAL
SUCCESS131.122685184s, OFF-r2 PID2374530 TERMINAL SUCCESS133.85573861s.
All6outputs all9arraysbits torespectivefirstarm; all3ON524incoming/524outgoing,
no fallback. ThreeBA/AB/BA matchedpairs, timeout1800all. E2Emedians133.85573861
->131.122685184s,1.020843483x/2.04%less. Eachpair1.01546x,1.01996x,1.02084x.
Modelmedians109.598211918->106.988372646s. OldRustcrossarm1e-3comparisonstill
failscoordinate/confidenceandwithholdsperformance; preserved3paircomparison.
User0.1Aalignedcriterionpassesallpairs; no defaultpromotion. Evidence
provenance/protenix_base_torch_lt_three_pairs.json copied locally, uncommitted.
Remote aggregation /tmp/fireprot-aggregate-torch-lt.py verifies receipts/binary,
timeouts/all9exactandroutecounts, runsRustcompare-runs thenreports separate
user-policy descriptive medians. No genericgaterelaxation.

CURRENT LIVE triangleprofile PID2375680; root
outputs/experiments/protenix-base-triangle-profile-20261007, measurement/launch.log/pid.
Output outputs/protenix-base-triangle-profile. HIP7/FP32C10D200S1seed101,
scopedLtON/publishedAMD+CK; wraps extension calls and module inputnorm/linear_z/g.
WaitsamePID; require base-triangle-profile.json andcompareall9bits vs
outputs/protenix-base-torch-lt-on. THEN interpret inclusiveeventrows; do notcall
instrumentedE2Espeedup. No otherGPUjobs. Newprobe/3pairprovenance/docs remain
uncommitted; master7ee9ec7. Userdocs/isa untouched. Goalactive, notplateaued.
Triangleprofile2375680 TERMINAL SUCCESS; all9arraysBITS vs scopedLtON anchor.
Evidence provenance/protenix_base_triangle_profile.json. Inclusiveeventtotals:
packedBMM1048calls8355.046ms;dual_projection_tiled2096calls6104.667ms;
triangle_residual1048calls1780.417ms;packed_norm_vector1679.220ms;
linear_g1358.067ms;linear_z1341.198ms;layer_norm_in1266.008ms.
NoGPUprocesslive. Nexttarget HIP dual_projection_tiled kernel (6.1s), rescreen
rowtile/waves/LDSreuse with exactsameMFMA accumulationorder; must comparelogical
paddedoutputs, include allocation/fill in relevantmicrobench, thenfulltrained
E2Etrialonlyifwin. Existingkernel kernels/amd/fireprot_amd_ops.cu:787 uses
R128/C128/K16/W8, paddedchannelstride rows+128. No candidatekernelwrittenyet.
Thisturnmadeprogress:3pairLt trend andqualifiedprofile. Goalactive, notblocked.

## 2026-10-07 dual-projection tile and A/B fusion screens

Master415ab45. New isolated kernels/screen sources only, no production edits.
HIP7 dualtiles PID2377055 TERMINAL SUCCESS at
outputs/experiments/protenix-base-dual-tiles-20261007. Seven R/C/K/W variants
all14 ordinary/large-gate comparisons exact to maintained. Baseline2.888323ms;
identicaltile CABI0 2.865683ms; alternatives1..6 3.455763,3.020403,3.021362,
3.690163,3.897683,4.003603ms. No newtilewin; baselineclone0.8%noise notpromotion.
Evidence provenance/protenix_base_dual_tiles.json. Rust nativebuild receipt
screen.build.json; library SHAab7baa17db0f6973dd7c39b101387467efe7563e9d5ae6a4d97d93027fa73a41.

Fused A/B quadprojection PID2377914 TERMINAL SUCCESS, root
outputs/experiments/protenix-base-quad-projection-20261007. Fivevariantsall10
ordinary/largegatecases exact. Two maintainedcalls5.786562ms vsquadvariants
8.893204,8.017283,7.539523,10.443444,5.773362ms. Bestonly0.2%faster, no material
win orfullmodeltrial. Evidence provenance/protenix_base_quad_projection.json.

Nextscreen retains initialized dualoutputstorage toavoidrepeatallocation/fill;
deterministicfill staysTrue. CURRENT PID2379098, rootdualtiles/reuse-screen.json,
reuse-launch.log/reuse-pid. Publishedoriginaltile0viaRust at
runtime/kernels/protenix-base-dual-tiles-20261007, manifest
5122da38f866545270fac913bdfda28bebc1d221181871b67116c63b6cac3105,
libraryab7baa17db0f6973dd7c39b101387467efe7563e9d5ae6a4d97d93027fa73a41.
Publication succeeded; preparationinitiallylookedformanifest.json(nonexistent),
fixedusingpublicationreceiptmanifest_sha256, no GPUjobwasstartedbyfailedprep.
Currentreusemicrochecks3changinginputs againstreferenceexact andcomparessteady
retainedstoragevsmaintainedfreshallocation. Retainedinitialallocationexcluded;
E2Emustincludeit. Onwin nextfullmodeladaptermustguardA/Bpairlifetimesandstream:
retain2buffers, reuseonlyaftermatchingpacked_bmm consumptionenqueuedonsamestream,
rejectunexpectedcalls/shapes/training/capture ratherthanallowaliascorruption.
No fullmodelreuseadapterwrittenthisturnyet. Allnewcode/evidenceuncommitted.
Reusemicro2379098 TERMINAL SUCCESS: maintained2.884561ms vs retained2.481680ms,
14%lesstime, all3changinginputcasesexact. Evidence
provenance/protenix_base_dual_reuse_screen.json. Initialretainedallocationexcluded
frommicroonly; no E2Eclaim. New model_dual_reuse.py replacesdualprojectionwith
same tile0 andtwoinitializedslots. GuardsFP32N1000gfx90a/noautograd/TF32off,
streamidentity, A/Bsameinputidentity/version, pairconsumptionbeforeoverwrite;
intercepts existingLtBMMto release slots onlyafterconsumptionenqueued. Restores
bothwrappers in finally; requires2096projectioncalls/1048pairs. Grad-enabled
parameterflagsalonearenotrejectedbecauseinferenceusesno_grad; inputmustnograd.

CURRENT LIVE fullBasecandidatePID2379733, root
outputs/experiments/protenix-base-dual-reuse-model-20261007;on-measurement,
on-launch.log/on-pid. Output outputs/protenix-base-dual-reuse-on. Frozenworker
fromTorchLt snapshot nesteddual_install inside lt_install, FP32N1000C10D200S1
seed101HIP7timeout1800, publishedAMD+CK+newdualtilebundle. OFFcontrolprepared
NOTlaunched; retainsTorchLt andsameworker/bundleenv exceptFIREPROT_BASE_DUAL_REUSE.
Onterminal inspectbase-dual-reuse.json counts/error, all9arraysBITWISE vs
outputs/protenix-base-torch-lt-on. OnlyifcorrectstartOFFafterterminal; measure
actualE2Eincludinglazybufferallocation/libraryload/guards. NootherGPUjobs.
Kernel tile/quad sources andscripts, reuseadapter and3provenancefiles allnew
uncommitted; productionunchanged. No speedupclaimfornewfullmodelyet.

## 2026-10-07 dual storage full-model rejection and diagnosis

DualreuseON2379733 TERMINAL SUCCESS133.04689037s/model108.595448331s,
2096projection/1048consumedpairs/1retainedallocation/noerror. All9arraysBITS
vs scopedTorchLtON. OFF2380778 TERMINAL SUCCESS131.688015749s/model107.224376964.
Matchedpair0.989786498x,1.03%slower. Rustcompare-runs unchangedgatesPASS between
theseidenticalmathroutes; do notpromote. Evidence
provenance/protenix_base_dual_reuse_first_pair.json; fullreceipts+comparison in
outputs/experiments/protenix-base-dual-reuse-model-20261007. Micro14%doesnot
translateE2E; needseparateinitialization/steadytrainedoperatorcostbeforeclaimcause.

CURRENT LIVE diagnosticON PID2382655, root
outputs/experiments/protenix-base-dual-reuse-profile-20261007;
on-measurement/on-launch.log/on-pid. OFFpreparedNOTlaunched. Frozenworkercloned
fromdualreuseworker, firstrecordscommonAMDpreload, thenextra dual_install setup,
then triangle_measure nested inside dual_install inside lt_install. Writes
base-dual-setup.json andbase-triangle-profile.json. HIP7standardFP32C10D200S1,
N1000umemseed101timeout1800. Onterminal requireall9bitsvsuninstrumentedON, inspect
setup+eventrows, thenOFFsameworkerafterterminal. InstrumentedE2Enotspeedup.
NootherGPUjobs; productionunchanged; newkernels/screens/modeladapter+negative
andmicroscreenresults beingcommittedthisturn. docs/isauserfileuntouched.

## 2026-10-07 receipt parsing bottleneck and buffered candidate

DualreuseprofileON2382655/OFF2383727 bothTERMINAL SUCCESS/all9bits torespective
uninstrumentedoutputs. CommonAMDpreload6.771/6.747s; additionaldualsetupON2.169354s,
OFF0.000007s. Dualprojectionevents6.007510s OFF ->5.280494s ON, so0.727s actual
operatorgain offsetby2.169s setup. Evidenceprovenance/protenix_base_dual_reuse_profile.json.
Rootcause candidate found inRustkernel_bundle.rs:read_json uses
serde_json::from_reader(File::open) withoutBufReader. NewlocalRustexample
receipt_read_probe.rs countsactualFile.readcalls whileparsingreal2.4MBreceipt:
unbuffered2,436,810readcalls,1.638/1.654/1.658s; buffered299calls,.00658/.00598/.00623s.
AllsixdecodedBuildReceiptvaluesexact. Evidenceprovenance/kernel_receipt_read_probe.json,
localCPUonly, NOT DiamondHillE2E. Bothpublisherandacceptcallread_json (freshimpact
HIGHdue3modules,2directcallersreviewed). Minimalproductionedit importBufReader and
wrapFileonly; retainedsize/hash/schema/compatibilitychecks. Existing3kernel_bundle
testsPASSincludingcorruption/symlink/mismatchrejects. Newprobeandedituncommitted.

Built TWO localRustbridgesfromsameisolatedgitarchiveccf6b19workspace at
/tmp/fireprot-buffered-receipt-src withCARGO_TARGET_DIR=/home/husrcf/Code/fireprot/target.
Oldfirstthennew; verifiedonlyonefilediffkernel_bundle.rs inentiresnapshot.
Artifacts/tmp/fireprot-buffered-receipt-libs/{unbuffered,buffered}.so plusbuild-comparison.json.
SHAold0dcc10a43b01e9261094f69638b5d67d58d4747d13bbeb5575885b0776b0703f;
new e8f166bb0a7f13f2600702207a7d07660f5193dbcf0de177fddd21bdfb51f965.
BothremoteCPUimportABI1PASS; originalDiamondHillrootbridge andRustCLI untouched.

CURRENT LIVE bufferedcandidatePID2386359; root
outputs/experiments/protenix-base-buffered-receipts-20261007, buffered-measurement,
buffered-launch.log/buffered-pid. Output outputs/protenix-base-buffered-receipts.
SamefrozenTorchLtworker T/python andpublishedAMD/CK, no dualreuse; onlyprofile
FIREPROT_BRIDGE_LIBRARY choosespairedsame-sourcebuiltlibs atthisroot. FP32N1000
umemC10D200S1seed101HIP7timeout1800. Unbufferedrequest/profile/specpreparedNOTlaunched.
Pollcandidate; requireall9bits tooutputs/protenix-base-torch-lt-on andpublished
kernelcounts, thenstartunbufferedcontrolsameworker. Compare-runs expectedchanges
mustdeclarelaunch.inputs.rust-bridge (pinnedinputobject), launch.environment.
FIREPROT_BRIDGE_LIBRARY andinvocation.profile_sha256. Do notcomparetooldfrozenbridge
asifonlyBufReaderchanged; usebothnewlybuiltpairedartifacts toisolatetreatment.
NootherGPUjob. Updatedbuild-comparison.jsonincludesbasecommit/compiler/lockSHA,
sourcearchiveSHA andverifiedsinglechangedpath. Goalactive, concreteprogress.
Bufferedcandidate2386359 TERMINAL SUCCESS122.355360071s/model98.130984192s;
all9arraysBITS vs scopedLtON. CURRENT unbufferedcontrolPID2387269 samebuffered
experimentroot, unbuffered-measurement/unbuffered-launch.log/unbuffered-pid.
No pairedratioyet; onlyhistorical~131scomparison isnotmatched. Awaitcontroland
Rustcompare-runs withexplicitbridgeinput/env/profile treatment. Worktreecorefix
anddiagnostics beingcommitted; rootDiamondHilldefaultbridge stilldf7d1537,
frozenCLI stilldfbbe303. NootherGPUjob.


## 2026-10-07 buffered first pair and CK scheduling preparation

Bufferedfirstpair bothterminal: oldbridge131.150439245s/model106.907066886,
newbridge122.355360071s/model98.130984192. 1.071881438x/6.71%lesstime,
all9arraysbits andunchangedRustcompare-runsPASS. Evidence copiedto
provenance/protenix_base_buffered_receipts_first_pair.json. Sourcecomparison
provesonlyBufReaderdiffbetweenpairedbridges; frozenoriginalbridgeuntouched.

CURRENT LIVE Rustbenchmark-pairs PID2388403 at
outputs/experiments/protenix-base-buffered-receipts-20261007/repeat-measurement;
repeat-spec.json/repeat-launch.log/repeat-pid. TwoadditionalpairsABthenBA;
combinedwithfirstmanualBA gives3pairsBA/AB/BA. Bothoriginalandnewinvocations
alltimeout1800. Do NOT restart orlaunchGPU/remoteheavybuilduntilterminal.
Journal order.jsonl (notjournal.jsonl); benchmark.json oncompletion. Newoutputs
protenix-base-{unbuffered,buffered}-receipts-r{1,2}. Finalaggregate usesmanual
unbuffered/buffered-measurement plusrepeatmeasurement000/001 and003/002.
Allsameprofilepins. Declareexpectedbridgeinput/env/profilechanges asfirstcompare.

Prepared ONLY nextretaineddualtrial at
outputs/experiments/protenix-base-dual-reuse-buffered-20261007,
on/off-request/profile/spec.json; usesoriginaldualreusefrozenworker withbotharms
newbufferedbridge. NO GPUrunyet. Intendedretest0.727strainedkernelgainafter
eliminating2.169sadditionalJSONloadcost. PriorunbufferedreuseE2Erejectedremains.

Newlocaluncommitted CKscheduling files prepare_ck_swizzle.py,ck_swizzle.cpp,
screen_ck_swizzle.py. Remote prepared outputs/experiments/protenix-base-ck-swizzle-20261007
includinggeneratedfmha_swizzle.hpp/provenance.json; NOTcompiled/run.
ReviewedupstreamheaderSHAaf3ae7102bb6c6439ce8501fec214571076eaec9540c1f02ac1a9d16bb0e1ef1.
GeneratorrequiresSHA, keepsAMDcopyrightMIT, renamesclass/helpernamespace andadds
BatchGrouptemplate, replacesonlyGetTileIndex; remainderpipelineunchanged.
Fourbatchgroupvariants1(currentorder),16,64,1000, fixedB1000H4N1000D32.
All32000tile/head/batchtuplescheckedbijective locallyincludingpartiallastgroup.
HypothesisgroupbatchCTAorderreusesbroadcastbiasbetter; couldtradeKVlocalityaway,
NOT measuredgain. Requirebitwiseoutputsnormal/largebiasbeforetiming. Rustbuild
fingerprint includesgeneratedheader+provenancerecord. Defaultheadersuntouched.
Onlycompile/microbenchaftercurrentRustpairedbenchmarkterminal. Bothnexttrials
prepared; runseriallyandchoosepriorityfromrepeatevidence. Goalactive.

Bufferedrepeat2388403 TERMINAL SUCCESS; correct3pairaggregationpassedvia
/tmp/fireprot-aggregate-buffered.py. Newprovenance/protenix_base_buffered_receipts_three_pairs.json.
All6outputsall9bits/route524each, Rustperrun/pair/aggregategatesPASS. E2Emedians
131.744855106->122.458591367s,1.075831868x/7.05%less. Baseline131.150-138.963,
candidate122.355-123.521; retainbaseline138.963outlier. All3pairsfaster. README
andBase docsupdateduncommitted. No overallnative-ratiomultiplication.

CURRENT LIVE CKschedulingmicro/build PID2392609, root
outputs/experiments/protenix-base-ck-swizzle-20261007;launch.log/pid/screen.json.
Startedonlyafterpairedbenchmarkterminal. Usesbufferedbridge, HIP7, Rust-owned
nativebuild includesgeneratedpinnedCKheader. PollsamePID; oncompilefailurefixonly
isolatedsource withfreshartifact/run preservinglogs. Onnumericalpassinspect4group
varianttimings; onlyclearwinshouldgettrainedmodeltrial. No fullGPUmodeljobactive.
Retaineddualwithbufferedbridge pairremainspreparedNOTlaunched.
CKschedulingv1 PID2392609 TERMINAL COMPILE FAILURE missingfmha_swizzle.hpp:
Rustbuildstagesonlycpp, extra_fileshashesdoesnotcopyheaders. Preservedv1failure.
Fixednewscreen with explicit-Igeneratedheaderdirectory, freshv2root
outputs/experiments/protenix-base-ck-swizzle-v2-20261007;PID2393983 TERMINAL SUCCESS.
All8random/largebiasexact; group1/16/64/1000 medians20.368997/20.603715/
20.488755/20.314756ms. Best0.27%notmaterial; rejectpromotion/fullmodeltrial.
Evidenceprovenance/protenix_base_ck_swizzle.json. InstalledCKheadersuntouched.

CURRENT LIVE retaineddual+bufferedbridge candidatePID2394664;
root outputs/experiments/protenix-base-dual-reuse-buffered-20261007,on-pid confirms2394664;
on-measurement/on-launch.log.
StartedafterCKmicroterminal. OFFpreparedNOTlaunched. SameFP32N1000C10D200S1seed101
HIP7, latestbufferedbridgebotharms. Requireall9bits againstpriorbuffered/scopedLt
anchorand2096projection1048paircoverage; thenmatchedOFF. Thisretestsreusewithout
2.169sreceiptsetupoverhead. NoE2Egainclaimyet. New3pairbufferedevidence/README/
CKschedulingcodeandreportuncommittedthisturn. Goalactive, meaningfulprogress.

## 2026-10-07 buffered dual reuse positive pair; diffusion bias audit

RetaineddualbufferedON2394664 TERMINAL SUCCESS121.635853086s/model97.410903361;
OFF2395668 TERMINAL SUCCESS123.362992166s/model98.966769612. Singlepair1.01419926x,
1.40%lesstime. All9arraysbits tobufferedanchor/eachother; unchangedRustgatesPASS.
ON2096projections/1048consumedpairs/oneallocation/noerror. Evidence
provenance/protenix_base_dual_reuse_buffered_first_pair.json uncommitted.
No repeatedtrend/promotionyet. Earlierunbufferednegativepairretained.

Newisolated diffusion_bias_cache.py supports off/audit/on, globaldiffusion24blocks
only (not6localatomblocks). Upstreampair_z suppliedbyenable_cache is cloned by
DiffusionConditioning wheninplace, thennormalized andper-blockF.conv2d biasrepeated
200steps. Hypothesisbiasstatic; audit RETAINS ORIGINALCOMPUTATION andcompares
4776laterbiases bitwise to24firstbiases, sourcepair_zidentity/versionandweight
identity/versionguarded. Expected200conditioningcalls/4800globalattentioncalls.
Cachemodeonlyafterauditpass; preservesadaptiveQKV/outputgate/timeconditioning.
Newcodeuncommitted, ASTpasses. Auditstorageretains24*64MB=1.536GBbiases.

CURRENT LIVE auditPID2396798 confirmedlive02:00. Root
outputs/experiments/protenix-base-diffusion-bias-cache-20261007,
audit-measurement/audit-launch.log/audit-pid. Outputoutputs/protenix-base-diffusion-bias-audit.
FrozenTworker addsbias_install nestedinLt; bufferedbridge; no dualreuse; HIP7
strictFP32N1000umemC10D200S1seed101timeout1800. ON/OFFrequest/profile/specprepared
NOTlaunched. Onterminal inspectbase-diffusion-bias-cache.json counts/auditexact,
requireall9outputarraysbits vsoutputs/protenix-base-buffered-receipts. Onlythen
startONcachingfullmodel; afterONbitsvalidationstartmatchedOFF. AuditE2Enotspeedup.
Ifsource/shapeinvariantfails preservefailureandinspectactualinputs; don'tassume
biasconstantwithoutproof. NootherGPUjob, master471d2c8 current.
Diffusionbiasaudit2396798 TERMINAL SUCCESS:200conditioningcalls/4800global
attentioncalls/24firstbiases/4776laterbiasesBITWISEEQUAL; originalcomputations
retained, all9predictionarraysbits vsbufferedanchor. Storedcache1536000000bytes.
Evidenceprovenance/protenix_base_diffusion_bias_audit.json uncommitted.
CURRENT LIVE ONcachePID2397709 atsamebiasexperimentroot/on-measurement,
on-launch.log/on-pid; output outputs/protenix-base-diffusion-bias-on.
OnlyafterONterminalverifyrecordhits4776/created24/calls4800andall9arraysbits,
thenlaunchpreparedOFFcontrol. NootherGPUjobs. NewcachecodeauditedbutfastON
E2E/correctnessnotyetqualified. Worktree master471d2c8 plusuncommittednewsource,
2provenancefiles/docs/memory. Keepgoalactive; concreteprogress, notplateaued.


## 2026-10-07 global diffusion cache first pair complete; local bias audit live

GlobalcacheON2397709 TERMINAL SUCCESS119.582363404s/model95.130140742;
OFF2399502 TERMINAL SUCCESS122.429573457s/model98.352285347. Onepair1.023809615x,
2.33%lesstime. All9arraysbits; unchangedRustcompare-runs gatesPASS. Created24,
hits4776,total4800; conditioning200; cache1536000000bytes. Evidence
provenance/protenix_base_diffusion_bias_first_pair.json. No defaultpromotion or
threepairtrendyet; retainedmathsameasbuffered/scopedLtanchor. Source/audit/firstpair
andpreviousbuffereddualpairdocs stilluncommitted. No oldreceipt rewritten.

Prepared NEW diffusion_local_bias_cache.py, separateoff/audit/oncontext for6atom
attentionblocks (3encoder+3decoder). Sourceinspection: encoder c_l augmentation
usesstatic s_trunk; r_noisy entersq_l separately, thenpairfeaturep_lm update uses
c_l_q/k andsmallMLP, so localbias MAY bestatic. Auditmustprove1194laterbiases
bitwise against6firstbiases. Denoiserprehookbindssignature andguards inputp_lm,
c_l,s_trunk identity/version, stream, andallencoder/decoderparameter versions.
Original calculationsretainedinaudit; expected200denoiser/1200attentioncalls.
GlobalbiascacheONalreadyactiveinalllocalexperimentarms; no dualreuse here.

CURRENT LIVE localauditPID2400638 confirmedlive01:47; root
outputs/experiments/protenix-base-local-bias-cache-20261007; audit-measurement,
audit-launch.log/audit-pid. Output outputs/protenix-base-local-bias-audit.
Frozenworkerclonedglobalbiasworker, nestedlocal_bias_install insideglobalbias
insideLt. Bufferedbridge, FP32N1000C10D200S1seed101HIP7timeout1800. ON/OFFprepared
NOTlaunched. Onterminal inspectbase-local-bias-cache.json andall9arraysbits vs
outputs/protenix-base-diffusion-bias-on; onlythenlaunchlocalON andverifybeforeOFF.
IfinvariancefailsrecordanddoNOTcachelocalbias. NootherGPUjobs. Newlocalauditcode
uncommitted; globalcachecodeawaitingcommitwithitsqualifiedfirstpair.
Localaudit2400638 TERMINAL SUCCESS:200denoiser/1200localattention/6firstbiases/
1194laterbiasesALLBITWISEEQUAL; originalcomputationsretained. Cache98304000bytes.
All9predictionarraysbits vs globalcacheON. Evidence
provenance/protenix_base_local_bias_audit.json. CURRENT LIVE localON PID2401548,
samelocalexperimentroot/on-measurement,on-launch.log/on-pid; output
outputs/protenix-base-local-bias-on. Validatehits1194/created6/calls1200, global
cachealsohits4776, all9predictionarraysbits toglobalONanchorbeforelaunchingOFF.
OFFpreparedNOTlaunched. NootherGPUjobs. FastlocalE2E/resultstillpending.

## 2026-10-07 local bias first pair passes; graph combination constraints

Local ON2401548 and OFF2402623 TERMINAL SUCCESS. Matched E2E120.024906411 ->
112.687053399s,1.06511709x/6.11%less, all9arraysBITWISE and unchanged Rust
compare-runs gatesPASS. ON200denoiser/1200attention/6created/1194hits/errornull;
globalcacheON inbotharms, bufferedbridge, no dualreuse. Evidence
provenance/protenix_base_local_bias_first_pair.json; remote first-summary.json,
compare.json/comparison.json at outputs/experiments/protenix-base-local-bias-cache-20261007.
Aggregation/preparation script /tmp/fireprot-local-bias-pair.py onbothhosts.

CURRENT LIVE repeat schedulerPID2404870 confirmedlive, sameexperimentroot;
repeat-spec.json/repeat-measurement/repeat-launch.log/repeat-pid. Rustbenchmark-pairs
runs2additionalpairsABthenBA afterfirstmanualBA, allinvocationtimeout1800,
executiontimeout7200. Requestoutputs protenix-base-local-bias-{off,on}-r{1,2}.
PollsamePID; don'trestart/overlapGPUjobs orheavyremoteCPUbuilds. HIP7only.
Requireall6outputsall9bits andperpairRustgates; aggregate3pairswithfirstmanual
on/off thenrepeat000/001 and003/002, retainallobservations. Repetitionnotyetclaimed.

UseraskedHIPGraph: currentdefaultoff, oldPairformer1.0107x anddenoiser0.9769x
singlepairsdocumented, all9bitsexact. Existingwrapperscannotnaivelystack:
model_torch_lt.py rejects current_stream_capturing andexpects524each; cachehooks
expect200Pythonconditioningcalls/4800or1200attentioncalls, butreplaybypasseshooks.
Localcachealsofixesstreamownership. Neednewexplicitcomposedcapturecontractwith
staticcachelifetime/version validation andlogicalstepcounts, preservingselected
BLASroute; doNOTjustremoveguards. No newgraphtrialstarted. Firstfinishrepeat.
Currentgoalprogress: localcachegain/fulloutputcorrectness, notplateaued.
Master7242778 includespriorglobal/localaudits; thisappend/localpairdocsawaitcommit.

## 2026-10-07 cache-aware denoiser graph prepared during repetitions

Repeat scheduler2404870 stillLIVE confirmedelapsed04:33: repeatbaselineindex0
completed119.916s, candidateindex1completed~113.799s, unchangedRustgatesPASS;
index2candidateONr2 startedat234.46s. Index3baselineOFFr2 stillpending.
No overlappingGPUexperiment. Completeaggregation script prepared onbothhosts
/tmp/fireprot-aggregate-local-bias.py, adaptedvalidatedbufferedaggregator; require
benchmark.json phase=succeeded beforeexecution. It builds3pair-compare andchecks
all6outputsall9bits againstlocalON anchor +524Ltcounts. Firstpairalreadycommitted
20d26a2; currentnewsourceuncommitted.

Prepared root outputs/experiments/protenix-base-cached-diffusion-graph-20261007
on/off-request/profile/spec.json andfrozenpython clonedlocalbiasworker; NOTLAUNCHED.
Preparation /tmp/fireprot-prepare-cached-graph.py, exactsourcescopiedfromlocal
experiments/protenix_base/{diffusion_graph,diffusion_bias_cache,diffusion_local_bias_cache}.py.
FreshGitNexusimpact on3install symbols, typedindexmetaHEAD20d26a2/schema4current;
MCPstalenessprosewrong(14behind), noindexedcallers; textsearch+frozenworkersreviewed.
NewoptionalCacheReplayContract keepsoldnon-graphcachebehaviorunchanged.
Workercreatescontract, nestsLt>globalcache>localcache>graph, graphinnermostso
localhookinspect.signature seesoriginalforward. GraphONwarmactualstep1 andcapture
step2,199replays; graphchecksallnamedparameteridentity/version, fixedinputidentity/
version, andcachedbiasversions eachlogicalcall. Localprehookstillfires200times
outsidemodule.forward; globalconditioner2/globalattention48/localattention12,
cachehits24global/6local. ThesePythoncountscheckedseparatelyfrom200logicalcalls.
Graphcontextmarkssuccessonlyafter200/1/199coverage, cachesrequireitbeforeaccepting
reducedPythoncounts. OFFexpectsoriginal200/4800/1200and4776/1194hits.
Biasesfirstcreatedduringwarm(notcapture); nosyncfinitechecksinsidecapture.
Lttrianglepathoutsidegraph, still524each/nofallback. Nodualreuseinthistrial.
CPUcontractchecksrejectincompletecompletionandpropagateretainedmutationguard;
AST/diffchecksPASS. FullGPUcorrectnessNOTYETtested. DoNOTpromoteorclaimspeedup.

NextfinishrepeatPID2404870,aggregate/copysummary/updatedocs. ThenlaunchnewgraphON
viaRustmeasure-inferenceon-spec.json/on-measurement, HIP7onlytimeout1800.
Requiregraph200calls/1capture/199replays, global2/48/24hits,local200/12/6hits,
all9arraysBITWISEvs outputs/protenix-base-local-bias-on. Onlyifpassedlaunchsame
frozenworkergraphOFFbeforeclaimingincrement. Onfailurepreservereceiptanddiagnose;
donotdisableinvariantchecks. Graphnewsource/docsremainuncommittedpendingtrial.

## 2026-10-07 local cache repeated gain; cache-aware graph rejected

Repeat2404870 TERMINAL SUCCESS. Aggregate via/tmp/fireprot-aggregate-local-bias.py
savedthree-pair-summary.json andcomparison inlocalbiasexperimentroot, copiedto
provenance/protenix_base_local_bias_three_pairs.json. ExactofficialE2Emedians
120.024906411->113.317563216s,1.059190676x/5.59%less. Baseline119.818-123.845,
candidate112.687-113.708; all3pairedratios>1; all6outputsall9bits andunchanged
Rustpair/aggregategatesPASS. Earliercommentaryusedjournalintervals(~0.09slarger);
finaldocsuseinvocationelapsedonly. READMEupdatedwiththisincrement, nodirectnativeclaim.

Cache-awaregraphON2409214 andOFF2410991 bothTERMINAL SUCCESS, root
outputs/experiments/protenix-base-cached-diffusion-graph-20261007. On113.569438927s/
model88.841963322 vsOff113.316117284s/model88.917637080;0.997769456x,0.22%slowerE2E.
No usefulgain; graphnotselected. All9arraysBITSoff/onandearlierlocalONanchor;
RustgatesPASS. Graph200logicalcalls,1capture,199replays,capture.078879s,peakalloc
10644840960B. Globalcondition2/attention48/hits24; localdenoiser200/attention12/
hits6 exactlyasnewcontractexpects. OFForiginalcounts200/4800/1200,hits4776/1194.
Evidenceprovenance/protenix_base_cached_diffusion_graph.json; aggregation
/tmp/fireprot-cached-graph-pair.py onbothhosts. New3filesoptionalcontractchanges
nowfullmodelvalidated; docs/source/provenanceawaitcommit. Defaultsremainoff.

CURRENT LIVE combineddualreuse+biascacheON PID2412262 confirmedlaunchedafterboth
grapharmsterminal. Newroot outputs/experiments/protenix-base-combined-reuse-20261007;
on-measurement/on-launch.log/on-pid. Frozenworkerclonedpregraphlocalbiasworker,
addsoriginalqualifiedbase_dual_reuse.py fromdual-reuse-model snapshot; nest
Lt>dual>globalcache>localcache. BothcachesON, bufferedbridge, graphOFF, FP32N1000
umemC10D200S1seed101HIP7timeout1800. Profilesaddpublisheddualtilebundleidentical
inbotharms; onlyFIREPROT_BASE_DUAL_REUSE differs. OFFpreparedNOTlaunched.
Onterminal requireall9bitsvsoutputs/protenix-base-local-bias-on, dual2096/1048,
global4800/4776local1200/1194,nofailures. OnlythenlaunchmatchedOFF. Nextmeasure
whetherseparatelyqualifiedstoragegainstackswithcachegain; no combinedclaimyet.
RootremoteRustCLI/bridgeunchanged. NootherGPUjob. docs/isauntouched; goalactive.

## 2026-10-07 combined reuse first pair; cached diffusion profile live

CombinedON2412262 andOFF2413517 TERMINAL SUCCESS. All9bitsagainstlocalcacheanchor
andcrossarm; unchangedRustcompare-runsPASS. E2E113.479850396->112.678683511s,
1.00711019x/0.71%less; model88.864033266->88.112037746. Onepaironly, no repeated
combinedtrend/directoriginalratio. ONdual2096projection1048consumed; botharms
global4800/4776hits/local1200/1194hits, allerrorsnull. Evidence
provenance/protenix_base_combined_reuse_first_pair.json. Aggregator onbothhosts
/tmp/fireprot-combined-reuse-pair.py; fullreceiptsroot combined-reuse-20261007.

CURRENT LIVE diagnosticPID2414790, root
outputs/experiments/protenix-base-diffusion-static-profile-20261007,
measurement/launch.log/pid. Frozenworkerclonedlocal-bias-cache worker (bothbiasON,
scopedLt/bufferedbridge; NOdualreuse orgraph), addsnewdiffusion_static_probe.py
asbase_diffusion_static_probe.py andmeasurecontext innermost. Newlocalsource
ASTPASS, stilluncommitted. HIP7N1000strictFP32C10D200S1seed101timeout1800.
Selectedmodulesdenoiser/conditioning/globaltransformer/encoder/decoder plus
encoderatom_transformer,small_mlp,linear_no_bias_cl/cm/s/r/q,layernorm_s;
requires200calls each. Outputoutputs/protenix-base-diffusion-static-profile.
Onterminal requirebase-diffusion-static-profile.json ANDall9bits vs
outputs/protenix-base-local-bias-on; onlytheninterpretinclusiveoverlappingrows.
NoE2Espeedupclaimforinstrumentedrun. No otherGPUjob.

Nextpossibleoptimization ONLYifprofilewarrants: encoderstaticpairupdate still
computeslinear_no_bias_s,cl,cm andsmall_mlp eachstep despitecachedattentionbias.
Source transformer.py880-949 showsstaticc_laug usesfixeds_trunk; noisyr_lenters
q_l separately. Couldauditinput/outputinvarianceforfourstaticmodules before
memoization, withdenoisersources/atommap/weightsversions guarded andcachedoutput
mutationchecks. DoNOTcachelinear_no_bias_r,q_l,adaptiveconditionorattention.
No newmemoizationcodeyet. Avoidcopyingupstreamforwardbody; useisolatedmodule
wrappersandretainlicenseboundaries. Currentmaster8f81f7b; newprofile/evidence/docs
awaitcommit, docs/isa untouched. Goalactive, nextactionpollsamePID2414790.
Staticprofile2414790 TERMINAL SUCCESS, all9bitsvsuninstrumentedlocalON. Evidence
provenance/protenix_base_diffusion_static_profile.json; root summary.json.
InclusiveD200denoiser16.021498s,globaltransformer7.769712,encoder3.048196,
encodertransformer1.803023,decoder1.681288,conditioning.352574,smallMLP.309219,
linearcl.024642/cm.008942/s.007036s. NestedrowsNOTadditive. StaticMLP/projections
only~.35ssoDEFERmemoization; priorhypothesisoflargercostnotborneout.

CURRENT LIVE deeperglobalprofilePID2416052, root
outputs/experiments/protenix-base-diffusion-global-profile-20261007,
measurement/launch.log/pid;outputoutputs/protenix-base-diffusion-global-profile.
Frozenworkerclonedstaticprofileworker,switchesmeasureimporttobase_diffusion_global_probe.
SameHIP7strictFP32N1000C10D200S1seed101,bothbiasON,scopedLt/buffered,no dual/graph.
Newdiffusion_global_probe.py groups24blocksattention/adaNorm/QKV/O/gate/conditioned
transition andwrapsF.scaled_dot_product_attention onlyqshape[1,16,1000,48];
requires4800observationspergroup. NewcodeASTcheckedbutfullGPUprofileNOTYETqualified.
Onterminal requirebase-diffusion-global-profile.json ANDall9bits vs localbiasON;
onlytheninterpretrows. Failurecountsrecordinworkerlog; retainfailurebeforefix.
No otherGPUjob. Source/evidence/docsbeingcommitted; no newoperator speedupclaim.

## 2026-10-07 global profile qualification; HIP transition fusion first pair

Globalprofile2416052 TERMINAL SUCCESS/all9arraysBITSvs eagerlocalcacheanchor.
Evidenceprovenance/protenix_base_diffusion_global_profile.json. 4800calls/group,
inclusiveattention6.660825s,attentionwithprojections4.040036,conditionedtransition
2.708735,SDPAcore1.975685,adaNorm.861349,Q.408210,K.353779,V.362794,gate.282909,
outputlinear.274896s. NestedinstrumentedrowsNOTadditive; moreprobeoverheadthan
previousglobaltransformer7.77s. SDPAqFP32[1,16,1000,48],stride[768000,48,768,1].

PublishedAMDlibraryalreadyhassilu_mul_f32/sigmoid_mul_f32. HIP7synthetic16checks
shapes[1,1000,1536],[1,1000,768],[1,7979,256],[1,7979,128],randomscale1/20,
seed101strictfillallBITWISE vsTorch. Evidence
provenance/protenix_base_diffusion_transition_synthetic.json; remote
outputs/experiments/protenix-base-diffusion-transition-20261007/synthetic.json.
No newkernelbuild. Newisolateddiffusion_transition_fusion.py usesexistingHIPkernels
for30conditionedtransitionblocks, keptnormalization/linearoperations, fresh
projectedgate/outputstorageconsumedinplace; rejectsgrad/autocast/capture, strict
N1000global768/local7979x128; requires200calls/block=6000. Prototype hardcoded
7979atomscannotyetserve5ZMW7711; generalizewithseparatequalificationbeforepromotion.

TransitionON2418291 andOFF2420551 TERMINAL SUCCESS. All9arraysBITSagainstcombined
anchorandcrossarm; RustgatesPASS. ON111.950568443s/model87.935992179,
OFF118.374006712s/model93.792648997. Rawpair1.05737745x isPROVISIONAL becauseoff
outlierslowversusprior~113s. Preserveit, DONOTclaimstable5.7%. Evidence
provenance/protenix_base_diffusion_transition_first_pair.json; aggregator
/tmp/fireprot-diffusion-transition-pair.py onbothhosts. Frozenworkerclonedcombined
reuseONworker; addsfusioninnermost. Bothcaches/dualreuse/Lt/bufferedON, graphOFF.

CURRENT LIVE repeat schedulerPID2421662, root
outputs/experiments/protenix-base-diffusion-transition-20261007,
repeat-spec.json/repeat-measurement/repeat-launch.log/repeat-pid.
2additionalABthenBApairs afterfirstmanualBA, eachtimeout1800total7200. HIP7only;
nooverlappingGPUjobs orremoteheavybuild. Finaljournalorder.jsonl/benchmark.json.
Outputprotenix-base-diffusion-transition-{off,on}-r{1,2}; mapbaselineidx0,3,
candidate1,2. Prepareaggregatefrom/tmp/fireprot-aggregate-local-bias.py byreplace
root/outputnames/anchors; verifyall6outputsall9bits +Lt524each andfusionon6000.
Awaitterminal, no restartonobservationtimeout. Goalactive, gainsnotflatproven.

PotentialnextHIPcandidateafterrepeat: attention _wrap_up multiplies transposed
SDPAoutput bysigmoid(linear_g) thenflattens. Could fusegatewithlayoutconversion
andoverwritefreshlinear_gstorage, retaininglinear_o andFP32operationorder;
requireactualo shape/stride/alias guards andsyntheticexacttests beforemodeltrial.
No newgatingkernelwritten. Biggerdeliverableremainsintegratingqualifiedroute
into maintainedopt-inentry/profile anddirectoriginalTorchE2Erepeats; current
fastestpathsfrozenexperimentalworkers, don'tclaimdefaultprofile~112s.

## 2026-10-07 transition repetitions underway; attention gate kernel prepared

Repeat scheduler2421662 stillLIVE confirmedelapsed04:15. Baselineindex0 and
candidateindex1 terminal/perrunRustgatesPASS; candidateindex2 running; baseline
index3pending. Journalintervalbaseline112.706/candidate112.127 includecontroller
overhead; useinvocationelapsedinfinalreport. Baselinebacknear112.6s supports
cautionaboutfirstOFF118.374outlier. Newaggregate onbothhosts
/tmp/fireprot-aggregate-diffusion-transition.py copiedfromvalidatedlocalbias
aggregatorwithpaths/descriptionupdated; additionallyrequiresallON30blockcounts
200each/errornull. Awaitbenchmark.json succeeded thenaggregate/copyevidence.

NewUNCOMMITTED sources experiments/protenix_base/diffusion_gate.cpp and
screen_diffusion_gate.py. HandwrittenHIPvectorfloat4 kernel forglobaloutputshape
[1,1000,16,48], supportsdense tokenmajorstrides768/48 orheadmajor48/48000.
Fuses exactsigmoidgate*mappedattentionoutput into freshlinear_gbuffer; rejects
null/misaligned/identicalpointers andunexpectedstrides atCABI. DoesNOTchange
linear_g/linear_o. Microcloneofgateinputcommonbotharmsrepresentsfreshlinearoutput,
4numericalchecks2layouts*2scales1/20,40timedalternatingmeasurementsperarm/layout;
strictfillseed101,HIP7; sourceASTPASSbutNOTcompiledorGPUtested.

Preparedremote rootoutputs/experiments/protenix-base-diffusion-gate-20261007,
onlycpp/screenfiles; NO pid/outputyet. Launchscript/tmp/fireprot-launch-diffusion-gate.py
onbothhosts assertspriortransitionbenchmarkphase succeeded beforelaunch, uses
bufferedbridge/profileenv/PYTHONPATHroot/HIP7 andRust-ownednativebuild.
DO NOT runmicro/buildbeforecurrentrepeatterminal. Nativec_abi
base-diffusion-gate-v1, packagefireprot_base_diffusion_gate, -ffp-contract=off.
Onmicrofailurepreservelog/receiptbeforefix; onmaterialwinpublishviaRustandonly
thenmodelwrapperfulltrainedtrial. No modelgateadapterwrittenyet. No speedupclaim.

Afterthiskerneldecision, prioritizepromotingqualifiedfastpathinto maintained
opt-inBaseentry/profile anddirectnativeE2Erepetitions; fastest~112s stillfrozen
workers, notpublicdefault. Needpreservelegacycoordinate1e-3 failures anduser
0.1AalignedRMSDcriterionseparately, confidencevisible; no multiplyratios.
Currentmaster405878f, onlynewgatefiles+thismemorydirty, docs/isauntouched.

## 2026-10-07 transition repeated small gain; gate kernel micro passes

Transitionrepeat2421662 TERMINAL SUCCESS. Aggregatevia
/tmp/fireprot-aggregate-diffusion-transition.py: all6outputsall9bits, allON30*200
calls, Lt524each, unchangedRustgatesPASS. E2Emedians112.610351267->112.040998359,
1.005081648x/0.51%less. Baseline112.483-118.374,candidate111.951-112.386;
pairedratios1.057377/1.005082/1.000863. Firstoutlierretained; finalpaironly.086%
gain, notbroad5.7%. Evidenceprovenance/protenix_base_diffusion_transition_three_pairs.json.

GateRustbuild/microPID2425335 TERMINAL SUCCESS. All4checks2layouts*scale1/20
BITWISE. Commonfreshgateallocationincluded; standalonegate tokenmajor.05112->
.03960ms, headmajor.07152->.04112ms. LinearprojectionsEXCLUDED; notE2E.
Evidenceprovenance/protenix_base_diffusion_gate_screen.json. PublishedviaRust:
runtime/kernels/protenix-base-diffusion-gate-20261007 manifest
2de8bbf95c19e263f7a3aaf0a7953ef7ba33ea201e0a963e0b04d51d70df1509,
libraryaf596ed23eb8cb0310b05d062bfea07f2926dfcb977d73bdc3d983b5ee7b7989,
packagefireprot_base_diffusion_gate,c_abi base-diffusion-gate-v1.
Root outputs/experiments/protenix-base-diffusion-gate-20261007 includes
screen.json/screen.build.json/publication.json. Newcpp/screen/modeladapterlocal
uncommitted; AST/diffchecksPASS. Newadapterdiffusion_attention_gate.py replaces
24globalattention._wrap_up only; retainslinear_g/o, demands4800totalcalls,
strides768/48or48/48000,FP32N1000heads16D48; overwritesfreshlineargbuffer,
rejectscapture/grad/autocast. No maintainedpathpromotion.

CURRENT LIVE fullgateONPID2426494, samegateroot/on-measurement/on-launch.log/on-pid.
FrozenworkerclonedtransitionONworker, nests gate_install inside transition_install.
Bothbiascaches+dualreuse+transitionfusion+scopedLt+bufferedbridgeON, graphOFF,
HIP7N1000umemC10D200S1seed101timeout1800. Outputoutputs/protenix-base-diffusion-gate-on.
OFFpreparedNOTlaunched. Onterminal requirebase-diffusion-gate.json24counts200each/
errornull, all9bits vsoutputs/protenix-base-diffusion-transition-on. Onlythenlaunch
OFFsameworker/profileexceptFIREPROT_BASE_DIFFUSION_GATE_MODE, samebundleenv.
Needoff/onE2Ecomparebeforegainclaim; eventmicrodoesnotprovefullmodelgain.
NootherGPUjob. READMEBaseguideupdatedthreepairresult/microqualuncommitted.
GateON2426494 andOFF2427354 TERMINAL SUCCESS. All9arraysBITSvs transitionONanchor
andcrossarm; RustgatesPASS. 24*200globalcalls allobservedo token-majorstride
(768000,768,48,1), nothead-major. E2EOFF112.048547859->ON112.477773152s,
0.99618391x/.38%slower; model88.035941213->87.814772557 improves.221s butfull
invocationdoesnot. REJECTgatepromotion; source/library/micro retained.
Evidenceprovenance/protenix_base_diffusion_gate_first_pair.json, aggregation
/tmp/fireprot-diffusion-gate-pair.py onbothhosts. No GPUjobsactiveafterthispair.

NEXT priority: maintained opt-in Base N1000 route, then directnativecomparison.
Qualifiedselection: scopedLt + bufferedRustbridge + global/localbiascache +
dualstorage + transitionfusion. NO graph ornewgatelibrary. Bothlastincremental
winsare small (.71%dual singlepair,.51%transition3pairs), kernelphaseflattening;
notcompletebecausemaintainedentry/directcurrentnativeE2E/secondinputqualpending.
Promoteexperimentalhelpers to python/fireprot/adapters/protenix maintained
modules with explicit opt-inworkercontext andpublishedprofile. Keeporiginal
experimentalfilesasversionedevidence; no upstream checkoutmutations. Preserve
shape/cycle/dtype/device/alias/version/countguards. Currenttransitionprototype
hardcodes7979atoms: needderiveexpectedatomcountfromrequestfeatures andqualify
5ZMW7711 beforegeneralN1000claim. BaseN600/nonBasefallbackmustremainusable;
donot silentlyenableunsupportedshape. ExistingnativevsCK legacycoordinategate
stillfails, user0.1Aalignedgate+confidenceassessmentmustremainseparate.
Newuser-facingentrymustnotpointtofrozenexperimentalworkerpaths. Bridgecanbe
published/copytopinnedruntimepath preservingqualifiedbufferedbinaryhash; leave
historicaldefaultremoteCLI/bridgeuntoucheduntilqualifiedmigrationchosen.
Currentuncommitted3gatefiles+3pairevidence/gatemicro/gatepair/docs/memory ready
forcommit afterindex/gate. docs/isauntouched. Goalactive, noimpasse.

## 2026-10-07 maintained Base composition integration

New maintainedpackage python/fireprot/adapters/protenix/base_n1000 has5helpers
promotedfromqualifiedexperimentalversions:triangle_lt,dual_storage,global_bias,
local_bias,transition. Environmentreadreplacedbyexplicitmodeargument(defaulton);
transitionlocalatom_count nowpassedfromvalidatedinputfeatures;originalprototypes
andprovenance/layout_path_map.json preserved. Mapping/sourcehashes innew
provenance/protenix_base_promotion_map.json. New__init__.py explicitflag
FIREPROT_ENABLE_BASE_N1000_OPTIMIZATIONS=0(default)/1; strictBaseFireProtFP32
C10D200S1N1000contract,nonemptyatommap, ExitStackrestoresallhooks. Defaultoff
returnsbeforeGPUimports. Workerprotenix.py wrapsoriginal_predictonly; fresh
GitNexusimpactpredictreportedunknown/0resolved, manuallyreviewed__main__caller;
localtypedindexmetaHEAD4d047dc/schema4current, staleMCPprosehintsignored.

Newprofileconfigs/diamondhill/protenix-base-n1000-optimized.json usesmaintained
rootpython, publishedAMD/CK/dual andnewruntime/bridges/protenix-base-buffered-e8f166bb.so.
CopiedqualifiedB/buffered.so afterSHAverification e8f166bb0a7f13f2600702207a7d07660f5193dbcf0de177fddd21bdfb51f965;
rootoldCLI/defaultbridge untouched. Targetedrsyncdryrunreviewedthenonlynewpackage,
workerandprofiletransferred. Oldfrozenworkersnotmodified.

InitialmaintainedUMEM2429457 TERMINAL112.637190625s/all9bitsvstransitionanchor;
5ZMW2430326 TERMINAL113.527006311/all9bitsvsold5zmwoptimized. Counts7979/7711
nowderivedfromfeatures. Evidenceprovenance/protenix_base_maintained_{umem,5zmw}.json.
ThenwiredcompositionselectionthroughexistingRustregistry toretainpolicyboundary:
operationprotenix_base_n1000_context,implementationamd-gfx90a-base-n1000-cached-v1,
Pythonmetadata supports callback, contexttensorwork/stacksremainPython.

Registryv1UMEM2431714 TERMINALFAILUREbeforemodelcomputation: mistakenlycalled
registry.select instead ofexistingresolve. Preservedoutput/nativeERR/measurement
andcopyofoldsourceatmaintainedroot/failed-registry-v1.py beforeoverwriting.
Correctedresolve, addedREALRustregistrytest(fakehardwaremetadata only). That test
alsofoundduplicateregistration; registrationnowcachedonce, testedmultiplequeries
andCPUrejection. All32 targetedCPUtestsPASS:
python -m pytest -q tests/adapters/test_protenix_base_n1000.py tests/workers/test_protenix.py.
Testscoverdefaultoff,wrongmodels/dtype/cycles/steps/samples/backend/N,atomcount,
exceptionunwind,invalidflag,capabilityrejectionandrealregistrybinding.

Correctedregistryv2UMEM2433476 TERMINALSUCCESS112.085780s/all9bits vs frozen
transitionON, selectedimplementationrecordpresent. Evidence
provenance/protenix_base_maintained_registry_umem.json. Rootallmaintainedtrials:
outputs/experiments/protenix-base-maintained-20261007.
CURRENT LIVE 5ZMWregistryv2 PID2434640, same root/5zmw-registry-v2-measurement,
5zmw-registry-v2-launch.log/5zmw-registry-v2-pid; output
outputs/protenix-base-maintained-5zmw-registry-v2. HIP7N1000FP32C10D200S1seed101,
onlyjob. Onterminal requireselectedimplementation,errornull,atom_count7711,
all9bitsvsoutputs/protenix-base-5zmw-optimized, thenupdatequalificationdocs/map.
Sourcepackage,worker/profile/tests/docs/map/3pilotprovenancefilesuncommitted.

PreparedNOTLAUNCHED directnativestudy root
outputs/experiments/protenix-base-maintained-native-pairs-20261007 via
/tmp/fireprot-prepare-base-native-study.py (bothhosts). study-plan.json,sixpinned
request/specfiles,native/optimizedprofiles,run.sh(checkedbash-n),anchors.sha256.
AB/BA/AB order; sameoptimizedprofilebotharms exceptnewflag0/1 andbackendnative/
fireprot, sharedfastinit andpinnedbufferedbridge,bundlesenvsame,maintainedworkers.
EachinvocationRustmeasure-inference1800; serialshelldriverRustcompare-npz vsown
anchorplusjqrequire9bitwise/passed; checksanchorSHAeveryrun. Nativeanchoroutputs/
protenix-base-strict1000-init-control, optimizedanchoroutputs/protenix-base-maintained-umem.
EachpairRustbase-drift checksuser0.1Aalignedcriterion, usingmatchingCIFpaths.
Legacycompare-runsfixedgatesWILL remainseparateunchangedfailure; notbenchmark-pairs
becauseitscross-arm1e-3gatewouldabortscientificallyauthorized0.1Astudy. No gate
relaxation. ThesearepredictionreferencesNOTexperimentaltruth. Driver-status.json
traprecordsfullcompletionexitcode;order.jsonlperrunstarted/verified. Needinspect
run.sh/specsagainbeforelaunch; after5ZMWpassandcommitstartwithoveralltimeout7200
anddetachedPID, nootherGPUjob; expect~24minfor3*(native~358+fast~112). Requireall6
terminal/bitwiseanchorchecks+3alignedgates andfinalcondition/confidencecomparison
beforeclaimingdirectE2E. Source/profilemustnotchangewhilethisstudyruns.
5ZMWregistryv2 PID2434640 TERMINAL SUCCESS112.727518159s,selectedimplementation
andatomcount7711correct, all9arraysBITSvsold5zmwoptimized. Evidence
provenance/protenix_base_maintained_registry_5zmw.json. Bothfinalregistry-mediated
GPUqualifications nowPASS; directstudyNOTYETstarted. All32CPUtestsPASS.
CommitgateinitiallyHIGH47symbols12flowsbecausepublicProtenixpredictchanged;
reviewedcontextactualcallscheck_request/runtimeinstall/manifest plus__main__
callandnewcapture->contract/select/registerpaths. Existingpathsunchangedexcept
explicitcontext(default0 returnsbeforeheavyimports); enabledpathstrictlylimited.
Retained32tests/defaultoff/errorunwind/realRustpolicyand2fullGPUexactchecks address
actualchange. Re-stage/re-gateafterdocs/evidenceupdatesbeforecommit. NoGPUlive.

MaintainedintegrationCOMMITTED/PUSHEDmastera0e86fe. BothfinalregistryGPUchecks
complete; source/profilefrozenfordirectstudy. Postcommitindexrefreshed andfinal
all-diffgateclean(exceptuserdocs/isauntracked).
CURRENT LIVE directnative/maintainedstudyPID2436859 (timeoutwrapper), confirmed
launchedAFTERbothqualificationsandcommit. Root
outputs/experiments/protenix-base-maintained-native-pairs-20261007,
pid/launch.log/order.jsonl/driver-status.json attermination. Baselineindex0native
currentlyrunning; sequence native-r0,optimized-r0,optimized-r1,native-r1,
native-r2,optimized-r2. Preparedcontrolpinsscientificsettingscheckedidentical,
profilesidenticalexceptopt-inflag; backenddiffdeclaredscientifictransaction.
Rustmeasure-inferenceeach1800,overalltimeout7200INTthenkillafter60, serialshell
callsRustcompare-npz+all9bitwisejqandRustbase-drift+.1Aalignedjqbeforecontinuing.
DoNOTlaunchGPUjobs/heavyremoteCPUbuilds/editremotePython/profileduringstudy.
Expected~24min. PollsamePID/order; missingwrapperorstatusmustinspectRustchildren
andmeasurementsbeforeanyrestart. driver-status exit0 +sixinvocationexit0 +six
exactchecks +threealignedgatesrequiredforcompletion; stillneedsfinalRust
compare-runsconditions/summaryconfidence report (retainlegacycoordinatefail)
andrecorduserpolicyseparatelybeforeE2Eclaim.

Scriptreviewedhead/invocationpaths/CIFpaths, bash-nPASS, anchors.sha256verified.
Studyplanstoredperjobpin/reference; sourcecommit a0e86fe, upstreambb3be993,
Baseweight9ea20b0a...,N1000umeminputf4208459...,FP32C10D200S1seed101HIP7,
sharedfastinit/nonflushedcaches. No newdirectspeedupclaimed. Userupdatedabout
~24minandmaintainedentrycommit. LocalREADME/docsnowaccurate; remotePython/profile
synced,butdoNOTsyncotherassetsmidtiming. Goalactive; optimizationkernelphase
flattening, finaldirectevidence/READMEfinalizationremain.

## 2026-10-07 direct native study stopped for actual initialization mismatch

IMPORTANT supersedespriorLIVE2436859 entry. FirstnativeE2E430.392625852s/all9bits
andfirstoptimized111.838676394s/all9bitscompleted; firstalignedpairpassed.
Reasonnative430not358: maintainedworkeronlyruntime.install() onfireprotbackend,
whichinstalledconstructionhook; nativeFIREPROT_ENABLE_FAST_MODEL_INIT=1 envwas
IGNOREDbecausehookneverinstalled. Thisviolatedstudy'sSHAREDfastinitdescription.
We sentSIGINTonlytostudyPGID2436859afterverifyingitsmembers. Bythenindex2optimized
r1started; itwascancelled. Wrapper/bash/RustchildrenTERMINAL, driver-statusexit1,
psshowedNO remainingfireprot_workers/measure-inference. Preservealloldreceipts,
no restartatoldroot. Remote stopped-initialization-mismatch.json copiedto
provenance/protenix_base_native_study_initialization_mismatch.json. Oldrawpair
NOTa validshared-fastinitclaim, norcompletedthreepairresult.

FIXlocal+remote workerprotenix.py: new_install_native_construction(backend)
installs ONLY construction hookfornativewhenexplicitfastinitflag1. FireProt
runtimeinstallunchanged. Bothactivepathswriteconstruction-runtime.json after
inference, fromexistingconstruction.manifest. Defaultnativeflagabsent/0 unchanged.
Added2CPUtests proveexplicitnativeopt-inandconstructor-only install; total34
worker+basecontexttestsPASS. FreshGitNexusimpactpredict/readactualruntime45-46
andconstruction.install214-244 verifiedcause; typedindexa0e86fe/schema4current.
TargetedrsyncdryrunreviewedthenworkeronlysyncedAFTERallGPUchildrenstopped.
Numericalimplementationnotchanged; actualnativefastpathstillneedsfullstudyqual.

NewUNLAUNCHED root outputs/experiments/protenix-base-maintained-native-fastinit-pairs-20261007
preparedvia/tmp/fireprot-prepare-base-native-fastinit-study.py (bothhosts).
Freshoutputsprotenix-base-current-fastinit-{native,optimized}-r{0,1,2}; noGPUrunyet.
SameAB/BA/AB controls/profiles/anchors asprior,butdriver ALSO jqchecksactual
constructionenabled+installed,unavailablenull,selected1,recovered0,errors0 BEFORE
perrunverifiedevent. Driverbodypinnedinstudy-plan.driver afterpreparation.
run.shsyntaxchecked. Sourceprofileflagsunchangedbutworkerfixnowapplieshooknative.
Needcommitfixthenlaunchonlyaftercheckingallcontrolpins/profiles/scientificsettings
andnootherGPUjob. Overalltimeout7200INT/killafter60; eachRustinvocation1800.

Newlocaluncommitted summarize_native_study.py (ASTvalid) preparesfinalRust
comparison withdeclaredbackend/opt-in/profilehashchanges, readsall6checks,
confidence<=.001anduseraligned<=.1, retainslegacyreport/performance-null failure
andemitsseparatepolicytimings ONLYafterdriverexit0/fulljournal. Construction
selectionanddriverpinchecked too. NOTexecutedoncompleteinputs yet; reviewandrun
whennewstudycomplete. Copy scriptto/tmpremoteonlyaftercommit (notworkerpaths).
ExpectedmayneedfreshreportTAGifRustconditiondiagnosticunexpected; neveroverwrite
failedreport. Provisionaldocsupdatedmismatch/constructor reporting. No GPUlive.
Worktreea0e86fe +worker/tests/aggregator/mismatchprovenance/docs/memorydirty.
DoNOTclaimgoalcomplete: correctedshared-initdirectstudy/READMEfinalratio pending.

## 2026-10-07 corrected shared-init direct study launched

Fix/aggregator committed and pushed on master 0881f0a; 34 targeted CPU tests pass.
Fresh typed GitNexus status was current/schema4/incompleteReasons empty; staged
HIGH gate covered expected worker construction/report flows and aggregator only.
Reviewed actual native helper, runtime.install, construction.install and worker
entry before re-stage/re-gate/commit. Postcommit index refresh completed and final
detect_changes(all) returned no changed symbols (user docs/isa remains untracked).

CURRENT LIVE wrapper PID/PGID2442330, verified ps at elapsed6s, root
outputs/experiments/protenix-base-maintained-native-fastinit-pairs-20261007.
All six spec/request/profile/reference pins and driver hash verified before launch;
scientific settings equal, HIP7, both fastinit1, only backend/compositionflag differs.
All destinations absent, no GPU job active before launch. Serial timeout7200 driver,
1800 per run; AB/BA/AB. No remote source/profile changes or overlapping GPU/build
work until terminal. Poll same PID and order.jsonl; do not relaunch on observation
failure. Check first native actual constructor selected1/errors0/recovered0 and
all9 bitwise anchor match. Final driver exit0/six receipts/three aligned gates,
then run committed summarize_native_study.py with fresh TAG; this aggregator has
not yet been run on a completed direct study. README final native ratio pending.

User asked about Graph. Audited three existing graph experiments and reported
single-pair E2E results, not new measurements. Maintained composition excludes
Graph. Next graph investigation must preserve current fast path: dual_storage
rejects capture and binds its two slots to one stream; triangle_lt currently
falls back under capture and expects 524 incoming+524 outgoing Python calls.
Old recycle_graph warms side stream and captures complete48blockstack, which
conflicts with a dual workspace already owned by default stream (e.g. MSA).
Do not remove guards or silently fall back to other math. A qualified experimental
composition needs explicit stream ordering/ownership, logical replay coverage
separate from Python counters, input/weight versions and stable-address guards.
Partition capture alone does not establish lower total capture cost: 48 distinct
weighted blocks still require their own captured launches. Prefer measuring
setup vs replay on current route before promising block partition gains.
Local investigation only during live direct study; no new graph GPU experiment yet.

## 2026-10-07 local composed trunk Graph prototype prepared during direct study

LIVE direct wrapper2442330 recheckedelapsed03:05, nativeindex0running; no restart.
NewLOCALUNCOMMITTED experimentstrunk_graph.py,graph_dual_storage.py,
graph_triangle_lt.py andtests/adapters/test_protenix_trunk_graph_contract.py.
ASTall3PASS;10CPUcontracttestsPASS (notGPUqualification). Derivedtrianglehelpers
frommaintained0881f0a; originalsunchanged. Entiremodelinvocationbotharmssameseparate
side streamavoidsMSA/defaultworkspaceownershipcollision;captureper4/12/48blocks,
firsttrunkwarm,secondcapture/replay,8remainingreplay. Global/localbiascachesand
transitionfusionkeptoutsidecapturedtrunk. Graph poolsharedwithstrictcaptureorder.
Parameter/bufferversion/shape/RNGguards; partitionstaticinputsretained.
Ltpermitcaptureonlydeclaredcontract (noalgorithmfallback); dualstoragepermits
captureonlysamephase/streamandregistersbefore-replayownershipguard. Expected
Pythoncontractionsgraph280=88other+96warm+96capture vs1048off;logicalboth1048.
Incoming/outgoing140eachgraph vs524off;2projectionsperpair. No GPUbuildneeded.
Unqualifiedprototype READMEexplicit. Needfreshindex/commitgatebeforecommit.
DoNOTsync/editremotePythonorlaunchuntilcurrentdirectstudyterminal. Afterwards
freezeindependentworker snapshot andreplacebase_optimizationsimportwithtrunk_graph,
include3newmodules+diffusion_graph.pyidentityhelperinPYTHONPATH. Usemaintained
publishedbundleprofile;graphofffirstmustall9bitsvsmaintainedanchor;thenonG12
sameworker/profileexceptflag, fullcoverage+all9bitsandE2E. Recordcaptureandfirst
replaysubmissiontime separately (instantiationmaybe deferred). No claims yet.

ComposedGraphprototypeCOMMITTED/PUSHEDmastera288ecd. All44targetedCPUtestsPASS;
postcommitindexrefreshandfinaldetect_changes(all)clean; onlyuserdocs/isauntracked.
No experimentfiles syncedremote and no GraphGPUrun yet.

Directstudy2442330 LIVE; nativeindex0 TERMINALVERIFIED, optimizedindex1 started.
NativeE2E382.844250605s/model358.508248045s; all9arraysBITSvsnativeanchor. Actual
constructionenabledinstalledtrue/selected1/recovered0/errors0, init3989.245ms,
checkpoint1070.912ms. Thus currentnative~383 vs historical358 is MODELforward
~358.5 vs333.3, not oldmissinginitializationhook. Preserveobservation; awaitall
repeatsbeforeinterpretingvarianceorcurrentdirectratio. DoNOTdiscardorreplaceit.
Studyroot/orderdriverasabove; wrapper2442330confirmedliveelapsed06:27. Index1
RustmeasurePID2444964/worker2444966 observed; nooverlap/sourceschanged.

## 2026-10-07 first corrected direct pair verified; graph deployment prepared locally

Directwrapper2442330 confirmedLIVEelapsed08:35. Index0native382.844250605s,
index1optimized112.639672091s bothTERMINALVERIFIED(all9bits/actualfastinitselected).
Pair0alignedRMSD0.0060998239421515815A/gatePASS. Approx3.40xFIRSTPAIRonly, notfinal
threepairclaim. Index2optimized-r1started;remainingnative-r1/native-r2/optimized-r2.
No remote changes/builds/GPUoverlap. Keep samePID/study, preserveallobservations.

LOCALONLY /tmp/fireprot-prepare-composed-graph.py prepared(ASTPASS), sourcebundle
/tmp/fireprot-composed-graph-source-a288ecd with manifest and4files (trunk_graph,
graph_dual_storage,graph_triangle_lt,diffusion_graph). NOTtransferred/executed.
PreparationrequirespriorDIRECTdriver-statusexit0 andnoactiveworker/measureprocess.
Createsfreshrootoutputs/experiments/protenix-base-composed-trunk-graph-20261007,
copyrootpython/fireprot/kernels; snapshotworkerchangesbase_optimizationsimportto
.fireprot_workers.trunk_graph(relative.trunk_graph actual). Copiesall4helpersinside
python/fireprot_workers (Rust-fingerprinted source tree), rewritesonlyhelperimports
torelativeinFROZENcopies; originsmanifestpinsoriginal+frozencopies. Maintainedremote
workeruntouched. CreatesG12off/onprofiles,requests,specs,run.sh,study-plan,
anchors.sha256. EachRustmeasure1800; serialoffthenonrequiresall9exactanchorbefore
nextarm. PreparationdoesNOTlaunch. Futurelaunchonlyafterinspection/pinchecks and
currentdirectstudyfinished. Use overalltimeout~4000/INT/kill60; then comparepair
viaRustexpectedonlyFIREPROT_BASE_COMPOSED_GRAPHoff/on+profilehash. NoGraphperformance
claimbeforeall9bitwise+completecoverage and E2E. Native-summaryaggregator keyformats
spotcheckedagainstoldscopedLtRustreport; full6runaggregationstillpending.

## 2026-10-07 historical native baseline route audit

Directindex2optimized-r1 TERMINALVERIFIED112.021993353s/all9bits;index3native-r1
nowrunning(samewrapper2442330). Importantnewfinding: oldstrict1000-init-control
requestbackendwasFIREPROT withtriangleopsdisabled, notbackendnative. Itsmanifest
1048nativeattention+1048nativemul, modulehooksdisabled, AMDextensionnotloaded.
Upstreampinnedtriangular.py715 omitsallvalidmaskONLYfireprotbranch;nativeconstructs
maskthenzeromaskbias. layers.py276_attentionaddsEVERYbiasinplace. Thus old357.921s
controlTorchmathalreadyhaszerobiaselision; currentnativeactuallyretainsit. This
explainsaKNOWNexecutionpathdifference; isolated25stimingshareNOTmeasured, don't
attributeallslownesstoitordescribeoldcontrolasidenticalnativeentry. Newlocal
provenance/protenix_base_native_baseline_route_audit.json pinsbothrequest/execution,
oldmanifestcountersandupstreamsourcehashes. README/Baseguideclarifybaseline.
No directstudycontrolsmodified, noGPUoverlap; pendingdoc/evidencecommit.

## 2026-10-07 two direct native pairs complete, last pair active

Wrapper2442330confirmedLIVEelapsed16:44;index3native-r1TERMINALVERIFIED
383.350351039s/all9bitsandconstructorselectionpass. Nativefirst382.844250605,
second383.350351039; optimizedfirst112.639672091,second112.021993353. Thuscurrent
native~383isreproducibleacrossfirsttworuns, notisolatedslowoutlier. Index4native-r2
STARTED, finalindex5optimized-r2stillpending. Keepallmeasurements; finalthreepair
aggregation/READMEratio/GraphGPUqualificationnotcomplete. Currentmasterf711328
pushed, postcommitGitNexuscurrentanddetect_changesallcleanbeforethismemoryappend.
No remotechangesorsync/GPUoverlap. SamepreparedGraphscript/bundlelocalonly.

Local /tmp/fireprot-summarize-composed-graph.py nowprepared(ASTPASS),NOTexecuted
ortransferred. ReadscompleteGraphdriver/order/control/sourcepins, checksG12off/on
sameworkload/profilesexceptflag, actual280vs1048Pythoncontractionsandlogical1048,
Lt140vs524each/no fallback, dualoneallocation, on4captures36replays. Rustcompare-npz
rechecksall9exacttoanchor; compare-runsexpectedgraphflag/profilehashonly, unchanged
gatesmustPASS. FreshTAGoutputs preserveallfailedreports. Scriptneedsrealartifact
qualificationafterGraphstudy; doNOTtreatASTvalidationasperformance/correctness.

## 2026-10-07 maintained direct native study COMPLETE: 3.42210x

Wrapper2442330 TERMINAL SUCCESS(driverexit0), sixverifiedruns and3alignedchecks.
NativeE2E[382.844250605,383.350351039,387.153919299]; optimized
[112.639672091,112.021993353,111.644131692]. Medians383.350351039->112.021993353,
3.4220989965x/70.77817%less. Modelmedians358.508248045->87.516323107,4.09647x.
All6all9bitsvsownpredictionanchors; actualfastinitselected1/noerrors/norecovery.
EachcrossarmallatomalignedRMSD.00609982394215158A;summaryconfmax4.959106445e-5,
exactmetadata/CIFidentitiespass. UserpolicyPASS; genericlegacy.001AFAIL retained,
performance null. FullconfidencearraysdiagnosticmaxPAE.0409927368/PDE.1253623962;
NOTcrossarmbitwise. Rawmaxcoordinatecomponent.13372135A isNOTuserRMSDcriterion.

Aggregatorfinal-v1FAILEDconditionsbeforemetrics: missingexpectedcandidateonly
launch.inputs.rust-bridge. PreservedemptyJSON/stderr/spec. Addedexplicitdescriptor
fromverifiedprofilepinnedbridge (baseline null), no gatechanges. final-v2SUCCESS
on same6archives. Copiedfullsummaryto
provenance/protenix_base_maintained_native_three_pairs.json. README/Baseguide
updateddirect3.422xentryandcorrectnessscope, uncommitted. FreshGitNexusimpact
summarizeLOW1caller afterdisambiguation; onlyscript__main__. localASTPASS, real
sixrunaggregationvalidateschange. Needcommitdocs/evidence/aggregatorfix.

Graph followup rootoutputs/experiments/protenix-base-composed-trunk-graph-20261007
preparedusinglocalgenerator and4filestransferafterdirectterminal. Snapshothelpers
insideRustfingerprintedworkerswithrelativeimports; maintainedremoteuntouched.
Wrapper2456043TERMINALFAILURE: eagerOFF rejectedfirsttrunk signature;16dual
projections/8MSAcontractionsalreadyran;onNEVERstarted. Alllogs/outputsretained.
FailuregenericUnsupportedtrunkgraphinputsorstream; no guardsrelaxed. Addedlocal
rejected_signature detailedmetadata totrunk_graph.py onlyexceptionbranch after
freshindex/impact(noresolvedcallers, manuallyreviewedstack.forwardassignment).

CURRENT LIVE diagnostic-onlyOFFwrapper2457206, root
outputs/experiments/protenix-base-composed-trunk-graph-diagnostic-v2-20261007.
Freshsnapshotwithdiagnostic-onlysource, oldrootunchanged; onlyOFFprepared; HIP7.
Sameguardfailureexpectedbutnowbase-composed-trunk-graph.json records exact flags,
shapes,strides,devices,stream,kwargs. Inspectwhen terminal beforefix; noGraphON.
Local/tmp/fireprot-prepare-composed-graph-diagnostic-v2.py andsourcebundle
/tmp/fireprot-composed-graph-source-diagnostic-v2. Sourceoriginlabela288ecd+diagnostics.
No remoteactivejobsbeforelaunch, timeout1900 each1800. Goalactive: Graphqualification
andfinalplateaudecisionpending; directREADMEresultnowcompleteawaitingcommit.

## 2026-10-07 Graph FP32 autocast guard corrected, fresh pair running

DirectnativeREADME/evidence/aggregatorfixCOMMITTED/PUSHEDmaster2e29d5f,3.42210x.
Diagnosticwrapper2457206 TERMINALFAILUREexpected. Detailedrejectionproved
s[1000,384]/z[1000,1000,128],FP32,gradfalse,streammatches,kwargsallcorrect;
ONLYautocasttrue. SeparateHIP7Torchprobe confirmedtorch2.12.0a0+git78258b9/
HIP7.14.60850 reportsautocasttrue AND targettorch.float32 insideupstreamexplicit
FP32context. ExistingupstreamInferenceRunner.predict creates that context.
NoBF16modelrequest. Fixexperimental_trunk.forward acceptsdisabledorFP32target,
stillrejectslowerprecision; autocasttupleboundintosignatureacrosscycles.
FreshGitNexusimpactforwardunknown0; manuallyverifiedstack.forwardcaller. Evidence
provenance/protenix_base_graph_fp32_autocast_diagnosis.json pinsbothfailures,
signaturesandworkerhashes. Failedmeasurementsusefailure.json, noinvocation.json.

CURRENT LIVE Graph pair wrapper2458570, root
outputs/experiments/protenix-base-composed-trunk-graph-fp32-v3-20261007.
Samefrozenfourhelpers withcorrectedguard, originslabela288ecd+fp32-autocast-guard.
Allcontrol/source/referencepinscheckedandnootherGPUjobbeforelaunch. OffthenonG12,
HIP7 FP32C10D200S1seed101, timeout4000overall/1800each. Ononlyrunsafteroffall9bits.
Currentoffoutputoutputs/protenix-base-composed-graph-fp32-v3-off-g12;onanalogous.
Oldv1/v2rootsuntouched;maintainedremotePythonunchanged. Newlocal/remote/tmp scripts:
fireprot-prepare-composed-graph-fp32-v3.py,fireprot-summarize-composed-graph-fp32-v3.py,
sourcebundlefireprot-composed-graph-source-fp32-v3. AggregatorfreshTAGonly after
fullsuccess. Local10CPUcontracttestsPASS; actualGPUqualificationpending.
Sourceguard/diagnosisevidence/README/memoryuncommitted. Nextpollsame2458570,
inspectterminalreceiptsbeforeanyrepeat. No otherGPUwork/builds whiletiming.

## 2026-10-07 composed G12 Graph passes; G48 candidate running

GraphFP32v3 wrapper2458570 TERMINAL SUCCESS, off113.765412431/model89.189898546,
on112.868515248/model88.866562517s. Bothall9BITSvssamemaintainedanchor; unchanged
Rustcompare-runsgatesPASS. G12on4captures36replays/280Pythoncontractions1048logical;
capture11.654718172s/firstreplaysubmit.000166301s,peakallocated13181979648.
Onepair1.007946389x/.79%less; modelgainonly.323s, no stablepromotionclaim.
Aggregator/tmp/fireprot-summarize-composed-graph-fp32-v3.py first-v1SUCCESS;
copiedevidenceprovenance/protenix_base_composed_graph_g12_first_pair.json.
DocsupdatedG12resultuncommitted; sourceguardfix9c8d9f5COMMITTED/PUSHED.
Post9c8d9f5indexrefresh/finalgatecleanbeforethesedocadditions.

CURRENT LIVE G48candidatewrapper2461333; root
outputs/experiments/protenix-base-composed-trunk-graph-g48-20261007,
plan.json/spec.json/request.json/profile.json/run.sh/pid/launch.log/driver-status.
UsesSAMEfrozenworkerfromFP32v3; onlyFIREPROT_BASE_GRAPH_GROUP_SIZE12->48 vsG12on.
Outputoutputs/protenix-base-composed-graph-on-g48; HIP7FP32N1000C10D200S1seed101.
SinglecandidateNOTpairedspeedupclaim. DriverRustmeasure1800 (overall1900), then
compare-npzall9bitsandgraphcalls10/captures1/replays9/logical1048. Allpinschecked,
nootherGPUactivebeforelaunch. Needterminalandallchecks thencompareG48/G12setup/E2E;
ifmaterialgainrepeatmatchedbestgraph/eager. CurrentG12onlymarginalandstill~112.9,
notbetterthanmaintained~112median; doNOTmultiplyratiosinto3.422directresult.
No maintainedsource/profilechangeswhilecandidateactive.

## 2026-10-07 G48 trial complete; direct maintained-vs-Graph confirmation next

G48wrapper2461333 TERMINAL SUCCESSdriverexit0. E2E113.415505891s, capture7.098824058s,
firstreplaysubmit.000096581s,1capture9replays280Python1048logical;all9BITSanchor,
Lt140incoming140outgoingnofallback;dual280pairs. Peakallocated11636760576.
LowercapturecostthanG12(11.65)butnotlowerE2E(G12on112.869). Evidencecopiedlocal
provenance/protenix_base_composed_graph_g48_trial.json,currentlyuncommitted.
NO GPUjobsactiveafterthiscandidate. Masterd57b25b pushedG12docs/evidence; refreshed
GitNexusfinalgatecleanbeforeG48record. Goalactive; Graphnotpromoted.

Next meaningfulstep: repeatBESTG12 directlyagainstMAINTAINED eagerentry, notjust
experimentaloffside-streamguardedcontrol. Prepare newfrozenworker/profile pair
withsameallsourcefiles; addexperimental mode 'maintained' totrunk_graph.install
that delegates to maintainedbase_n1000.install and returns BEFOREside-stream/
graphwrappercreation; modeon unchanged. Both useflagBASE_N1000_OPTIMIZATIONS1.
DeclareonlyFIREPROT_BASE_COMPOSED_GRAPH maintained/on +profilehashdifference.
ThreeAB/BA/ABpairs(~12min), eachall9exacttooptimizedanchor before next, fullRust
compare-runsconditions/gates. This testsactualentryE2Eandavoidsmistakingguard/
streamcontrol overhead for a production gain. Do NOT rerun expensive nativearm.
If Graphno stablegain, retaincurrentmaintained3.422xdirectresultanddocumentplateau.
No suchmodeadded/runslaunchedyet; beforeeditinginstall freshindex/impact/manual
callercheck (frozenworker explicit import). PreserveallG12/G48/failureartifacts.

## 2026-10-07 preparing direct maintained-eager versus G12 repeats

VerifiednoactiveGPUworker/measurebeforework. Newexperimentalmaintainedmode in
trunk_graph.install delegates to maintainedbase_n1000.install beforeTorch/helper
imports/side-stream/graphguards; requiresopt-inflag1. Added2CPUtests delegation+
exceptionunwindanddisabledflagrejection;46targetedtestspass. Modesoff/onunchanged.
Initialimpactinstallunknown0callers; manualfrozenworkerimportreviewed. Subsequent
CLIstatussaidstale despiteHEADd57+runnercurrent, so refreshingindexbeforefinal
review/commit instead ofrelyingonpreviousgraph. NoGPUrunlaunchedyet. G48evidence
andREADMEcontrolscopependingcommitwiththesechanges. Needfrozennewworker/common
profileAB/BA/AB sixruns withmode maintained/on;same4helperfiles/controls/pins,
HIP7, all9anchorcheckeachbeforecontinuing. Noextraexpensivenativerunsneeded.

## 2026-10-07 maintained-vs-G12 three-pair study LIVE

Maintainedcontrolmode/tests/G48evidenceCOMMITTED/PUSHEDmaster35a6409.46CPUchecksPASS.
Freshindexstatusup-to-date/current/incompleteReasons[]; highstagedgate6existing
install/loadflowsreviewed(manualearlyreturntomaincontext,nograph setup), re-gated
thencommitted. Postcommitindexrefresh/finaldetect_changesallclean.

CURRENT LIVE wrapper2462871, root
outputs/experiments/protenix-base-maintained-vs-graph-pairs-20261007.
Preparedusing/tmp/fireprot-prepare-maintained-graph-pairs.py andsourcebundle
/tmp/fireprot-composed-graph-source-maintained-pairs onbothhosts; frozenhelpers
origin35a6409. All6controls/reference/sourcepinschecked, identicalscience/profiles
exceptmode,6freshoutputs,noGPUactiveatlaunch. HIP7 C10D200S1N1000seed101FP32.
Order maintained,on,on,maintained,maintained,on; eachRustmeasure1800,overall4000.
Driverall9BITSanchorbeforeverified/next. Script/study-plan/frozen-origins/pid/log
atroot; driver-statusonlyterminal. DoNOTeditremoteoroverlapGPU/builds.
Outputsprotenix-base-maintained-graph-{mode}-{index}-g12. Profilesmodeonlyeacharm;
measurement{mode}-{index}-measurement; seeplanratherthaninferfilenames.

NewLOCAL /tmp/fireprot-summarize-maintained-graph-pairs.py (ASTPASS,NOTexecuted or
transferred) authenticatesall6pins/journal/constructor/counts, maintainedcontext
implementationvsgraph4captures36replays, all9exactanchorchecksagain, thenRust
compare-runsunchangedgateswithonlymode/profilehashdeclared. FreshTAGdefaultfinal-v1.
Requiresdriverexit0; fullsummarygives3pairmedians/modelmedians/pairedratios. Useonly
whenall6terminal. GoalactiveuntilGraphstabledecisionandfinalREADMEplateauaudit.
Currentheadline3.42210x belongsnoGraphmaintainedpath, donotmultiplyincrements.

## 2026-10-07 Graph repeats underway; independent Base template candidate

Wrapper2462871 confirmedLIVEelapsed04:28; maintainedindex0TERMINALVERIFIED
112.761228699s, Graphindex1TERMINALVERIFIED113.684114227s (Graphslowerfirstpair),
index2Graphstarted. Allcompletedall9bits. No remote changes/GPUoverlap.

Read-onlyauditfoundunexploredCPUpreprocessingtransferfromMini: features.py supports
onlyMini-default0.5; Base-default0.5 configalso template_embedder.n_blocks0,
use_templatefalse. UpstreamTemplateEmbedder.forward guardexitbeforeinputreads;
ASTsourcehash74cf338e... matchesverifiedMini consumerpin exactly. Nativeinference
stillcallsmap_to_standard/assemble evenwithuse_templateFalse; skippingmayreduce
~featureworkbutNOmeasuredBasegainyet. Datasetcreatesexistingdummyfeaturesfor{}.

NewLOCALUNCOMMITTED experiments/protenix_base/unused_templates.py context wraps
entireworkerpredict (beforefeatures), temporarilyextendsfeatures._supportedONLY
Base-default0.5FP32/templateunused/zero blocks/verifiedsource/flag1. Othermodels
useoriginalpredicate, exceptionrestoreshook. Newtests/adapters/
test_protenix_base_template_experiment.py +existingtest_protenix_features.py:
17CPUchecksPASS. No maintainedcodechange, no remote transfer/GPUrunyet.
Needfreshindex/impactfeatures._supported+build_template_features reviewbefore
commit/frozenworkerpreparation. AfterGraphrepeatsterminal, canqualifyseparately:
clonemaintainedworker, wrap__main__ bundle=predict in unused_templates.install;
keepnoGraphmaintainedprofile andonlyfeatureflag0/1diff. Bothoutputsall9bits to
maintainedanchor; inspectfeature_preparation skip/nativecounts andE2E. Neither
assumeignoredfeaturesnorCPUtestsprovepredictionRNGequivalence. Ifno materialgain,
reject; ifgainrepeatandsecondinput5ZMWbeforepromotion. This closesremaining
Mini-to-Base preprocessingcandidate beforedeclaringE2Eplateau.

TemplatecandidateCOMMITTED/PUSHEDmaster6cafde8; postcommitindexrefresh/finalgate
clean. Freshfeature._supportedimpactLOW directbuild_template_features reviewed,
manualupstreamdatasetcallverified(theProtenixcheckoutisindependentofgraphindex).
17CPUtestsPASS. Graphrepeatwrapper2462871 stillrunning, lastobservedindex3maintained.
NewLOCALONLY /tmp/fireprot-prepare-base-template-pair.py (ASTPASS) andsourcebundle
/tmp/fireprot-base-template-source/unused_templates.py+manifest. RequiresGraph
repeatdriverexit0 andnoGPUjobsbeforepreparation. Createsfreshrootoutputs/experiments/
protenix-base-unused-templates-20261007; clonesmaintainedpython/fireprot/kernels,
wrapsfrozenworkers/__main__.py entirepredictwithunused_templates.install, leaves
modelworker/base_n1000maintainedcontextunchanged. Off/ononlyfeatureflag0/1; HIP7,
usesUMEMrequestandmaintainedanchor. Driverall9BITSandfeature_preparationcounts
skip0/native1 orskip1/native0beforeeachverifiedevent. PreparationdoesNOTlaunch;
reviewpins/requests/profiles thenlaunchaftercurrentstudy. NoBaseGPUparttested yet.

## 2026-10-07 Graph direct confirmation complete: reject promotion

Wrapper2462871 TERMINAL SUCCESS,6verifiedruns. Aggregator
/tmp/fireprot-summarize-maintained-graph-pairs.py final-v1 SUCCESS. MaintainedE2E
[112.761228699,113.444521726,112.632747294]; Graph
[113.684114227,112.623723071,113.374821634]. Medians112.761228699->113.374821634,
ratio.9945879259 (Graph~.54%slower), pairedratios.991882/1.007288/.993455.
Modelmedians88.361868094->88.477236631. All6all9BITSanchor, allGraph4captures36
replays1048logical; unchangedRustcompare-runsgatesPASS. Copiedevidence
provenance/protenix_base_maintained_vs_graph_three_pairs.json. README/Baseguide/
experimentguide updateddecisionNO GRAPHpromotion; priorG12singlepairgainwas
experimentalcontrol, notproductiongain. Directnative3.42210xunchanged.

CURRENT LIVE unused-templatepairwrapper2468516, root
outputs/experiments/protenix-base-unused-templates-20261007, run.sh/plan/pid/log.
Preparation+source+aggregatortransferredafterGraphterminal; noactiveGPUatlaunch.
Allsource/entry/control/referencepinschecked; frozenworker__main__contextcovers
features, maintainedGPUcompositionunchanged. HIP7N1000umemFP32C10D200S1seed101,
offthenon, onlyFIREPROT_SKIP_UNUSED_TEMPLATES0/1 differs. EachRustmeasure1800,
overall4000INT/kill60; all9BITSandactualfeaturecountsrequired before next.
Outputsprotenix-base-unused-template-{off,on}; driver-statusatcompletion. Candidate
ONrequiresverifiedconsumerhashandskip1/native0; OFFskip0/native1.

/tmp/fireprot-summarize-base-template-pair.py isonbothhosts,ASTPASSNOTyetexecuted.
Afterdriverexit0, freshTAGfirst-v1 verifiescontrols, consumer/counts, Rustexact
arrays+conditions/gates, emitsfirst-v1-summary.json. No performanceclaimyet.
CurrentGraphdocs/evidence/memoryuncommitted; master6cafde8 pushed. Needindex/gate
thencommitwhiletiming(remoteuntouched). NootherGPU/buildwhile2468516live.

## 2026-10-07 unused-template first pair PASS; repetitions LIVE

Templatewrapper2468516 TERMINAL SUCCESS. Aggregator/tmp/fireprot-summarize-base-
template-pair.py first-v1SUCCESS; copiedevidence
provenance/protenix_base_unused_templates_first_pair.json. OFFE2E112.364168339/
model87.901271572; ON110.163983404/model87.474230924;1.019971908x onepairONLY.
Bothall9BITSmaintainedanchorandRustgatesPASS. ActualfeaturecountsOFFskip0/native1,
ONskip1/native0, consumerSHA74cf338e...matches. No GPUsemantic/RNGchangeobserved.
Docsupdatedfirstpair, no maintainedpromotion; main3.422xdirectresultstillflag0.

CURRENT LIVE template repeatwrapper2470512 same root
outputs/experiments/protenix-base-unused-templates-20261007. repeat-plan.json,
repeat-run.sh/repeat-pid/repeat-launch.log/repeat-status.json/repeat-order.jsonl.
Samefrozenworker/profiles asfirstpair; new4requests indexed2-5:pair1ON/OFF,
pair2OFF/ON, completingAB/BA/AB across6runs. HIP7only;allpinscheckedandnootherGPU
jobbeforelaunch. Eachmeasure1800,overall4000; eachall9BITS+actualskipcounts before
next. Outputsprotenix-base-unused-template-{on,off}-r{1,2}. Newmeasurements
{mode}-r{pair}-measurement; referencesmaintainedumem. No remotesourcechanges.

Needaggregatesixruns afterrepeat-statusexit0 plusfirstdriverexit0; authenticate
bothfirststudyplan and repeatplan/driver/controlpins, combine firstrows(pair0)
withrepeatrows(pair1/2), checkall9bits/counters/sourceconsumer andRustcompare-runs
expectedonlyFIREPROT_SKIP_UNUSED_TEMPLATES0/1+profilehash. UsefreshTAG,e.gthree-v1.
SixrunaggregatorNOTyetwritten. Onlyiftimingshold, qualify5ZMWsameflag1 against
outputs/protenix-base-maintained-5zmw-registry-v2 orold5zmwoptimized (all9bits)
beforepromotingfeatures.pyBaseguard andprofileflag1. Do not assumeperformance
fromonepairormultiply2%into3.422headline. CurrentGraphdocscommit a82bbae pushed;
newtemplatefirstpairdocs/evidence/memorydirty. Goalactiveuntilthesechecks/decision.

Local /tmp/fireprot-summarize-base-template-three-pairs.py nowprepared(ASTPASS),
NOTtransferred/executed. Requiresfirstdriver/repeat-statusexit0, pinsfirst/repeat
plans/drivers/origins, checksbothjournals, combinesfirst2rows(pair0)andrepeat4rows
(pair1/2), all6actualfastinit/featurecounts/consumer/context/Lt/dualguards, reruns
Rustall9bitwiseanchorchecksandunchangedcompare-runsfor3pairs. FreshTAGthree-v1.
Defaultscriptrootcurrentunused-templates20261007, expectedonlyfeatureflag0/1+
profilehash. Needaftercompletion transfer/run/copyevidence,then5ZMWqual.
Read-onlyverified5ZMWrequestsourceexistsatmaintained20261007/5zmw-request.json,
inputhash938324d4... sourcebb3be993 BaseFP32C10D200S1seed101HIP7. Clonefreshoutput
anduseSAMEon-profile oftemplateexperiment; anchor maintained-5zmw-registry-v2.
No5ZMWjob/configcreatedyet. Rootmaster7407693pushed(firstpairdocs), postcommit
indexrefresh/finalgatecleanbeforethisappend. Currenttemplate repeat2470512live.

## 2026-10-07 template three-pair PASS; 5ZMW qualification LIVE

Template repeat2470512 TERMINAL SUCCESS. Aggregator
/tmp/fireprot-summarize-base-template-three-pairs.py three-v1SUCCESS; copied
provenance/protenix_base_unused_templates_three_pairs.json. OFF
[112.364168339,111.93632923,111.763280051], ON[110.163983404,110.272235805,110.248044419].
Medians111.93632923->110.248044419,1.0153135125x; paired1.01997/1.01509/1.01374,
allthreefaster. Modelmedians87.711947716->87.653227046 (preprocessinggain).
All6all9BITSanchor/actualskipcounts/consumerpin andRustconditions/gatesPASS.
Docsupdated3pairincrementalresult; NOTmultipliedintodirectnativeheadline3.422x.

CURRENT LIVE 5ZMWwrapper2474958, root
outputs/experiments/protenix-base-unused-templates-5zmw-20261007.
Preparedusing/tmp/fireprot-prepare-template-5zmw.py (bothhosts), afterrepeatand
aggregatecomplete/noGPUjobs. Samefrozenon-worker/profileas3pairs; fresh
outputs/protenix-base-unused-templates-5zmw, HIP7 FP32N1000/7711atomsC10D200S1
seed101, input938324d4... . plan.json/spec.json/request/profile/reference/run.sh/
pid/launch.log/driver-status atroot; measure1800/overall1900. Pinnedreference
outputs/protenix-base-maintained-5zmw-registry-v2/sample-0.npz; driverchecksall9
bits, feature skip1/native0, contextatom_count7711/errornull. Singlequalification,
notpaired5ZMWspeedup. Onterminal collectevidenceonlyifallchecksPASS.

Nextafter5ZMWPASS: considerpromotingBase0.5 into maintainedfeatures._supported
allowedmodels, keepingexplicitflag/source/dtype/templateuse/nblocks guards and
Mini behavior; updateBaseoptimizedprofilefeatureflag1. Needfreshindex/impact
beforeexistingfeature/testedits, CPUtestsparameterizedMini+Base+unsupportedBase1.0,
maintainedUMEM/5ZMWqualificationbeforeclaimingnewentryresult. No promotioneditsyet.
Currentmaster7407693; new3pairdocs/evidence/memoryuncommitted. Goalactive until
qualification/promotion/finalE2Escope+plateaudecision. NooverlapGPU/builds.

## 2026-10-07 template 5ZMW PASS; maintained promotion qualification LIVE

5ZMW2474958 TERMINAL SUCCESSdriverexit0, E2E110.769354252/model88.075113835s.
All9BITSmaintained5ZMWanchor;feature skip1/native0+consumerpinPASS; context7711.
Copiedprovenance/protenix_base_unused_templates_5zmw.json. Notpairedspeedup.

Newmaintainedsourcefeatures.py._supported addsBase-default0.5alongsideMini-default0.5;
allotherguardsunchanged. BaseN1000optimizedprofilefeatureflag0->1. New7Baseguard
CPUcases +existingtestsTOTAL70PASS (workers/Basecontext/Graph/featureexperiments).
Freshindex/impact_supportedLOWdirectbuild_template_features reviewed. Targeted
rsyncdryrunreviewedthenONLYfeatures.py+profiletransferredAFTER5ZMWterminal/noGPU.
Frozenexperimentalworkersunchanged. Source/profile/tests/docs/evidenceUNCOMMITTED.

CURRENT LIVE maintainedqualificationwrapper2476381 root
outputs/experiments/protenix-base-maintained-templates-20261007. plan.json/run.sh/
pid/launch.log/order.jsonl/driver-status. Samepublicrootworkers/profile,UMEMthen
5ZMW freshoutputsprotenix-base-maintained-templates-{umem,5zmw}. BothHIP7FP32
N1000C10D200S1seed101;atom7979/7711. EachRustmeasure1800,overall4000. Allsource/
control/referencepinschecked,nootherGPUatlaunch. Driverall9BITSoldmaintainedanchors,
actualtemplatecounts1/0andcontextatomcount/errornull beforeverified. DoNOTedit
remotePython/profileoroverlapGPU/builds whileactive.

NeedterminalbothPASS thenrecordmaintainedmigrationevidence,commitpromotionwith
READMEscope. READMEalreadyclarifies3.422xdirectseriespredatesfeatureomission;
latestincrement1.0153xseparate, donotmultiply. Considerfinaldirectnative3pairs
usinglatestmaintainedprofileafterpromotionfixed/committed toreportupdatedentry
E2E honestly (oldnativebaselinehasfeatureflag1ignoredbybackendnative; sharedfast
constructoractualselectionmandatory). Canreuse native-study generator/aggregator
withFRESHroot/outputnames andexplicitlatesttemplatecandidate scope; oldroots
neveroverwrite. No finalstudy prepared/launchedyet. Goalactive; afterfinal
entryqualification/E2E and documentedoperator/Graphplateau do completionaudit.

## 2026-10-07 maintained template promotion qualified on both inputs

Re-polled wrapper2476381: handle absent and driver-status exit0; exact journal
contains both started/verified pairs. Independent aggregation re-authenticates
plan/driver/features/spec/request/profile/reference/acceptance pins and reruns
Rust compare-npz. Both all9 arrays bitwise, counts skip1/native0, consumer SHA
74cf338e... and maintained context atom7979/7711 verified. UMEM E2E110.200394837;
5ZMW111.005110345. Evidence provenance/protenix_base_maintained_templates_qualification.json.
No paired speedup inferred from these two migration runs. 70 targeted CPU checks
PASS again. README and guides now mark entry qualification complete.

Fresh typed GitNexus status up-to-date/current/incompleteReasons[] at7407693;
impact summarize LOW, sole direct file caller (__main__) reviewed. Added actual
feature count/hash verification to direct-native aggregator only when candidate
profile requests omission; historical flag0 remains supported. Remote python
command absent, retried harmless aggregation with python3; no GPU rerun.
Final direct-native generator remains /tmp/fireprot-prepare-base-final-native-study.py,
not launched yet. Need commit/push promotion, then fresh final study3pairs with
latest maintained profile, shared actual fastinit, noGraph HIP7. Goal active.

## 2026-10-07 promotion committed; final direct-native three-pair study LIVE

Promotion/docs/evidence/conditional actual-omission aggregator COMMITTED/PUSHED
master1b1825ac6b0b6bab76056a688ff76d954b9c118a. Staged GitNexus gate MEDIUM,
8 symbols/2 expected summarize pin/read flows, not partial/truncated; reviewed.
Postcommit index refreshed with PDG, typed status up-to-date/current/incompleteReasons[];
final detect_changes all clean before this append. 70 targeted CPU checks PASS.

CURRENT LIVE wrapper2479402, root
outputs/experiments/protenix-base-final-native-pairs-20261007.
Authoritative ps confirmed timeout4000 live at00:16, journal native pair0 started.
Prepared by /tmp/fireprot-prepare-base-final-native-study.py on both hosts;
all6 spec/request/profile/reference/driver pins checked, output/measurement dirs
fresh, AB/BA/AB order. HIP7 only FP32N1000C10D200S1seed101, published kernels,
shared actual fastinit, candidate template omission, no Graph. Runtime profiles
identical except Base composition0/1; backend native/fireprot differs. Each run
must all9BITS match its respective qualified anchor, constructor selected1/no
recovery/error, candidate skip1/native0+consumer hash; each completed pair runs
aligned0.1A drift check before continuing. No competing GPU jobs at launch.
Do NOT modify remote Python/profile or start GPU/build work while active.

Remote Protenix Git HEAD is historical d3b4db6a because sync excludes Git metadata;
all248 actual source files rehashed MATCH source manifest bb3be993 exactly.
Use manifest verification for deployed source, not unsynchronized remote Git HEAD.
Frozen CLI SHA dfbbe303... unchanged; maintained features SHA16cdca60229f... .

Updated aggregator transferred with reviewed targeted rsync to
experiments/protenix_base/summarize_native_study.py on remote. After driver-status
exit0 and exact12 journal events, invoke python3 SCRIPT STUDY
/media/IntelSSD/fireprot/target/release/fireprot final-v1. It verifies candidate
actual omission when flag1, authenticates Rust conditions and archives, preserves
legacy .001A gate failure separately from user .1A/summary .001 policy. Never
relax generic gates. Copy final summary to new provenance filename, update README
and Base guide direct ratio with actual latest medians, then completion audit.
Latest published direct3.42210x still predates omission; donotmultiply1.01531.
Need poll SAME2479402; observation timeout is not terminal. Goal remains active.

## 2026-10-07 completion audit drafted while final native pairs run

Previous goal turn classified progress: committed/pushed qualified feature change
and launched a fresh direct-native study. Re-polled SAME2479402 twice, latest
elapsed02:15 confirms live; journal only native pair0 started. No remote changes
or competing GPU work. Final ratio remains unmeasured until all six complete.

New docs/protenix-base-optimization-audit.md maps workload/entry/correctness/
publication requirements to actual sources and separately records plateau evidence
(CK scheduling, projection tiles/fusion, transition, gate, Graph, templates).
It explicitly leaves final E2E/terminal authentication pending, distinguishes
prediction parity from experimental accuracy, and does not claim global kernel
optimality. Source/profile inspection confirms N1000 scope and preserved C10D200S1.
Local documentation link checks PASS; corrected obsolete direct-series-pending
sentence in Base entry history. No new runtime or kernel edits in this turn.

## 2026-10-07 final direct-native pair0 verified; remaining pairs LIVE

Authoritative re-poll SAME wrapper2479402 at elapsed08:41 confirms live.
Journal indices0native/1optimized verified; index2optimized(pair1) started.
First native E2E382.136964420s, optimized110.025357019s, single pair ratio
3.47317x (not final median). Both exact-NNN reports have9entries allbitwise/pass.
Aligned pair0 gatePASS, allatom RMSD0.0060998239421515815A, rawcomponentmax
0.13372135162353516A. Driver additionally verifies actual constructor and
candidate feature counts/consumer before verified. No benchmark restarted and
no remote source/profile modified. Remaining4runs pending; final full Rust
condition/summary-confidence aggregation only after complete driverexit0.
Masterf03015f auditdocs pushed; postcommit typed indexup-to-date/current/no
incompleteReasons and finaldetect_changesclean beforethisappend. Goalactive.

## 2026-10-07 final maintained Base direct-native study complete

Wrapper2479402 TERMINAL SUCCESS, handle absent, driver-status exit0 and exact12
journal events. No run restarted. Aggregator experiments/protenix_base/
summarize_native_study.py STUDY CLI final-v1 completed successfully, including
actual template omission and shared construction checks, authenticated Rust
conditions/archives, all6 own-arm all9 bitwise checks and all3 aligned policies.
Evidence copied provenance/protenix_base_final_native_three_pairs.json.
Native E2E[382.136964420,381.967820082,381.903691933]; optimized
[110.025357019,109.572206027,110.057017543]. Medians381.967820082->110.025357019,
3.4716344526x,71.19512%less. Paired3.47317/3.48599/3.47005. Modelmedians
357.794972017->86.894093936,4.11759828x. All3alignedRMSD.00609982394215158A;
summaryconfidence max4.9591064453125e-5. Exactmetadata/CIFidentitiespass.
Legacyexit1/coordinate.001AFAIL/performanceNULL preserved; separate user.1A+
summary.001 policyPASS. Crossarmfullconfidencearrays notbitwise. No genericgate
relaxation or scientificsettingchange; HIP7FP32N1000C10D200S1seed101,unflushed.

README/guide/experimentguide/audit updated latest direct3.472x with correctscope;
old3.42210x remains historical pre-template evidence. All54 recorded own-arm
arraychecks and3numericpolicies rechecked locally; statisticsrecomputed and
alllocaldoclinksPASS. 70CPUchecks lastpassedwithpromotion, runtime unchanged
since. GraphOFF. CKtile/scheduling/projection andgate/Graph followups lackuseful
E2Egain; latesttemplate1.53%gain qualified. Audit documents observedplateau,
notglobaloptimality; no unsupportedworkload or experimentalaccuracyclaims.
Remaining: commit/push final docs/evidence, refreshindex/finalgate, verifyremote
noactivejobs andfinalrepositorycondition, then completionaudit/status. Master
andGitHubdefaultmasterverified; userdocs/isa remainsuntoucheduntracked.

## 2026-10-07 preferred SSH interface and new CU/batch investigation

User now prefers pc@10.120.16.5 for DiamondHill; .9 is another interface.
Both returned hostnameDiamondHill and boot_id2239217e-fa3c-4c2a-9a99-7009ecbaa73f,
and .5 sees SAME live batch PID2519608. Future SSH/rsync use .5. Updated sync
script and checkpoint guide; latter nowcorrects obsoleteall8 authorization to
latest HIP_VISIBLE_DEVICES6or7. No fullsource sync run; onlysyntaxcheckneeded.

Previous Base optimization goal completed atmasterbe976e7, directN1000 latest
381.967820->110.025357s/3.471634x, model357.794972->86.894094/4.117598x; repeated
userreport nowdelivered. NEW usertask: profileCU saturation/memoryvscompute and
assessbatchthroughput. OnlyHIP7 used. Maintainedruntime/config unchanged.

Experimentroot outputs/experiments/protenix-base-cu-batch-20261007 onDiamondHill.
Tracewrapper2515531 andcounterwrapper2516990 TERMINALSUCCESS/all9bitwiseanchor.
Usedrocprofv3 explicitprofileinjection (sameworkerPID; no supervisorweakening),
SDK1.3.5core10.0 ROCmTorch7.14. Basicprobe passed. Counterskerneloccurrences10-12,
88dispatches/all9metrics; profilerfinalizationwarned77outstandingkernels, documented
as samplecoverage, notfullcountertimeline. Localprovenance/protenix_base_hardware_profile.json
newuntracked copiedfromhardware-summary-v1.json. Trace537608dispatches,sum84.419s,
union84.416s over106.231s=79.46% (instrumentedwholekernelspan, NOTmodelutilization).
HIP7Agent9/PCI0000:93:00.0/SMI5 has104CU. SDK SIMD_UTILIZATION expression
SQ_BUSY_CU_CYCLES/GRBM_COUNT/104 (noadditional4) giveshotkernels99-99.8%active.
CKMFMA75.05%,waveoccup49.18%,HBM482GB/s;LtBMM98.65-98.86%MFMA,24-36%occup,
HBM290-302GB/s; dual70.56%MFMA415GB/s;QKV73.03%/720GB/s;transition68.02%/408.
LayerNorm~1398-1403GB/s,residual1216; diffusionattn61.5%active/19.5%MFMA/7.7%occup.
EmpiricalHIP7copy1249.57GB/s, FP32square8192hipblas36.879TFLOPS/hipblaslt35.485.
Metrics/roofreferencescalibration.json inroot, noGPUoverlap.

CURRENT LIVE batch wrapper2519608; batch-study underroot, pid/run.sh/plan.json/
pins.json/launch.log/driver-status. Rustexecutable root/batch-throughput owns
4groups serial/concurrent/concurrent/serial, twoindependentfreshprocesses each,
allHIP7,N1000C10D200S1FP32seed101. Pythonmembercallsunchangedpredict; Rustcohort
monitorallowsonlylaunchedPIDs onSMI5/PCI93, checksall9bits andsummaryconfidence
exactaftereachgroup. Thisis controlledmultiprocessconcurrency, NOTtensorbatching
andNOTmaintainedRustinferE2E (excludespreflight). No acceptance receipts forged.
CompiledRustlocally/remotelywithserde_jsonrlib; sourcesnewuntracked:
experiments/protenix_base/batch_throughput.rs,batch_member.py. DoNOTeditremotesources
orstartGPU/build whilelive. Group0serialcompleted211.595585s for2requests, member
105.573/105.771s,bothgatesPASS. Group1concurrentbothchildren2522984/2522985live
atlastpoll (groupelapsed1:07;wrapper4:26). Needall4groups +8exact/summary checks,
thenindependentCIFbinding/alignmentchecks, aggregate throughput andlatencies.
Prepare script /tmp/fireprot-prepare-batch-study.py onbothhosts. Hardwareaggregator
/tmp/fireprot-summarize-hardware.py onbothhosts. Calibration/tmp/fireprot-roof-calibrate.py.
Docs/protenix-base-hardware.md draftedwithhardwarefindings andpendingbatchresult.
Potentialresult: coreBMMcompute-bound,CK/projectionnotHBMsaturated; memorykernels
streamingbound, diffusionattentionunderfilled. Awaitbatchbeforeclaimingbenefit.

## 2026-10-07 Base hardware/batch direction closed; move to OpenFold3

User explicitly requested stop digging CU/batch and switch to OpenFold optimization.
Interpret OpenFold as OpenFold3/OpenBind-0, the existing optimized adapter (not
future OpenFold2). No further Base batch trials or promotion. PreferredSSH .5;
HIP6/7only. Batch wrapper2519608 TERMINALSUCCESS exit0; aggregator
/tmp/fireprot-summarize-batch.py completed summary-v1 plus8independent Rust
CIFbinding/alignmentchecks. All8all9arrays exact, summaryconfidenceexact, CIF
atomidentities/bindingspass, allmaintainedcontexts/construction/templateactual
countsverified. Localprovenance/protenix_base_batch_throughput.json copied.
Two-request wallmedians211.678971715(serial)->193.385373845(concurrent),
throughput1.0945965949x;paired1.08825496/1.10100749. Perrequestlatency105.7223194
->192.5420612;model86.79722652->172.85670367s. Independentfreshprocessscope
excludesmaintainedRustpreflight andisNOTtensorbatching. GPUcomputethroughput
roughlyflat; smallgainconsistentwithhost/loadingoverlap. Hardwaredocsupdated
complete/closeddecision; maintainedBaseE2E3.471634xunchanged. Needcommit/push
newbatchsources/hardwaredocs/evidence withGitNexusgates; noGPUjobcurrentlyactive.
OpenFold3 next: re-anchor latesttriangle-memory combo N1536E2E416.226/model321.699,
notolderpublishedprofile. Lastfullprofilepredatestriangle-memory, so refresh
hotspotprofile on actualfastestfrozenworker before selectingnewkernel work.

## 2026-10-07 OpenFold3 latest-combination stage profile LIVE

BaseCU/batchclosure COMMITTED/PUSHEDmasterdefefc2; postcommitGitNexus typedindex
up-to-date/current/noincompleteReasons andfinaldetect_changesclean beforethisappend.
No furtherBaseexperiments. NewOpenFold3request authorized byuser; onlyHIP7.

CURRENT LIVE wrapper2527534, root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-current-hotspots-20261007.
Frozen copy of /tmp/fireprot-triangle-e2e-v1/{python,fireprot,kernels,licenses,...},
actualfastesttriangle-memorycombo. Onlyfrozenworkerfinalnullcontext changed to
model_probe(model,'openfold3',output), probe copiedfromqualifiedstage-profile-v3.
Scopes retaincondition/atomcache, D16, workspace, trianglecopies/keypacking.
OldabsoluteHIPcopylibraries andcontrols retainedunchanged, frozenoldRustCLI/bridge.
Environmentrecapturedwithsameinventoryspec: content EXACTLYequaloldinventory,
so nohostlibrarydrift. SourceOpenFold3pin7de748b7...; weightsOpenBind0bd43301c...;
N1536input5fa81bf... strictFP32R3D200S1seed101HIP7. Allpreparation/frozen/source/
reference/CLI/spec pinsverified, nootherGPUjobsatlaunch. driverRustmeasure1800,
timeout1900; all11bitwiseandfullCIFagainsttriangle-e2e-n1536-candidate-0 required.
Outputoutputs/openfold3-current-hotspots-n1536. driver-statusonlyterminal,
operator-profile.json diagnostics; timesinclusiveoverlap,notnewE2Emeasurement.
Prepare script /tmp/fireprot-prepare-openfold3-current-profile.py onbothhosts.
DoNOTeditfrozenfiles oroverlapGPU/build; pollSAME2527534. Nextaggregatefreshstages,
inspectactualhotspots, then choose optimization withstrict11array/CIFparity.
Read-onlyupstreamtemplate_module.py locatedatOpenFold3/core/model/latent; cannot
omittemplate networkjustbecauseuse_templatesFalse: learnedpairupdateusesz.

## 2026-10-07 OpenFold3 hotspot refresh passed; duplicate-template trial live

Previous wrapper2527534 terminal exit0. All11 arrays and fullCIF exact; frozen
preparation, CLI and reference pins rehashed. Local evidence now
provenance/openfold3_current_hotspots_20261007.json. Inclusive diagnostics:
trunk262.373s, pairformer190.612, triangle attention105.698, multiplication65.081,
template49.415, MSA18.466, diffusion46.465, confidence6.891. These overlap and
include tuning; published uninstrumented E2E416.226s remains unchanged.
Actual template-stack shape[1,4,1536,1536,64]. Fresh candidate experimental file
experiments/triangle_memory/template_reuse.py guards identical input/mask bits,
FP32/inference/strict/shape/inplace, computes one native block slot and copies to
other3; stack final LayerNorm and four-slot sum stay original. Distinct or
unsupported calls fallback. Source-pinned upstream template module, no upstream
or maintained runtime change. No claim that no-template allows skipping network.

CURRENT LIVE wrapper2535783 (shell parent2535782), remote root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-template-reuse-20261007.
Outputoutputs/openfold3-template-reuse-n1536. Same frozen latest combination plus
newscope and stageevents; HIP7only FP32N1536R3D200S1seed101. Onlyjobatlaunch.
Prepare script /tmp/fireprot-prepare-template-reuse.py onbothhosts. run.sh terminal
writes driver-status; require all11 bits/fullCIFagainsttriangle-e2e candidate0.
Do NOT overlap GPU/build or edit frozen source; poll SAME2535783. Need inspect
actual eligible/distinct/unsupported counts and all output gates before any
uninstrumented paired E2E. Newscope and report/docs are local uncommitted.

## 2026-10-07 template reuse model and guards passed; E2E pairs live

Wrapper2535783 TERMINAL exit0. Rehashed frozen source/CLI/reference and upstream
module pins; all11 arrays bitwise and fullCIF match accepted triangle-memory
anchor. Nine eligible calls,27reusedslots, zero distinct/unsupported. Template
inclusive49.414924->14.191791s; other stages vary, NOT E2E claim. Evidence copied
provenance/openfold3_template_reuse_model_20261007.json. Separate GPU guard screen
passed identical slot/alias, signed-zero difference fallback, distinctmask
fallback, training/noninplace fallback and method restoration. Screen result
provenance/openfold3_template_reuse_guard_screen.json. No maintained promotion.

CURRENT LIVE wrapper2543911; remote root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-template-pairs-20261007.
Rust benchmark-pairs AB/BA/AB, six fresh processes strictFP32N1536R3D200S1seed101,
HIP7only; reference includes triangle-memory. Shared frozen worker, no profile;
only FIREPROT_TEMPLATE_REUSE0/1 and profile digest differ. Frozen pins checked and
HIP7/SMI5idle atlaunch. Prepare /tmp/fireprot-prepare-template-pairs.py bothhosts.
run.sh writesdriver-status onterminal, then exactall11/fullCIF checks6against
triangle-e2e-n1536-candidate-0. paired/order.jsonl tracks progress; benchmark.json
onlyatcompletion. Do NOT restart onobservationtimeout, editfrozenfiles or overlap
GPU/build. Need collect actual counters(candidate9/27), confirm no diagnostic
profile, authenticateRust condition comparison and report medians onlyafterend.

Nextkernel hypothesis: PyTorch actualbuild source /home/pc/pytorch HEAD
78258b9de6f4e44e7b8df165d4651bf2e14acf62 matches installedTorch; source
aten/src/ATen/native/cuda/layer_norm_kernel.cu unmodified. FP32C128aligned path
uses256threads/row, four-value perlane Welford and inter-wave empty reductions.
A multirowHIP kernel preservingthatorder may improve triangle normalization;
not implemented/measured yet. Previous Protenix mean/squaresum kernel would
changeWelford and cannot be blindlyused forOF3 strictbitwisegate.

## 2026-10-07 active OpenFold3 goal; norm prototype and transient SSH failure

Previous goal turn made progress: exact template reuse model/guard qualification
committed/pushed c76170b. Postcommit index current, no incompleteReasons and final
GitNexus detect_changes clean except untouched user docs/isa. Active goal is now
'consistently optimize the openfold3 optimization'; no completion claimed.

Re-polled SAME E2E wrapper2543911 LIVE at elapsed12:34. Baseline0 complete and
verified at order offset420.875s; candidate0 started. No other benchmark restarted.
Subsequent .5 SSH timeout twice and .9 fallback timeout once. These are observation
failures, NOT terminal evidence. Do not relaunch. Need retry .5 and inspect SAME
PID plus order/driver-status. Last actual live observation is authoritative;
no complete first-pair ratio or three-pair median yet.

New unqualified local prototype experiments/openfold3_norm/{norm.cpp,build.py,
screen.py,README.md}: handwritten WelfordC128, four wave64/eight subwave32 rows
per256threadblock, retains native empty-state merges and affine operation order.
Local Rust-owned HIP build succeeded with HIP_VISIBLE_DEVICES empty, receipt
/tmp/fireprot-of3-norm128-local-build-v1/build.json, librarySHA
8d918fe24e6567dbbd8e25daab12ec2c3426850d2591581e0e035444b9732a3f.
No localGPU used; localTorch2.9.1HIP7.2 vs compilerHIP7.14. No remote deployment,
GPU test or performance claim; source and screen syntax pass. Do not run remote
build or GPU screen until E2E wrapper truly terminal. Screen covers output+mean+
rstd exactness, random/constant/lowvariance/large/signedzero/tail/affine variants,
then optional realcapturedattentionz/weights at triatt-profile-n1536-v1. Onlyexact
variants timed, includingstrictoutputallocation. Fullmodelparity/E2Estillneeded.

## 2026-10-07 host reboot recovered; fresh E2E handle15499

User requested wait120s; waited two60s sleeps, .5 SSH recovered. Confirmed actual
host reboot: bootIDafbe7b4e-c398-45b8-86ad-45cded194221, uptime2min; old2543911
absent and GPUidle. Old firstpair has invocation420.113059237/386.645935697s but
candidate predictions.json/template-reuse.json/sample-0.npz are EMPTY afterboot.
Do NOT use that pair as speedup evidence. accepted.json hash survives, but its
referenced output bytes do not; old run preserved with interruption.json.

Reboot cleaned /tmp frozen CLI/bridge/libraries/controls. Restored496 files from
5 SHA-verified archives into runtime/recovered-openfold3-20261007. Temporary
compatibility aliases recreated only to authenticate prior pins; ALL prior
frozen pins/reference/CLI now match. New environment capture byte-identical old.
Recovery evidence provenance/openfold3_reboot_recovery_20261007.json. Script
persisted runtime/recovered-openfold3-20261007/recovery.py; receipt.json records
all recovered hashes. Oldpreparation at /tmp helper local still available.

CURRENT LIVE wrapper15499, root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-template-pairs-recovery1-20261007.
Fresh Rust AB/BA/AB with runIDs template-pairs-r1-n1536-{arm}-{i}, HIP7 only.
Last live1:30 firstbaseline0started; no runtimeerror. New profile uses persistent
CLI/bridge/control/bundle/probe paths, worker hardcodedcopylibraries relocated to
persistent recoveredpath, same bytes/arithmetic. Newcontrols only relocates paths,
allmanifest/numericalscreenSHAchecked beforelaunch. environment unchanged. New
prepare.py persisted alongsidefrozen worker. driver-statusonlyterminal. Do NOT
launch GPU or remoteheavybuild/editfrozen while active; poll SAME15499. Must
collectnew3pairs and gates, not salvage damagedoldfirstpair.

Normprototype still LOCAL ONLY/GPUUNVERIFIED. Buildsuccess syntaxchecks described
above; local source and README pendingcommit with explicitunqualifiedstatus.
Currentgoalturnprogress: newHIPprototype compiled, actualhostreboot identified,
archive recoveryverified, measuredjob restarted onlyafterauthoritativeterminal.
Goalactive; no plateau/completion evidence yet.

## 2026-10-07 norm trained-input guard strengthened while pairs remain live

Previous turn progress: recovery and HIPprototype committed/pushed4e290e8; goal
stillactive. This turn re-polled SAME15499 LIVE. No competing GPU/build. Norm
screen previously could silently skip an empty capture directory; now requires
exacttwo filenames,size,SHA256 from captures.json derived from historical
provenance/openfold3_triangle_triatt_profile.json beforeGPU execution. Report
records capturepins/count; omitted argument explicitly synthetic-only. CPUtest
passes validinput plusmissing/same-size-corrupt/truncated rejection. Numerical
HIPkernelstillunqualified; no GPUtest/dispatchchange. Need pending E2E completes
before norm GPU screen. Preserve strict all11/CIF gate and previousmeasuredE2E.

## 2026-10-07 template E2E complete; HIP normalization model gate live

SAME wrapper15499 terminal exit0; collector at recovered-pairs-root/collect.py
completed with all frozen pins, Rust benchmark/comparison, 6 exactall11/fullCIF
checks against triangle-memory anchor. Evidence local
provenance/openfold3_template_reuse_e2e_n1536.json. E2Emedians419.239143733->
387.984735169s,1.080555769x,7.455%less; model323.661357742->291.027215039s,
1.112134333x. Paired1.08056/1.05435/1.10963. All candidates9eligible/27reusedslots;
copy430/7740 andkey5064 retained; nativefallback counts72baseline/18candidate
because repeated C64 template work is removed. No scientificsettingchange.
Archive outputs/experiments/openfold3-template-pairs-recovery1-20261007.tar.gz
SHA777d45a6dd3abaa3c6f6d1057822f19e5585d737f615984cb396943902ba321d.
Includes recovereddependencies plus all6outputs; fsynced. Damagedoldseriesexcluded.
README/performanceguide/experimentguide updated completedincrement, not direct
originalTorch ratio. Three observations do not establish productiontail latency.

Operatornorm trials now completed, HIP7only, afterE2Eended. Root outputs/experiments/
openfold3-norm-20261007 v1:24checks; allstatistics/trainedinputs exact, only bias-only
outputfails4.768e-7. Firstshellwrapper EXITtrap hadquotingerror, screen JSONcompleted
withpassed_widths[]/no timing; no successreceiptclaimed. v2(rootopenfold3-norm-v2-
20261007,wrapper60450) samebiasfailure despiteexpression rewrite. Pointwise
roundingdiagnostic: native==addcmul, candidate==separate multiply/add;37250/131072
differ. Explicitfmaf fixesbranch in v3, wrapper62666 terminal0, all24 checks exact,
passedwidths[64,32], pluslargeoutputchecks. Rust-ownedbuild, memoryfill/FP32strict.
M393216 Torch2.125403ms vsW64 1.007742 vsW32 .464152. M2359296 Torch12.634741 vs
W64 5.955017 vsW32 2.705953 (~4.67x operator). Eightbalancedforward/reverseorder
observations includingoutputallocation. Evidenceprovenance/openfold3_norm_operator_screen.json
recordsall3trials/rejections/pins, noE2Eclaim. QualifiedlibrarySHA
a06828d77438cd417298f19221dca4cff371ed81ba4df041283eb2ab8ebbf03a
runtime/torch_extensions/rust-v1/fireprot_of3_welford128_v1_c7a1740414e6c7729a29ca7798d414d891cd55814ff14b41171f5b8cef5aac3b/...

CURRENT LIVE wrapper65113; root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-norm-model-20261007.
Outputoutputs/openfold3-norm-model-n1536. Frozenlatesttemplatecandidate plusnew
norm_scope. Librarycopiednorm.so pinned. Newexperiments/openfold3_norm/model_scope.py
wrapsonlyupstreamLayerNorm C128, M393216or2359296,eps1e-5,strictFP32inference;
retainsnativecontiguousconversion, fallbackothers. UsesW32. SourceSHA
normalization.py e39b9599c1711a232b7a52767732920e45e069dc37850192090c1f59440b3d79.
Driverrequiresall11bits/fullCIFagainsttemplate-pairs-r1-n1536-candidate-0 andpositive
actualnormcalls. R3D200S1seed101HIP7. No concurrentGPU/build. Lastverifiedlive3:04,
noerror. NeedpollSAME65113, authenticateactualcounts+fullgate before pairedE2E.
Prepare /tmp/fireprot-prepare-norm-model.py local; remote savedv3root/prepare-model.py.
Driverlocal/tmp/fireprot-run-norm-model.py andfrozenroot/driver.py. No maintained
promotion. LocalCPPfix/notes/modelscope/evidence pendingcommit. CPUcaptureguardtest,
Python syntax and all recordedtemplate/lastnorm operatorgates pass.

Timinggap observation forfuturestartupdiagnostic: firsttemplatepair model323.661/
291.027,request360.256/329.352,E2E419.239/387.985. Requestoutsidedelta58-59s;
candidatesupervisor369.843s,thus~40.5swithinsupervisionoutsiderequest. These are
boundary differences, not isolated initialization/import measurements. Profile
beforeoptimizing; do not bypass source/weight/environment verification.

## 2026-10-07 normalization full-model passed; new E2E pairs live

Wrapper65113 TERMINAL0. Rehashed all frozen/source/CLI/reference/operator pins;
all11arrays and fullCIF exact vs template-pairs-r1-n1536-candidate-0. Actual
normalization11202accepted/7846fallback/4730contiguouscopies. Singlemodel260.606504s,
fullinvocation355.960196s; qualificationobservation,NOTpairedspeedup. Persisted
provenance/openfold3_norm_model_n1536.json; outputfilesystemsynccompleted.

CURRENT LIVE wrapper74134; root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-norm-pairs-20261007.
Rustbenchmark-pairs AB/BA/AB, strictFP32N1536R3D200S1seed101HIP7only. Sharedfrozen
worker inclnormalizationcontextconditionalonFIREPROT_NORM1280/1; botharmsretain
allpriortriangle-memory/templatecache. Newprofilehashdeclaredtreatment. Allsix
requests inspected; nootherGPUjobatlaunch. Driver.py subprocessargs (no shell
trap); terminaldriver-statusrequires6all11bits/fullCIFagainsttemplatecandidate0,
candidatenormselectedpositive/baselinefileabsent. NeedpollSAME74134, noGPU/heavy
build orfrozeneditswhilelive. Preparelocal/tmp/fireprot-prepare-norm-pairs.py,
remotepersistednorm-modelroot/prepare-pairs.py. Driverlocal/tmp/fireprot-run-norm-pairs.py.

Formaltemplate3pairdocpublication+qualifiednormsource/evidence stilllocalpending
commit; allrecordednumericalchecks/syntax/docslinks passed. Modelscope nowverified.
No maintainedscopepromotion orfinalnativeTorchratio claimed. Goalactive; next
stepscollectnormpairedE2E, then continuehotspot/initializationinvestigation.

## 2026-10-07 normalization E2E complete; strided model trial live

Norm paired PID74134 TERMINAL0. Three AB/BA/AB pairs re-read all11 bits/fullCIF
against template-pairs-r1-n1536-candidate-0. Complete invocation medians
391.161906097 -> 364.159347770 (1.074150392x); model287.825230250 ->
260.433601489 (1.105177015x). All six qualify. Note earlier commentary387.71/
364.75 used journal offsets, NOT invocation times; corrected official firstpair
386.877629842/364.159347770. Candidate1 363.943045050, candidate2 366.417228946.
provenance/openfold3_norm_e2e_n1536.json + README/performance/norm notes now local
pending git gates/commit. Master pushed b303bb0. User docs/isa remains untracked.
Archive /media/IntelSSD/fireprot/outputs/experiments/openfold3-norm-series-20261007.tar.gz
SHA f4489b27fbfd09a294835d7a30ef898870be47f645076eade2995256d694cc27 size458085545.
Archive includes six outputs, both experiment directories and strided library;
completed before the later model-scope preparation. Raw archive receipt in pairedroot.

New strided operator root outputs/experiments/openfold3-norm-strided-20261007.
Initial queued PID98380 failed before any build/screen because SMI idle sentinel
was string 'No running processes detected', not emptylist. driver-v2 PID112439
TERMINAL0; preserves rejected wrapper receipt. All25 numerical tests output/mean/
rstd exact. Real captures columnchunk copy+HIP0.909ms vsstrided0.508-0.511ms;
transpose12ms vs10.2ms; narrowtransposed2.02ms vs1.73-1.76ms. Dense slower2.86vs2.70
and densechunk0.50vs0.464: retain original HIP on contiguous input. No E2Eclaim.
Library88d43a5129b3e0abb2ecfcc321a9e8bf29e00f409a611ec0f1adca542511e65e under
runtime/torch_extensions/rust-v1/fireprot_of3_welford128_strided_v1_3c8883fea17957663e043d33d23f85c4dfa42d1af95fd72dd03344436b6cc3ff/.
Local prototype /tmp/fireprot-norm-strided-prototype contains norm_strided.cpp,
screen_strided.py,norm_helpers.py,captures.json,build.py,model_scope.py,drivers.
Sources not yet integrated into repository; remote screen/build frozen and archived.

CURRENT LIVE PID114134 timeout2000; root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-norm-strided-model-20261007.
Output outputs/openfold3-norm-strided-model-n1536. StrictN1536R3D200S1seed101HIP7.
Copies qualified normpairs candidateworker; baseline contiguous norm preserved;
strided only three measured noncontiguous shape/stride signatures. Requires all11
bits/fullCIF vs norm-pairs-n1536-candidate-0 plus counts11202accepted/7846fallback/
4730strided_calls/0contiguous_copies. No profiler. Poll SAME114134; no GPU/build
competition. Preparelocal /tmp/fireprot-prepare-strided-model.py; driverlocal
/tmp/fireprot-run-strided-model.py. Remote copies in strided operatorroot. Source
pins checked, request inspected, SMI5idle before launch. No maintained promotion.
Next: authenticate fullmodel trial, ifpasses uninstrumented paired E2E; archive
and integrate qualified strided source/receipts. Commit pending norm docs/evidence
with GitNexus current-index and staged detect_changes gates. Goal active.

## 2026-10-07 strided normalization model passed; paired trial running

Committed/pushed norm E2E docs/evidence as51e092c616139793f9d9d7b9d0406c163617e3ac.
Postcommit GitNexus index-current, runner-current, incompleteReasons[], final
all-change gate clean (only user docs/isa untracked before this journal entry).
Strided fullmodelPID114134 TERMINAL0: all11bits/fullCIFexact vs norm-pairs
candidate0; model257.120749320s singleobservation, not pairedspeedup. Counts
11202accepted/7846fallback/4730strided_calls/0contiguous_copies. Before pairing,
preparation rehashed allsource/CLI/reference/operatorpins, reran compare-npz
all11exact and compared current completeCIF. No reduction/precision changes.

CURRENT LIVE PID121367 timeout10900. Root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-norm-strided-pairs-20261007.
ThreeAB/BA/AB pairs with shared frozenworker; onlyFIREPROT_NORM_STRIDED0/1and
profiledigest differ. BotharmsHIPWelford+allpriorcache/template/trianglechanges.
Baselinecopies4730/strided0; candidatescopies0/strided4730 required, all11bits/
fullCIFagainststridedfullmodelanchor. StrictN1536R3D200S1seed101HIP7only,noprofiler.
NeedpollSAME121367; nooverlappingGPU/build. Localprepare
/tmp/fireprot-prepare-strided-pairs.py;driver/tmp/fireprot-run-strided-pairs.py;
remotecopiesinprecedingstrided-modelroot. SMI5idle checkedbeforelaunch; allsix
requests inspected. Next authenticate/collectpairedresults; integrate qualified
stridedprototype sources+operator/fullmodel evidence fromarchived/remote artifacts
into repo, thenfreshhotspotselection. No new E2Eclaimyet; maintainedpromotionpending.

## 2026-10-07 strided sources published; post-pair diagnostic queued

Master pushed ebc3e44d63bbd40b08bbeb714284a396de4b8395. Qualified strided kernel,
build/screen/model scope and STRIDED.md are under experiments/openfold3_norm;
operator and fullmodel provenance published. HIP kernel, build recipe, shared
helpers and modelscope hashes match executed bytes. Screen publication changes
helper import wiring only; CPU imports pass. No maintained-profile promotion.
Local GitNexus incremental DB reached16GiB duringrefresh. Explicit --force full
rebuild succeeded, compacted DBto274MiB. Postcommit index-current at ebc3e44,
runnercurrent,incompleteReasons[], finaldetectallclean. Future ordinaryanalyze
canbeused; rerun--forceifincrementalbloatreturns. No GitNexussourcechanges.

Stridedpairs PID121367 stillLIVE last14:53, secondcandidate running. Firstpair
actualinvocation362.900108031 baseline vs363.842374558 candidate; model260.743900671
vs260.547456001. Firstpair passes unchangedgates but NO observedE2Egain yet.
Do not infer gain fromremovedcopycount. Collectorprepared
/tmp/fireprot-collect-strided-pairs.py andfrozenroot/collect.py; all6currentarrays/
fullCIF/pins/copycounts rechecked onterminalsuccess, no partialresultpublication.

NEW LIVE postpairqueue PID140539 timeout10900; root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-post-norm-profile-queue-20261007.
Currentlywaiting_for_pairs predecessor121367. Sourcehashespinned,bootchecked;
onlyafterpairsTERMINAL0andPIDgone,SMI5idle: collectall6,preparefreshstageprofile,
thenrunonefullmodeldiagnostic onHIP7. Keepsallscientificsettings; requiresall11bits/
fullCIFandactualnormcounts. UsesarmwithfasterobservedE2EmedianforDIAGNOSTIC,not
maintainedpromotion. OldstageprobeSHA
67fe7a78d841bc14d47568ef019c0247f2741a0625bcd2c444e740eb752b12c1.
Nextprofileroot /media/IntelSSD/fireprot/outputs/experiments/openfold3-post-norm-hotspots-20261007;
outputoutputs/openfold3-post-norm-hotspots-n1536. InclusiveGPUmoduleevents, no
Kineto, noinstrumentedE2Eclaim. Queuechecksfullprofileoutputnumericparity.
Local /tmp/fireprot-post-norm-queue.py,
/tmp/fireprot-prepare-post-norm-profile.py,/tmp/fireprot-run-post-norm-profile.py.
Remoteprofilepreparation/runner in stridedpairsroot, queue driver inqueueroot.
PollSAME121367and140539; do notmanuallycollectorlaunchprofilewhilequeueownssteps.
Ifqueuedstepfails, inspectterminalreason andrepair withoutrestartingbenchmarks.

## 2026-10-07 queued profiler preparation corrected before execution

IMPORTANT current queue PID153902, root
outputs/experiments/openfold3-post-norm-profile-queue-v2-20261007.
Old queue140539 was deliberately SIGTERM'd WHILE waiting_for_pairs, after proving
its stage and matching /proc command; it is gone, no GPU/profile step ran. Receipt
oldroot/cancelled-for-preparation-fix.json. Actual pairedPID121367 uninterrupted,
last27:54 thirdbaseline running (4/6 completed). Secondpair officialE2E baseline
367.711244813 vs candidate368.235660590; firstpair362.900108031 vs363.842374558.
Both completed pairs exact but no E2Egain. Wait remainingpair before conclusion.

Reason forqueuefix: frozenworker already imports model_probe insideforward, unused.
Oldprepare assertedimportabsent andwouldfailafterpairs. v2preparer permits existing
import; actualworkercontextinsertion syntax checked without GPU. Queue153902 pins
newpreparer andsamecollector/profiledriver,boot/terminal/idlechecks unchanged.
Local /tmp/fireprot-prepare-post-norm-profile-v2.py, /tmp/fireprot-post-norm-queue-v2.py.
Remotev2preparer in stridedpairsroot; newqueue/driver.py. PollSAME121367/153902;
do not manually collect/profile whilequeueownssteps.

New profilecollector /tmp/fireprot-collect-post-norm-profile.py; remotecopyat
OLDqueueroot/collect-profile.py (not automatically executed). Runafterprofilestatus0.
Checks allsourcepins/currentarrays/fullCIF, totalsrun_trunkcalls1 (contains4passes),
pairformer_stackcalls4, andstageevents/normcounts; outputsprofileroot/collected.json.
Do not sum inclusive parentandchildGPUtimes. No instrumentedE2Eclaim.

Future originalTorch comparison caveat: forward_with_backend(native) bypasses
attention/workspace, BUT pair_cache/atom_cache/triangle/template/norm scopes wrap
outside it in forward. Thus backend=native alone DOES NOT disable optimizations.
Explicitly zero relevant profile flags for a genuine original path; verify no
operator/cache selection artifacts. No neworiginalTorchspeedup claimed yet.

## 2026-10-07 strided E2E complete: reject; post-normalization profile live

PairedPID121367 TERMINAL0. Collector completed and freshly re-read all six
arrays/fullCIF/pins/counts. All11 bits and completeCIFexact. Actual baseline
E2Emedian365.068917259 vs strided368.235660590 (0.991400226x); modelmedians
261.806940818 vs264.060917817 (0.991464178x). AllthreepairedE2Es slower
0.997410234/0.998575869/0.989765533. Candidatecopies0/strided4730 verified but no
E2Egain. Decision: do notpromote, retain preceding HIPnorm+contiguouscopies.
No overallplateauclaim. Rawpairedroot/collected.json; localprovenance
openfold3_norm_strided_e2e_n1536.json addsdecision. STRIDED.md/normREADME/perfdoc
updated locallypendingcommit; evidencearchivefortheseoutputs stillpending.

CURRENT LIVE queue153902; stageprofile childtimeout161348. Root
outputs/experiments/openfold3-post-norm-hotspots-20261007; selected_armbaseline
fromobservedpairedmedians. Originalrun settings strictN1536R3D200S1seed101HIP7.
Queuev2owns invocation; do notstartanotherGPUjob. Requiresall11bits/fullCIFand
norm11202accepted/7846fallback/4730copies/0strided. PollSAME153902 and child161348.
Onsuccess run /tmp/fireprot-collect-post-norm-profile.py (remoteOLDqueueroot
collect-profile.py), then archive via /tmp/fireprot-archive-strided-completed.py
(remotenewqueuev2root/archive-completed.py). Archiveonlyafterqueueexit toavoidIO
competition; itincludesmodel/pairs/profileandall8outputsplusdependencyarchivepin.
Thenpublishupdatedhotspots andchoosebottleneck-driven nextkernelwork.

## 2026-10-07 latest hotspots complete; QK scheduling sweep live

Latestcommitted/pushedHEAD93e1e486da7526ab0c98594b3951e02f415febbf; postcommit
GitNexusindexcurrent/runnercurrent/incompleteReasons[], finaldetectallclean.
Queue153902/profile161348 TERMINAL0. Collector passesfreshpins/all11/fullCIF.
Postnormbaseline diagnostic: run_trunk203.470125s, pairformer169.991555s,
trunktriatt101.772243s (start49.786770/end51.985473), trimul50.238590s,
pairtransition13.502159s, templates12.826254s, MSA16.407269s, diffusion44.345363s,
confidence6.322280s. Inclusiveoverlap/noprofileE2Eclaim. Newlocalprovenance
openfold3_post_norm_hotspots_20261007.json and perfdocpendingcommit.
Completedarchive /media/IntelSSD/fireprot/outputs/experiments/openfold3-strided-and-postnorm-20261007.tar.gz
SHA2e52f7537943f63f189525c56442ac6c4bb290418a001b22a7648f6071ce1a42 size550620385.
Containsstridedmodel/all6pairs/profile/queues/outputs anddependencyarchivepin;
localstridedE2Eprovenance nowaddsarchive (pendingcommit).

Triattcomponentreplay root outputs/experiments/openfold3-triatt-components-20261007
PID168785 failedbeforecandidate: native chunk1024 reference OOM allocating36GiB
softmax probabilityon63.98GiBcard (scoresalready36GiB). Preservedfailure. v2root
openfold3-triatt-components-v2-20261007 PID169933 TERMINAL0. NativeREFERENCEonly
useschunk256; candidatekeepscapturedchunk1024; bothtrainedstart/endoutputsbitexact.
Currentworkspace/norm/keypacking actuallyselected; warm2thenoneKineto+GPUevents.
End: normalization11.98ms (includestransposecopy), QKV10.56ms, PV75.55ms,
workspace225.52ms; QKsoftmaxGPUkernels144.14ms. Start: norm2.827,QKV10.54,PV75.09,
workspace358.10ms. Instrumentedreplaysarenotstablelatencyclaims; ROCTracer emitted
duplicateflowstartwarnings. Rawtracesretained; unprofiledtimingrequired. Local
provenance/openfold3_triatt_components_postnorm.json createdpendingcommit. Local
/tmp/fireprot-triatt-components-v2.py,/tmp/fireprot-run-triatt-components-v2.py.
Nexttarget QK+softmax itself, not peripheralprojection/copies.

CURRENT LIVE PID171387 timeout3600; root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-qk-reuse-sweep-20261007.
Threeblockinterleavevariants1/4/16versusaccepted8; basedonEXACTcombined-v12
sourcefromruntime/recovered-openfold3-20261007/fireprot-qk-softmax-combined-v12.
Floatingbodybyteidentical; onlyintegerblockmappingchanges. CPUenumerationbijective
forB1,7,8,9,32,64,128,256,512,1024,1536. Local/tmp/fireprot-qk-reuse-sweep holds
baseline.cpp,variantdirs,mapping.json,build.py,driver.py. Driverpinsallsource,
checksbaselineSHA,buildsviaRust, publish-kernelwithpersistentrecoveredCLI,
then21packedchecksperB32/B128eachvariant (nonfinite/subnormal/signedzeroincluded).
GPUHIP7/PCI93/SMI5only;buildHIPempty;serial,nootherGPU/build. ThisisNUMERICALONLY,
notperfclaim. PollSAME171387,checkdriver-status andvariantprobes. Nextinspect
code-objectresources, benchmarkcompleteattention/realcapturesagainst8including
packing/PV/strictallocation and balancedorder. Onlypromotewithfullmodel/E2E.

## 2026-10-07 QK reuse sweep numerical and operator timing complete

Masterpushed282634f4896e34ecf3ed894029e0f362a7940ebb (hotspot/component evidence).
PostcommitGitNexuscurrent/runnercurrent/incompleteReasons[], finalallgateclean.
NumericalPID171387 TERMINAL0, all3variants groups1/4/16 each21exactpackedcases at
B32andB128 =>126passes, samefloatbodyonlyintegerschedulingchanged. Published
bundles/controlpins recordedroot/driver-status.json. Numericalscreen no perfclaim.
BalancedrealTriangleAttention benchmarkPID175422 TERMINAL0. Rootsameqk-reuse-sweep;
benchmark/report.json alloutputsbitexactvsnativechunk256; candidatecapturedflags
chunk1024unchanged. Eacharmwarm2, eightbalancedGPUeventsamples/capture, noKineto.
Fourworkspacearms retainedsimultaneously; inputclone/exactcheckoutsideinterval,
actualnormalization/QKV/packing/scoreallocation/PV/output inside fullmoduleinterval.
End mediansms: reuse8=267.037887573, reuse1=282.674591064,
reuse4=266.189346313, reuse16=270.396957397.
Start: reuse8=256.957672119,reuse1=273.488677979,reuse4=256.204315186,
reuse16=260.146484375. Only~0.3%operatorgainfor4; no E2E orpromotionclaim.
Do notspendfull3modelpairssolelyonthissmallvariant. Newdiagnosticbenchmark source
/tmp/fireprot-qk-reuse-benchmark.py and/tmp/fireprot-run-qk-reuse-benchmark.py;
remote benchmark.py/benchmark-driver.py inroot. Capturecorrectness includesall80
calls(4arms*(2warm+8timed)*2captures). Rootnotarchivedyet, noGPUjobnow.
Next archive/collectsweep (includingresourceinspection if useful), thenexplore
1024thread16querytile'sbarrier/residencycost. CandidateideaUNIMPLEMENTED:
512threads/8queries,12bands*128keys,discardduplicatedMFMAqueryrows8..15;
keep8Kfragments/order, mapvalues[band*2+j], j<2, N1536softmaxorderunchanged.
LDS8*128=4KiB, gridx192, mappingheadquerycount192 ->reuse8group6144; queryoperand
coordinate&7 duplicatesupper8rows, onlylower8store. DoublesMFMAworkbutallows
independentblocks/barriers; hypothesisrequirescompile/resources/exactscreen/timing.
Do notclaimgain. Existingfullsourcebaseline in/tmp/fireprot-qk-reuse-sweep/baseline.cpp
andremote recoveredcombined-v12. Allrawmathpatchesdocumentedinexperiments/mfma_qk.

## 2026-10-07 half-query tile rejected after exact checks and timing

CURRENT noGPUjob. Halfv1 PID178022, halfv2 PID179430, benchmarkPID181908 allTERMINAL0.
Roots outputs/experiments/openfold3-qk-half-tile-20261007,
openfold3-qk-half-tile-v2-20261007,openfold3-qk-half-tile-benchmark-20261007.
Local /tmp/fireprot-qk-half-tile and-v2 containbaseline,mapping,build,driver,
resources_probe.py andcandidatefolder. Do NOT namehelperinspect.py (stdlibshadow);
renamedbeforelaunch,noGPUfailurefromthat. One shellpreparationcommand quoting
failedbeforeexecution; repairedwithlocalPythonfiles, no experiment rerun.

Both512thread8queryversionspass21packedrecords B32andB128 (84total across2variants).
v1 retains4KiBLDSandconditionalcoordinate<8stores: softmax130VGPR31SGPR;
v2usesfull16x1288KiBLDSandunbranchedstores,upper8rowsunused:60VGPR34SGPR.
No spills/privatebytes both. Both96MFMAand23barriers vsbaseline48/11.
Fullrealtriangleattention benchmark3arms,2warm+12balancedtimedsampleseach/capture,
all84fullmodulecallsbitexact vsnativechunk256; candidatecapturedchunk1024unchanged.
Endmsbaseline267.007584,v1=398.435059,v2=298.202957.
Startmsbaseline256.985535,v1=388.231277,v2=288.187302. REJECTboth; no E2Etrialneeded.
Benchmarkreportroot/benchmark/report.json; qualification.json aggregatestrue
completedprobes, notafaked GPUreceipt. Local /tmp/fireprot-qk-half-benchmark.py,
/tmp/fireprot-run-qk-half-benchmark.py,/tmp/fireprot-prepare-qk-half-benchmark.py.

ResourceboundruntimeoccupancyqueriedviaHIPmoduleAPIonHIP7,nokernellaunch:
baseline1024threads max2blocks/CU=32waves;v1 512threadsmax1block=8waves;
v2 512threadsmax4blocks=32waves. Thusno wave-residencyupperboundgainforv2;
DO NOTcallthese measuredCUutilization/counters. Evidencebenchmarkroot/occupancy.json;
script/tmp/fireprot-qk-occupancy.py andremoteroot/occupancy.py. Usesarchivedactual
HSACOsandrecordstheirSHA. Allperformanceandoccupancyrootsstillneedcollection/archive.
Reusegroup1/4/16sweepalsoawaitsarchive/publication (priornotehasmedians).

NextideaUNIMPLEMENTED:32querytile/1024threads,2independentMFMAaccumulators perwave
reuseeachKfragmentacrosstwoquerygroups,32x256LDS32KiB,16wavesconsume2rows/wave;
values[2][32],same24scoreelements/row andsameWelford? NO: sameSOFTMAXordering;
noLayerNormchange. Gridquerytiles48,mappingreuse8group1536;querystarttile*32;
queryvectorscoordandcoord+16;inputN1536divisible32. MoreVGPRmayreduceoccupancy,
so compile/resourcesbeforedeciding. Unlikehalf-tile, no redundantMFMA:2xoutputs
for2xcompute andKloadsreused. Needexactpackedtests/trainedfulloperator timing,
thenfullmodel/E2Eonlyifmaterialgain. Baseline acceptedcombined-v12source copiedin
/tmp/fireprot-qk-reuse-sweep/baseline.cpp. Goalnotcomplete/globalplateaunotproven.

## 2026-10-07 wide QK rejected; PV backend screen rejected

Wide-query QK v1/v2 and v2 full-attention timing are terminal. Roots:
outputs/experiments/openfold3-qk-wide-tile-20261007 and
openfold3-qk-wide-tile-v2-20261007. Both pass 42 packed numerical records.
V1 (32 queries, 1024 threads, 32 KiB LDS) spills 56 VGPRs / 228 private
bytes per thread; no timing trial. Moving mask/bias additions into each completed
band retains output arithmetic order and removes spills in v2 (actual remote
95 VGPRs / 34 SGPRs). Full trained attention, two warmups and eight balanced
samples per capture: end baseline267.757950 vs candidate293.721268 ms;
start257.019058 vs283.115570 ms. All40 full-module outputs exact. Reject;
no full-model/E2E trial. Local compile-only scheduling variants under
/tmp/fireprot-qk-wide-scheduling are not GPU-qualified. Raw archives still pending.

Network wait120 completed; endpoint10.120.16.5 recovered without restarting jobs.
PV-only Torch hipBLASLt preference screen completed terminal0 at
outputs/experiments/openfold3-pv-lt-20261007. Scripts/pinned deployment and raw
report remain there; local copy /tmp/fireprot-of3-pv-lt. Trained N1536 full
attention with accepted norm/key-pack/QK/workspace scopes, HIP7, strictFP32,
seed101, two warmups/eight balanced samples each arm/capture, no profiler.
End nativePV267.813919 vs Lt269.828644 ms; start256.983368 vs258.790466 ms.
Native arm bitwise exact, Lt arm fails bits: maxabs4.1961669921875e-05 and
0.00010633468627929688 respectively. Reject/no promotion/no E2E gain.
Recorded tensor examples are native-reference chunk256 only (first4 calls);
not evidence of the Lt kernel name. Backend preference is scoped to the PV
expression, but internal algorithm identity was not profiled. Do not claim it.

Next experiment CURRENT LIVE child190424 (wrapper190413), root
outputs/experiments/openfold3-trimul-chunks-20261007. Tests native chunk256
against copy-patched/HIP-normalized full triangle multiplication with chunk
256/128/384/512/768 on two trained captures. StrictFP32 HIP7, two warmups/four
balanced timed samples. Records numerical failures instead of weakening gates;
terminal exit0 means screen completed, not all candidates qualified. Check
report per-arm bitwise results and driver correctness_passed separately.
Local scripts /tmp/fireprot-of3-trimul-chunks; no maintained source changes.

## 2026-10-07 triangle chunk screen exposed a composition discrepancy

All latest GPU jobs TERMINAL, no live job. Chunk screen190424 completed;
copy-only screen191403 at outputs/experiments/openfold3-trimul-copy-screen-20261007
and per-normalization audit192158 at openfold3-trimul-norm-audit-20261007 also
completed. Local script/report copies /tmp/fireprot-of3-trimul-{chunks,copy-screen,norm-audit}.
Captured flags omit _inplace_chunk_size; upstream default verified256.

Combined copy+HIPnorm chunk256 fails full-module bits vs fresh native reference,
maxabs0.0001220703125 on BOTH trained captures. Chunk128 passes but ~242/244ms,
versus combined256~127.54/135.01ms; chunks384/512/768 slower and nonexact.
No candidate promoted. The copy-only isolation at256 is exact on both;128 also
exact, largerchunksnonexact. Copy-only timing has just one sample, not a perfclaim.

IMPORTANT follow-up normalization audit compares EVERY layer_norm_in/out output
against its original method on the SAME input within the candidate execution.
ALL these local outputs are bitwise exact, yet final triangle output differs
from the earlier native module reference. Thus do NOT blame Welford arithmetic
or claim a failing normalization tensor was saved: the save-on-local-failure
branch never fired. Difference arises elsewhere in composition/execution;
allocation/layout-dependent downstream GEMM selection or baseline variability
remain hypotheses only. Audit instrumentation timings invalid for performance.
Previous qualified full-model11array/fullCIF results remain evidence for their
measured workload, not universal tensor guarantees.

NEXT: localize FIRST divergent projection/contraction using reference versus
candidate intermediate hooks, also repeat unchanged native module in same
process to check baseline stability. Record input/output strides and pointer
alignment and compare equal inputs at the first divergent operator. Avoid
prematurely changing norm arithmetic. Keep strict gates and do not claim new
E2E speedup. PV preference and all new QK/triangle roots await durable archive
and publication; .agents journal uncommitted. Master remains282634f4896e34ecf3ed894029e0f362a7940ebb.

## 2026-10-07 first divergence localized; canonical norm model qualification live

First-divergence audit child193154 TERMINAL0 at
outputs/experiments/openfold3-first-divergence-20261007. Native repeat and
copy-only paths both exact on both captures. First copy+norm mismatch is
linear_a_g (index1), with BITWISE IDENTICAL inputs but singleton batch stride
301989888 instead of native50331648, shape[1,256,1536,128]. Both have alignment0
mod4096. Projection maxabs1.33514404296875e-05 incoming /1.52587890625e-05 outgoing.
Norm outputs themselves match; layout-mediated downstream behavior is supported.

Canonical metadata-only normalization-output view restored exact chunk256/128
full-module results; largerchunksstillnonexact/slower. Root
outputs/experiments/openfold3-canonical-norm-20261007 child193922 terminal0.
Then actual source change Kernel.__call__ allocates torch.empty(x.shape,...)
instead of empty_like (canonical singleton strides, no extra copy). Source replay
root openfold3-canonical-source-20261007 TERMINAL0 correctness_passedtrue;
12 trainedfullmodulecalls(two warm/four timed on eachcapture) chunk256 exact.
Local source experiments/openfold3_norm/screen.py modified; README notes pending
fullmodel qualification; new test_output_layout.py detects exact values+native
strides on small sliced singleton input. TEST NOT YET RUN (GPU model owns7).
Test deployed canonical-source root; original-screen expected failure and fixed
screen success should be recorded after modeljob. No arithmetic/kernel change.

GitNexus beforeedit currentHEAD282634f index/runnercurrent/incompleteReasons[].
Context resolved Method:experiments/openfold3_norm/screen.py:Kernel.__call__#4;
impact found0resolvedcallers (UNKNOWN; textcheckedmodel_scope/model_scope_strided/
screen_strided imports). Initial name-only Kernel.__call__ target notfound,
correctedtoUID. Afteredit analyze--index-only--pdg succeeded; followed required
query after rg hook. Newtest added afterwards, so finalrefresh required before
commit gates. No commitsyet; docs/isa remains user-owned. MCP staleness prose
wrong42commitsbehind; CLI typedreceiptcurrent. No graphfailure.

CURRENT LIVE fullmodel qualification wrapper195559 /timeout195560 /driver195561 /
measure195562 /infer195563 /worker195573. Root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-canonical-model-20261007.
Output /media/IntelSSD/fireprot/outputs/openfold3-canonical-norm-n1536.
StrictN1536R3D200S1seed101HIP7, no profiler; exactscreen.py source deployed in a
fresh copy of qualified normpairs frozenworker. Other source/artifacts unchanged;
preparation rehashes baselinepins and CLI inspect passes. No modified maintained
profile. Driver compare-npz all11bitwise and fullCIF againstnorm-pairs-candidate0,
counts11202accepted/7846fallback/4730copies. Source modifiesoutputlayout only;
fullmodel resultpending. Poll SAME handles/root; no concurrentGPU/heavybuild.
Preparation local /tmp/fireprot-prepare-canonical-model.py,driver
/tmp/fireprot-run-canonical-model.py, remote copies in canonical-source root.
Needcollect/gate/archive; earlier QK/PV/chunk rootsstillawaitarchive/publication.

## 2026-10-07 canonical layout qualified; gate+packing candidate improves operator

Canonical model195560/195573 TERMINAL0: all11arrays/fullCIFexact, normcounts
11202accepted/7846fallback/4730copies. Fresh collector rehashed sourcepins and
reran compare-npz/currentCIF. Single E2E351.033575462s, model257.847478328s;
NOT a paired speedup. Local provenance/openfold3_norm_canonical_layout.json.
Regression test proven discriminating: oldscreenfails stride(7680,640,128,1)
versusnative(1280,640,128,1); fixedscreenpasses. Rawlogs canonical-source root
layout-regression/result.json. README pending statement replaced with qualification.

Completed durable archive (before gate model preparation):
/media/IntelSSD/fireprot/outputs/experiments/openfold3-qk-layout-gate-followups-20261007.tar.gz
SHA9aed59a6554343c2244946dffb45a5359536d2ffdbf2b56376725c4c1b858903 size64045792.
Includes20roots listedarchive.json: completed QK variants, PV, triangle chunks,
firstdivergence/canonical sources/model/output, gate operator builds/screens/replays
and both componentprofiles. Captures/checkpoints/externalbaselines remain pinned
externaldependencies. Later gate model scope/preparation additions are NOT covered;
archive this later model separately. Earlier archive-pending journal notes superseded.

New handwritten HIP gate fusion: sigmoid(linear_g) * linear_p * mask in one
kernel, exactoriginalFP32operationorder. Rootopenfold3-gate-fusion-20261007;
all9operatorcases bitexact(random/extreme/nonfinite/stridedmask at small and real
shapes). First fullreplay fails BEFORE any moduleexecution due source-dedent
pattern indentation assertion; preservedfailure. Correctedreplayroot
openfold3-gate-fusion-replay-v2-20261007 terminal0: all40fullmodulecalls exact,
twowarm/eightbalancedtimedsamples perarm/capture. Baselinecanonicalnorm+copy:
in126.250946 ->120.538685ms; out133.043861 ->127.438160ms. No E2Eclaim.

Furthergate+packing uses32rowx128channelLDStranspose (padded132width), writing
channel-first physicalstorage directly. Rootopenfold3-gate-pack-20261007.
All9operatorcases exact; completeattention? NO: full TRIANGLE MULTIPLICATION
replay all40calls exact. In126.355713 ->116.203289ms; out133.069656 ->120.993801ms.
LibrarySHA b7a980bf469a0f13581985033f904541cae5783eab01d4579054588e75c3614c.
Source/build/numerical/fullmodule artifacts archived above. Local prototypes
/tmp/fireprot-of3-gate-fusion and /tmp/fireprot-of3-gate-pack. Not yet repo source.

CURRENT LIVE gate model wrapper208797/timeout208798/driver208799/measure208800.
Root outputs/experiments/openfold3-gate-model-20261007; output
outputs/openfold3-gate-pack-n1536. Fresh frozencanonicalworker, import swaps only
triangle_scope togate_experiment; sameoldcapabilityexactN1536chunk256/c128 mask
semantics andnativefallback. Gate librarypinned, newmoduleprofile/sourcepins.
StrictFP32N1536R3D200S1seed101HIP7, no profiler. Driver requiresall11/fullCIFvs
canonicalmodel, normcountsunchanged, gateaccepted430/fallback18/calls5160.
Poll SAME handles/driver-status; do notlaunchGPU/buildwhilelive. Modelscope copied
from successful replay transformation with correctedindentation. Prep/driver local
/tmp/fireprot-{prepare,run}-gate-model.py; remotegate-pack root.
Ifpasses, measure paired E2E against canonicalbaseline beforeclaiming benefit.
Fullmodelgatearchive/publication and operator source integration stillpending.

## 2026-10-07 gate+packing full model passed; three paired E2E runs live

Canonical-layout fix committed/pushed3bd6b9fc7dcadeca11122bce283b9a0122ebca25.
Postcommitindexcurrent/runnercurrent/incompleteReasons[], detectallclean.
The gate model208798/208799/208800 is TERMINAL0: all11arrays and fullCIFexact,
430accepted/18fallbacktrianglemodules,5160gatecalls,5160copycalls/1290copyfallbacks;
normalizationcountsunchanged. Freshcollector independently rehashedpins and
rechecked currentarrays/CIF. Single E2E346.785288351s/model252.691552714s,
notpairedgain. Qualifiedsource now copied locallytoexperiments/openfold3_gate
(gate.cpp/build.py/runtime.py/model_scope.py/screen.py), README+provenance
openfold3_gate_pack_qualification.json; pendingcommit. Kernel/runtime/modelscope
bytesmatch executedqualified sources. Modelscope is for frozenworkerpackage,
not automaticallywired into maintainedprofiles. Comparison script/rawreports
stay in archived experimentdirs.

Gate model durable archive:
/media/IntelSSD/fireprot/outputs/experiments/openfold3-gate-model-20261007.tar.gz
SHA097f3f18e8e2aa83efead3f15fbf3538b7b7cb61805bc2676f28ce4862cde2f8 size49453376.
Includes frozenmodel/output andmodelpreparation/drivers, depends on prior
9aed59a6 archive foroperatorbuilds/replays. Canonical andoperator archivealreadydone.

First pairedpreparer failed its compile assertion beforeanyGPUjob because import
insertion was insideforward andmultiline replacement indentationwrong. Preserve
incompleteoutputs/experiments/openfold3-gate-pairs-20261007; no benchmarkranthere.
Fixedpreparer createsNEW root openfold3-gate-pairs-v2-20261007. Sixrequestsinspect
andallsource/reference/operatorpinsverified; sourcechoice is localalias inside
forward: gate_triangle_scope vsbaseline_triangle_scope controlledsolelyby
FIREPROT_GATE_PACK0/1 (plusprofilehash declaredtreatment). Shared canonicalnorm
andallpreviouscache/attention/template changes. No profiler; strictFP32N1536
R3D200S1seed101HIP7only; Rustsupervisor monitors physicalGPU.

CURRENT LIVE timeout216757 ('timeout10900 python3driver.py'); root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-gate-pairs-v2-20261007.
ThreeAB/BA/AB pairs; driver requiresall6all11bits/fullCIFvsgate-model anchor,
normalizationcountssame, baseline triangle-copy430accepted/18fallback andno
 gate-fusion.json; candidategate430/18/5160 andno triangle-copy.json.
NoGPU/heavyremotebuildwhilelive; pollSAME216757 androotdriver-status. Lastconfirmed
live56seconds withfirstbaselineinprogress. Localprepare/run
/tmp/fireprot-{prepare,run}-gate-pairs-v2.py; copiesrootprepare.py/driver.py and
operatorroot. NextcollectpairedE2E andarchive, donotmultiplyhistoricalincrements.
Afterthiscycle nextpossibleoperator target is finaltriangle sigmoid/gated residual
write fusion (UNIMPLEMENTED); preserveseparate rounding toavoidFMAchanges.

## 2026-10-07 paired timing continues; residual epilogue screen queued

Latestmaster d38df7b934568474fbe2c66ac6b529b329f51519 pushed; postcommitGitNexus
index/runnercurrent, incompleteReasons[], finaldetectallclean. Gate sources and
operator/fullmodel evidence now committed. User docs/isa remains untracked.

Paired216757 stillLIVE, last11:56. Firstbaselineaccepted.json exists; global
paired/benchmark.json is only written atterminal, so its currentabsence is NOT
failure. measurement-000 andmeasurement-001 exist; no partialratio claimed.
New collector /tmp/fireprot-collect-gate-pairs.py andpairedroot/collect.py freshly
validateall6sourcepins/arrays/CIF/routecounts andRustperformance beforemedians.
Baselinecopy430/18/7740/0; candidategate430/18/5160gatecalls,5160copies/1290fallbacks;
normalization11202/7846/4730 andattentionkey5064/template27reused unchanged.

NEW LIVE epilogue queue timeout229343, root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-epilogue-20261007.
Currently waiting_for_pairs predecessor216757. MustpollSAMEbothhandles; doNOT
manuallycollectpairs orlaunchscreen whilequeueowns these steps. Queuechecks
predecessorPID+cwd+cmdline+boot, requires terminalsuccess AND PIDgone, SMI5idle;
rechecksallscript/sourcepins, collectsverifiedpairedresults, thenRustHIPbuild
withGPUhidden, then12 numericalcases onHIP7afterindependentPCI93/gfx90aprobe.
No otherGPU/heavyremotebuild duringpairedtiming. Queuefilesimmutablewhilelive.

Prototype /tmp/fireprot-of3-epilogue contains epilogue.cpp/runtime.py/build.py/
screen.py/queue.py; persistedremote. Python syntaxpassed ONLY. HIP compilation,
resources/numerical/performance/fullmodel allPENDING. Kernel fuses finalsigmoid,
x*sigmoid(g), andoptionallyz+=product directlyinto stridedzslice. Uses
#pragma clang fp contract(off) toretainseparate multiply/add rounding; native
operandorder x*sigmoid thenz+product. HIP __fmul_rn header availabilitywasnot
confirmed; do notclaimuseofthatintrinsic. Screen planned12cases: small/realshapes,
random/extreme, addfalse/true; comparesENTIREpaddedstorage sooutside-slice writes
fail. No alterationofexistinggate kernel orpairedworker. Futuretrainedreplay
requires replacingonly chunked finalg_chunk.sigmoid_/x_chunk*=g/zupdate block,
with originalview/aliasingcontract andexisting capabilityfallback preserved.

Collector runs BEFORE queuedbuild; ifcollector/build/screenfails, inspectterminal
reason and repair successoronly, neverrestartcompletedpairedbenchmark. A terminal
screen with correctness_passedfalse means numericalrejection, not successfulop.
Rawepilogueprototype hasnotbeenintegratedintorepo/archivedyet. Pairedarchive and
publicationpendingterminalcollection. Overallgoalactive; globalplateaunotproven.

## 2026-10-07 four-arm epilogue replay prepared and queued

Paired216757 stillLIVE, last21:10; threeof6invocationscomplete. Actual completed
invocationseconds: measurement000baseline0=351.629082030,
measurement001candidate0=348.148546485,measurement002candidate1=350.102864205.
No completed-series median orspeedupclaim. Currentfourthbaseline running.
Numericalqueue229343 stillwaitingon216757; immutablequeueinputs unchanged.

Four-arm trainedreplay preparedin/tmp/fireprot-of3-epilogue-replay andpersistedto
outputs/experiments/openfold3-epilogue-replay-20261007. Source-pinned ASTextraction
uses actualnativefileSHA612caf7abd07a86a7fdf4eda7e0a2170c04b407f9121fc286ab864a648db1ba7.
CPUcheck proves generatedbaselineBYTEIDENTICALtoqualified copy substitutions in
experiments/triangle_memory/replay.py; all4generatedfunctionscompile. Gateonly,
epilogueonly,andcombined substitutions eachoccur exactlyonce. ThisisCPU/source
validation ONLY, not numerical orperformance. Upstream filelocalcopy
/tmp/fireprot-of3-native-trimul.py. No fullrepo sourcechanges inthisturn.

NEW LIVE replayqueue240876 timeout11200, root
/media/IntelSSD/fireprot/outputs/experiments/openfold3-epilogue-replay-20261007.
Currentlywaiting_for_numerical_screen predecessor229343. Requiresparentterminal0
ANDcorrectness_passedtrue ANDPIDgone, samebootandpinnedsourcebytes, SMI5idle.
Thenrunsdriver.py to revalidatefrozenpostnormsources anddeviceidentity, setsHIP7,
launchesbenchmark.py. NoGPUworkbeforepairedcompletion+numericalscreenpass.
Localqueue/driver/benchmark/transform in/tmp/fireprot-of3-epilogue-replay;
preparer/tmp/fireprot-prepare-epilogue-replay.py. PollSAME216757/229343/240876;
doNOTmanuallycollect/build/screen/replay whilethesequeuesownsteps.

Replay comparesbaseline,gatepack,epilogue,combined with2warm+8Latin/reverse-balanced
timedsamples perarm/capture (80modulechecks). Freshnative moduleoutputreference,
canonicalnorm inallarms; fullmodule includespacking/allocations, checksoutside
GPUevents; unflushedcaches. Rehashes2trainedcapturebytesagainstqualifiedgate
replaycapturepins beforeGPUtensorcreation. Recordsallarmnumericalfailures;
correctness_passedfalse meansreject, evenifprocess exits0. Noautomaticfullmodel
launch beyondthisreplay. Nextinspectnumericalcompiler/result, pairedcollector
medians, thenchoosenextcandidateusingverifiedfulloperatorresults. Pairedarchive/
publication and epilogueartifactsarchive remainpending.

## 2026-10-07 gate paired E2E verified; epilogue exact; data-worker trial live

Paired216757 TERMINAL0. Collector (owned by229343 queue) completed successfully:
all6 currentarrays/all11bits/fullCIF/sourcepins/countspass. E2Emedians
352.949171123 ->348.257930396,1.0134705926772885x,1.329154765%less time.
Modelmedians258.904371057 ->254.220328252,1.0184251308194168x.
AllthreeE2Epairsfaster1.00999727/1.00812992/1.01787462. Newlocalprovenance
openfold3_gate_pack_e2e_n1536.json; README/performance/gateREADME updatedpendingcommit.
Thisincrementisagainstcanonicalnorm+prioroptimizations, NOT directoriginalTorch.

Queues229343/240876 nowTERMINALfailure BEFORE GPU epiloguework: queuedbuild
Torchimportresolved localqueue.py instead ofstdlibqueue, re-executedcoordinator
andfailed'Collection already exists'. Downstreamcorrectlyrefusedparentfailure.
DoNOTrerunpaired/collector. Sources/failurelogs preserved. Avoid queue.py or
inspect.py besidePythonexperimententrypoints. Fixednewdirectory
openfold3-epilogue-v2-20261007 containsrun_experiment.py,noconflictingqueue.py.
Numericalwrapper256079/child256681 terminal0;12casesexactentirestorage incl
untouchedslice/signedzero/subnormal/nonfinite/addfalse/true. #pragma clang fp
contract(off) compiles andpreservesseparateproduct/add; no intrinsicclaim.

Four-armreplayNEW rootopenfold3-epilogue-replay-v2-20261007,wrapper257290 terminal0.
All80fullmodulecalls exact(two warm/eightLatin+reversetimedsamples/capture/arm).
Incoming mediansms baseline126.230049,gate116.201469,epilogue122.825230,
combined112.740013. Outgoing133.024551,121.049664,129.590950,117.630695.
Epilogueadds~3.4ms/operatorgainovergate but NO fullmodel/E2Equalificationyet.
No automaticpromotion. Localnewprototypes /tmp/fireprot-of3-epilogue-v2 and
/tmp/fireprot-of3-epilogue-replay-v2; originalswithfailurekept separately.

Completedarchive /media/IntelSSD/fireprot/outputs/experiments/openfold3-gate-pairs-epilogue-20261007.tar.gz
SHA63c213999cf40505d16d17ddff6498da1f44a7e1a340807afe371a05df40d18f size490666381.
Includespairedsixoutputs/frozenworker,failedqueues,successfulv2epiloguebuild/
12screens/80replaycalls,dependencygate-modelarchive. Localarchive/script
/tmp/fireprot-gate-pairs-archive.json,/tmp/fireprot-archive-gate-pairs-epilogue.py.
Remote scriptpairedroot/archive-completed.py. DoNOT archiveagain samefilename.

Newread-onlynativecontrolaudit /tmp/fireprot-openfold3-native-control-audit.json:
backendnativebypassesattention/confidencecap butoutercache/triangle/template/norm
scopesneedexplicit0; finalfrozenworkerreaudit requiredbeforetrueoriginalbaseline.
No newnativecomparisonrun. CPUauditunderlyingsource /tmp/fireprot-gate-pairs-worker.py.

LargerpotentialE2Etarget: firstbaseline351.629082 total,supervisor333.539054,
request293.289425,model257.389746. ~94soutsideforward, not allattributedtostartup.
VerifiedupstreamDataModuleArg.num_workers default10; predictiondata_module_config
passesitthrough; generate_dataloader usesnum_workers forpredict andforkserveron
Linux. DataLoadergeneratorseedseparatefrommodelRankSpecificSeedCallback.
Reducingworkersmuststillpassall11/CIF; noassumptionofsemanticparityorbenefityet.

## Data-loader follow-up (2026-10-07)

The zero-worker trial finished inference but failed exact-output validation:
`outputs/experiments/openfold3-dataworkers0-model-20261007/driver-status.json`
records exit 1. Nine of eleven arrays differ (raw coordinate maximum absolute
error 50.008374691 Angstrom); it is not an accepted optimization. Its single
345.634980388 s E2E / 257.813005604 s forward observation cannot establish a gain.
Changing the process context may alter feature generation or RNG behavior;
this is a hypothesis, not an established cause.

The one-worker trial finished with all eleven arrays and full CIF exact.
E2E 351.664853578 s; forward 257.647426726 s. It has no paired speedup claim;
the default remains ten. Both loader trials are recorded in
`provenance/openfold3_dataworkers_n1536.json`.

The added output epilogue passed full-model qualification: all eleven arrays
and full CIF exact; 2,580 fused calls, 430 accepted modules, 18 native fallbacks.
Single E2E 349.815232659 s / forward 255.062962609 s. No E2E benefit claim yet.
Original model driver 271306 is terminal with a collector-only KeyError for
implicit `data_module_args`; separate `verified-status.json` passes full config,
all output and route checks. Preserve both records. Evidence is in
`provenance/openfold3_epilogue_model_n1536.json`.

Completed model/loader archive:
`outputs/experiments/openfold3-epilogue-model-dataworkers-20261007.tar.gz`, SHA
`7eee4f1d7d3fdcb483f01169074f801149929ef074a082d3ae904909a8ed55f5`,
153687155 bytes. It depends on the earlier paired/epilogue operator archive.

## Resumed optimization (2026-10-09)

The epilogue three-pair driver completed successfully; all six outputs retain
all eleven arrays and full CIF bytes. Independent collection is complete.
E2E medians 369.435519670 -> 371.197392122 s (0.995253543x); model medians
270.865807241 -> 269.710602969 s. Only one of three E2E pairs improves.
Do not select the added epilogue in the best composition. Retain gate/packing.
Evidence: provenance/openfold3_epilogue_e2e_n1536.json.
Archive outputs/experiments/openfold3-epilogue-pairs-20261009.tar.gz,
SHA99ebf832434a1ff4f298564fae9cc2b207831bddea8f8b8f9a72747dacedd505,
503458943 bytes. Collector and archive are finished; do not rerun on same paths.

The first startup diagnostic (589551) failed before worker launch: the pinned
environment inventory rejected 48 changed file identities, all system-library
entries (including /lib aliases). The changed families are FreeRDP/WinPR,
FreeType, Poppler, librsvg/pixbuf and libsoup. No inventoried Python/Torch/ROCm
files changed. Full capture and delta are preserved under
outputs/experiments/openfold3-environment-audit-20261009. Original pins retained.

The v2 startup diagnostic (590705) completed and retains all eleven arrays and
full CIF exactly. E2E 355.372791913 s, forward 260.541610430 s. Checkpoint read
1.366948292 s, model construction 2.914024332 s, runner setup 4.521564986 s.
DataLoader construction is negligible; the gap to first predict_step is
23.42157 s, including worker startup and preparation. No speedup claim.
Evidence: provenance/openfold3_startup_profile_n1536.json. Full audit included.
Current environment manifest SHA:
da808963739db3ca61d4177fd67256a5b3b742bdfaa7f835423ac2047e6147d6.

New direction: single-item in-process featurization with worker-0 initialization
and isolated Python/NumPy/Torch CPU/CUDA RNG, CPU thread and deterministic state.
Upstream DataModule explicitly passes pl_worker_init_function; normal num_workers=0
never invokes it. Candidate uses the iterator's actual _base_seed, worker_id=0,
and the original callback, then restores parent state before model execution.
CPU worker context uses one thread and default nondeterministic algorithms as
fresh worker processes do; the model retains strict determinism and memory fill.
No input tensors may be on GPU during this scoped data preparation.

First isolated trial 600012 failed before model execution because Lightning wraps
batch_sampler and sets loader.batch_size=None. Preserve this failure. The new
check requires len(dataset)==1 and len(loader)==1 plus num_workers=0 instead.
The v2 isolated-data model (601564) completed with all eleven arrays and full
CIF exact. E2E 351.541369164 s, model 256.045595740 s; one CPU preparation
15.876904323 s, base_seed302501347963651595, parent128 threads/strict restored.
No speedup claim from this one run. Maintained defaults unchanged.
Repository experiment: experiments/openfold3_data/{isolated_scope.py,test_scope.py}.
Source is byte-identical to the executed runtime. CPU test passes success and
failure restoration, including Lightning batch_size=None. Evidence in
provenance/openfold3_isolated_data_n1536.json.

Model/startup/environment/failed-trial archive completed:
outputs/experiments/openfold3-dataworker-followup-20261009.tar.gz,
SHA25e911b17a78d50dd891871bd9e8d3b83a3e3d080bd82e8451dc0a735ca35037,
121156359 bytes; depends on gate-model archive. Do not recreate same archive.

CURRENT LIVE: isolated-data pairs timeout PID 607660, root
outputs/experiments/openfold3-isolated-data-pairs-20261009. HIP7 only,
strict N1536 R3/D200/S1 seed101, baseline ten workers vs candidate isolated
worker-0 initialization with zero subprocess workers. Shared frozen source and
new environment inventory; only FIREPROT_ISOLATED_DATA=0/1 and profile SHA differ.
Three AB/BA/AB pairs, all eleven arrays/full CIF required; no epilogue added.
Local /tmp/fireprot-{prepare,run,collect}-isolated-data-pairs.py, deployed in
old gate-model experiment root. Collector not executed. Check existing driver;
never duplicate jobs or run heavy remote work during measured intervals.
Latest direct original-Torch comparison and maintained promotion remain pending.


## Queued GPU follow-up (2026-10-09)

Prior goal turn made progress: the isolated-data model and CPU state-restoration
gates passed and were committed; paired E2E is still live (607660, observed again).
A PV accumulation-order prerequisite probe is now prepared in experiments/mfma_pv.
No GPU probe or build has run yet, and it has no numerical/performance result.
It checks native PV kernel identity against the prior trace, three MFMA K-order
hypotheses on six adversarial/random bounded cases, and full output rows only
for hypotheses passing every bounded case. Target B128/B512, N1536/K1536/D32.

Queued timeout PID 617750 runs
outputs/experiments/openfold3-pv-order-20261009/run_after_pairs.py. It waits for
607660 to complete with six exact outputs, then executes the isolated-data
collector and archive exactly once, verifies the new full environment inventory,
checks GPU5/PCI93 idle, uses the Rust HIP compiler service with no visible GPU,
and finally runs the PV-only numerical probe on HIP7. Logs/status/preparation
are under the new root. No overlap with paired timing is allowed. A predecessor
failure or missing process without a terminal receipt stops the queue.

Local source directory /tmp/fireprot-of3-pv-order-20261009; same five source/pin
files are now in experiments/mfma_pv. The coordinator remains in that local
folder and the remote root. Paired archive script is
/tmp/fireprot-archive-isolated-data-pairs.py, deployed in old gate-model root;
expected new archive name openfold3-isolated-data-pairs-20261009.tar.gz.
Next inspect BOTH existing handles/status files before work. Do not duplicate
collection, archival, compiler or GPU work. Even a passing PV-only screen is not
a full attention implementation, model qualification or speedup result.

Local static PV follow-up: archived native excerpt hash verified; reconstructed
A/B LDS stores and reads support even-then-odd K8 for every thread/accumulator.
The CPU address audit passes 4,096 relationships (256 threads x 2 groups x 8
instructions). It is not a GPU arithmetic proof. Script
experiments/mfma_pv/analyze_native_lds.py and evidence
provenance/openfold3_pv_lds_address_audit.json. Local extracted excerpt
/tmp/fireprot-pv-native.asm; verified 2MB archive
/tmp/fireprot-native-mfma-review-v2.tar.gz. Original persistent archive remains
outputs/experiments/conditioning-cache-20261006/native-mfma-review-v2.tar.gz,
SHAecffe69c8447537ed43ddcb875dadf621454d730657b8ac3f3f8df8948fdbfd2.
This audit does not change the queued three-hypothesis GPU probe.

## Conditional full attention fusion prototype (2026-10-09)

A QK/softmax/PV prototype is prepared in experiments/mfma_qkv and remote
outputs/experiments/openfold3-qkv-fused-20261009. It retains the accepted QK source
SHA b89129b5... (SOURCE hash, not library hash), full softmax evaluation order,
then stages normalized probabilities through the existing 16 KiB LDS in six
bands and accumulates PV with two waves/even-odd K8. The untested performance
tradeoff is removed probability HBM traffic versus staging and limited PV waves.
Both ordinary and probability-capture templates will compare against native
Torch on eight packed cases with exact output/probability bits and guard checks.
B32/B128 outer, N1536, D32, FP32; no model or performance claim.

CURRENT QUEUED: timeout PID 628440 runs run_after_pv.py in that new root. It waits
for PV predecessor 617750, requires order 1 and all full PV checks to pass,
revalidates the full environment inventory and physical GPU, then builds and
screens. On predecessor failure it stops without running the fusion. Local
/tmp/fireprot-of3-qkv-fused-20261009 contains sources and coordinator. Do not
launch it again or overlap remote builds/GPUs with the existing paired sequence.

At the latest read, isolated-data E2E measurements 000..003 completed:
349.125556821 (baseline0), 341.782122796 (candidate0), 339.599098009 (candidate1),
349.971491766 (baseline1). Both complete pair checks passed. These are partial
observations, not a final median/speedup selection. Paired PID607660 still live;
PV queue617750 waiting; fusion queue628440 waiting. Poll all existing handles.

## Completed data pairs and fusion numerical gates (2026-10-10)

Supersedes the queued status above: isolated-data PID607660, PV617750,
fusion628440 and index32 PID638555 all finished successfully. Do not restart.
Three AB/BA/AB pairs: 349.971491766 -> 341.782122796 s E2E medians,
1.02396078795x, all six outputs/all 11 arrays/full CIF exact. Retain isolated
data preparation in the experimental best composition. Archive SHA
ea825abf416ab4745fe920d7328839d608e77672c3eca616affbed8c08e1449d
at outputs/experiments/openfold3-isolated-data-pairs-20261009.tar.gz.
This is incremental, not a new direct original-Torch comparison.

PV order 1 uniquely passes all bounded cases and full B128/B512 random/probability
outputs. Initial fusion and index32 both pass eight packed cases x two templates,
including probability bits and guards. Synthetic only. Evidence is recorded in
provenance/openfold3_qkv_synthetic_20261010.json. Local compile resource evidence
is separate: normal fusion 68 VGPR, index32 58 VGPR, 16 KiB LDS, zero scratch;
these are not occupancy/performance measurements.

CURRENT: complete trained TriangleAttention three-arm replay running under
outputs/experiments/openfold3-qkv-replay-20261010, driver shell643226, worker643321.
Local sources /tmp/fireprot-of3-qkv-replay-20261010. Baseline, initial fusion,
index32; canonical HIP normalization and K packing shared; same retained score
workspace/Rust leases; two warmups and twelve balanced samples per capture/arm;
all calls exact against native trained outputs. Source pins and current full
environment inventory verified before launch, HIP7/PCI93/SMI5. Check terminal
receipt before any further GPU work. No new model or E2E result yet.

Replay worker643321 finished0: all84 full trained calls exact. End baseline
267.045303345 ms / initial357.776504517 / index32 258.432830811; start baseline
256.973297119 / initial347.627792358 / index32 248.387313843. Reject initial
fusion; index32 is 1.0333x/1.0346x at module level. No E2E claim. Source/scripts
now under experiments/mfma_qkv, evidence openfold3_qkv_replay_20261010.json.

CURRENT full-model trial: timeout PID644920, root
outputs/experiments/openfold3-qkv-model-20261010; output
outputs/openfold3-qkv-index32-n1536-20261010. Frozen source copies qualified
isolated-data worker, inserts qkv_scope after HIP key_scope, enables
FIREPROT_QKV_INDEX32=1, pins library SHA
4bafb83b9056d8f80d5215bb2ae3a204b3baef5abbb61a0bbd32b829fc2eeab0.
Rust measure-inference followed by exact11/fullCIF/count/runner checks.
Reference is qualified isolated-data model output. Prep/run scripts in
/tmp/fireprot-{prepare,run}-qkv-model-20261010.py and remote replay root.
Physical GPU idle verified before launch. No simultaneous benchmark/archive.

## Index32 full-model qualification and live paired follow-up (2026-10-10)

Model PID644920 finished0. 5,064 index32 calls, no bounded-batch fallback, all11
arrays and full CIF exact, expected norm/gate counts unchanged. Single E2E
335.676654701 s / model249.373961893 s. Collector independently rechecked
all sources, NPZ arrays and CIF; collected.json exists. Do not rerun collector
or preparation. Evidence provenance/openfold3_qkv_model_n1536.json.

CURRENT paired coordinator timeout PID648410, child paired runner649275.
Coordinator script fireprot-run-qkv-followup-20261010.py in remote replay root
outputs/experiments/openfold3-qkv-replay-20261010. Its followup-status.json says
paired_e2e. Full model collector and paired preparation already succeeded.
Pairs root outputs/experiments/openfold3-qkv-pairs-20261010; six outputs named
outputs/qkv-pairs-n1536-{baseline,candidate}-{0,1,2}. AB/BA/AB, current full
environment inventory, HIP7 only. Both arms share copied model worker, isolated
data preparation and retained workspace; FIREPROT_QKV_INDEX32=0/1 is the only
execution flag difference. Full11/fullCIF and 5064 fusion-call checks required.
Coordinator will collect pairs exactly once after driver success; it does not
archive. Do not duplicate collectors/preparation/launch. Local prep/run/collect
files /tmp/fireprot-{prepare,run,collect}-qkv-pairs-20261010.py; coordinator
/tmp/fireprot-run-qkv-followup-20261010.py. All copied to remote replay root.

Next: inspect existing handles/terminal receipts, collect paired medians, decide
whether index32 improves E2E consistently. After GPU queue ends, archive the PV,
fusion, index32, replay/model and paired evidence. No archive overlaps timing.
Latest original-Torch direct remains earlier 949.999 -> 633.712 s; don't divide
that old baseline by current model qualification. Active goal continues.

## PV scheduling follow-up prepared while paired E2E runs (2026-10-10)

Previous goal turn was progress: exact full-model index32 qualification and
paired scheduling; no blocking condition. Current paired coordinator648410 and
child649275 were revalidated live. Do not restart either.

Local-only PV schedule variants generated by experiments/mfma_qkv/pv_variants.py
from pinned index32 source12467c8b...: unroll4 normal58/capture58 VGPR, unroll16
normal69/capture82, prefetch1 normal58/capture55. All16KiB LDS/zero scratch.
Prefetch1 disassembled instruction lists exactly equal baseline; skip GPU trial.
Unroll16 statically deprioritized, no measured latency rejection. Unroll4 chosen
for next GPU screen/replay; no float-order changes and no performance claim.
Static archive outputs/experiments/fireprot-qkv-pv-scheduling-20261010.tar.gz,
SHA bc5db919ff4cbd458d2885bb6b223f9ca0d138a1e0587edd1418aa28647bf6ae,
366628 bytes, uploaded and hash verified. Local files
/tmp/fireprot-qkv-pv-scheduling-20261010; provenance in repo.

CURRENT ADDITIONAL QUEUE: timeout PID653434 runs run_after_pairs.py in
outputs/experiments/openfold3-qkv-unroll4-20261010. It waits for coordinator648410
followup-driver-status0 and the collected six exact paired outputs, then checks
GPU5/PCI93 idle and full environment inventory, builds unroll4 via Rust without
visible GPU, screens eight cases x ordinary/capture, then runs three-arm trained
replay (baseline, unroll4, index32). Same84 exact calls and balanced12 samples.
Local sources /tmp/fireprot-of3-qkv-unroll4-20261010. Source SHA
dd7bbb785dae1afed98ad6fd357fa8e6822f0e49310a87530a296ee7c0d9ff9f.
Screen receipt is screen-driver-status.json; final driver-status.json covers
synthetic+module only. Any failed prerequisite stops the queue. No archive or
additional benchmark should overlap the current paired series or queued replay.
Next inspect BOTH existing paired and unroll4 handles. Whole-root archives for
PV/fusion/index32/replay/model/paired still pending until all GPU work terminal.

## QK-only index-width control added (2026-10-10)

The fusion comparison combines PV fusion and index narrowing. A useful separate
control narrows only QK/softmax and retains native PV. Local cross-compile from
qualified QK sourceb89129b5... gives QK+softmax VGPR62->44, rawQK51->44,
16KiB LDS/zero scratch. New source SHA
d1c0893ec7d4dc971ea4ab7c26ef1ed497e475b7468c70c152595de106f81261.
Patch experiments/mfma_qkv/qk_index32.patch; no GPU/performance claim.
Static archive outputs/experiments/fireprot-qk-index32-local-20261010.tar.gz
SHA f07fb158ed4327b99df8de49d6fc151edbea71b5622d4c7d6d76d1be213bfb5e
85236bytes; uploaded/hash verified. Local /tmp/fireprot-qk-index32-20261010.

CURRENT ADDITIONAL QUEUE timeout PID657408 in
outputs/experiments/openfold3-qk-index32-20261010/run_after_unroll4.py.
Waits for predecessor653434 unroll4 driver-status0, then verifies same full
environment/idle physical GPU, builds QK32 via Rust without visibleGPU, screens
eight cases x rawQK/softmax outputs with guard checks, runs four-arm full trained
attention: baseline,QK32,QKV32,QKV-unroll4. Two warmups+16 balanced samples
each of2captures =>144 calls, all exact. QK32 workspace entry replaced only in
that arm; inherited oldQK receipt describes fallback, explicit override binary
pin is recorded. No model/E2E qualification for QK32. Local complete scripts
/tmp/fireprot-of3-qk-index32-20261010, copied to same remote root.

Latest paired measurement000 baseline0=341.625921611s; only one completed
invocation seen at10min; this is not a speedup result. Coordinator648410,
child649275, unrollqueue653434, QKqueue657408 all revalidated live. Observe these
existing handles; do not restart or overlap any new GPU/build/archive work.

## Qualified-source integration audit (2026-10-10)

Previous goal turn was progress (local scheduling/index-width candidates and
serial GPU queues). This turn revalidated all four existing handles live and
observed first paired result: baseline0 E2E341.625921611/model254.552379405;
candidate0 E2E364.559315330/model277.183544621. Candidate slower in first pair,
mostly inside model; do not discard or attribute to external load without proof.
Remaining two pairs continue unchanged. Last child649275 still live at15min;
653434 and657408 wait on their pinned predecessors. No current blocker.

Read-only integration audit maps qualified isolated-data frozen source to current
repo in provenance/openfold3_qualified_source_map_20261010.json and
 docs/openfold3-deployment-audit.md. Most helper sources already identical; real
integration differences: maintained attention lacks frozen D16 selector/counters,
worker composition absent, artifact roots/import mappings, and transport omits
new gate/key/data/QKV diagnostics from completed artifact hashes (collectors
check them separately). Do not copy unrelated frozen Protenix drift. A new
maintained entry still requires exact whole-model and paired timing gates.
No maintained profile or active frozen source was changed by this audit.

Direct-native control correction: isolated-data standalone model and derived
QKV-model/pairs workers enable data isolation unconditionally. Merely setting
FIREPROT_ISOLATED_DATA=0 does NOT disable it there. Original isolated-data paired
worker does read the flag. Future direct-original comparison needs audited
conditional data scope/num_workers configuration plus all outer optimization
flags0 and backendnative. Do not reuse prior flag-only advice blindly.

## QKV E2E diagnosis and final archival queue (2026-10-10)

Compared baseline0, candidate0, candidate1 and original full-model qualification
operator-runtime records. Chunk tuners exactly match; all have workspace calls
{'1024':4,'512':4,'256':2508}, total5064, D16/D32 fallback launch counts100/192,
peak allocated51757716992 and reserved57139003392 bytes. Model times254.552379405,
277.183544621,253.474888237,249.373961893 respectively. Thus current evidence does
not support a changed chunk/operator path or different peak memory as the cause
of candidate0 slowdown. No historical clock/power telemetry proves a cause.
Local raw diagnosis /tmp/fireprot-qkv-run-diagnosis-20261010.json; authoritative
per-run operator-runtime.json remain in remote outputs.

First two pairs complete and Rust pair checks pass: baseline0=341.625921611,
candidate0=364.559315330; candidate1=342.071570544, baseline1=342.858609975.
Second pair is only ~0.23% faster. Last pair remains active; no final decision.
Coordinator648410/child649275 and successor653434/657408 revalidated live.

NEW FINAL QUEUE: timeout PID669126 waits after QK32 replay PID657408 in
outputs/experiments/openfold3-qkv-family-archive-20261010. Script
archive_after_replays.py (local /tmp/fireprot-qkv-family-archive-20261010/).
It requires the last terminal receipt, drains recorded coordinator identities,
checks SMI5 idle, then archives8 experiment roots plus model/six paired outputs.
It verifies dataworker/isolated-data dependency archive hashes and publishes
outputs/experiments/openfold3-qkv-family-20261010.tar.gz exclusively with digest.
Archive permits terminal failed trials and never changes their acceptance.
External checkpoints/resources/captures remain external and are not claimed
standalone. Inspect this final queue too before starting another GPU run or
heavy remote build; let archival finish before new model qualification. Do not
start a duplicate archive or relaunch any predecessor on observation timeout.

## QKV E2E terminal result (2026-10-10)

Paired coordinator648410/child649275 finished0; collector ran exactly once and
collected.json exists. All6 outputs/all11/fullCIF exact; candidates5064 fusedcalls.
Raw E2E: baseline0 341.625921611, candidate0 364.559315330, candidate1
342.071570544, baseline1 342.858609975, baseline2 340.641587154, candidate2
339.245447973. Medians341.625921611 ->342.071570544 (0.99869720558x). Model
254.552379405 ->253.474888237 (1.00425087935x). REJECT initial index32 QKV
for promotion; no stable E2E gain. Keep qualified isolated-data composition.
No outlier discarded and slowdown cause remains unproven. Evidence
provenance/openfold3_qkv_e2e_n1536.json; local raw
/tmp/fireprot-qkv-results-20261010/paired-collected.json.

Next queues653434(unroll4),657408(QK32 control),669126(family archive) remain
active; unroll4 should advance from its completed paired prerequisite. Inspect
current status and existing handles. Do not rerun paired collector/archive or
start overlapping GPU/heavy remote work. Wait for669126 before next model run.

## Unroll4 wins replay; QK32 rejected; next model/E2E active (2026-10-10)

Prior queued653434 and657408 both finished0. Unroll4 synthetic8x2 pass and84
trained calls exact: end baseline267.022659302/unroll4 240.266105652/index32
258.412902832 ms; start256.957168579/230.792106628/248.350906372.
Four-arm replay144 calls exact: end baseline266.899047852,qk32 271.389434814,
index32 258.322479248,unroll4 239.579216003; start256.865890503,261.594680786,
248.300537109,232.345024109. RejectQK32 for latency. Unroll4~1.11x module
gain repeated, advances to fullmodel. Evidence under provenance, raw local
/tmp/fireprot-qkv-results-20261010/{unroll4,qk32}-{screen,replay-report}.json.

Archive queue669126 finished0. Archive
outputs/experiments/openfold3-qkv-family-20261010.tar.gz
SHA eb995051a45c7b46324cb434948b2aaa7bb3f7b7d559b29624bb48b8026344ba
size614267969 bytes; includes8 experiment roots and7 outputs, plus dependency
archive pins. Receipt provenance/openfold3_qkv_family_archive_20261010.json.
Do not overwrite or rerun. Original PV/fusion/index32/replay/model/pairs and
unroll4/QK32 evidence now archived.

CURRENT coordinator timeoutPID685102 in
outputs/experiments/openfold3-qkv-unroll4-followup-20261010/driver.py; model child
685112. Model root openfold3-qkv-unroll4-model-20261010 and output
outputs/openfold3-qkv-unroll4-n1536-20261010. Reuses existing qkv_scope with
new unroll4 library pin, selected after four-arm winner/exactness checks. Root
prepared once, archive completion/GPUidle verified. Coordinator runs model,
collector, prepares3AB/BA/AB pairs in openfold3-qkv-unroll4-pairs-20261010,
executes/collects them only after all11/fullCIF/count gates. Scripts local
/tmp/fireprot-of3-qkv-unroll4-followup-20261010 and same remote control root.
No new model/E2E result yet; inspect existing handles instead of relaunch.

NEXT queue timeoutPID688389 in outputs/experiments/openfold3-qkv-unroll8-20261010.
Waits on685102 final driver-status0 and all6 collected outputs, then environment/
GPUidle checks, Rust build without visibleGPU, synthetic8x2 gate and three-arm
trained replay (baseline,unroll8,unroll4). Local sources
/tmp/fireprot-of3-qkv-unroll8-20261010. Source SHA
c065c4db5275e2bccae133eebfac8d8e6ecd56438b5d8d27b412f4eadde64f4d.
Local-only resource compile normal58VGPR/capture77,16KiB/zero scratch; no GPU
claim. Patch and provenance in repo. Static archive
outputs/experiments/fireprot-qkv-unroll8-local-20261010.tar.gz SHA
fa4d37fae63fe5a31a350de0d0fe7b4adb7b695c92bab43db2f8974ece624110,
36697bytes, uploaded/hash verified. Respect both active queues before GPU/build/
heavy archival. Old queues648410/649275/653434/657408/669126 all terminal.

Unroll4 model child685112 finished0; collector/preparer completed once. Full11
and CIF exact,5064 calls, unroll4 library SHA
80a619ecf7206d729b6f4a6ae95e2009f8c7cced8b0a042555e5893aedbccb3e.
Single E2E333.859070741 s/model241.828974545 s; not a paired ratio.
Evidence provenance/openfold3_qkv_unroll4_model_n1536.json, raw local
/tmp/fireprot-qkv-results-20261010/unroll4-model-collected.json.
CURRENT coordinator685102 stagepaired_e2e, child690424. Pairs root
outputs/experiments/openfold3-qkv-unroll4-pairs-20261010; outputs
outputs/qkv-unroll4-pairs-n1536-{baseline,candidate}-{0,1,2}. Do not duplicate
collector/preparation/launch. Unroll8 queue688389 still waits on coordinator
685102; no concurrent GPU/heavy build/archive. Goal remains active.

## Steady diffusion profiling queued (2026-10-10)

Previous goal turn was a verified wait:685102/690424 paired run and688389 next
queue were confirmed live. This turn identified a concrete measurement gap:
post-norm sample_diffusion44.345s has inclusive stage timing, while operator
profiling covers trunk only. New experiments/openfold3_diffusion/probe.py samples
DiffusionModule.forward call5 with Torch CPU/GPU trace and immediate child
ranges/input shape-stride metadata. All200 calls keep native arithmetic; event
instrumentation only elsewhere. It excludes outer augmentation/noise/scheduler
work and does not establish E2E improvement or hardware utilization/bound status.
Upstream DiffusionModule source is the same file SHA3b5ed9fb... as SampleDiffusion.

NEW queue timeoutPID701093 in
outputs/experiments/openfold3-diffusion-probe-queue-20261010/run_after_unroll8.py.
Waits for688389 terminal0/exactness, then checks SMI5/PCI93 idle, prepares fresh
outputs/experiments/openfold3-diffusion-profile-20261010 from the qualified
isolated-data baseline (no QKV fusion), and runs Rust measure-inference. Output
outputs/openfold3-diffusion-profile-n1536-20261010. Requires all11/fullCIF exact,
unchanged runner/norm/gate counts,200 denoiser calls, one nonempty call5 trace,
seven expected child names and a verified trace hash. Diagnostic elapsed/model
times cannot be used as E2E evidence. Source/scripts local
/tmp/fireprot-of3-diffusion-profile-20261010, copied to remote queue root. New
probe is in repo; no maintained default changed. No result yet. Inspect queue
701093 before any further GPU work/archival after unroll8.

Unroll4 paired first result observed during this turn: baseline0 E2E339.417633232,
candidate0 329.915480453 (~1.0288x); Rust pair-check-000 correctness_passed=true.
Only first pair is complete; do not report a final median or stable gain yet.


## Unroll4 E2E accepted, unroll8 rejected, diffusion profile active (2026-10-10)

Coordinator685102 and paired690424 finished0; collected.json exists and was
independently checked locally (six complete outputs, all11/fullCIF exact).
Medians345.752828869 ->330.998593305 s E2E (1.04457491924x), model254.904648186
->243.406796663 (1.04723718352x). Pair ratios1.0288/1.0446/1.0256. Retain
unroll4 in the experimental best composition; no maintained promotion or new
original-Torch ratio. Provenance openfold3_qkv_unroll4_e2e_n1536.json.

Unroll8 queue688389 finished0: synthetic8x2 and84 trainedcalls exact, but slower
than unroll4 (ending257.943/240.712, starting247.681/229.770 ms). Rejectunroll8
at module gate; no fullmodel trial. Provenance openfold3_qkv_unroll8_replay_20261010.json.

Network: 10.120.16.5 timed out after120s retry; same-host old endpoint10.120.16.9
worked, hostnameDiamondHill, boot_idafbe7b4e-c398-45b8-86ad-45cded194221.
Prefer.5 when healthy; .9 can inspect existing jobs without restarting them.
CURRENT diffusion coordinator701093 confirmedlive, stageprofile child718563,
rootoutputs/experiments/openfold3-diffusion-profile-20261010. No other GPU/build/
archive until it is terminal. Previous turn was verified wait followed by a
transient connection failure; this turn obtained new terminal evidence. Goalactive.

Read-only source audit points to repeated DiffusionTransformer.layer_norm_z and
per-block attention_pair_bias.linear_z for global/local stacks. Fixedinputs may
allow rollout-local reuse, but actual profile cost, consumer immutability and
retained memory must be checked before implementing. Time-dependent s/a remain
native. Local source snapshot /tmp/fireprot-openfold3-diffusion-sources-20261010.txt.


## Diffusion profile accepted; global projection reuse launched (2026-10-10)

Diffusion coordinator701093/child718563 finished0,200calls,all11/fullCIFexact.
Raw profile/trace SHAec6edcc9.../63e7a25c...; evidence in
provenance/openfold3_diffusion_profile_20261010.json. Samplecall5: globalstack
kernel sum166.156ms, including24pair-projection GEMMs39.286ms and24bias adds
63.052ms. Atomenc/dec27.316/21.635ms. One HIPnorm1.796ms has no externalID,
retained asunassigned. 2874kernels accounted; diagnostic only, noE2Egain/CUbound.
Other199 denoiser eventtimes median218.334ms. Globalbiasreuse isnowpriority.

NEW coordinator timeoutPID725420 confirmedlive, stage model_verified child725432,
root outputs/experiments/openfold3-global-bias-followup-20261010. It runs fresh
openfold3-global-bias-verified-20261010 then openfold3-global-bias-cache-20261010
only on numerical success; local scripts /tmp/fireprot-of3-global-bias-20261010.
Experiment scope source experiments/openfold3_diffusion/bias_cache.py SHA
3d5e6d8e902826515d076cced2bc92508d32114858ba1f3e61cf082f21ffc973.
Copies qualified unroll4 worker, wraps only global normalization+24linear_z;
keeps original layouts, native time-dependent work and200steps. Expected25misses/
4975hits; verification4975native+5000postconsumer checks; cache mode no comparisons.
Result storage4.5GiB; diagnostic extra references not performance evidence.
CPUtests6pass for reuse acrossrequests,invalidation,mutationdetection,cleanup.
No GPU acceptance yet. Read current receipts/handles; do not relaunch/overwrite.
No otherGPU/heavy archive while this coordinator runs. Earlier allqueues terminal.
Archives for unroll4model/pairs,unroll8,profile/currentcache remainpending until
allnewGPUworkends. Sourcefileincludeschecks; don'teditdeployedbytes in place.


## Global reuse recomputation passed; packed layout queued (2026-10-10)

Previous goal turn was progress (accepted unroll4 E2E, profiled diffusion,
launched globalcache). This turn global verified child725432 finished0:
all200steps,25misses/4975hits,4975native checks+5000consumer checks,all11/fullCIFexact.
No fallback/invalidation;4.5GiBretainedresults. Wholemodelpeakallocated51757716992,
reserved57139003392; doesnotmeasureisolatedcacheoverhead. Evidence
provenance/openfold3_global_bias_verified_n1536.json. Diagnostic time notspeedup.
CURRENT timeout725420 nowstage model_cache child730200; bothconfirmedlive.
Frozenoriginalsource remainsSHA3d5e6d8e...; don'teditthatcopy orreruncollector.

NEW queued timeoutPID732542 in outputs/experiments/openfold3-packed-bias-followup-20261010,
script run_after_cache.py; local /tmp/fireprot-of3-packed-bias-20261010.
Waits for725420terminal0 andbothverified/cacheall11/fullCIF receipts, fullinventory/
GPUidle, fourcaseGPUadd-layoutscreen thenpackedverified andpackedcachemodels.
UsesHIP7only; nootherGPU/build/archiveuntilqueueends. Modelroots
openfold3-packed-bias-{verified,cache}-20261010, outputs
openfold3-packed-bias-{verified,cache}-n1536-20261010. No new paired E2E scheduledyet.

Repo bias_cache.py adds optional pack_bias=False; currentqueuedpackingcopySHA
3d8e8436a6a4757e0d1f8b8ab6b7b4109cfe4a4b3af1434c0eac7b70ac9a735f.
Onlyprojectionmisses usepermute/contiguous/inversepermute, preserveslogicalbits,
returnsheadcontiguousbiasafterupstreampermute. Expectedprojectionstrides[-3:]
[1536,1,2359296];same4.5GiBretainedstorage. Tests9pass includingpackedstrides and
specialbits; fullGPUgatespending. Original activecompute-cacheusesoldfrozenbytes.
GitNexusimpactunknown(noindexedcallers), textauditfoundonlyexperimenttests plus
explicitfrozenworker injection; defaultfalsepreservescallers. No maintainedchange.
Nextobserve725420 and732542, collectrealqualification, thenpreparematchedE2E
forqualifiedwinner. Archivesnewunroll4/unroll8/profile/cachefamily remainpending.


## Compute-skipping global cache accepted; layout7x; E2E successor queued (2026-10-10)

Previous turn wasprogress(newpackedcandidate andverifiedglobalcache). Thisturn
725420/730200 finished0; original-layout compute-skipping globalcache all11/fullCIF
exact,200calls,25misses/4975hits,0recomputechecks/fallbacks/invalidations. Single
E2E323.335368624/model235.354797350; no pairedratio. Evidence
provenance/openfold3_global_bias_cache_n1536.json. Do notrerun.

Packedqueue732542 proceeded: fullinventory+HIP7identity verified, fourcaseadd
layoutscreen112checks pass. Native/packed medians2.659921/.375601,2.645360/.373680,
2.640801/.375280,2.641120/.374960ms. Onlyadd timed; packingoutside, notE2E.
Evidence provenance/openfold3_packed_bias_layout_20261010.json. Currentlystage
model_verified child736880; observeoriginalhandle thenautomaticpackedcachemodel.

NEW successor timeoutPID737448 confirmedlive in
outputs/experiments/openfold3-packed-bias-pairs-queue-20261010/run_after_packed.py,
waitingfor732542terminal0 +bothpackedmodelqualifications. Localcomplete scripts
/tmp/fireprot-of3-packed-bias-pairs-20261010. Willpreparefreshpairsroot
outputs/experiments/openfold3-packed-bias-pairs-20261010, threeAB/BA/AB, outputs
outputs/packed-bias-pairs-n1536-{baseline,candidate}-{0,1,2}. Botharmssharecopied
packedworker withconditionalFIREPROT_GLOBAL_BIAS_CACHE0/1; QKVunroll4always1 and
BIAS_VERIFYalways0; pairedflag+profileSHAonlydifferences. Baseline skipscache,
candidatecachesnormalization+24projectionsandpacksbias. Existingdataisolation
shared; notraworiginalTorch. Collectorchecksall11/fullCIF,currentbyteNPZ,counts,
cache/sidecarhashes,peakmemoryandchunk_tuners. No pairsresultyet.
Respect732542AND737448beforeotherGPU/build/archive. NoindependentGPUworkqueued.
Newfamilyarchivesstillpendinguntiltheseterminal. Goalactive,noblocker.

Read-only next-candidate audit: global diffusion uses5D[1,1,16,N1536,D48]
attention andhigh_precision=True; maintained/frozen softmax adapter eligibility
covers4D/H4/D16or32, so global attention remainsnative. Native ._attention does
FP32einsumQK, addsbothbiases, softmax, einsumPV. Potentialfutureexperiment:
keepnativeQK/PV, usefourH4slices ofpublishedstrictFP32HIPsoftmax (maskshared,
packedbiasslicecontiguous), oneoutputallocation. NeedsCABI/source/layoutaudit,
synthetic/trained/fullmodelcorrectness andperformancegates; notimplemented orqueued.
Source snapshot /tmp/fireprot-of3-native-attention-20261010.py; maintained
python/fireprot/adapters/openfold3/attention.py andbackends/amd/softmax_fp32.py.
Publishedkernel restrictsH4; do notclaimD48alreadyenabled.


## Packed model qualified, pairs active, global HIP softmax queued (2026-10-10)

Previous goalturnprogress: originalcachequalified,packedlayoutscreenpassed,
pairedsuccessorqueued. Thisturnpacked732542/736880/741965 allfinished0. Packed
verified200calls,4975nativechecks/5000consumerchecks,all11/fullCIFexact; compute
cache25misses/4975hits,0checks,all11/fullCIFexact. SinglecacheE2E321.920079837 /
model234.036924092. Original-layoutcache323.335single cannotprovidepairedratio.
Evidence provenance/openfold3_packed_bias_{verified,cache}_n1536.json.

CURRENT737448 stagepaired_e2e child745651 confirmedlive. Prepcompletedonce;
pairsroot outputs/experiments/openfold3-packed-bias-pairs-20261010. No resultsyet.
Do notrerunpreparer/collectororchangeexistingfrozenbytes. Same3AB/BA/ABflagonly
comparison; rawcurrentbytechecks andsidecarhashesafternativerustgates.

NEW successor timeoutPID750495 in
outputs/experiments/openfold3-global-softmax-screen-20261010/run_after_pairs.py,
local /tmp/fireprot-of3-global-softmax-20261010. Waitsfor737448terminal0 andsix
exactoutputs plus everyE2Epairfaster, then fullenvironment/physicalGPU/sourcepins,
synthetic5probability+140completeattentionchecks, >=1%gain onpacked/maskedcases,
thenfreshwholemodelqualification. Modelrootopenfold3-global-softmax-model-20261010,
outputopenfold3-global-softmax-n1536-20261010. Not yetGPUexecuted.
Do notstartotherGPU/build/archiveuntilBOTH737448and750495terminal.

Newcode experiments/openfold3_diffusion/global_softmax.py SHA
6d793830e559092e6bc4802f926c2bd480fa4eca908d1fcb11864cc7719e863d:
GlobalSoftmaxDispatcher wrapsexisting WorkspaceDispatcher (no monkeypatch),
N1536/H16/D48/5D/FP32guard; nativeeinsumQK/PV, fourH4publishedsoftmaxentrylaunches
batch1/waves2/reuse1 intoonefreshstrictoutput. CABIrowheadfixed4; pointeroffset
4*N*N*4byteseachgroup. Packedbiasnocopy; mismatchesfallback. ExistingSoftmaxFP32
inworkerisOwnedScoreSoftmaxsubclass, inheritsentry/device/receipt; sharedlaunch
counterincrementedbygroups, owned-scorecounts stilldescribeoriginal path only.
Fullmodelexpects4800calls/19200groups,0biascopies/layoutfallbacks,unchangedcache/
norm/QKVcounts andall11/fullCIFexact. CPUfallbacktests2pass; numericsnotyettestedGPU.
Apache2headerandTHIRD_PARTYinventoryaddedfornativearithmeticadaptation.

Screenqueueusespairedoutput/request.json forsource.root; actualcurrentrequest
keys/sourceverifiedcompatible. Allpreparedscriptscheckedsyntax. Source/prep/run/
benchmarkpins inremote launch.json. No maintainedprofilepromotion. Next inspect
existingpairs,softmaxqueue; newfamilyarchivepending afterallGPUtasksend.


## Direct-native route prepared locally while packed E2E runs (2026-10-10)

Previous goalturnprogress(packedmodelgates,globalsoftmaxcandidate+queue). Current
paired737448/745651 andsuccessor750495 revalidatedlive. Firstpair completed:
baseline0E2E328.562508105,candidate0 320.447456840 (~1.0253x), pair-check000true.
No finalmedianyet. Continueexistinghandles; noextraGPU/build/archive.

Read-onlynativeaudit found maintained python/fireprot_workers/openfold3.py SHA
c30ab522f9a18cb5fa0f21a182d5d2ac33e85173becb4676489485f6eb7d140e. Nativeearly
return invokesoriginalforward, nooutercache/isolateddata injection; configkeeps
upstreamdatamoduledefaults. New experiments/openfold3_diffusion/native_router.py
isLOCALONLY/unconnected: commonworkerentry choosesbackend-specific module,
lazyimportsone, sourcehash+originchecks, hardpinnednativeSHA, forwardsrequest
unchanged, emitsopenfold3-route.json onsuccess. CPUtests5pass(actualimports,
unselectedmodule exclusion,tamperchecks,referencepin). No GPUclaim/newnativejob.
Futurepreparer mustcopyexactnativeworker andqualifiedoptimizedworker intoone
sharedtree, routerasopenfold3.py, manifestopenfold3_routes.json; route+newsidecars
mustbehashedincompletedtransportoutputs, baselineflagsdisabled andRustexpected
backend/environment/bridge differencesdeclared. Bothroutesneedfullworkloadgates
andfreshmatchedE2E. Neverreuserawflag-onlyadviceunconditionaldataworker.
Docs/openfold3-deployment-audit.md updated withconcretesourcepin/routercontract.

Biaslayout qualification notes: originalcacheprojection0 strides
[37748736,37748736,24576,16,1], packed[37748736,37748736,1536,1,2359296];
packingreallychangeslayout. Isolatedadd~7x doesnotpredictwholemodelgain. Single
originalcache323.335vspacked321.920 notmatched; currentfirstcombinedpair~2.53%.
No cause claimed for gap between isolatedkernel andoverallgains.
