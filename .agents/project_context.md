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

The next handwritten FP32 HIP mask/bias/softmax candidate now passes 36 operator
configurations and all 11 model arrays exactly at N1000 and N1536. It packs bias
and uses wave64 row reuse; operator gains are not E2E gains. See
`experiments/hip_softmax/README.md` and `provenance/openfold3_hip_softmax_firstpass.json`.
Three sequential N1536 pairs are running under Rust at DiamondHill
`/tmp/fireprot-hip-softmax-model-v2/paired-n1536`; inspect the live benchmark
process and its order journal before taking further GPU actions. Do not launch
other FireProt GPU jobs during this measurement, or mutate its worker trees,
published bundle, environment inventory, CLI or bridge. The valid bundle is
`runtime/kernels/experimental-softmax-f32-v2`. The v1 bundle was accidentally
rewritten by an ISA extraction command and is invalid; both initial model trials
were rejected/cancelled and excluded. Always give objcopy a separate output ELF.
The user supplied local `docs/isa/CDNA2.pdf`; its GPT-6-luna xhigh review is complete.
Further work includes the paired E2E verdict, maintained integration if justified,
and additional exact HIP/layout/conditioning optimizations. The goal is active.
The source audit in `docs/openfold3-conditioning-reuse.md` identifies timestep-
independent pair conditioning and defines chunking, aliasing, lifetime and exact-
output gates. It is a proposed experiment, not implemented or measured caching.
`experiments/conditioning_reuse` now contains a diagnostic overlay for sampled
pair equality, downstream mutation, RNG and chunk observations. Five CPU tests
pass; no GPU/model validation has run. Deploy only to a fresh copied worker after
the active softmax benchmark finishes. Diagnostic timings are not E2E results.
An additional owned-score in-place softmax screen is prepared in
`experiments/hip_softmax/screen_inplace.py`; see `INPLACE.md` there. It has only
passed syntax compilation, not GPU checks. It uses the unchanged published
kernel and leaves active copied workers untouched. Test after the paired run.
The first N1536 softmax pair has passed all 11 arrays exactly: complete
invocation 953.331 -> 694.973 s (1.37175x), model 858.251 -> 599.567 s
(1.43145x). This is insufficient-repeat evidence only; two pairs remain.
`provenance/openfold3_hip_softmax_pair_000.json` pins that immutable pair report.
The same benchmark PID 1596375 has started candidate-1 (reverse-order pair).
Prepared (unconnected) maintained FP32 softmax tensor boundary:
`python/fireprot/backends/amd/softmax_fp32.py`, with byte-identical kernel source
copy `kernels/amd/fireprot_softmax_fp32.cpp`. Twelve CPU/meta contract tests pass.
It requires a new pinned package/ABI (`fireprot_softmax_fp32`/`softmax-fp32-v1`)
that is not built/published/validated yet. No OpenFold3 HIP request route has been
enabled. Follow the integration gates in `experiments/hip_softmax/README.md`.
Prepared `python/fireprot/adapters/openfold3/attention.py` adds source-pinned,
context-scoped dispatch through the Rust registry. Four CPU tests cover thread
isolation, duplicate installation, exceptional cleanup, escaped contexts and
native fallback (16 total with tensor-contract tests). Model workers still do
not import it; selected GPU execution and full-model parity remain unverified.
Prepared HIP runtime validation entry points now live beside the original screen:
`build_runtime.py`, `screen_runtime.py`, `worker-runtime-overlay.patch`. They are
syntax/patch checked only. The new worker overlay enables dispatch only with
`FIREPROT_RUNTIME_SOFTMAX=1`; both arms load the new pinned bundle. A separate
submitted-launch counter prevents confusing Rust selection with a HIP launch.
The second experimental candidate has passed the benchmark anchor gate at
695.721 s E2E / 600.051 s model; its baseline-1 is now running. No pair-1 ratio yet.
Update: pair-1 is now accepted, including all 11 bitwise comparisons and CIF
atom identity checks. E2E 957.824 -> 695.721 s (1.37673x), model
861.974 -> 600.051 s (1.43650x); see
`provenance/openfold3_hip_softmax_pair_001.json`. The same PID 1596375 is running
baseline-2 for the last pair. Full three-pair aggregate remains pending. Latest
prepared-runtime commit `7e82c13` passed remote CI; no GPU validation of that
prepared route has run. A narrow deployment dry-run was reviewed but not applied.

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
