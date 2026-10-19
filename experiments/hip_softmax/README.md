# Experimental FP32 HIP attention softmax

This is a new OpenFold3 experiment, not a promoted production backend.
The user requests handwritten HIP / CK-style optimization; Triton enablement
is outside this workstream. Rust owns native builds, publication, environment
inventories, model supervision and numerical acceptance. Python files here are
the explicit Torch tensor boundary and operator diagnostics.

## Arithmetic and scope

`fused_softmax.cpp` combines native-order mask and pair-bias additions with
softmax for FP32 `[B,4,N,N]` scores at N1000/N1536. It follows the installed
Torch `PersistentSoftmax.cuh`: 16/32 lane-local slots, wave64 XOR reductions in
32/16/8/4/2/1 order, `std::exp`, division and the zero-sum NaN branch.
No fast-math compiler option is requested. QK and PV remain upstream einsums.
All score/base address arithmetic uses 64-bit offsets. The boundary checks
dtype, shape, device, contiguity, bias broadcasting and inference mode.

The existing BF16 AtlasFold kernel supplies a row-scheduling reference; its BF16
rounding semantics are not copied into FP32. The user-provided local
`docs/isa/CDNA2.pdf` was reviewed with a GPT-6-luna xhigh agent. Relevant PDF pages
are 91–92 (addressing), 133 (`V_EXP_F32` is base-2 and flushes denormals), and 180
(`DS_BPERMUTE_B32` and EXEC). These correspond to printed pages 83–84, 125 and 172.
The ISA does not establish performance or equivalence of `V_EXP_F32` to `std::exp`.

## Completed operator checks

- `screen.py`: 36 bitwise-equal configurations, including zero/partial/nonzero
  masks, special floating values, both sequence lengths and a >4 GB N1536 score.
- `time_operator.py`: every variant is compared bitwise before timing, for
  starting and ending bias layouts. Two warmups and five GPU-event observations
  follow, alternating variant order. QK/PV and native score reset are excluded;
  output allocation and candidate bias packing are included.
- Code object metadata: wave64, no fixed LDS or scratch allocation; N1000 uses
  36 VGPRs/56 SGPRs and N1536 uses 57 VGPRs/48 SGPRs. These are compiler counts,
  not an occupancy measurement. Disassembly contains DS_BPERMUTE and 64-bit
  address carry operations.

Latest isolated measurements on HIP5 / PCI b3 / SMI7, gfx90a:

| Shape / bias layout | Native median | Packed HIP median, 2 waves | Ratio |
| --- | ---: | ---: | ---: |
| N1000 B256, starting | 33.811 ms | 10.973 ms | 3.08x |
| N1000 B256, ending | 52.841 ms | 11.006 ms | 4.80x |
| N1536 B128, starting | 39.974 ms | 9.885 ms | 4.04x |
| N1536 B128, ending | 85.644 ms | 10.269 ms | 8.34x |

These synthetic-input operator results are not model or E2E gains. The selected
candidate packs the O(N^2) bias and reorders independent rows for reuse across
eight outer rows, without changing arithmetic within each row.

## Full-model experiment and retained setup failure

The numerical/benchmark source deployment is `/tmp/fireprot-hip-softmax-v1` on
DiamondHill. The successful Rust build cache ends in
`44fe1e728065b85aab6ed52ce947694423a167f1b8e663a529569790bd5801da`.
The current immutable bundle is
`/media/IntelSSD/fireprot/runtime/kernels/experimental-softmax-f32-v2`, manifest
SHA-256 `c67bca20fd630e0e6f3e688a2ae24b71df0a5571be057c4896d19bb1d541b47d`.

An ISA extraction command omitted an output ELF path, causing llvm-objcopy to
rewrite the first published library's host ELF. A later loader rejected its hash;
the first N1000 model failed final environment revalidation, and the first N1536
model was explicitly cancelled. Both are excluded from accepted correctness and
timing results. The build cache remained untouched. A fresh v2 bundle restores
the exact original library. Subsequent extraction uses a distinct output copy;
the original/extracted device fatbins match, and the v2 library hash is unchanged.
Do not use or repair the invalid v1 bundle in place.

The current model deployment is `/tmp/fireprot-hip-softmax-model-v2`. It copies
the maintained worker/shim/kernel trees, applies `worker-overlay.patch` with
`git apply -p0`, adds `boundary.py` and `worker_overlay.py` alongside the workers,
and places the HIP source in its kernel identity tree. Both arms load the same
Rust-verified bundle. A new strict environment inventory binds the bridge and
published bundle in addition to the existing OpenFold3 environment.

N1000/N1536 full-work candidate validation passes all 11 arrays bitwise against
strict native references: FP32, R3, D200, S1, seed101, original source and weights.
Successful HIP selections are 1,702 and 5,592 respectively. Calls count function
entries; selections are counted after the HIP boundary returns, so tuning errors
before that point can prevent the dispatch counters from summing to all entries.
Evidence is in `provenance/openfold3_hip_softmax_firstpass.json`.
The candidate flag is
`FIREPROT_FUSED_SOFTMAX`; it is absent from maintained production profiles.
Three sequential N1536 AB/BA/AB pairs have completed on HIP7/PCI93/SMI5 with no
other FireProt GPU job launched during the experiment. Initial candidate validations
ran concurrently on separate GCDs and do not establish E2E gains.
Keep those worker trees, inventories and bundles intact for reproduction.

## Completed paired evidence

All three pairs pass all 11 arrays bitwise and CIF atom identity checks. The
scientific workload remains FP32/R3/D200/S1/seed101. Fresh model processes use
strict environment inventories and the same published library; filesystem cache
is not flushed. Rust reports `faster_median_and_p95` for full invocation time.

| Scope | Native median | HIP median | Speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 956.112 s | 694.973 s | 1.37575x |
| Model computation | 859.950 s | 599.567 s | 1.43429x |

Per-pair E2E ratios are 1.37175x, 1.37673x and 1.38226x. Full raw-report pins,
request parameters, physical devices, acceptance records and route counts are
in `provenance/openfold3_hip_softmax_e2e_n1536.json`. This is one fixed long input
and an experimental worker; the prepared maintained route still needs its own
GPU/model acceptance. Other host jobs were not continuously audited.

### Historical partial observations

The first N1536 pair passes all 11 arrays bitwise and CIF atom identity checks.
Complete invocation time is 953.331 -> 694.973 seconds (1.37175x); model time is
858.251 -> 599.567 seconds (1.43145x). Rust reports `insufficient_repeats`.
The immutable first-pair report is pinned
in `provenance/openfold3_hip_softmax_pair_000.json`; no aggregate verdict or
production promotion follows from this one pair.

The second, reverse-order pair also passes all 11 arrays bitwise and CIF atom
identity checks. Complete invocation is 957.824 -> 695.721 seconds (1.37673x);
model time is 861.974 -> 600.051 seconds (1.43650x). Its report is pinned in
`provenance/openfold3_hip_softmax_pair_001.json`. These records predate and are
superseded by the complete aggregate above.

## Maintained integration requirements

Read-only inspection confirms that the installed upstream
`openfold3/core/model/primitives/attention.py` (SHA-256
`30fe5433982ee73e3d7ce5341fa6b510b1ea332dbde072d56129632cc53f0746`)
calls the module-level `_attention` function from multiple attention classes.
The experimental global function replacement must not become an unrestricted
public hook. A maintained integration must scope dispatch to the active model
invocation, preserve unrelated callers, restore hooks on failure, and reject
conflicting/nested installations it cannot safely support.

Rust request validation currently admits native/Triton OpenFold3 requests, while
the worker only executes native OpenFold3. Enabling a HIP route requires explicit
changes at both boundaries. The existing `HipFused` enum value can be reused, but
schema admission alone is not integration. Rust's `OperationRegistry` must own
selection and counts, with tensor eligibility recomputed before every call and
included in cache keys. Pin the observed Torch/HIP and gfx90a compatibility,
upstream arithmetic source, and published library manifest. Unsupported tensors
must use the unchanged native path; kernel failures after launch must surface.

Before rollout, validate the maintained route's own N1000/N1536 outputs and
paired performance: the experimental worker results do not automatically cover
different dispatch overhead, artifact names, hook lifetimes or fallback behavior.
In-place storage reuse and diffusion caching remain separate experiments.

Local integration preparation now includes
`python/fireprot/backends/amd/softmax_fp32.py` and an exact source copy at
`kernels/amd/fireprot_softmax_fp32.cpp`. Twelve CPU/meta tests cover tensor
contracts, real bias strides, broadcast masks and rejected inputs. The loader
requires a new pinned `fireprot_softmax_fp32` / `softmax-fp32-v1` publication,
the measured Torch/HIP version and gfx90a. That new publication has now passed
the operator screen below. No maintained model worker imports this boundary and
no new request backend is enabled; full-model acceptance remains pending.

Prepared scoped dispatch is in `python/fireprot/adapters/openfold3/attention.py`.
It verifies the attention source, routes only the active execution context,
restores the captured original on exit, and rejects simultaneous installations.
Other threads and retained wrapper references after scope exit use native calls.
Rust `OperationRegistry` owns selection, cache decisions and counters; the tensor
predicate is reevaluated before every call. Registry selection counts precede
QK execution and must not be reported as successful HIP launches. Four CPU tests
cover scope/fallback behavior (16 together with the tensor boundary tests).
The dispatcher remains unconnected to maintained model workers. GPU operator
acceptance now passes; copied experimental workers are undergoing model checks.

## Prepared runtime validation entry points

The build, publication and operator screen have now executed after the original
paired benchmark completed. A fresh copied worker is undergoing model validation.

- `build_runtime.py NEW_DIRECTORY`: describe the renamed source and ABI to the
  existing Rust HIP build service. Use the validated model environment and hide
  GPUs during this build. Retain `build.json`, then publish using the documented
  Rust C ABI artifact workflow and a new immutable bundle directory.
- `screen_runtime.py NEW_DIRECTORY`: use `FIREPROT_SOFTMAX_FP32_BUNDLE` and its
  `_SHA256` pin to test the new loader and scoped dispatcher. It compares full
  attention outputs bitwise at both target lengths, both bias layouts and both
  precision flags, including masks, >4 GB scores and an unmatched-length native
  fallback. Each supported call must increment the submitted-HIP-launch counter;
  bitwise equality alone would allow an accidental all-native run to pass.
- `worker-runtime-overlay.patch`: apply with `git apply -p0` to a fresh copy of
  the maintained worker tree. Both arms load the same new bundle; only
  `FIREPROT_RUNTIME_SOFTMAX=1` enables the candidate. The native arm uses the Rust
  registry's native fallback. Existing transport seals `operator-runtime.json`.
  Both arms still use experimental `backend: native` requests, not a promoted
  OpenFold3 HIP request route.

`launches_submitted` counts immediate-success C ABI returns separately from Rust
registry selections. It is not proof of asynchronous GPU completion; full-model
synchronization and acceptance must pass. Failed screens preserve their partial
records, and full-model trials must retain all 11 arrays and the unchanged gates.

The published bundle is `runtime/kernels/softmax-fp32-v1`, manifest SHA-256
`4914cb71a64c42bf54495f1c9ee0eeb27808f0571b6a50fed81e8067f56f01be`.
All 32 operator checks pass bitwise with 28 actual HIP submissions and four
native fallbacks. Evidence: `provenance/openfold3_softmax_runtime_operator.json`.
N1000 and N1536 full-model acceptance and all 11 bitwise native comparisons pass,
with 1,702 and 5,592 submitted HIP calls respectively. These runs used
`/tmp/fireprot-softmax-runtime-model-v1` and overlapped on
separate GCDs, so their timings are diagnostic only. See
`provenance/openfold3_softmax_runtime_models.json`.
Keep the copied workers for reproduction. The library, inventory and bridge
are also bound by the still-running conditioning diagnostic and must remain
unchanged. Maintained request admission and its E2E result remain pending.

## Precision gfx1100 wave32 experiment

The 2026-10-17 Precision experiment is separate from the historical gfx90a
qualification above. `gfx1100_softmax.hip` retains the mask-add, bias-add,
exponential and division sequence but reproduces the installed Torch wave32
reduction order. It is compiled for gfx1100 only, with contraction and fast math
disabled. The maintained wave64 kernel and runtime gates are unchanged.

The quick screen passes 32 probability checks. The full sweep passes all 288
checks across four wave counts, optional eight-row bias reuse, direct/packed
biases, both triangle orientations, N1000/N1536, noncontiguous masks and fully
masked rows. All ten complete upstream attention comparisons are bitwise exact.
Alternating warm QK/softmax/PV timings improve by 1.253--1.638x in the tested
synthetic cases. The probability-only benchmark restores the native input with
a clone, so its larger ratios are not realistic attention-speed claims.

The initial `gfx1100_scope.py` provided an explicitly hash-pinned experimental dispatcher.
It accepted only FP32 inference at the recorded Torch/HIP/gfx1100 runtime with
four heads, head width 32, and `(outer batch, length)` in
`{(1,1000), (1,1536), (8,1000), (8,1536), (128,1536)}`. Other signatures use the
upstream function. The scoped screen passes 28 cases: 20 submitted HIP launches,
eight fallbacks, both precision flags, broadcast masks, restoration after a
deliberate exception and rejection of a wrong library digest. The Rust registry
still owns dispatch selection; this loader is experimental and is not a
published production artifact.

Reproduce on Precision from `/media/990Pro/fireprot`, selecting unused output
directories. Keep other GPU jobs and checkpoint transfers out of timed screens:

```bash
mkdir -p runtime/kernels/openfold3-gfx1100-softmax-rebuild
/opt/rocm/bin/hipcc -O3 -std=c++17 --offload-arch=gfx1100 \
  -fno-fast-math -ffp-contract=off -shared -fPIC \
  experiments/hip_softmax/gfx1100_softmax.hip \
  -o runtime/kernels/openfold3-gfx1100-softmax-rebuild/wave32.so
sha256sum runtime/kernels/openfold3-gfx1100-softmax-rebuild/wave32.so
/home/husrcf/anaconda3/envs/BIO/bin/python \
  experiments/hip_softmax/gfx1100_softmax_screen.py \
  runtime/kernels/openfold3-gfx1100-softmax-rebuild/wave32.so NEW_SCREEN_OUTPUT
/home/husrcf/anaconda3/envs/BIO/bin/python \
  experiments/hip_softmax/gfx1100_scope_screen.py \
  runtime/kernels/openfold3-gfx1100-softmax-rebuild/wave32.so \
  SHA256_FROM_PREVIOUS_COMMAND NEW_SCOPE_OUTPUT
```

The scripts pin GPU visibility to HIP0 / PCI `0000:53:00.0`, strict deterministic
FP32 and seed101. The broader screen records idle-device snapshots before and
after execution, not continuous Rust-supervisor isolation. Timings are device
events on warmed operators, include deterministic allocation filling, and use
four alternating samples per arm, each averaging three calls. Clocks and thermal
state are not fixed. Bias packing remains inside the candidate timing.

The pinned OpenBind-0 checkpoint was transferred under separate user authorization and its
complete source/destination hashes verified; see
`provenance/precision_openfold3_checkpoint_20261017.json`. The general asset
sender still waits on the unrelated failed AtlasLM finalization gate.
A separate venv now runs the complete strict deterministic FP32 N1000 model.
Its first maintained native invocation passed Rust acceptance in 326.240543 s,
including 288.395946 s of model forward. See the full-model follow-up below;
synthetic operator results do not replace its numerical and E2E gates.
See `provenance/precision_openfold3_softmax_gfx1100_20261017.json` for hashes,
raw report paths, environment preparation and representative timings.

### Real model chunks and complete invocation follow-up

The first frozen-worker candidate was exact but performed zero HIP launches:
all 8,167 attention calls fell back to native. Its actual N1000 chunks included
256 and 232 outer rows, outside the synthetic whitelist. That benchmark was
cancelled after its first pair and is retained as a coverage diagnostic, not
an optimization result (`outputs/experiments/gfx11-of3-pairs-20261017-v1`).

`gfx1100_model_shapes.py` checks these two real chunk sizes with both bias
orientations and zero, partial and nonzero masks. All 12 complete-attention
checks are exact, with 1.23--1.40x synthetic device-event ratios. The dispatcher
now additionally accepts `(232,1000)` and `(256,1000)` with H4/D32 FP32.
The 1000-row autotuner probes, D16 heads and other unmatched calls remain native;
no upstream chunk tuning or scientific settings are changed.

`prepare_gfx1100_model.py` freezes both arms into one worker tree, binds the
existing library hash, adds a zero-HIP-launch failure guard, and prepares a
three-pair AB/BA/AB specification. Rust owns all execution and acceptance.
`audit_gfx1100_model.py` independently rehashes outputs and replays Rust array
comparisons, additionally checking complete CIF bytes and actual HIP coverage.

The corrected `gfx11-of3-pairs-20261017-v2` series passes all three fresh-process
AB/BA/AB pairs with strict FP32/R3/D200/S1/seed101 unchanged. Complete invocation
medians are **322.388348 -> 285.978419 s (1.127317x)**, and forward medians are
**290.049754 -> 254.290548 s (1.140623x)**. Every E2E pair improves. All six
accepted predictions match the maintained native reference in all 11 arrays and
complete CIF bytes; each candidate executes 1,696 HIP calls. Independent
rehashing and Rust comparison replay pass. Full timings, identities and limits:
`provenance/precision_openfold3_model_gfx1100_20261017.json`.

Reproduce using a new absolute experiment directory, preserving the pinned
library and native reference. The preparation refuses to overwrite a directory:

```bash
python3 experiments/hip_softmax/prepare_gfx1100_model.py \
  /media/990Pro/fireprot /media/990Pro/fireprot/outputs/experiments/NEW_EXPERIMENT
target/release/fireprot benchmark-pairs \
  outputs/experiments/NEW_EXPERIMENT/pairs.json \
  outputs/experiments/NEW_EXPERIMENT/paired
python3 experiments/hip_softmax/audit_gfx1100_model.py \
  /media/990Pro/fireprot \
  /media/990Pro/fireprot/outputs/experiments/NEW_EXPERIMENT \
  /media/990Pro/fireprot/outputs/experiments/gfx11-openfold3-model-20261017-v1/native-prediction \
  /media/990Pro/fireprot/outputs/experiments/NEW_EXPERIMENT/audit.json
```

This remains experimental: the raw library is not a Rust-published production
bundle, the inherited venv needs a complete environment inventory, and N1536
trained-model qualification is separate. Clocks/caches were not fixed; AMD SMI
reported throttling in samples from both arms. Do not automatically enable the
maintained gfx90a profile on gfx1100.

### Precision gfx1100 rollout-local reuse cycles (2026-10-18)

`prepare_gfx1100_reuse.py` copies the already qualified wave32 worker into a
new immutable experiment tree. Mode 0 retains softmax only; mode 1 adds the
source-pinned pair-conditioning cache, mode 2 adds static atom representations,
and mode 3 adds exact duplicate-template reuse. The original conditioning
helpers and N1536 template experiment are unchanged. Only the frozen template
copy's five N1536 shape literals are specialized to N1000.

The pair/atom caches live for one diffusion rollout and must each record 200
calls, one miss, 199 hits and no fallbacks. Time-dependent single conditioning
and noisy-coordinate projections still execute at every diffusion step. These
caches require the audited immutable-input source contract; identity keys do
not detect arbitrary in-place mutation. Template reuse compares all bits and
masks (including signed zero and NaN payloads), preserves the first native
calculation and final reduction, and falls back for distinct slots. This
query-only/no-template workload still has internal padded template slots.

`gfx1100_reuse.py` composes the contexts and restores them on exit, including
exceptions. It fails on missing cache coverage or zero template reuse in mode 3.
The frozen transport seals all reuse diagnostics into accepted outputs.
`audit_gfx1100_reuse.py` independently rehashes frozen code/library/controls and
accepted outputs, replays Rust comparison of all 11 arrays against the original
maintained native result, and compares full CIF bytes. Forward-scoped allocator
peaks are diagnostic, not whole-invocation or hardware memory measurements.

Run each cycle serially and require its audit to pass before adding the next
optimization. Substitute a new absolute root for `NEW_REUSE` below:

```bash
python3 experiments/hip_softmax/prepare_gfx1100_reuse.py \
  /media/990Pro/fireprot /media/990Pro/fireprot/outputs/experiments/NEW_REUSE
# First run the copied N1000 template guard screen on idle HIP0.
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/media/990Pro/fireprot/outputs/experiments/NEW_REUSE/python/fireprot_workers \
runtime/venvs/openfold3-gfx1100/bin/python \
  outputs/experiments/NEW_REUSE/template_guard_screen.py \
  /media/990Pro/fireprot/outputs/experiments/NEW_REUSE/template-guard
# Repeat these two commands for i=1, then 2, then 3; inspect each audit.
target/release/fireprot measure-inference \
  outputs/experiments/NEW_REUSE/measure-1.json \
  outputs/experiments/NEW_REUSE/measurement-1
python3 experiments/hip_softmax/audit_gfx1100_reuse.py \
  /media/990Pro/fireprot /media/990Pro/fireprot/outputs/experiments/NEW_REUSE \
  /media/990Pro/fireprot/outputs/experiments/NEW_REUSE/audit-cycle1.json
# Only after all three cycle audits pass:
python3 experiments/hip_softmax/prepare_gfx1100_reuse_pairs.py \
  /media/990Pro/fireprot/outputs/experiments/NEW_REUSE
target/release/fireprot benchmark-pairs \
  outputs/experiments/NEW_REUSE/pairs.json outputs/experiments/NEW_REUSE/paired
python3 experiments/hip_softmax/audit_gfx1100_reuse.py \
  /media/990Pro/fireprot /media/990Pro/fireprot/outputs/experiments/NEW_REUSE \
  /media/990Pro/fireprot/outputs/experiments/NEW_REUSE/audit-final.json
```

Both paired arms use the same frozen worker, library and `hip_fused` backend.
Only the mode variable and corresponding profile digest differ. The AB/BA/AB
comparison is incremental against the previous softmax-only route, **not** a
direct native comparison; do not multiply historical ratios. Rust owns launch,
device isolation, scientific-setting checks and acceptance. The existing
production-publication, environment-inventory and N1536 limitations still apply.

The completed `gfx11-of3-reuse-cycles-20261018-v1` experiment passes all three
cycle gates and all three paired comparisons. Cycle E2E observations are
263.543519, 260.604323 and 241.414822 seconds; these are not repeated ablations.
Paired invocation medians are **283.614466 -> 239.334872 s (1.185011x,
15.6126% less time)**, with forward medians **252.733066 -> 208.135715 s
(1.214271x)**. All three paired ratios improve: 1.190806, 1.183716 and 1.193164x.
No observation is discarded. Rust reports `faster_median_and_p95`.

All nine predictions pass independent rehashing, all 11 bitwise array checks
and full CIF equality against the maintained native reference. Every run
submits 1,696 HIP calls. Each mode-3 run records 199 pair hits, 199 atom hits
and 27 reused template slots across nine eligible calls. Allocator peaks are
identical across all nine runs: 38,636,892,160 allocated / 40,909,144,064 reserved
bytes. No memory reduction is established. The six template-guard checks and
50 CPU tests pass. Full controls, timings, pins and limitations are retained in
`provenance/precision_openfold3_reuse_gfx1100_20261018.json`, produced by
`collect_gfx1100_reuse.py` after the final audit and test log are present.

## Maintained gfx1100 publication and default qualification (2026-10-19)

The Precision default promotes the earlier **out-of-place softmax plus
pair/atom/template reuse** composition, not the later layout-copy/private-score
experiments. It uses maintained modules under
`python/fireprot/adapters/openfold3/gfx1100`, the published-only loader
`python/fireprot/backends/amd/softmax_gfx1100.py`, and
`kernels/amd/fireprot_softmax_gfx1100.hip`. Rust owns build/publication, registry
selection, full-model supervision, numerical acceptance and array comparison.
Python helpers only describe the build, execute tensor screens and audit evidence.

To build on this prepared host, use a fresh receipt directory:

```bash
HIP_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/media/990Pro/fireprot/python \
FIREPROT_BRIDGE_LIBRARY=/media/990Pro/fireprot/target/release/lib_fireprot_bridge.so \
runtime/venvs/openfold3-gfx1100/bin/python \
  experiments/hip_softmax/build_gfx1100_published.py /absolute/NEW_BUILD_RECEIPT
# The helper prints the Rust cache entry. Substitute it and a NEW destination:
target/release/fireprot publish-kernel /absolute/CACHE_ENTRY /absolute/NEW_BUNDLE
```

Record the returned manifest digest in a new profile before testing. Never
overwrite the qualified bundle or substitute a new digest without qualification.
The maintained loader requires an explicitly pinned bundle and never falls back
to a raw library or just-in-time compilation. Its `softmax-gfx1100-v1` ABI is
out-of-place; the later private-score ABI is distinct and not interchangeable.

The installed bundle can be screened on idle HIP0 with a fresh JSON destination:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/media/990Pro/fireprot/python:/media/990Pro/fireprot/runtime/repos/openfold3 \
FIREPROT_BRIDGE_LIBRARY=/media/990Pro/fireprot/target/release/lib_fireprot_bridge.so \
FIREPROT_SOFTMAX_GFX1100_BUNDLE=/media/990Pro/fireprot/runtime/kernels/openfold3-gfx1100-softmax-v1 \
FIREPROT_SOFTMAX_GFX1100_BUNDLE_SHA256=dc8716ef683a43fc5c1c4438966754b7d50e0f3dae46dc24ec5ec7524fa7acf8 \
runtime/venvs/openfold3-gfx1100/bin/python \
  experiments/hip_softmax/screen_gfx1100_published.py /absolute/NEW_SCREEN.json
```

The 54 exact cases include N124 fallback, all seven HIP batch/length signatures,
an unsupported N1000 batch, both orientations, strided nonzero/partial/all-masked
biases and exception restoration. Incorrect manifest pins must fail. The first
screen's wrong-pin assertion harness failed to catch the Rust bridge's `OSError`;
its numerical cases passed. The corrected `published-screen-v2.json` is the
successful complete receipt; the first attempt is retained, not overwritten.

The retained qualification root is
`/media/990Pro/fireprot/outputs/experiments/gfx11-of3-promotion-20261019-v1`.
Its `pairs-standard-n124.json`, `pairs-standard-n1000.json` and
`pairs-standard-n1536.json` contain exact request/profile pins for one matched
pair each. The user canceled the N1536 baseline and waived further long-model
reruns; none of the ordinary-profile pairs completed. Only the strict N124
pair completed. Do not treat prepared controls as successful evidence or resume
them without a new request. Rust `benchmark-pairs SPEC NEW_OUTPUT` owns
execution, isolation and treatment checks. For read-only independent replay of
the completed strict short-sequence evidence, use a fresh audit destination:

```bash
python3 experiments/hip_softmax/audit_gfx1100_promotion.py \
  /media/990Pro/fireprot \
  /media/990Pro/fireprot/outputs/experiments/gfx11-of3-promotion-20261019-v1 \
  /absolute/NEW_AUDIT.json 124
```

The auditor rehashes sealed outputs and maintained source bytes, replays all 11
array comparisons in Rust, checks complete CIF equality, and requires actual
HIP/cache/template coverage for both long sequences. N124 must record zero HIP
launches and no feature reuse. `ready_for_default` requires all three lengths;
single-pair timings retain `insufficient_repeats`, not a stable speedup claim.
`collect_gfx1100_promotion.py REPO ROOT NEW_PROVENANCE` additionally validates
the successful strict-N124 audit, published screen, final tests, inventory digest
and equality of installed profiles with the prepared controls. The explicit
`--user-waived-long-reruns` collector flag records this user's deployment override
without changing the auditor's gates or inventing results; it additionally
requires the N1536 default example to select `native`. Without that flag, all
three completed ordinary-profile length gates remain mandatory.

Ordinary verification is the user-selected default; strict verification adds
the complete recorded Python/ROCm/system dependency-content inventory on every
invocation. The successful inventory is v3. Earlier captures rejected a legacy
dangling Qt symlink; v3 partitions that tree and checks absence of its missing
target ancestor. No system library was changed. This inventory is not a hermetic
OS lock and does not detect arbitrary newly added top-level system files.
All failed attempts, discovery, exclusions and successful inventory remain under
the root. The strict N124 matched pair verifies this separate deployment mode.
See [Precision usage](../../docs/precision-setup.md#openfold3-maintained-precision-default)
and `provenance/precision_openfold3_promotion_gfx1100_20261019.json`.
