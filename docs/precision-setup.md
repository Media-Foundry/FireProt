# Precision initialization and independent baselines

Initialization started on 2026-10-16 at `Precision:/media/990Pro/fireprot`.
The top-level checkout tracks `master` from the FireProt GitHub repository.
This host has Navi 31 / gfx1100 GPUs; DiamondHill's gfx90a binaries and measured
speedups do not establish compatibility or performance here.

## OpenFold3 maintained Precision default

The recommended Precision entry is now the maintained published **out-of-place
wave32 softmax plus length-independent pair/atom/template reuse** route. The user explicitly
canceled additional long-model runs and authorized promotion based on existing
evidence. This is a deployment decision, not a completed three-length packaged
qualification. The subsequent user policy permits generalizing semantic reuse
after two exercised, correct length points; N1536's default example now selects
`hip_fused`. This route does not include the
later experimental layout copies or private-score ABI. The host-specific
default does not change DiamondHill profiles or the generic Rust backend default.

```bash
cd /media/990Pro/fireprot
cp configs/requests/precision-openfold3-n1000-default.json /tmp/my-openfold3.json
# Set a fresh run_id and unused output_dir; retain or explicitly update input pins.
target/release/fireprot inspect /tmp/my-openfold3.json
target/release/fireprot infer /tmp/my-openfold3.json configs/precision/openfold3-default.json
```

Additional examples are `precision-openfold3-n124-default.json` and
`precision-openfold3-n1536-default.json` under `configs/requests/`. The latter
uses backend `hip_fused` under that two-point policy. No fresh gfx1100 N1536
full-model result is claimed; the operator screen alone is not a model test. The short
input is the original 124-residue `cases/1izq_L124/sequence.fasta`, not a prefix
of a long sequence. It now admits semantic reuse too; its attention remains
native outside the screened kernel signatures. The earlier N124 native-fallback
qualification describes the previous revision, not a new reuse-enabled run.

The three execution choices are:

| Choice | Profile | Request backend |
| --- | --- | --- |
| Recommended ordinary verification | `configs/precision/openfold3-default.json` | `hip_fused` |
| Extra environment-content verification | `configs/precision/openfold3-strict.json` | `hip_fused` |
| Explicit native escape hatch | `configs/precision/openfold3-native.json` | `native` |

The ordinary profile retains full checkpoint/source checks, published-kernel
hash/compatibility verification, GPU isolation and Rust numerical acceptance.
The strict profile differs only by adding the recorded Python/ROCm/system-library
content inventory, revalidated on every invocation. The user selected ordinary
verification as the default because that additional scan is costly. Neither
choice relaxes deterministic FP32, deterministic fill or qualification gates.

The strict inventory is installed at
`runtime/environments/precision-openfold3-gfx1100-v1.json`, pinned by SHA-256
`f2e67a1616c713fbb109a4f3e5dad86e247597aff16ea44c37017faffac0c0ff`.
It covers the recorded current dependency trees, not a hermetic operating system.
An existing dangling Qt chooser link required partitioning its containing tree;
the inventory separately checks that `/usr/share/qtchooser` is absent. No system
files were changed. Discovery, scope and exclusions remain in the evidence root.
An environment upgrade requires recapture and qualification, not silently
replacing the trusted digest.

The kernel bundle is `runtime/kernels/openfold3-gfx1100-softmax-v1`, manifest
SHA-256 `dc8716ef683a43fc5c1c4438966754b7d50e0f3dae46dc24ec5ec7524fa7acf8`.
Rust builds/publishes/verifies it. The loader checks actual gfx1100, Torch
`2.13.0+rocm10.0.0` and HIP `7.15.26333`; an invalid explicit bundle fails closed.
No raw-library or compiler fallback is used by this maintained route.

Qualification uses strict FP32, R3/four trunk passes, D200/S1, seed101,
query-only inputs without supplied templates. Earlier N1000 full-model evidence
records pair and atom reuse with one rollout-local miss and 199 hits each, zero fallbacks;
only bitwise-identical padded template slots are reused, preserving reduction.
Positive sequence lengths now share the same reuse admission policy; exact-N
allowlists were removed from pair, atom and template eligibility. Workload,
batch/channel/layout, source and immutable-rollout conditions still apply, and
unlisted attention signatures use native attention. Two-point evidence comes
from N1000/N1536 pair/atom reports, N1536 template reuse on gfx90a and N1000
composition on gfx1100. This is semantic generalization, not blanket measured
correctness/performance for every input, version or platform. No new GPU run
was started; 76 CPU condition/lifecycle tests pass. See the recorded policy in
`.agents/project_context.md` and `provenance/openfold3_reuse_generalization_20261019.json`.

The published operator screen passes 54 exact checks (42 HIP and 12 native
fallbacks), both bias orientations and nonzero/partial/fully-masked strided masks,
exception restoration and rejection of an incorrect manifest digest.
The strict-profile N124 packaged pair passed all 11 arrays and complete CIF
equality with intentional zero HIP coverage. The ordinary-profile N1536 pair was
canceled during its baseline at the user's request; its candidate never ran.
Ordinary-profile N1000 and N124 pairs were not started. No new long-model or
ordinary-profile timing result is claimed; the earlier N1000 three-pair study
remains the composition evidence. The independent auditor still requires all
three lengths to return `ready_for_default`; that historical flag is NOT set by
either deployment decision. Its source-map replay describes revision `fedff98`,
not the subsequent generalized code. See `provenance/precision_openfold3_promotion_gfx1100_20261019.json`
and `outputs/experiments/gfx11-of3-promotion-20261019-v1/`.

Build/publication reproduction and audit commands are in
[`experiments/hip_softmax/README.md`](../experiments/hip_softmax/README.md).
Weights, binaries, environment inventories and predictions remain outside Git.

## Source and CPU entry

The Rust CLI/runtime/bridge build locally with Cargo 1.97.1 and the checked-in
lockfile. The initial online crates.io index request stalled. A dependency bundle
created by `cargo vendor --offline --locked` from the existing local cache enables
a clean offline build, with no global Cargo configuration changes. The bundle is
under ignored `runtime/cargo-offline/vendor`. CLI help and CPU-only request
inspection pass. Create the ignored `outputs/` parent before the first inference.

Independent model sources are initialized as follows:

| Model source | Precision path | Pinned commit |
| --- | --- | --- |
| Protenix | `Protenix/` | `bb3be993b52713795796c72cf0a46ef8a7323442` |
| Biohub ESM | `runtime/repos/esm/` | `43b4548b86762edfa747b07d5f440aad3c33acee` |
| AtlasFold / AtlasLM | `runtime/repos/atlasfold/` | `8ab3aca0e18c8b814d5ca6756b2617a07d72c68d` |
| OpenFold3 | `runtime/repos/openfold3/` | `7de748b7bc93adb5af0a4032d5e9f208b6e1325f` |

Protenix is cloned from a bundle of the clean independent local repository, with
its Git history retained and its origin set to the upstream URL. It is not a
top-level submodule/gitlink. The other repositories are shallow checkouts of
specific upstream commits. All model sources remain outside top-level Git.

## Existing home-directory weights

The home scan found usable published checkpoints under `~/Code/ProtBind`:

- ESM2-3B and original ESMFold v1 in
  `pypropel_unibio/tmp/uniprot_membrane_esmfold_20260721/esmfold_local/torchhub/hub/checkpoints/`.
- ESM3-open in
  `ESM/hub/models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/weights/`.
- ESM3 structure/function decoders also exist in that snapshot; the current
  sequence-only FireProt command does not require them.

Many other `.pt`/`.npz` hits were cached embeddings, task-specific training
checkpoints or predictions. They are not interchangeable with the published
model weights. One container-overlay system directory was inaccessible; the
scan does not claim exhaustive visibility into that directory.

Full SHA-256 checks identify the reusable ESM2, ESM3 and ESMFold v1 files.
They are copied into stable FireProt asset paths, rather than referenced through
another project's temporary directory. Original files are preserved.

## Asset layout and migration

The explicit manifest contains **49 files / 52,778,669,198 bytes**. Of this,
**11,253,416,040 bytes** are reused locally and **41,525,253,158 bytes** require
network transfer. DiamondHill is the initial transfer source. While its benchmarks
run, a separate local-host relay downloads the pinned official ESMFold2 revision
directly and streams it to Precision. Precision itself could not connect directly
to Hugging Face; the local relay can. Each streamed file must match the same
source-manifest size and full SHA-256 before atomic installation. Existing valid
files are preserved, and subsequent rsync skips them. The relay reads no
DiamondHill files and starts no process there. The manifest includes model
weights, snapshot metadata and required CCD
resources. Corrupt/partial historical downloads are excluded.

| Assets | Destination relative to workspace |
| --- | --- |
| Protenix Base, Mini, Mini-ESM, Tiny; ESM2-3B and regression companion | `runtime/checkpoint/` |
| AtlasFold 260703 | `runtime/weights/atlasfold/` |
| AtlasLM-3B | `runtime/weights/atlaslm3b/` |
| ESMFold2 and bundled ESMC, all six shards plus config/index/CCD | `runtime/weights/esmfold2/` |
| ESM3-open | `runtime/weights/esm3-open/` |
| OpenFold3 / OpenBind-0 | `runtime/weights/openbind0/` |
| AF2 model_1_ptm through model_5_ptm and license | `runtime/weights/af2/` |
| Original ESMFold v1 | `runtime/weights/esmfold-v1/` |
| Protenix CCD/common inputs | `runtime/common/` |

The source manifest is fully hashed and cross-checked against existing request
pins. Source and destination identities are retained under
`runtime/migration-20261016/`. Transfer status is not equivalent to hash
verification; `verified-assets.json` is written only after every destination
file passes complete SHA-256 verification. No checkpoint is added to Git.

**Migration status: transfer and final destination verification are in progress.**

### Detached Wi-Fi transfer

The user selected continued Wi-Fi transfer. As of 2026-10-17, the transfer and
final verifier run in detached tmux sessions and survive SSH disconnection:

| Host | Session | Purpose |
| --- | --- | --- |
| DiamondHill | `fireprot-refresh-finalize` | Wait for the remaining GPU studies, independently verify their evidence, then release the transfer |
| DiamondHill | `fireprot-precision-transfer` | Copy missing assets after release; retain partial rsync data for retry |
| Precision | `fireprot-assets-verify` | Wait for all 49 files, then verify every complete SHA-256 |

Inspect the sessions with:

```bash
ssh -t pc@10.120.16.5 tmux attach -t fireprot-precision-transfer
ssh -t Precision tmux attach -t fireprot-assets-verify
```

Detach with `Ctrl-b`, then `d`; completed sessions exit. Durable logs and status
files remain under DiamondHill's
`outputs/experiments/e2e-refresh-20261016/` (`asset-transfer-tmux.log`,
`asset-transfer-v2-status.json`, `finish-and-release-status.json`) and Precision's
`runtime/migration-20261016/` (`verification-tmux.log`,
`verification-progress.json`, `verified-assets.json`). A successful transfer
status alone is insufficient: only `verified-assets.json` with status
`all_files_sha256_verified` confirms migration completion.

The replacement sender was moved into tmux while it was still waiting, with no
active file-copy child. The old waiting verifier was replaced separately. The
active public-download stream and model jobs were left running. The release
watcher preserves the failed original experiments and requires successful
replacement studies plus independent evidence checks. If it fails, inspect its
log before resuming; do not bypass numerical acceptance to publish a speedup.

## Native GPU qualification

The local HIP identity probe maps HIP0 to SMI0, PCI `0000:53:00.0`, gfx1100.
AMD SMI reported no running processes on either device before the first qualification. The native ESM3 B4/N1000
entry passes Rust acceptance on this GPU, with four exported predictions.
That functional run overlapped asset copying and is excluded from performance
claims. A missing `outputs/` parent caused an earlier startup failure before
any GPU work; creating it resolved the failure without changing validation.

The native ESM3 profile uses `/home/husrcf/anaconda3/envs/BIO/bin/python`,
Python 3.12.7, Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, and the pinned
Biohub source above. It selects the official pretrained factory and offline
checkpoint loading. The separate FireProt profile uses the same environment but
selects the qualified sequence-only direct-GPU loader described below. No
DiamondHill HIP/CK operator bundle is loaded.

Independent native timing is recorded under
`runtime/precision-baselines-20261016/`, with fresh workers, complete hashes and
asset transmission paused. Optimized gfx1100 inference and its numerical parity
must be qualified before reporting a Precision speedup. At that stage ESM2's
fair-esm namespace, portable execution profile and model-specific route still
required separate setup and validation; the qualification below completes that
work without treating checkpoint availability alone as working inference.

### Qualified native isolation monitor

The maintained ESM2 and FireProt ESM3 profiles use the native AMD SMI helper at
`runtime/fireprot-smi-probe` with the explicit `snapshot-v1` protocol. Each
sample still opens a fresh AMD SMI session, enumerates every device, verifies
all BDF mappings and reads every device's process list. It combines the former
`amd-smi list` and `amd-smi process` subprocesses; it does not cache isolation
state or weaken the external-process check.

The Precision binary is built from
`kernels/amd/tools/fireprot_smi_probe.cpp` (SHA-256
`b73cdc829285498df5abf5654232a370410963d04708eaadfc403ad64b46d088`)
against AMD SMI 27.0.0. Rebuild it from the repository root with:

```bash
/opt/rocm/llvm/bin/clang++ -O2 -std=c++17 -Wall -Wextra -Werror \
  --ld-path=/opt/rocm/llvm/bin/ld.lld \
  kernels/amd/tools/fireprot_smi_probe.cpp \
  -I/opt/rocm/include -L/opt/rocm/lib -lamd_smi \
  -Wl,-rpath,/opt/rocm/lib -o runtime/fireprot-smi-probe
sha256sum runtime/fireprot-smi-probe
```

The qualified binary SHA-256 is
`29e550384593d7880fb591213391ada0d728f3e589fd5edae7165d0449653c91`.
Its normalized two-GPU list exactly matches `/usr/bin/amd-smi`, and a controlled
GPU0 holder was detected by both implementations. The helper's combined query
takes about 20 ms on this host, compared with about 220 ms for each official
Python CLI query.

Three fresh-process AB/BA/AB pairs isolate this monitoring treatment. ESM2-3B
B4/N1000 improves from 19.437104 seconds median and 19.570121 seconds p95 to
17.625852 and 17.641967 seconds: **1.102761x median** and **1.109294x p95**.
ESM3-open B4/N1000 improves from 10.502244 seconds median and 11.142026 seconds
p95 to 9.174322 and 9.208305 seconds: **1.144743x median** and **1.209998x
p95**. Every one of the six paired ratios is above one. All 24 ESM2 and ESM3
prediction archives remain bitwise exact, while median isolation sample counts
increase from 19 to 44 for ESM2 and from 10 to 22 for ESM3. This is an E2E
supervisor-overhead improvement, not a model-operator speed claim. Full evidence
is in
[`precision_plm_snapshot_monitor_20261017.json`](../provenance/precision_plm_snapshot_monitor_20261017.json).

### Qualified concurrent preflight verification

The maintained ESM2 and FireProt ESM3 profiles set
`FIREPROT_PREFLIGHT_CONCURRENT_SOURCE=1`. Preflight still reads and verifies
every byte of the pinned model source, input, checkpoint and named resources.
The only scheduling change is that source verification runs on a scoped thread
while the caller hashes the independent artifacts. Both tasks are joined and
must succeed before a prepared request can exist; a panic, changed source file
or corrupt artifact still rejects the run. The default remains serial for
profiles that do not explicitly select this treatment.

On Precision, source verification alone takes about 0.20 seconds for Biohub ESM
and 0.25--0.27 seconds for fair-esm. Three fresh-process AB/BA/AB pairs isolate
the overlap while retaining the qualified snapshot monitor and model routes.
ESM3-open B4/N1000 improves from 9.789266 seconds median and 10.225945 seconds
p95 to 9.305478 and 9.312227 seconds: **1.051990x median** and **1.098120x
p95**. ESM2-3B improves from 13.800290 seconds median and 15.225389 seconds p95
to 12.977287 and 13.739414 seconds: **1.063419x median** and **1.108154x p95**.
All six paired ratios are above one, and all 24 paired prediction archives are
bitwise exact. The observed E2E differences include normal fresh-process
variation and should not be relabeled as source-verification time alone. Full
evidence is in
[`precision_plm_concurrent_preflight_20261017.json`](../provenance/precision_plm_concurrent_preflight_20261017.json).


### First independent native baseline

ESM3-open, four N1000 inputs / B4, native BF16, seed101, HIP0/W7900:
three fresh invocations take **12.593587, 11.883476 and 11.885215 seconds**.
The complete invocation median is **11.885215 seconds**; model-forward median
is **2.370584 seconds**. All 12 measured prediction archives match the four
same-host functional-reference outputs byte-for-byte. Complete checkpoint
hashing and output sealing remain inside E2E. Asset transmission is paused;
DiamondHill has no GPU experiment running during these observations.

This is a native baseline, with **no optimized speedup ratio**. Torch/HIP differ
from DiamondHill, so cross-host timing is not an operator-only comparison.
[Timing and acceptance receipts](../provenance/precision_esm3_native_baseline_20261016.json).


### Run the qualified native entry

On Precision, copy the request and choose a fresh run ID/output directory:

```bash
cd /media/990Pro/fireprot
cp configs/requests/precision-esm3-open-b4-n1000-native.json /tmp/my-esm3.json
# Edit run_id and output_dir in /tmp/my-esm3.json before each invocation.
target/release/fireprot esm /tmp/my-esm3.json configs/precision/plm-esm3-open-native.json
```

The profile is byte-identical to the measured native profile. It selects HIP0
through the request and contains no gfx90a operator resource. This example is
specific to the initialized Precision paths and Python environment above.

## Qualified gfx1100 ESM3 frameless route

Sequence-only ESM3 supplies no structure frames, so its affine mask is entirely
false. The upstream geometric-attention block still constructs pairwise geometry
before masking its result to positive zero and applying `out_proj`. The explicit
FireProt route selects `esm3_frameless_exact_v1` only for contiguous BF16
`[B,1002,1536]` inputs on gfx1100 with the initialized Torch/HIP build. It skips
the work forced to zero and retains the native output projection. Unsupported
lengths, batch sizes, runtime versions, devices, layouts or masks are rejected by
the qualified route.

Warm-forward diagnostics passed exact token, embedding and logit checks for B1
and B4. B1 improved from 0.411104440 to 0.070079198 seconds (`5.866283459x`),
and B4 improved from 1.616895031 to 0.235678678 seconds (`6.860591059x`). These
warm observations explain the operator opportunity; they are not E2E claims.

Three fresh-process B4/N1000 native/FireProt pairs then passed complete Rust
acceptance and all-array exact comparison across 24 prediction archives. Native
E2E median and p95 were 12.436207255 and 13.135625181 seconds; the maintained
route measured 10.953459786 and 10.969215349 seconds. Median speedup is
**1.135367957x**, p95 speedup is **1.197499070x**, and all three paired ratios
improve. Model-forward medians are 2.438680 versus 0.987543 seconds. Filesystem
and GPU library caches were not flushed, and no asset transfer or concurrent GPU
workload was present.

The FireProt profile also replaces the general official factory with the same
meta-device ESM3 constructor restricted to the sequence-only interface. Unlike
the rejected CPU-mmap loader, it restores the pinned checkpoint directly to the
GPU before assignment, while avoiding unused structure/function codecs and
tokenizers. An incremental three-pair AB/BA/AB comparison measured the official
factory at 10.550715826 seconds median and 11.474607739 seconds p95, versus
10.509555259 and 10.533975665 seconds for direct-GPU loading. Median speedup is
**1.003916490x**, p95 speedup is **1.089295068x**, and all three pair ratios are
above one. All 12 paired predictions remain bitwise exact. This is a small
loading improvement, not an operator-speed claim. Full evidence is in
[`precision_esm3_direct_cuda_gfx1100_20261017.json`](../provenance/precision_esm3_direct_cuda_gfx1100_20261017.json).

The route is qualified only for sequence-only ESM3, N1000, B1/B4, native BF16,
Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, and gfx1100. It does not establish
performance or compatibility for generation, structure/function inputs, other
lengths, other batches, other runtimes or other architectures. Full evidence is
in [`precision_esm3_frameless_gfx1100_20261017.json`](../provenance/precision_esm3_frameless_gfx1100_20261017.json).

Follow-up searches did not find another production-worthy exact operator. The
four dominant BF16 GEMMs account for about 132 ms of the 240 ms profiled warm
forward, but rocBLAS changes their results and TunableOp selects the existing
default hipBLASLt solution for every shape under exact validation. A forced
memory-efficient SDPA is exact and looks 1.157x faster in isolation, but the
complete 48-layer forward is a tie at 231.861 ms versus 232.007 ms. The exact
RoPE/SwiGLU pointwise route also misses the E2E median gate at 0.995675x.

Omitting five output heads unused by this sequence-only caller has a descriptive
three-pair trend of 1.007781x median and 1.052730x p95, but one pair regresses
and absolute model-forward changes are only 0.3--11.9 ms. It remains a diagnostic
rather than a deployment change. An exact derived safetensors checkpoint is
also rejected because its warm load median is 1.803598 seconds versus 0.839171
seconds for the pinned PyTorch checkpoint; the 2.7 GB derived file was removed.
All follow-up artifacts and decisions are recorded in
[`precision_esm3_followup_screens_gfx1100_20261017.json`](../provenance/precision_esm3_followup_screens_gfx1100_20261017.json).

A further resident-model screen captures the original sequence-only forward in
a HIP Graph, preserving all output heads and validating the constant frameless
mask before capture. Tokens, embeddings and logits remain bitwise exact at B1
and B4, including replay with different tokens and restoration of the original
tokens. Eight alternating measurements per arm give B1 medians of 69.722 ms
eager versus 67.039 ms replay (1.0400x), and B4 medians of 231.260 versus
229.226 ms (1.0089x). Replay includes copying the input tokens. Additional
side-stream warmup and capture cost 175.502 ms at B1 and 506.676 ms at B4;
simple break-even estimates are 66 and 250 replays, respectively. These are
resident diagnostics, not single-request E2E improvements. Graph is not enabled
in the maintained worker.

Handwritten HIP residual kernels fuse scaling and addition while retaining the
intermediate BF16 rounding. All 97 residual sites and complete exported arrays
pass byte equality. Scalar/unrolled variants are slower; 16-byte vector loads
produce only a sub-percent whole-model improvement. That margin does not justify
a production route. Changing the four dominant BF16 projection weights to
column-major storage is also rejected: none of the 48 calls per shape is exact,
and isolated GEMMs are slower. No production configuration changes result from
these screens. Evidence, limitations and reproduction commands are recorded in
[`precision_esm3_graph_residual_layout_20261017.json`](../provenance/precision_esm3_graph_residual_layout_20261017.json).

Run the explicit B4 entry after choosing a fresh `run_id` and unused
`output_dir`:

```bash
cd /media/990Pro/fireprot
cp configs/requests/precision-esm3-open-b4-n1000-fireprot.json /tmp/my-esm3-fireprot.json
# Edit run_id and output_dir in /tmp/my-esm3-fireprot.json.
target/release/fireprot esm /tmp/my-esm3-fireprot.json \
  configs/precision/plm-esm3-open-fireprot.json
```

## Qualified gfx1100 ESM2 exact-loading route

Precision has the published ESM2-3B checkpoint at
`runtime/checkpoint/esm2_t36_3B_UR50D.pt` (5,678,116,398 bytes, SHA-256
`7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500`) and
the contact-regression companion at
`runtime/checkpoint/esm2_t36_3B_UR50D-contact-regression.pt` (SHA-256
`4da500eab246481dc9c8c95bc7b1d02f2803d761c380b0e95186d4a07d0fc84e`).
The request uses the official fair-esm `v2.0.0` source at commit
`0b59d87ebef95948c735b1f7aad463dc6dfa991b`; all 30 package Python files match
the original pinned manifest. This isolated checkout prevents the installed
fair-esm and Biohub ESM distributions from colliding in the shared `esm`
namespace.

The FireProt entry constructs ESM2 on the meta device, assigns the published
FP16 checkpoint tensors without a duplicate initialized model, transfers them to
the GPU, and then performs the native FP32 promotion. It also omits the
per-layer attention-head mean that the caller discards. The capability is
fail-closed to ESM2-3B, N1000 B4, gfx1100, Torch `2.13.0+rocm10.0.0`, HIP
`7.15.26333`, inference mode and the absence of a separate `hip_esm` library.
GEMMs, attention, GELU and all exported heads otherwise remain native FP32.

The qualified route now also opens the main checkpoint with PyTorch's
memory-mapped deserializer. This is selected by the same fail-closed gfx1100
capability check; native ESM2 and unqualified FireProt runtimes retain ordinary
loading. The checkpoint tensors are still assigned to the meta-constructed
model and materialized by the existing GPU transfer, so inference arithmetic is
unchanged.

Three fresh-process AB/BA/AB pairs passed complete Rust acceptance. Native E2E
median and p95 were 38.907886942 and 39.040355459 seconds; the exact-loading
route measured 22.430914554 and 22.693239074 seconds. Median speedup is
**1.734565341x**, p95 speedup is **1.720351834x**, and all three paired ratios
improve. Model-forward medians are 10.346957 versus 10.192840 seconds, confirming
that most of the gain is loading rather than a substituted compute path. All 12
paired prediction comparisons are bitwise exact for tokens, embeddings and
logits. The verifier independently rehashed all six accepted invocations and 63
declared artifacts. Full evidence is in
[`precision_esm2_exact_loading_gfx1100_20261017.json`](../provenance/precision_esm2_exact_loading_gfx1100_20261017.json).

An incremental three-pair AB/BA/AB comparison isolated ordinary versus
memory-mapped deserialization inside that exact-loading route. Ordinary loading
measured 22.421679226 seconds median and 22.534246987 seconds p95;
memory-mapped loading measured 21.642948619 and 22.174811107 seconds. The
incremental median speedup is **1.035980800x**, p95 speedup is
**1.016209197x**, and all three paired ratios improve. All 12 paired prediction
comparisons remain bitwise exact for tokens, embeddings and logits. The verifier
independently rehashed all six invocations, 66 declared artifacts and 24
prediction archives. Full incremental evidence is in
[`precision_esm2_mmap_gfx1100_20261017.json`](../provenance/precision_esm2_mmap_gfx1100_20261017.json).

The exact gfx1100 compute route additionally replaces the 216 qualified FP32
linear projections with a periodic-stagger HIP GEMM whose reduction order
matches the native hipBLASLt result bit-for-bit. The first three-pair comparison
measured 21.742772 seconds median and 21.860997 seconds p95 for native GEMMs,
versus 19.102696 and 19.511886 seconds for the exact route: **1.138204x median**
and **1.120394x p95**. All 12 output archives were bitwise exact. Its full
operator evidence is in
[`precision_esm2_exact_gemm_gfx1100_20261017.json`](../provenance/precision_esm2_exact_gemm_gfx1100_20261017.json).

The initial maintained profile loaded that same library from the Rust-published
bundle at `runtime/kernels/esm2-gfx1100-exact-v2`, pinned by manifest SHA-256
`9394d872c9642bb0a617cd27268cb331b9bb76f2df88ceda85431112e3dec433`.
Acceptance rehashes the manifest, 2,084,033-byte build receipt and 24,088-byte
library, and checks the package, gfx1100 target, Torch, HIP and C ABI identities.
It therefore avoids rescanning compiler executables and the complete ROCm,
Clang, system and GCC header trees in every fresh inference process without
omitting the deployed-artifact checks.

Three fresh-process AB/BA/AB pairs compared source-cache acceptance with the
published bundle. E2E median and p95 improve from 16.980327 and 17.569626
seconds to 13.322593 and 13.738595 seconds: **1.274551x median** and
**1.278852x p95**. All three paired ratios improve. Model-forward medians,
which include first use of the loader, decrease from 7.327940 to 3.459760
seconds. All 12 prediction archives remain bitwise exact and both arms execute
216 calls with build identity
`415b14fac6e155f7f87778a42a426e075b622fe9802dc45b438adc181b2e0c9c`.
This is loader/verification overhead removal, not a new arithmetic kernel.
Full evidence is in
[`precision_esm2_published_gemm_gfx1100_20261017.json`](../provenance/precision_esm2_published_gemm_gfx1100_20261017.json).

The current maintained kernel doubles its K tile from 16 to 32 without changing
the per-accumulator reduction sequence. This halves the block-wide
synchronization count while preserving exact FP32 output. The published bundle
is `runtime/kernels/esm2-gfx1100-exact-v3`, pinned by manifest SHA-256
`fc030bdb248b64e3270d5599d257664042cffe5cfa62da1b545379b18931e065`.
Acceptance also rehashes its 2,084,033-byte build receipt and 26,904-byte
library.

Three fresh-process AB/BA/AB pairs compared the published K16 and K32 bundles.
E2E median and p95 improve from 13.496007 and 13.599282 seconds to 12.528526
and 13.295886 seconds: **1.077222x median** and **1.022819x p95**. One paired
wall-clock ratio is below one because the first arm in every pair paid the same
startup/order penalty; the declared acceptance trend is nevertheless
`faster_median_and_p95`. All three candidate model-forward measurements improve
over their paired K16 measurements, from 3.443067/3.426590/3.476819 seconds to
3.228557/3.257890/3.249851 seconds. All 12 prediction archives remain bitwise
exact and both arms execute 216 qualified GEMMs. Full evidence is in
[`precision_esm2_tile32_gfx1100_20261017.json`](../provenance/precision_esm2_tile32_gfx1100_20261017.json).

Run either explicit B4 entry after choosing a fresh `run_id` and unused
`output_dir`:

```bash
cd /media/990Pro/fireprot
cp configs/requests/precision-esm2-3b-b4-n1000-native.json /tmp/my-esm2-native.json
cp configs/requests/precision-esm2-3b-b4-n1000-fireprot.json /tmp/my-esm2-fireprot.json
# Edit run_id and output_dir in both files.
target/release/fireprot esm /tmp/my-esm2-native.json \
  configs/precision/plm-esm2-3b-native.json
target/release/fireprot esm /tmp/my-esm2-fireprot.json \
  configs/precision/plm-esm2-3b-native.json
```

## OpenFold3 gfx1100 preparation and experimental softmax

OpenFold3/OpenBind-0 development resumed on Precision on 2026-10-17. The pinned
source `7de748b7bc93adb5af0a4032d5e9f208b6e1325f` is present. The isolated
`runtime/venvs/openfold3-gfx1100` environment inherits BIO's ROCm Torch and adds
the missing OpenFold3 inference dependencies without changing BIO. Model,
runner and CLI imports pass. The prepared native request/profile are
`configs/requests/precision-openfold3-n1000-native.json` and
`configs/precision/openfold3-native.json`; request inspection is CPU-only and
does not establish that inference works.

The 2,287,872,989-byte OpenBind-0 checkpoint is now installed at
`runtime/weights/openbind0/of3-ob-2025-06-30-174k.pt`, with independently verified
source and destination SHA-256
`bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4`.
The original migration verified other assets, but its replacement sender waited
for a release that never arrived after `atlaslm-b4-refresh` finalization failed.
There was no OpenFold3 checkpoint or partial file on Precision at inspection.
The user subsequently authorized this single checkpoint transfer; rsync completed
and the staging file was hash-verified before exclusive installation.

The general transfer gate and waiting sender remain unchanged. Twenty-two other
manifest files are still pending by existence/size checks, so this does not claim
completion of the 49-file migration. The same-host native N1000 full-model
entry now passes; its experimental follow-up is described below. See
[`precision_openfold3_checkpoint_20261017.json`](../provenance/precision_openfold3_checkpoint_20261017.json)
for the transfer diagnosis and verification receipts.

The first operator experiment ports the FP32 mask/bias/softmax fusion to
Torch's **wave32** reduction order on gfx1100. The gfx90a wave64 implementation
and maintained compatibility checks remain unchanged. All 32 quick cases and
288 variant checks are bitwise exact, including both triangle-bias orientations,
strided masks, nonzero biases and fully masked NaN outputs. Ten complete
QK/softmax/PV comparisons also pass bitwise equality with the pinned upstream
attention. Representative alternating warm device timings are:

| Length / outer attention chunk | Bias orientation | Native attention | Candidate | Ratio |
| --- | --- | ---: | ---: | ---: |
| N1000 / 8 | Starting, zero mask | 2.926 ms | 2.320 ms | 1.261x |
| N1536 / 8 | Ending, zero mask | 8.352 ms | 5.926 ms | 1.409x |
| N1536 / 128 | Starting, partial mask | 95.117 ms | 72.408 ms | 1.314x |
| N1536 / 128 | Ending, partial mask | 118.610 ms | 72.421 ms | 1.638x |

The outer chunk is an attention batch dimension, not a protein batch count.
These are synthetic operator measurements, not trained-layer, model-forward or
E2E speedups. Full attention timing includes fresh QK, allocations, bias packing
when selected, softmax and PV. The separate softmax-only sweep includes an input
restoration clone in its native arm and must not be used as a realistic speedup.

An explicitly hash-pinned experimental dispatcher passes 28 additional checks:
20 submitted HIP calls and eight native fallbacks. It restores the upstream
function after normal execution and exceptions and rejects a wrong library
hash. The experiment accepts only screened N1000/N1536 FP32 signatures; no
maintained optimized OpenFold3 profile is enabled on gfx1100. Scripts, kernel
build command and next gates are in
[`experiments/hip_softmax/README.md`](../experiments/hip_softmax/README.md#precision-gfx1100-wave32-experiment).
The evidence is pinned in
[`precision_openfold3_softmax_gfx1100_20261017.json`](../provenance/precision_openfold3_softmax_gfx1100_20261017.json).

### OpenFold3 N1000 full-model follow-up

The first maintained native invocation completes in 326.240543 seconds, with
288.395946 seconds of model forward, and passes Rust output/isolation acceptance.
Settings remain OpenBind-0 N1000, strict deterministic FP32 with memory filling,
R3/four trunk passes, D200/S1, seed101, query-only MSA and no templates.

The first frozen candidate reproduces all outputs but submits zero HIP calls:
the real triangle chunks are 256 and 232 rows, outside the original synthetic
whitelist. That series was cancelled after the first pair; its small timing
difference is not an optimization result. The 1000-row autotuner probes and D16
heads remain native in the corrected experiment. A 12-case complete-attention
screen qualifies H4/D32 at the two real sizes, and a zero-launch guard prevents
another all-native candidate from being accepted as an accelerated forward.

Three fresh-process AB/BA/AB pairs pass all Rust gates. Complete invocation
medians are **322.388348 s native versus 285.978419 s candidate (1.127317x)**;
p95 values are 322.522234 and 287.990005 s (1.119908x). Every paired ratio
improves: 1.119908, 1.127317 and 1.128889. Model-forward medians improve from
290.049754 to 254.290548 s (1.140623x). No observation is removed.

The independent audit rehashes all accepted outputs and replays the Rust
comparison against the maintained native reference: all six predictions match
in all 11 arrays and complete CIF bytes. Each candidate submits 1,696 HIP calls.
The successful series is `outputs/experiments/gfx11-of3-pairs-20261017-v2`;
the superseded zero-coverage diagnostic remains under the corresponding `v1`.

The experiment uses one frozen worker tree and a hash-pinned wave32 library;
the maintained OpenFold3 worker and gfx90a profile remain unchanged. This is not
production promotion: the inherited venv still needs a complete environment
content inventory, the raw library needs publication, and N1536 full-model
qualification remains separate. Caches are unflushed, clocks are not locked,
and AMD SMI samples report throttling in both arms. Three-pair trends are
descriptive, not statistical significance. Raw controls, failed coverage trial,
successful run receipts and independent audits are linked from
[`precision_openfold3_model_gfx1100_20261017.json`](../provenance/precision_openfold3_model_gfx1100_20261017.json).

### OpenFold3 N1000 reuse follow-up (2026-10-18)

Three cumulative experiments add rollout-local pair-conditioning reuse, static
atom-representation reuse, and bitwise-identical template-slot reuse to the
previous wave32 worker. The caches require the source-pinned immutable rollout
contract and retain the first native calculation; time-dependent conditioning
and noisy-coordinate projections still execute at all 200 diffusion steps.
Template slots and masks must match bitwise, including signed zeros and NaN
payloads. Distinct slots remain native and the original final reduction stays.

All three full-model cycles pass. Single invocation timings are 263.543519,
260.604323 and 241.414822 seconds; these are diagnostics, not repeated ablations.
The final three-pair AB/BA/AB comparison uses the same frozen worker and library
in both arms, changing only the reuse mode and profile digest:

| Scope | Previous softmax only | Combined reuse | Median speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 283.614466 s | 239.334872 s | 1.185011x |
| Model forward | 252.733066 s | 208.135715 s | 1.214271x |

Invocation time decreases 15.6126%. All three pairs improve (1.190806, 1.183716,
1.193164x); the p95 ratio is 1.190806x, with three-sample p95 equal to the maximum.
This is incremental to the previous softmax route, not a direct native speedup;
do not multiply results from different studies.

Independent replay checks all nine accepted predictions against the original
maintained native result: all 11 arrays and complete CIF bytes are identical.
Every run submits 1,696 HIP calls; each enabled pair/atom cache records one miss
and 199 hits, and each template-enabled run reuses 27 slots in nine calls.
All nine forward-scoped allocator peaks are equal (38,636,892,160 allocated /
40,909,144,064 reserved bytes). This establishes no memory reduction. Six GPU
template guard/fallback checks and 50 CPU tests pass.

The maintained route remains unchanged. Only this N1000 query-only/no-template
workload is measured; padded internal template slots were identical. Identity
cache keys do not detect arbitrary in-place mutation. Clocks/caches were not
fixed and a retained SMI snapshot reports throttling. Complete environment
inventory, Rust kernel publication and N1536 qualification remain separate.
Reproduction: [experimental README](../experiments/hip_softmax/README.md#precision-gfx1100-rollout-local-reuse-cycles-2026-10-18).
Evidence: [`precision_openfold3_reuse_gfx1100_20261018.json`](../provenance/precision_openfold3_reuse_gfx1100_20261018.json).

## Exact gfx1100 triangle layout copies (2026-10-19)

A separately accepted diagnostic of the previous composition measured 151.445 s
in four pairformer passes and 30.265 s in the diffusion rollout. These inclusive
GPU-event intervals include instrumentation; denoiser time is nested within the
rollout. A sampled steady block attributes 12.5% of kernel time to its largest
direct-copy kernel, 8.1% to its largest fill kernel and 7.8% to LayerNorm.
[Profile and native-parity evidence](../provenance/precision_openfold3_profile_gfx1100_20261019.json).

The candidate adds a Rust-built gfx1100 tiled bit-copy kernel for N1000 triangle
projection writes and explicit native operand/layout packing. It retains native
FP32 arithmetic, strict deterministic memory filling, R3/four passes, D200/S1,
seed101, query-only MSA and the same no-template input. Twelve GPU operator cases
pass bit-pattern, tail232, canary, stream and fallback checks. Six local CPU/meta
tests cover scope restoration and admission failures.

| Scope | Previous softmax/reuse | Added HIP copies | Median speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 238.181973 s | 231.214273 s | 1.030135x |
| Model forward | 208.234450 s | 201.095808 s | 1.035499x |

Three fresh-worker AB/BA/AB pairs reduce complete invocation time by 2.9254%.
Paired ratios: 1.030135x, 1.028206x, 1.031701x. All seven accepted predictions (one diagnostic cycle
and six paired calls) match the maintained native reference in all 11 arrays
and full CIF bytes. Each candidate selects 430 triangle calls and submits
3,440 HIP copies; 1,720 copy calls retain native fallback. The existing
1,696 HIP softmax submissions and pair/atom/template reuse coverage are unchanged.

All six paired forward allocator peaks remain 38,636,892,160 allocated and
40,909,144,064 reserved bytes. No memory reduction is claimed. Caches were
unflushed and clocks unlocked; the retained cycle snapshot reports THROTTLED
(90 C hotspot, 98 C memory). This is an incremental comparison against the
previous optimized composition, not a new direct-native ratio. Three-sample p95
is the observed maximum. The route remains experimental; complete environment
inventory, maintained publication and N1536 qualification remain open.

[Reproduction and scope](../experiments/triangle_memory/GFX1100.md);
[all timings, hashes and independent audit](../provenance/precision_openfold3_copy_gfx1100_20261019.json).

## Exact gfx1100 private QK-score reuse (2026-10-19)

A separately built HIP entry overwrites the private QK-score tensor with softmax
probabilities. It removes the separate output allocation and deterministic fill.
The pinned dispatcher owns the fresh scores; full-storage/disjointness checks
reject unsafe aliases. Native QK/PV, all FP32 arithmetic, wave selection, existing
conditioning reuse and triangle copies stay unchanged.

| Scope | Previous copy composition | Added private-score reuse | Median speedup |
| --- | ---: | ---: | ---: |
| Complete invocation | 231.100820 s | 223.261027 s | 1.035115x |
| Model forward | 200.910893 s | 193.078869 s | 1.040564x |

Three fresh-worker AB/BA/AB pairs reduce E2E time by 3.3924%.
Paired ratios: 1.045908x, 1.035099x, 1.035535x.
All seven independently audited predictions match the maintained native reference
in all 11 arrays and complete CIF bytes. Each candidate submits 1,696 in-place
softmax calls, zero storage fallbacks, and the unchanged 3,440 HIP triangle copies.
Strict FP32/N1000/R3/four trunk passes/D200/S1/seed101, query-only MSA and no
supplied templates remain fixed.

Eight complete-attention operator cases improve 1.063–1.134x. The largest
N1000/chunk256 attention case saves 4,096,000,000 allocated bytes, but **full-model
forward peaks do not fall**: 38,636,892,160 allocated / 40,909,144,064 reserved
bytes in both arms. The diagnostic cycle is excluded from paired statistics;
a short unrelated GPU task was observed before it. The Rust supervisor verified
the accepted device and isolation for every completed prediction.

These are incremental measurements, not a fresh direct-native ratio. Caches are
unflushed, clocks unlocked, and a retained SMI sample reports THROTTLED. Three
samples do not establish a statistical confidence bound. The route remains a
frozen experiment; full environment inventory, publication and N1536 qualification
are separate outstanding work.

[Reproduction and ownership contract](../experiments/hip_softmax/GFX1100_OWNED.md);
[all timings and independent audit](../provenance/precision_openfold3_owned_gfx1100_20261019.json).
