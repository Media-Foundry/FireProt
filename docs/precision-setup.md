# Precision initialization and independent baselines

Initialization started on 2026-10-16 at `Precision:/media/990Pro/fireprot`.
The top-level checkout tracks `master` from the FireProt GitHub repository.
This host has Navi 31 / gfx1100 GPUs; DiamondHill's gfx90a binaries and measured
speedups do not establish compatibility or performance here.

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

The maintained profile loads that same library from the Rust-published bundle
at `runtime/kernels/esm2-gfx1100-exact-v2`, pinned by manifest SHA-256
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
