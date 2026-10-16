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
**11,253,416,040 bytes** are reused locally and **41,525,253,158 bytes** are copied
from DiamondHill. It includes model weights, snapshot metadata and required CCD
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

## Native GPU qualification

The local HIP identity probe maps HIP0 to SMI0, PCI `0000:53:00.0`, gfx1100.
AMD SMI reported no running processes on either device before the first qualification. The native ESM3 B4/N1000
entry passes Rust acceptance on this GPU, with four exported predictions.
That functional run overlapped asset copying and is excluded from performance
claims. A missing `outputs/` parent caused an earlier startup failure before
any GPU work; creating it resolved the failure without changing validation.

The current ESM3 profile uses `/home/husrcf/anaconda3/envs/BIO/bin/python`,
Python 3.12.7, Torch `2.13.0+rocm10.0.0`, HIP `7.15.26333`, and the pinned
Biohub source above. It selects the official pretrained factory and offline
checkpoint loading. No DiamondHill HIP/CK operator bundle is loaded.

Independent native timing is recorded under
`runtime/precision-baselines-20261016/`, with fresh workers, complete hashes and
asset transmission paused. Optimized gfx1100 inference and its numerical parity
must be qualified before reporting a Precision speedup. ESM2's fair-esm namespace,
other model environments, portable execution profiles and model-specific kernels
still require separate setup/validation; checkpoint availability alone is not
working inference for every model.


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
