# DiamondHill checkpoint inventory

Verified by remote file metadata on 2026-10-04. These are existing published
model checkpoint paths supplied by the user, not trained adaptation weights.
File existence and byte size have been checked. AtlasFold and AtlasLM full
SHA-256 hashes were also computed during baseline setup; publisher authenticity
has not been independently verified in this task.

Preferred SSH endpoint: `pc@10.120.16.5` (updated 2026-10-07).
`10.120.16.9` is another interface on the same DiamondHill host; hostname and
boot identity were checked before switching. FireProt workspace: `/media/IntelSSD/fireprot`.
The current user restriction is **HIP_VISIBLE_DEVICES=7**; it supersedes
the earlier authorization to use all eight GCDs.
Check device occupancy before execution and monitor benchmark isolation.

## ESMFold2 (current target, 2026-10-15)

The user explicitly selected **biohub/ESMFold2**. All six model shards include
the ESMC language model; do not substitute the separate original ESMFold v1 or
ESM2-3B checkpoints. Full hashes of the existing snapshot were recomputed and
matched its pinned download manifest on 2026-10-15.

- Snapshot: `/media/PM982/engramfold/folding_e2e_20260921/weights/esmfold2/snapshot`.
- Index: `model.safetensors.index.json`; six shards total 26,319,349,908 bytes.
- Configuration: `config.json`; CCD: `ccd.pkl` (417,306,584 bytes).
- Weight revision: `69869f737beffec5294845ede23db5fc0b4f509e`.
- Source: `/media/PM982/engramfold/folding_e2e_20260921/repos/esm`, official
  `v3.4.1.post1` commit `43b4548b86762edfa747b07d5f440aad3c33acee`.
- Python: `/media/PM982/engramfold/folding_e2e_20260921/venvs/esmfold2-amd/bin/python`.

See [ESMFold2 integration](esmfold2.md) and
[full artifact hashes](../provenance/esmfold2_assets_20261015.json).

## Previously recorded single-file checkpoints

| Model | Bytes | Absolute path |
| --- | ---: | --- |
| AtlasFold 260703 | 859549307 | `/media/PM982/engramfold/folding_e2e_20260921/weights/atlasfold/snapshot/weights/atlasfold-260703.pth` |
| AtlasLM-3B base | 6128042331 | `/media/PM982/engramfold/folding_e2e_20260921/weights/atlaslm3b/snapshot/weights/atlaslm_3b_base.pth` |
| AF2 model_3_ptm | 371746284 | `/home/pc/Code/colabfold/cache/params/params_model_3_ptm.npz` |
| OpenFold3 / OpenBind-0 | 2287872989 | `/media/PM982/engramfold/folding_e2e_20260921/weights/openbind0/of3-ob-2025-06-30-174k.pt` |

The user also reports AF2 `model_1_ptm` through `model_5_ptm` in the same
parameter directory. This task has checked only `model_3_ptm`; compatibility
with a future OpenFold adapter remains to be validated.

## AtlasFold source and environment

Computed SHA-256:

- AtlasFold: `cbadb227d40e801a3d268d6e884e6ed3c6b9fb89d558739fcf5970492732e5ea`.
- AtlasLM-3B: `8097cc4a234853f19ca97435844829d84fe7f13215c900e74659fa9682cc0ddb`.

HIP ordinals and SMI ordinals differ in the observed environment. In particular,
`HIP_VISIBLE_DEVICES=7` resolves to PCI `0000:93:00.0`, SMI GPU 5. Resolve the
HIP PCI address and match `amd-smi list --json` before selecting the monitored
physical device; do not assume ordinal equality.

- Independent source: `/media/PM982/engramfold/folding_e2e_20260921/repos/atlasfold`.
- Source HEAD: `8ab3aca0e18c8b814d5ca6756b2617a07d72c68d`; clean at inspection.
- Python: `/media/PM982/engramfold/folding_e2e_20260921/venvs/atlasfold-rocm/bin/python`.
- Observed Torch: `2.12.0a0+git78258b9`; HIP: `7.14.60850`.
- Both `atlasfold` and `atlaslm` import from this checkout's `src` directory.

Pass both checkpoint paths explicitly to `atlasfold.pretrained.load_model`.
Do not copy multi-gigabyte assets into Git or alter the shared checkpoint files.
Other study scripts and trained adapters beside this checkout are not the
FireProt native reference.

## Verified native execution semantics

The monomer runner defaults to four recycles, which execute five trunk passes
(`range(num_recycles + 1)`). Every pass samples a new MLM mask at probability
0.15 and evaluates AtlasLM again. Caching one LM result across passes changes
the model and is not an exact optimization.

The trunk has four LM-stack blocks and 48 main Pairformer blocks per pass.
AtlasLM returns attention logits used in pair features; replacing it with an
attention implementation that only returns the value output is insufficient.
Native loading uses BF16 AtlasLM and trunk weights with FP32 components and
BF16 autocast; diffusion sampling disables autocast. Preserve these boundaries.

Default sampling uses 30 diffusion steps for 1000 residues and 100 steps for
1536 residues. Record sample count explicitly (runner default: five). Baselines
must include all LM passes and preserve the requested recycle/step/sample
counts. Subsequent measured speedups and their timing boundaries are documented
in [AtlasFold performance](atlasfold-performance.md).

## Protenix Mini-ESM and ESM2-3B

Full file hashes were recomputed on 2026-10-05 for the folding checkpoint,
ESM2-3B checkpoint and contact-regression companion under
`/media/IntelSSD/fireprot/runtime/checkpoint/`. Exact paths and hashes are in
[Mini-ESM integration](protenix-mini-esm.md) and the checked-in requests.
These files are separate from AtlasFold's AtlasLM checkpoint. All three are
required by the Mini-ESM request contract.

## Protenix Base-default v0.5.0: next target

User-supplied checkpoint, remote metadata verified on 2026-10-06:

- Path: `/media/IntelSSD/fireprot/runtime/checkpoint/protenix_base_default_v0.5.0.pt`
- Size: **1,474,265,486 bytes**.
- The resolved path is identical to the supplied path.
- Full SHA-256, checkpoint contents and model loading have not yet been verified.
  Perform those checks after the running OpenFold3 benchmark to avoid additional
  checkpoint I/O during timing.

This is a separate target from the measured Mini-default model. The local
upstream model configuration selects 10 cycles and 200 diffusion steps for
Base-default v0.5.0, versus Mini-default's 4 cycles and 5 steps. Confirm the full
resolved configuration against the pinned runtime source before benchmarking;
do not silently apply the Mini workload or transfer its speedup claims.
The maintained worker currently accepts Mini-default, Mini-ESM and Tiny-default,
so Base requires an explicit integration and validation step.

Sequence: complete the current OpenFold3 cycle, the one additional attention
cycle, and README clarification; then integrate Base and profile representative
1000/1536-residue workloads with native and optimized outputs checked at matched
scientific settings. Preserve these weights unchanged and outside Git.
