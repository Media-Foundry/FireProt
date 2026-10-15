# Biohub ESMFold2 integration and optimization

Status, 2026-10-15: Native and opt-in N600 qualification passes on HIP7, with
all 12 arrays bitwise equal and identical complete CIF bytes. Repeated N1000
E2E measurement is running. This is the standard **biohub/ESMFold2**,
not ESMFold v1 or ESMFold2-Fast.

## Source, weights and execution boundaries

- Official source: `https://github.com/Biohub/esm`, tag `v3.4.1.post1`, commit
  `43b4548b86762edfa747b07d5f440aad3c33acee`. The official Git HEAD and tag were
  checked against the existing clean DiamondHill checkout on 2026-10-15.
- Model revision: `69869f737beffec5294845ede23db5fc0b4f509e`.
- Snapshot: `/media/PM982/engramfold/folding_e2e_20260921/weights/esmfold2/snapshot`.
  Its six safetensors shards total **26,319,349,908 bytes** and include ESMC.
  The model index, configuration, all shards and `ccd.pkl` are separately pinned
  request artifacts. Current full hashes match the existing download manifest;
  see [asset evidence](../provenance/esmfold2_assets_20261015.json).
- Python: `/media/PM982/engramfold/folding_e2e_20260921/venvs/esmfold2-amd/bin/python`.
- Source checkout: `/media/PM982/engramfold/folding_e2e_20260921/repos/esm`.
- Native reference uses `EsmFold2Model.from_pretrained(..., esmc_precision='bf16')`,
  `set_kernel_backend(None)` and `ESMFold2InputBuilder.fold`. It includes the
  ESMC forward, input preparation, every refinement pass, diffusion and confidence.
  The ROCm installation uses upstream PyTorch fallbacks where NVIDIA-specific
  dependencies are unavailable. It is not a Transformer Engine/CUDA benchmark.

The first integrated input contract is one protein chain containing the 20
standard amino acids. Modified residues, ligands, nucleic acids, complexes and
MSA inputs require separate qualification. Upstream supports broader inputs;
FireProt does not claim those paths are validated.

## Model architecture and precision

The pinned release contains an 80-layer ESMC language model, a four-layer LM
pair encoder, a 48-layer folding trunk, a two-layer final refinement stack,
diffusion and a four-layer confidence trunk. Each main pair block contains
outgoing/incoming triangle multiplication and a SwiGLU pair transition.
The four-layer MSA encoder also executes on the native query-only MSA when no
external alignment is provided; this work is preserved.

`loops` maps directly to upstream `num_loops`; this source executes **loops + 1**
main trunk passes. ESMC runs once per fold. LM pair refinement runs each pass
with the native per-loop embedding dropout: the builder's default `0.3` is
preserved explicitly. Caching one refined LM pair across passes would change
the model.

`precision: native` preserves the source's BF16 ESMC and autocast boundaries,
FP32 state and geometry operations, and native diffusion behavior. FireProt does
not add FP16/BF16 casting to diffusion. The optional path changes operand
packing inside triangle contractions only.

`sampling_steps` is the requested upstream schedule size. The pinned sampler
filters its schedule at the native sigma cap, so this field must not be reported
as the number of executed denoiser calls. The worker records the actual calls,
LM calls and main trunk passes in the sealed `operator-runtime.json`.

## Opt-in path

The Rust workload is `model: esmfold2`, `model_name: biohub/ESMFold2`, with explicit
`loops`, `sampling_steps`, `samples`, `chunk_size`, `precision` and
`deterministic`. The `native` backend invokes the pinned PyTorch implementation.
The experimental `fireprot` backend selects a guarded triangle-contraction
implementation through the Rust capability registry, retaining native fallback.

The first candidate retains the same 64-row BMM chunks and dtype while packing
the invariant right operand once per contraction. Native chunked einsum can
repeat this conversion for each output chunk. There is no cache across loops,
no reduced sampling workload and no Triton implementation.

The synthetic screen covers N600/N1000, incoming/outgoing contractions and
FP32/native BF16 autocast. All eight outputs are bitwise equal. Median complete
contraction times improve by approximately **3.6–6.3x**. These are synthetic
operator results, not trained-model or E2E results.
[Samples and conditions](../provenance/esmfold2_triangle_screen_20261015.json).

Full-model qualification uses the existing 600-residue development prefix,
loops3/four passes, requested sampling50, S1, chunk64, seed101 and deterministic
execution. It is a performance/regression case, not an independent biological
accuracy test. Promotion requires matching complete output arrays and native
CIFs, followed by repeated matched complete-invocation measurements.

**These are development workloads, not the upstream default recipe.** The
pinned `fold()` defaults to `num_loops=20` and `num_sampling_steps=200`.
Both arms of our current N600/N1000 experiment explicitly use loops3 and a
requested schedule of 50. No gain is attributed to reducing these settings;
quality or performance at the default recipe is not established by these runs.

### N600 full-model qualification

One instrumented run per arm passed supervised acceptance and exact comparison.
The optimized route completed 460 triangle dispatches. ESMC ran once, the main
trunk and LM encoder each ran four times, and diffusion executed 34 denoiser calls.

| Boundary | Native | Opt-in |
| --- | ---: | ---: |
| Model forward | 68.209 s | 29.861 s |
| Main trunk, four CUDA-event intervals summed | 48.754 s | 16.638 s |
| LM pair encoder, four intervals summed | 4.094 s | 1.381 s |
| ESMC | 0.157 s | 0.156 s |
| Diffusion denoiser intervals summed | 2.772 s | 2.765 s |

Peak allocated/reserved memory stayed at 19,724,909,568 / 33,420,214,272 bytes.
These diagnostic intervals do not sum to complete invocation time and are not
repeated E2E results. [Qualification evidence](../provenance/esmfold2_qualification_20261015.json).

## Output acceptance

Bundle schema 4 retains input bytes bound to the request, raw FP32 atom
coordinates, residue/atom pLDDT, PAE/PDE, pTM/ipTM, chain-pair confidence and token
metadata. Per-atom confidence retains the native padded dimension; a separate
boolean `atom_mask` identifies real atoms followed by padding (12 arrays total).
Rust verifies the monomer sequence, complete heavy-atom identities,
token-to-atom partition and the native writer's three-decimal CIF serialization.
Raw coordinates are not rounded before array comparisons. Existing model bundle
schemas retain their previous meaning.

## Run the opt-in pipeline

From the repository root on DiamondHill, use the frozen, qualified worker and
matching CLI. Copy the request and set a fresh `run_id` and unused `output_dir`:

```bash
cp configs/requests/esmfold2-n1000-fireprot.json /tmp/my-esmfold2.json
# Edit run_id/output_dir; retain HIP7 and the explicit development workload.
ESMFOLD2_CLI=/media/IntelSSD/fireprot/outputs/experiments/esmfold2-integration-recovery3-20261015/target/release/fireprot
"$ESMFOLD2_CLI" inspect /tmp/my-esmfold2.json
"$ESMFOLD2_CLI" infer /tmp/my-esmfold2.json configs/diamondhill/esmfold2-experimental.json
```

`configs/requests/esmfold2-n1000-native.json` selects the matching native reference.
The profile pins the existing environment inventory and fixed deployment paths;
it is not a portable installer. Recreate and verify the inventory when moving the
deployment. Rust sets `HIP_VISIBLE_DEVICES=7` from the request. The guarded
candidate currently targets N600/N1000, batch1, pair width256, chunk64, gfx90a and
the qualified ROCm Torch build; unmatched signatures retain native execution.

The current experiment snapshot is
`/media/IntelSSD/fireprot/outputs/experiments/esmfold2-integration-recovery3-20261015`.
The first snapshot without the recovery suffix is retained as a failed adapter
qualification: its forward completed, but Rust rejected a mistaken unpadded
per-atom confidence contract. That run is not an accepted performance baseline.

Native CIF validation reproduces the Biotite FP32 shortest-decimal intermediate
before three-decimal formatting. Eleven real half-boundary coordinates exposed
the earlier direct-promotion assumption; dedicated regression tests preserve
the native serialization without rounding raw output arrays. Recovery1 retains
that failed qualification, and recovery2 retains an aborted stale-binary run.
