# Project context

FireProt aims to be an AMD GPU protein structure inference framework. It may support Protenix, AtlasFold, OpenFold, ESMFold, and other models, with operator implementations based on CK, hipBLASLt, or other AMD technologies where measurements justify them. At present, only the public Protenix baseline runs. The shared framework, other adapters, and accelerated operators are not implemented. See `docs/architecture.md` for the proposed design and `.agents/protenix_baseline.md` for historical run details.

## Current paths

| Path | Role |
| --- | --- |
| `Protenix/` | Independent upstream Git repository and first model baseline |
| `cases/` | Eight 124–164 residue single-chain development cases |
| `scripts/` | Inference, synchronization, and smoke-test scripts |
| `provenance/` | Source, asset, case-hash, and smoke-test records |
| `outputs/` | Local smoke predictions, excluded from top-level Git |
| Remote `runtime/` | Weights, ESM, and CCD assets on DiamondHill |
| `docs/` | Proposed framework architecture |

## Run baseline

`scripts/infer.sh` defaults to Mini-ESM, four cycles, one diffusion step, one sample, FP32, seed 101, no external MSA/template search, and PyTorch triangle kernels. Its default Python is `/home/pc/anaconda3/envs/fold/bin/python` on DiamondHill. The local machine lacks the required weights.

On 2026-10-02, smoke runs on `1izr_L124` completed for Mini-ESM, Mini-default, and Tiny-default. Each prediction had 945 atoms with finite coordinates. The 68-second figure is wall time for concurrent runs, not single-model latency or a speedup.

## Next measurement

Establish repeatable baseline timings for load, preprocessing/ESM, Pairformer, diffusion, and end-to-end execution. Separate cold and warm cache runs. Accuracy comparisons must account for the native CLI sampler, augmentation, and dropout settings, and score only experimentally observed ground-truth atoms. Then implement the Protenix adapter and a first AMD operator selected by profiling.
