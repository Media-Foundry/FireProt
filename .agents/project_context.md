# Project context

FireProt is an AMD GPU protein structure inference framework. Its architecture separates model-specific adapters from semantic operation contracts and AMD implementations. Protenix is the first integration target; AtlasFold, OpenFold, ESMFold, and other model adapters remain future work.

## Implemented state

The top-level repository now provides a Protenix benchmark harness, same-seed numerical validation, a model-neutral capability registry, process-start BLAS policy, Protenix runtime bootstrap, and two promoted gfx90a FP32 inference paths for the measured 124-token Mini bucket. Unsupported signatures fall back to the native Protenix implementation.

Final measured sources are FireProt `d31013ac643bdb7e033dd378795d216e29ecf448` and independent Protenix integration commit `316f5778e80965015a80339de632291f1abbb7f5`, based on upstream Protenix v1.1.0 commit `d3b4db6a121dd4584edd85e93744239325a2b72e`.

On one exclusively checked DiamondHill MI250 gfx90a GCD, Mini-default FP32 C4/S1 and C4/S5 warm model medians improved by 1.213x and 1.184x. Both p95 values improved, maximum peak allocated and reserved memory decreased, and same-seed coordinate/confidence checks passed the locked 1e-3 thresholds. These are development results for one 124-token case.

## Current paths

| Path | Role |
| --- | --- |
| `fireprot/ops/` | Model-neutral operation registry and capability signatures |
| `fireprot/backends/amd/` | BLAS policy and promoted AMD operator paths |
| `fireprot/adapters/protenix/` | FireProt bootstrap for the independent Protenix checkout |
| `fireprot/validation/` | Coordinate, confidence, and ground-truth validation |
| `benchmarks/protenix/` | Repeated inference, rocprof import, BLAS matrix, and output comparison |
| `Protenix/` | Independent upstream repository with local FireProt integration commits |
| `cases/` | Eight short single-chain development cases |
| `runtime/` | DiamondHill-only weights and CCD assets, excluded from Git |
| `outputs/` | Predictions, manifests, and profiler traces, excluded from Git |
| `provenance/` | Compact versioned evidence records |

## Runtime defaults

`scripts/infer.sh` uses Mini-ESM, four cycles, one diffusion step, one sample, FP32, seed 101, no MSA/template search, TF32 disabled, and torch LayerNorm. `FIREPROT_BACKEND=native` selects upstream torch triangle kernels. `FIREPROT_BACKEND=fireprot` installs the registry and selects FireProt triangle multiplication and attention only when capability predicates match.

## Next measurement

Extend the matrix to untouched evaluation inputs, multiple token-length buckets, Base weights, complexes, ligands, and production diffusion-step counts. The final residual profile ranks GEMM, elementwise work, layout copies, and LayerNorm as the next categories, but no additional optimization should be promoted without operator and end-to-end numerical, p95, and peak-memory gates.
