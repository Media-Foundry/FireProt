# Project context

FireProt is an AMD GPU protein structure inference framework. Its architecture separates model-specific adapters from semantic operation contracts and AMD implementations. Protenix is the first integration target; AtlasFold, OpenFold, ESMFold, and other model adapters remain future work.

## Implemented state

The top-level repository now provides a Protenix benchmark harness, same-seed numerical validation, a model-neutral capability registry, process-start BLAS policy, Protenix runtime bootstrap, and four promoted gfx90a FP32 inference paths for the measured 124-token Mini bucket. The benchmark harness checks the selected physical GPU before launch and continuously monitors it for other processes. Unsupported signatures and unavailable extension builds fall back to native Protenix.

Final measured sources are FireProt `773999ae404c561309391fa72902641e5be3fb95` and independent Protenix integration commit `316f5778e80965015a80339de632291f1abbb7f5`, based on upstream Protenix v1.1.0 commit `d3b4db6a121dd4584edd85e93744239325a2b72e`.

On continuously monitored DiamondHill MI250 gfx90a GCDs 6 and 7, Mini-default FP32 C4/S1 and C4/S5 warm model medians improved by 1.369x and 1.359x. Both p95 values improved, maximum peak allocated and reserved memory decreased by 10.83% and 10.59%, and same-seed coordinate/confidence checks passed the locked 1e-3 thresholds. These are development results for one 124-token case.

The measured implementation combines a hipBLAS process policy, direct short-sequence triangle multiplication, ROCm efficient triangle attention, and handwritten HIP kernels for 128-wide pair LayerNorm and the gated transition epilogue. The HIP extension builds lazily into excluded `runtime/torch_extensions` and has native fallback.

The complete observed C4/S1 matrix contains 57 FP32 BLAS signatures and 3,132 calls. CK won five isolated signatures, was slower on the dominant 1,588-call signature, and did not support fifteen signatures. Global CK is rejected for this bucket. Per-signature CK or HIP remains available when an end-to-end gate supports it.

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

`scripts/infer.sh` uses Mini-ESM, four cycles, one diffusion step, one sample, FP32, seed 101, no MSA/template search, TF32 disabled, and torch LayerNorm before FireProt hooks are installed. `FIREPROT_BACKEND=native` selects upstream torch paths. `FIREPROT_BACKEND=fireprot` installs the registry and runtime hooks. `FIREPROT_BLAS_BACKEND=hipblas` is validated only for the measured Mini/124-token/FP32 bucket; other buckets retain `auto` until measured.

Recycle-cut measurements produced C1/C2/C3/C4 medians of 123.86, 189.20, 259.57, and 334.48 ms. C1/C2/C3 differed from C4 by 5.760, 2.691, and 4.221 Angstrom aligned RMSD. The result is not monotonic on the single development example, so C4 remains the default and lower cycle counts are explicit approximate requests.

## Next measurement

Extend the matrix to untouched evaluation inputs, multiple token-length buckets, Base weights, complexes, ligands, and production diffusion-step counts. Revisit the triangle-contraction BMM that regresses under global hipBLAS, then layout copies and residual epilogues. Build a recycle early-exit policy only after an independent quality set and convergence gate exist. No additional optimization should be promoted without operator and end-to-end numerical, p95, and peak-memory gates.
