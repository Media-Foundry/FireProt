# Project context

FireProt is an AMD GPU protein structure inference framework. Its architecture separates model-specific adapters from semantic operation contracts and AMD implementations. Protenix is the first integration target; AtlasFold, OpenFold, ESMFold, and other model adapters remain future work.

## Implemented state

The top-level repository now provides a Protenix benchmark harness, same-seed numerical validation, a model-neutral capability registry, process-start BLAS policy, Protenix runtime bootstrap, short-sequence gfx90a FP32 paths, and promoted long-sequence triangle paths for the exact 1,000-token Mini bucket. The benchmark harness checks the selected physical GPU before launch and continuously monitors it for other processes. Unsupported signatures and unavailable extension builds fall back to native Protenix.

The primary long-sequence comparison uses FireProt `523f922` and independent Protenix integration commit `77218da70a9c84b101049aa0adf87cb0be461d15`, based on upstream Protenix v1.1.0 commit `d3b4db6a121dd4584edd85e93744239325a2b72e`. The historical short comparison uses FireProt `773999ae404c561309391fa72902641e5be3fb95`; incremental direct CK and channel-first LayerNorm studies use `619db00e21d942f3ff4f3d202b0b02b93bfe5695` and `47850407da303a21ec84a6327300424082ac4875`.

On a continuously monitored DiamondHill MI250 gfx90a GCD 6, the exact 1,000-token Mini-default FP32 C4/S1 model median improved from 45.310 to 22.977 seconds (`1.972x`) and p95 improved by `1.988x`. The Pairformer/recycle trunk improved by `1.939x`; triangle attention improved by `3.839x`; triangle multiplication improved by `1.046x`. Peak allocated and reserved memory fell by `73.29%` and `65.65%`. A same-seed comparison across 7,979 atoms passed the locked `1e-3` coordinate and confidence gates. This performance-only input has no experimental-coordinate mapping, so the comparison establishes native agreement rather than structure accuracy.

On continuously monitored DiamondHill MI250 gfx90a GCDs 6 and 7, Mini-default FP32 C4/S1 and C4/S5 warm model medians improved by 1.369x and 1.359x. Both p95 values improved, maximum peak allocated and reserved memory decreased by 10.83% and 10.59%, and same-seed coordinate/confidence checks passed the locked 1e-3 thresholds. These are development results for one 124-token case.

The measured implementation combines a hipBLAS process policy, direct short-sequence triangle multiplication, ROCm efficient triangle attention, and handwritten HIP kernels. For 1,000 tokens, memory-efficient SDPA uses outer chunks of 64 and avoids the native full attention matrix; the Protenix seam omits an implicit all-ones mask, and HIP kernels fuse gates around the chunked triangle multiplication. For 124 tokens, the extension also provides contiguous 128-wide pair LayerNorm, exact channel-first pair LayerNorm, and the gated transition epilogue. The HIP extension builds lazily into excluded `runtime/torch_extensions` and has native fallback.

The complete observed C4/S1 matrix contains 57 FP32 BLAS signatures and 3,132 calls. Process-wide CK won five isolated signatures, was slower on the dominant 1,588-call signature, and did not support fifteen signatures. A direct CK wrapper supported 29 signatures and won only the eight-call OuterProductMean `124x64 @ 64x32` bucket. Its small isolated advantage did not improve end-to-end medians, so it requires `FIREPROT_ENABLE_CK_OPS=1` and is not promoted. Global CK is rejected for this bucket.

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

`scripts/infer.sh` uses Mini-ESM, four cycles, one diffusion step, one sample, FP32, seed 101, no MSA/template search, TF32 disabled, and torch LayerNorm before FireProt hooks are installed. `FIREPROT_BACKEND=native` selects upstream torch paths. `FIREPROT_BACKEND=fireprot` installs the registry and runtime hooks. `FIREPROT_BLAS_BACKEND=hipblas` is the fixed policy in the 1,000-token comparison, but only the 124-token all-call matrix establishes it as the best measured process-wide option. `FIREPROT_DISABLE_LONG_TRIANGLE_ATTENTION=1` and `FIREPROT_DISABLE_LONG_TRIANGLE_GATES=1` disable the exact 1,000-token paths for diagnosis. `FIREPROT_DISABLE_CHANNEL_FIRST_LAYERNORM=1` disables the short exact-stride kernel. Direct CK requires `FIREPROT_ENABLE_CK_OPS=1`.

Recycle-cut measurements produced C1/C2/C3/C4 medians of 123.86, 189.20, 259.57, and 334.48 ms. C1/C2/C3 differed from C4 by 5.760, 2.691, and 4.221 Angstrom aligned RMSD. The result is not monotonic on the single development example, so C4 remains the default and lower cycle counts are explicit approximate requests.

## Next measurement

Optimize the remaining 11.915-second long triangle-multiplication stage first. Benchmark its observed chunked BMM, layout-copy, and 128-wide normalization signatures with CK, hipBLAS, hipBLASLt, and targeted HIP fusion. Then extend measurements to multiple 1,000+ token inputs, length buckets, Base weights, complexes, other model adapters, and production diffusion-step counts. Build a recycle early-exit policy only after an independent quality set and convergence gate exist. No additional optimization should be promoted without operator and end-to-end numerical, p95, and peak-memory gates.
