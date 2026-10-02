# Project context

FireProt is an AMD GPU protein structure inference framework. Its architecture separates model-specific adapters from semantic operation contracts and AMD implementations. Protenix is the first integration target; AtlasFold, OpenFold, ESMFold, and other model adapters remain future work.

## Implemented state

The top-level repository now provides a Protenix benchmark harness, same-seed numerical validation, a model-neutral capability registry, process-start BLAS policy, Protenix runtime bootstrap, short-sequence gfx90a FP32 paths, and promoted long-sequence triangle, LayerNorm, and transition paths for the exact 1,000-token Mini bucket. The benchmark harness checks the selected physical GPU before launch and continuously monitors it for other processes. Unsupported signatures and unavailable extension builds fall back to native Protenix.

The primary long-sequence comparison uses FireProt code commit `14eb9f7` and independent Protenix integration commit `77218da70a9c84b101049aa0adf87cb0be461d15`, based on upstream Protenix v1.1.0 commit `d3b4db6a121dd4584edd85e93744239325a2b72e`. The historical short comparison uses FireProt `773999ae404c561309391fa72902641e5be3fb95`; incremental direct CK and channel-first LayerNorm studies use `619db00e21d942f3ff4f3d202b0b02b93bfe5695` and `47850407da303a21ec84a6327300424082ac4875`.

On continuously monitored DiamondHill MI250 gfx90a GCD 6, freshly repeated native and default FireProt Mini/1,000-token/FP32/C4/S1 model medians were 45.350 and 13.501 seconds (`3.359x`); p95 improved by `3.350x`. Pairformer improved by `3.334x`, triangle attention by `4.159x`, and triangle multiplication by `3.195x`. Peak allocated and reserved memory fell by `74.59%` and `67.46%`. Full benchmark-worker wall time improved by `2.261x`; this includes one warmup and three measured calls, not a single cold request. The final default-path comparison over 7,979 atoms passed the locked `1e-3` aligned-RMSD and confidence gates (RMSD `9.46e-5 A`, max confidence delta `7.63e-6`). The performance-only input has no experimental-coordinate mapping.

On continuously monitored DiamondHill MI250 gfx90a GCDs 6 and 7, Mini-default FP32 C4/S1 and C4/S5 warm model medians improved by 1.369x and 1.359x. Both p95 values improved, maximum peak allocated and reserved memory decreased by 10.83% and 10.59%, and same-seed coordinate/confidence checks passed the locked 1e-3 thresholds. These are development results for one 124-token case.

The measured implementation combines a hipBLAS process policy, direct short-sequence triangle multiplication, ROCm efficient triangle attention, and handwritten HIP kernels. For 1,000 tokens, memory-efficient SDPA uses outer chunks of 64 and avoids the native full attention matrix; the Protenix seam omits an implicit all-ones mask. Packed triangle multiplication normalizes once, fuses sigmoid projection with channel-first packing, uses a library BMM in each direction, and fuses unpacking with output LayerNorm. Its incremental same-source A/B improved model median from 20.865 to 13.495 seconds (`1.546x`) without increasing whole-model allocated peak memory. The extension also accelerates the measured long 128-wide LayerNorm row counts and the exact long pair transition. For 124 tokens, it provides contiguous 128-wide pair LayerNorm, exact channel-first pair LayerNorm, and the gated transition epilogue. The HIP extension builds lazily into excluded `runtime/torch_extensions` and has native fallback.

The complete observed C4/S1 matrix contains 57 FP32 BLAS signatures and 3,132 calls. Process-wide CK won five isolated signatures, was slower on the dominant 1,588-call signature, and did not support fifteen signatures. A direct CK wrapper supported 29 signatures and won only the eight-call OuterProductMean `124x64 @ 64x32` bucket. Its small isolated advantage did not improve end-to-end medians, so it requires `FIREPROT_ENABLE_CK_OPS=1` and is not promoted. Global CK is rejected for this short bucket. A separate long inventory contains 62 FP32 signatures: CK supported 48, direct CK 31, and dominant projection GEMMs were slower than hipBLAS. After packing, hipBLASLt was only 0.5% faster in a three-sample model comparison and used about 78 MB more allocated memory; the reported primary policy remains hipBLAS. The fused packing kernels are handwritten HIP, not CK library calls.

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

`scripts/infer.sh` uses Mini-ESM, four cycles, one diffusion step, one sample, FP32, seed 101, no MSA/template search, TF32 disabled, and torch LayerNorm before FireProt hooks are installed. `FIREPROT_BACKEND=native` selects upstream torch paths. `FIREPROT_BACKEND=fireprot` installs the registry and runtime hooks. `FIREPROT_BLAS_BACKEND=hipblas` is the fixed policy in the 1,000-token comparison, but only the 124-token all-call matrix establishes it as the best measured process-wide option. `FIREPROT_DISABLE_LONG_TRIANGLE_PACKED=1` returns to the previous chunked gate path. `FIREPROT_DISABLE_LONG_TRIANGLE_GATES=1` disables both long multiplication paths. Other diagnostic switches are `FIREPROT_DISABLE_LONG_TRIANGLE_ATTENTION=1`, `FIREPROT_DISABLE_LONG_LAYERNORM=1`, and `FIREPROT_DISABLE_LONG_TRANSITION=1`. `FIREPROT_DISABLE_CHANNEL_FIRST_LAYERNORM=1` disables the short exact-stride kernel. Direct CK requires `FIREPROT_ENABLE_CK_OPS=1`.

Recycle-cut measurements produced C1/C2/C3/C4 medians of 123.86, 189.20, 259.57, and 334.48 ms. C1/C2/C3 differed from C4 by 5.760, 2.691, and 4.221 Angstrom aligned RMSD. The result is not monotonic on the single development example, so C4 remains the default and lower cycle counts are explicit approximate requests.

## Next measurement

Profile the remaining 7.085-second triangle-attention stage first; it now dominates the 13.501-second model. Measure SDPA, projections and layouts before selecting the next CK or HIP kernel. The packed triangle stage is 3.914 seconds. Extend the workspace policy and validation to multiple real 1,000+ token inputs, larger length buckets, Base weights, complexes, adapters, and production diffusion-step counts. Build a recycle early-exit policy only after an independent quality set and convergence gate exist. No additional optimization should be promoted without operator and end-to-end numerical, p95, and peak-memory gates.

## Latest evidence

- Packed path: `docs/protenix-packed-triangle.md` and `provenance/protenix_amd_long1000_14eb9f7.json`.
- Long BLAS inventory: `provenance/protenix_amd_long1000_blas_a8bbfef.json`; contiguous dimension replay does not preserve real strides.
- Reproducible operator benchmark: `python -m benchmarks.protenix.benchmark_triangle` with explicit GPU 6 or 7.
- Final tests: local 94 passed / 11 skipped; DiamondHill GPU 7 104 passed / 1 skipped.
