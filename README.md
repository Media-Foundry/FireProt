# FireProt

FireProt is an AMD GPU focused framework for protein structure inference. It is designed around model adapters and capability-gated operator backends so Protenix, AtlasFold, OpenFold, ESMFold, and later models can preserve their own scientific semantics while sharing reproducible execution, validation, and AMD optimization infrastructure.

## Current status

Protenix v1.1.0 is the first working adapter target. The repository now contains:

- a reproducible Protenix MI250 benchmark harness with synchronized GPU stage timing, rocprof import, source and input hashes, continuous GPU process isolation checks, and per-run peak memory;
- same-seed coordinate and confidence validation;
- a model-neutral operator registry with deterministic capability checks and native fallback;
- measured process-start BLAS selection for PyTorch ROCm and a matrix covering every observed GEMM signature;
- validated FP32 gfx90a paths for short-sequence Protenix triangle multiplication, triangle attention, contiguous pair LayerNorm, channel-first pair LayerNorm, and the gated transition epilogue;
- validated FP32 gfx90a long-sequence paths for the exact 1,000-token bucket, including memory-efficient triangle attention, packed triangle contraction with fused HIP layouts, pair LayerNorm, and gated transitions;
- lazily compiled handwritten HIP kernels with exact capability gates and native fallback;
- a direct CK GEMM candidate for observed Protenix signatures, kept behind an experimental opt-in when end-to-end measurements do not justify promotion;
- an independent Protenix patch branch that exposes FireProt dispatch without changing native defaults.

AtlasFold, OpenFold, ESMFold, Base-model coverage, BF16 promotion, additional long-sequence buckets, and multi-GPU execution remain future work.

## Verified 1,000-token MI250 result

The latest attention refinement measures **13.124 s** model median, with unchanged peak allocated memory. Same-source A/B runs confirm a further **1.026–1.028x** improvement over the packed-triangle result below. The tuned CK attention candidate improved from **45.641 to 36.139 ms**, close to SDPA's **36.00–36.02 ms**, and remains experimental. See [SDPA/CK investigation and model validation](docs/protenix-attention-tuning.md).

### Packed triangle comparison

The primary optimization target is long-sequence folding. The measured workload is Protenix Mini-default, a real 1,000-residue protein sequence, FP32, four cycles, one diffusion step, seed 101, one sample, warm in-process caches, and no MSA/template search. Native and FireProt used the same physical gfx90a GCD, hipBLAS process policy, one warmup, and three measured repetitions. Continuous process monitoring found no external GPU PID.

| Metric | Native median | FireProt median | Speedup |
| --- | ---: | ---: | ---: |
| Model | 45.350 s | 13.501 s | 3.359x |
| Pairformer/recycle trunk | 41.530 s | 12.456 s | 3.334x |
| Triangle attention | 29.468 s | 7.085 s | 4.159x |
| Triangle multiplication | 12.505 s | 3.914 s | 3.195x |

Model p95 improved from 45.416 s to 13.556 s (`3.350x`). Peak allocated memory fell from 39,499,758,592 to 10,037,256,704 bytes (`74.59%`), and peak reserved memory fell from 42,756,734,976 to 13,912,506,368 bytes (`67.46%`). Across warmup and measured calls, the accelerated run selected long triangle attention and packed triangle multiplication 576 times each. These GPU model times exclude loading, preprocessing, and serialization. The complete benchmark worker, including one warmup and three measured calls, improved from 226.715 to 100.260 seconds (`2.261x`); this is not a single-request cold-start measurement.

The same-seed validation covered 7,979 output atoms and passed the locked `1e-3` coordinate and confidence gates: aligned coordinate RMSD was `9.46e-5 A`, maximum coordinate delta was `9.77e-4 A`, and maximum confidence delta was `7.63e-6`. The input is a performance workload without an experimental-coordinate mapping, so these checks establish agreement with native Protenix rather than structural accuracy.

The long attention path prevents materialization of the native `[1000,4,1000,1000]` attention matrix by using ROCm memory-efficient SDPA. The comparison above used outer chunks; the latest exact unmasked path uses one call with head-last output storage to remove chunk assignments and layout copies. When the pair mask is implicit, the Protenix adapter also omits the all-ones mask and its zero additive bias. The packed triangle path normalizes once, fuses sigmoid projection with channel-first packing, contracts through library BMM, then fuses unpacking with output LayerNorm. The preceding chunked paths remain available as diagnostic fallbacks. A wave64 HIP LayerNorm covers the measured 244,000-, 256,000-, and 1,000,000-row contiguous tensors, while the exact `[1000,1000,128]` transition fuses SiLU with its gate multiply. Exact capability predicates preserve native fallback for other lengths, dtypes, layouts, masks, architectures, training, and autograd. See [packed triangle design and results](docs/protenix-packed-triangle.md).

## Historical 124-token MI250 result

The primary comparison bucket is Protenix Mini-default, 124 tokens, FP32, four cycles, seed 101, one sample, warm in-process caches, no MSA/template search, and either one or five diffusion steps. Each latency result used five warmups and thirty measured repetitions on one exclusively checked gfx90a GCD. Incremental operator studies are reported below and are not folded into this historical native/FireProt table.

| Workload | Native median | FireProt median | Median speedup | Native p95 | FireProt p95 | p95 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C4/S1 | 457.86 ms | 334.48 ms | 1.369x | 475.15 ms | 351.86 ms | 1.350x |
| C4/S5 | 530.89 ms | 390.77 ms | 1.359x | 564.35 ms | 407.77 ms | 1.384x |

Maximum measured PyTorch peak allocated memory fell from 850,018,304 to 757,955,584 bytes. Maximum reserved memory fell from 1,029,701,632 to 920,649,728 bytes. Same-seed Mini-default C4/S1 and C4/S5 comparisons passed the locked 1e-3 coordinate and confidence thresholds.

The observed BLAS matrix covers 57 FP32 `mm`, `bmm`, and `addmm` signatures and 3,132 calls. hipBLAS won the dominant 1,588-call signature and is the validated process policy for this exact bucket. CK won five isolated signatures, was slower on the dominant signature, and did not support 15 signatures, so global CK is not promoted. CK and custom HIP remain per-operation candidates.

A later direct CK matrix found one winner among the 57 signatures: the eight OuterProductMean `124x64 @ 64x32` calls. Direct CK measured 26.39 us versus 30.22 us for hipBLAS, but formal C4/S1 and C4/S5 runs did not improve end-to-end medians. That route therefore requires `FIREPROT_ENABLE_CK_OPS=1` and is not a default optimization.

The promoted channel-first LayerNorm kernel targets the 144 triangle-multiplication outputs with exact shape `[124,124,128]` and stride `[124,1,15376]`. It fuses layout conversion, normalization, scale, and bias. The operator median improved from 117.48 to 82.74 us (`1.420x`) with a maximum absolute error of `2.86e-6`. A 30-repetition C4/S1 A/B saved 6.10 ms at the model median (`1.019x`) and improved p95 by `1.024x`. Two C4/S5 orders produced mixed whole-model results, so no stable incremental C4/S5 speedup is claimed.

Reducing four recycle cycles to one, two, or three measured 2.700x, 1.768x, and 1.289x speedups over C4. Those outputs differed from C4 by 5.760, 2.691, and 4.221 A aligned RMSD on this development case. Cycle reduction therefore remains an explicit approximate choice; the default stays at four cycles.

These are development-case results for one 124-token input and do not establish Base-model, complex, ligand-heavy, or production-dataset speedups. See [Protenix AMD performance](docs/protenix-amd-performance.md) for the complete method, stage results, numerical checks, residual profile, and limits.

## Run on DiamondHill

The default script uses the Mini-ESM baseline, FP32, four cycles, one diffusion step, one sample, and seed 101. Runtime weights and CCD resources must already exist under `runtime/`.

```bash
# Native Protenix reference
FIREPROT_BACKEND=native \
  bash scripts/infer.sh \
  --model_name protenix_mini_default_v0.5.0 \
  --input /media/IntelSSD/fireprot/cases/1izr_L124/input.json \
  --out_dir /media/IntelSSD/fireprot/outputs/native

# Validated AMD path with automatic operator fallback
FIREPROT_BACKEND=fireprot FIREPROT_BLAS_BACKEND=hipblas \
  bash scripts/infer.sh \
  --model_name protenix_mini_default_v0.5.0 \
  --input /media/IntelSSD/fireprot/cases/1izr_L124/input.json \
  --out_dir /media/IntelSSD/fireprot/outputs/fireprot
```

`FIREPROT_BLAS_BACKEND` accepts `auto`, `hipblaslt`, `hipblas`, or `ck`. It is fixed once at process start. `hipblas` is validated only for the measured Mini/124-token/FP32 bucket; use `auto` for unmeasured buckets until they have their own matrix and end-to-end gate. The promoted operator implementations select only validated gfx90a/FP32/inference signatures. Unsupported architectures, dtypes, shapes, layouts, masks, training/autograd calls, aliasing modes, or extension builds use the upstream torch path.

The exact channel-first LayerNorm path is enabled by the FireProt backend and can be disabled for A/B diagnosis with `FIREPROT_DISABLE_CHANNEL_FIRST_LAYERNORM=1`. The direct CK OuterProductMean candidate is experimental and requires `FIREPROT_ENABLE_CK_OPS=1`; `FIREPROT_DISABLE_CK_OPS=1` overrides that opt-in.

The validated 1,000-token paths are enabled by default after their end-to-end and numerical gates. Use `FIREPROT_DISABLE_LONG_TRIANGLE_PACKED=1` to return to the preceding chunked gate path. `FIREPROT_DISABLE_LONG_TRIANGLE_GATES=1` disables both long multiplication implementations. Other diagnostic switches are `FIREPROT_DISABLE_LONG_TRIANGLE_ATTENTION=1`, `FIREPROT_DISABLE_LONG_LAYERNORM=1`, and `FIREPROT_DISABLE_LONG_TRANSITION=1`. Other long lengths fall back until measured and promoted.

`FIREPROT_DISABLE_LONG_DIRECT_ATTENTION=1` restores outer-chunked SDPA for the exact unmasked 1,000-token bucket. The separate CK attention probe is invoked with `python -m benchmarks.protenix.benchmark_attention --include-ck --ck-set tuned --output outputs/benchmarks/attention.json`; it is not selected by model dispatch. Its version-pinned header overlay leaves installed ROCm headers unchanged.

## Benchmark

```bash
HIP_VISIBLE_DEVICES=6 python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250.yaml \
  --backend native

HIP_VISIBLE_DEVICES=6 python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250.yaml \
  --backend fireprot --blas-backend hipblas

HIP_VISIBLE_DEVICES=6 PYTHONPATH=. python -m benchmarks.protenix.benchmark_blas \
  --config benchmarks/protenix/configs/mi250.yaml \
  --operators-json outputs/benchmarks/<run>/torch-operators.json \
  --output-dir outputs/benchmarks/blas-all-calls

# Primary 1,000-token native/FireProt comparison
HIP_VISIBLE_DEVICES=6 python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250-long.json \
  --backend native --blas-backend hipblas

HIP_VISIBLE_DEVICES=6 python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250-long.json \
  --backend fireprot --blas-backend hipblas
```

Raw predictions, profiler traces, weights, and hardware-local tuning data remain under excluded `outputs/` and `runtime/` paths. Compact records include [the current packed-triangle result](provenance/protenix_amd_long1000_14eb9f7.json), [all 62 long BLAS signatures](provenance/protenix_amd_long1000_blas_a8bbfef.json), [the preceding long result](provenance/protenix_amd_long1000_1a95f90.json), [the original long attention result](provenance/protenix_amd_long1000_523f922.json), and [the historical short benchmark](provenance/protenix_amd_benchmark.json).

## Repository layout

| Path | Purpose |
| --- | --- |
| `fireprot/ops/` | Model-neutral operation signatures, dispatch, selection cache, reporting, and native fallback |
| `fireprot/backends/amd/` | AMD BLAS policy and validated operator implementations |
| `fireprot/adapters/protenix/` | Protenix runtime bootstrap and backend registration |
| `fireprot/validation/` | Coordinate, confidence, and ground-truth helpers |
| `benchmarks/protenix/` | Fixed short and 1,000-token workload configurations, inference profiler, BLAS matrix, and output comparison |
| `tests/` | Registry, backend, numerical, adapter, and profiler tests |
| `Protenix/` | Independent upstream Git checkout with FireProt integration commits |
| `patches/protenix/` | Exported patches that reproduce the independent Protenix integration branch |
| `cases/` | Eight short single-chain development cases; not an independent test set |
| `scripts/` | Inference and DiamondHill synchronization entry points |
| `provenance/` | Compact source, runtime, asset, smoke, and performance records |
| `.agents/` | Historical baseline notes and project context |
| `docs/` | Architecture, implementation plan, and measured performance report |

## Documentation

- [Architecture](docs/architecture.md)
- [Measured Protenix AMD performance](docs/protenix-amd-performance.md)
- [Long attention: SDPA, CK tuning, and incremental model results](docs/protenix-attention-tuning.md)
- [Implementation plan](docs/plans/2026-10-02-gitnexus-plan-protenix-amd-acceleration.md)
- [Historical Protenix baseline notes](.agents/protenix_baseline.md) (Chinese, retained as recorded)
- [Project context](.agents/project_context.md)

The top-level remote is `https://github.com/Media-Foundry/FireProt.git`. `Protenix/` retains its independent history and is neither a top-level submodule nor a gitlink.
