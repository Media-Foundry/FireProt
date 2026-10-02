# FireProt

FireProt is an AMD GPU focused framework for protein structure inference. It is designed around model adapters and capability-gated operator backends so Protenix, AtlasFold, OpenFold, ESMFold, and later models can preserve their own scientific semantics while sharing reproducible execution, validation, and AMD optimization infrastructure.

## Current status

Protenix v1.1.0 is the first working adapter target. The repository now contains:

- a reproducible Protenix MI250 benchmark harness with synchronized GPU stage timing, rocprof import, source and input hashes, exclusive-device checks, and per-run peak memory;
- same-seed coordinate and confidence validation;
- a model-neutral operator registry with deterministic capability checks and native fallback;
- process-start BLAS selection for PyTorch ROCm;
- validated FP32 gfx90a paths for short-sequence Protenix triangle multiplication and triangle attention;
- an independent Protenix patch branch that exposes FireProt dispatch without changing native defaults.

AtlasFold, OpenFold, ESMFold, Base-model coverage, BF16 promotion, long-sequence tuning, and multi-GPU execution remain future work.

## Verified MI250 result

The current promoted bucket is Protenix Mini-default, 124 tokens, FP32, four cycles, seed 101, one sample, warm in-process caches, no MSA/template search, and either one or five diffusion steps. Each latency result used five warmups and thirty measured repetitions on one exclusively checked gfx90a GCD.

| Workload | Native median | FireProt median | Median speedup | Native p95 | FireProt p95 | p95 speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C4/S1 | 453.53 ms | 373.94 ms | 1.213x | 457.51 ms | 406.06 ms | 1.127x |
| C4/S5 | 518.43 ms | 437.86 ms | 1.184x | 541.45 ms | 468.27 ms | 1.156x |

Maximum measured PyTorch peak allocated memory fell from 850,018,304 to 837,647,360 bytes. Maximum reserved memory fell from 1,029,701,632 to 1,000,341,504 bytes. Same-seed Mini-default C4/S1 and C4/S5 comparisons passed the locked 1e-3 coordinate and confidence thresholds.

These are development-case results for one 124-token input and do not establish Base-model, long-sequence, complex, ligand-heavy, or production-dataset speedups. See [Protenix AMD performance](docs/protenix-amd-performance.md) for the complete method, stage results, numerical checks, residual profile, and limits.

## Run on DiamondHill

The default script uses the Mini-ESM baseline, FP32, four cycles, one diffusion step, one sample, and seed 101. Runtime weights and CCD resources must already exist under `runtime/`.

```bash
# Native Protenix reference
FIREPROT_BACKEND=native \
  bash scripts/infer.sh \
  --input /media/IntelSSD/fireprot/cases/1izr_L124/input.json \
  --out_dir /media/IntelSSD/fireprot/outputs/native

# Capability-gated AMD paths with automatic native fallback
FIREPROT_BACKEND=fireprot FIREPROT_BLAS_BACKEND=auto \
  bash scripts/infer.sh \
  --input /media/IntelSSD/fireprot/cases/1izr_L124/input.json \
  --out_dir /media/IntelSSD/fireprot/outputs/fireprot
```

`FIREPROT_BLAS_BACKEND` accepts `auto`, `hipblaslt`, `hipblas`, or `ck`. It is fixed once at process start. The promoted operator implementations only select the validated gfx90a/FP32/inference signatures; unsupported architectures, dtypes, shapes, layouts, masks, training/autograd calls, and aliasing modes use the upstream torch path.

## Benchmark

```bash
python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250.yaml \
  --backend native

python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250.yaml \
  --backend fireprot

python benchmarks/protenix/benchmark_blas.py \
  --config benchmarks/protenix/configs/mi250.yaml
```

Raw predictions, profiler traces, weights, and hardware-local tuning data remain under excluded `outputs/` and `runtime/` paths. The compact measured record is [provenance/protenix_amd_benchmark.json](provenance/protenix_amd_benchmark.json).

## Repository layout

| Path | Purpose |
| --- | --- |
| `fireprot/ops/` | Model-neutral operation signatures, dispatch, selection cache, reporting, and native fallback |
| `fireprot/backends/amd/` | AMD BLAS policy and validated operator implementations |
| `fireprot/adapters/protenix/` | Protenix runtime bootstrap and backend registration |
| `fireprot/validation/` | Coordinate, confidence, and ground-truth helpers |
| `benchmarks/protenix/` | Fixed workload configuration, inference profiler, BLAS matrix, and output comparison |
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
- [Implementation plan](docs/plans/2026-10-02-gitnexus-plan-protenix-amd-acceleration.md)
- [Historical Protenix baseline notes](.agents/protenix_baseline.md) (Chinese, retained as recorded)
- [Project context](.agents/project_context.md)

The top-level remote is `https://github.com/Media-Foundry/FireProt.git`. `Protenix/` retains its independent history and is neither a top-level submodule nor a gitlink.
