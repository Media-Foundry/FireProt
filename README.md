# FireProt

An AMD GPU protein structure inference framework with model adapters and
capability-gated HIP/CK operators. The primary performance target is long proteins,
including the measured 1,000- and 1,536-residue workloads.

**Status: Rust framework migration in progress.** Rust owns request contracts,
operator selection, process supervision, GPU isolation, provenance preflight and
output validation. Native build/cache services now own maintained Torch and standalone HIP loaders.
All six maintained loaders support pinned Python/C ABI artifacts. Repository
layout, packaging and some model parity gates are unfinished. Rust itself is not claimed to accelerate GPU kernels.

## Model coverage

| Model | Connected workflow | Verified scope / limits |
| --- | --- | --- |
| Protenix | Mini-default FP32, native and FireProt CK/HIP paths | N1000/N1536 pass established coordinate and summary-confidence gates. Full raw confidence arrays are retained but are not bitwise identical. Other variants remain pending. |
| AtlasFold + AtlasLM-3B | Monomer, original dual weights, native/chunked/HIP/fused HIP | N1000 native and N1000/N1536 accelerated routes pass exact public-output parity. N1536 reference uses bounded native math. |
| OpenFold3 / OpenBind-0 | Native inference under Rust supervision | N1000 runs successfully; numerical migration parity remains unpassed, including observed native-repeat variation. No accelerated route promoted. |
| OpenFold2, ESMFold, other adapters | Planned | No verified integration claim. |

Protenix and OpenFold3 are separate implementations. Model adapters preserve each
model's own preprocessing, precision boundaries and cycle semantics. Recycle cuts
are changes to inference work and require an explicit, separately validated policy.

## Build and inspect

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
target/release/fireprot inspect configs/requests/atlasfold-n1000-native.json
cargo test --workspace --locked
```

Request inspection is CPU-only and does not open weights. Model execution requires
the pinned upstream checkout, model-specific Python/Torch ROCm environment,
checkpoints, HIP identity probe and AMD SMI. The checked-in deployment profiles
reference existing DiamondHill paths; they are not portable installation scripts.
The bridge must be built explicitly before importing the shared Python operator API.

## Run on DiamondHill

From `/media/IntelSSD/fireprot`, after building the CLI/bridge and preparing the
[HIP probe and worker environment](docs/worker-protocol.md):

```bash
target/release/fireprot infer configs/requests/atlasfold-n1000-native.json \
  configs/diamondhill/atlasfold-native.json
target/release/fireprot infer configs/requests/protenix-n1000-fireprot.json \
  configs/diamondhill/protenix-fireprot.json
target/release/fireprot compare-npz reference.npz candidate.npz
```

Use a fresh run ID and output directory for each request. Requests pin source,
weights, input, precision, cycles/recycles, steps, sample count, seed and device.
Rust verifies artifacts and maps HIP devices to AMD SMI through PCI identity before
accepting continuously monitored inference. Failure or missing outputs cannot
produce a successful run report.

See [commands, output schemas and exact model scope](docs/rust-running.md) and
[checkpoint locations](docs/diamondhill-checkpoints.md). Existing `scripts/infer.sh`
and historical benchmark tools remain available during migration; they have not
all been replaced by the Rust CLI.

## Repository structure

| Path | Responsibility |
| --- | --- |
| `crates/fireprot-core` | Requests, workload semantics, protocol and operator registry |
| `crates/fireprot-runtime` | Preflight, worker lifecycle, device monitoring, native build/cache services |
| `crates/fireprot-validation` | Output acceptance, array comparison and numerical metrics |
| `crates/fireprot-cli` | Rust command entry point |
| `crates/fireprot-bridge` | In-process CPython boundary for Rust policy and build services |
| `python/fireprot_workers` | Explicit upstream model invocation and tensor serialization |
| `fireprot/adapters`, `fireprot/backends/amd` | Current tensor glue and HIP/CK implementations; layout migration pending |
| `configs`, `tests`, `provenance` | Explicit workloads, validation and measured evidence |
| `docs`, `.agents` | Architecture, guides, migration decisions and historical context |
| `Protenix/` | Independent upstream Git checkout; never a top-level submodule |
| `runtime/`, `outputs/` | Local weights/build caches and generated predictions; excluded from Git |

Weights and predictions are not synchronized or committed with framework source.
Review destinations and dry-run changes before using the deployment sync script.

## Correctness and performance evidence

Comparisons must retain model, weights, input, dtype, cycle/step/sample counts,
random state and cache conditions. Report model time and complete-request time
separately. Migration checks run concurrently on separate GCDs do not establish
new E2E speedups. The eight short `cases/` examples are development cases; long
workloads without experimental coordinates establish native agreement, not
structural accuracy.

- [Rust migration acceptance and remaining work](docs/rust-migration.md)
- [Current architecture and dependency boundaries](docs/architecture.md)
- [Operator policy and bridge deployment](docs/rust-operators.md)
- [Published kernel artifacts](docs/kernel-artifacts.md)
- [Torch extension loader ownership](docs/torch-loaders.md)
- [Direct HIP builds and C ABI loaders](docs/hip-builds.md)
- [Native build/cache migration](docs/kernel-builds.md)
- [AtlasFold performance and exact-output evidence](docs/atlasfold-performance.md)
- [Protenix request performance and historical isolation correction](docs/protenix-native-requests.md)
- [Historical README measurements](.agents/readme_history_2026-10-04.md)
- [Original Protenix setup notes](.agents/protenix_baseline.md)

Finish the maintained Rust framework and its parity gates before resuming
OpenFold3 optimization. Existing validated GPU arithmetic remains available
through capability checks and native fallback throughout this migration.
