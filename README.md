# FireProt

An AMD GPU protein structure inference framework with model adapters and
capability-gated HIP/CK operators. The primary performance target is long proteins,
including the measured 1,000- and 1,536-residue workloads.

**Status: Rust framework migration validated for the workflows below.** Rust owns
request contracts, operator selection, process supervision, GPU isolation,
provenance, native build/cache services, numerical acceptance and paired benchmarks.
All six maintained loaders support pinned artifacts. Seven model requests pass
validation from an independent Linux installation. Upstream model execution and
tensor calls retain an explicit Python/Torch boundary. See the
[acceptance audit](docs/rust-acceptance.md) for evidence and limits.
Rust itself is not claimed to accelerate GPU kernels.

## Model coverage

| Model | Connected workflow | Verified scope / limits |
| --- | --- | --- |
| Protenix | Mini-default, Mini-ESM and Tiny-default FP32, native and FireProt paths | Mini-default N1000/N1536, Mini-ESM N1000 and Tiny L124 pass established coordinate/summary gates. Mini-ESM includes pinned ESM2-3B and exact feature parity. Full confidence arrays are retained, without a bitwise equality claim. |
| AtlasFold + AtlasLM-3B | Monomer, original dual weights, native/chunked/HIP/fused HIP | N1000 native and N1000/N1536 accelerated routes pass exact public-output parity. N1536 reference uses bounded native math. |
| OpenFold3 / OpenBind-0 | Native and opt-in HIP softmax under Rust supervision | Strict native and experimental HIP N1000/N1536 checks pass exact parity. Maintained `hip_fused` N1000/N1536 pass all 11 arrays bitwise. Experimental N1536 softmax measures 1.376x E2E; maintained-route paired timing is pending. Legacy default-mode parity remains unpassed because native repeats vary. |
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
reference existing DiamondHill paths. Use the [Rust installer](docs/installation.md)
to generate profiles for an independent deployment prefix.
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
[checkpoint locations](docs/diamondhill-checkpoints.md). `scripts/infer.sh` calls
Rust single-request inference; `scripts/smoke.sh` calls Rust batch inference.
See [entry-point migration](docs/entrypoints.md) for arguments and archived commands.
Historical benchmark tools remain explicit reproduction tools.

## Repository structure

| Path | Responsibility |
| --- | --- |
| `crates/fireprot-core` | Requests, workload semantics, protocol and operator registry |
| `crates/fireprot-runtime` | Preflight, worker lifecycle, device monitoring, native build/cache services |
| `crates/fireprot-validation` | Output acceptance, array comparison and numerical metrics |
| `crates/fireprot-cli` | Rust command entry point |
| `crates/fireprot-bridge` | In-process CPython boundary for Rust policy and build services |
| `python/fireprot_workers` | Explicit upstream model invocation and tensor serialization |
| `python/fireprot` | Model adapters, tensor inspection/calls and Rust bridge loading |
| `kernels/amd` | Maintained HIP/CK sources |
| `fireprot/__init__.py` | Import shim for upstream and historical entry points |
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
- [Prediction CIF validation and sample associations](docs/cif-validation.md)
- [Mini-ESM assets and feature validation](docs/protenix-mini-esm.md)
- [Pinned batch inference](docs/batch-inference.md)
- [Maintained and historical entry points](docs/entrypoints.md)
- [Current architecture and dependency boundaries](docs/architecture.md)
- [Independent Linux installation](docs/installation.md)
- [Repository layout and historical path compatibility](docs/repository-layout.md)
- [Worker launch and physical device provenance](docs/execution-provenance.md)
- [Explicit environment content inventories](docs/environment-inventories.md)
- [Accepted run archives and timing collection](docs/run-archives.md)
- [Complete inference invocation timing](docs/invocation-timing.md)
- [Paired correctness and timing comparisons](docs/paired-comparisons.md)
- [Rust paired execution and per-run correctness gates](docs/paired-execution.md)
- [Framework CPU continuous integration](docs/continuous-integration.md)
- [Operator policy and bridge deployment](docs/rust-operators.md)
- [Published kernel artifacts](docs/kernel-artifacts.md)
- [Torch extension loader ownership](docs/torch-loaders.md)
- [Direct HIP builds and C ABI loaders](docs/hip-builds.md)
- [Native build/cache migration](docs/kernel-builds.md)
- [AtlasFold performance and exact-output evidence](docs/atlasfold-performance.md)
- [OpenFold3 experimental HIP speedup and exact-output evidence](docs/openfold3-performance.md)
- [Protenix request performance and historical isolation correction](docs/protenix-native-requests.md)
- [Historical README measurements](.agents/readme_history_2026-10-04.md)
- [Original Protenix setup notes](.agents/protenix_baseline.md)

OpenFold3 optimization proceeds through the Rust framework and the same correctness
gates. Existing validated GPU arithmetic remains available through capability
checks and native fallback.
