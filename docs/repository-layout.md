# Repository layout

## Maintained code

```text
Cargo.toml / Cargo.lock     Rust workspace and locked dependencies
crates/                     Core, runtime, validation, CLI and CPython bridge
python/fireprot_workers/    Explicit upstream model calls and tensor serialization
python/fireprot/            Adapter/tensor boundary and native library loading
kernels/amd/                Maintained HIP/CK implementation sources
configs/                   Typed workloads and DiamondHill deployment profiles
tests/                     Contract, operator and integration checks
provenance/                Versioned identities and measured evidence
benchmarks/                Historical reproduction tools
```

Rust owns requests, shared operator policy, build/cache services, processes,
device isolation, source/artifact validation and maintained numerical acceptance.
Python retains model construction/invocation, tensor metadata and capability
checks, operator calls, runtime loading and serialization. These are explicit
PyTorch boundaries. Existing Python numerical/benchmark helpers are historical
reproduction tools, not the owner of new framework policy.

`Protenix/` is an independent checkout. Runtime weights, build caches, deployment
bundles, migration backups and generated predictions remain excluded from Git.

## Compatibility and historical paths

`fireprot/__init__.py` is a small import shim that points its package search path
to `python/fireprot`. It preserves upstream `import fireprot...` statements and
historical entry points using the repository root on `PYTHONPATH`. Both the root
and `python/` work as import roots, independently of the current working directory.
All maintained modules and native source files have regular canonical paths;
there are no directory symlinks in the maintained layout.

Maintained loaders use `fireprot._paths.KERNEL_ROOT` and `REPOSITORY_ROOT` directly.
The [path map](../provenance/layout_path_map.json) records all 38 relocated tracked
files, migration-time before/after hashes, and source-path changes in 19 historical
tools. All ten maintained native source files retain identical bytes. Benchmark
source fingerprints and the experimental C++ include now reference canonical
paths; calculations and workload configuration are unchanged. Modified Python loaders
include the shared path descriptor in native build identities.

The map records this migration, not an ongoing freeze on implementation bytes.
Older evidence still describes its original paths and revisions; consult the map
for the relocated files. Reproducing exact historical source requires its original
Git revision. The import shim contains no framework policy or duplicate model code.

## Deployment

Preview the standard source synchronization before applying it:

```bash
scripts/sync_to_diamondhill.sh --dry-run
scripts/sync_to_diamondhill.sh
```

It includes Cargo sources, Python, kernels, configs and evidence, and excludes
Git metadata, runtime assets, predictions and compiled output. It does not delete
unlisted destination files or remove stale files from a legacy source directory automatically.
For an existing deployment, first transfer/verify the new trees, archive the old
`fireprot` directory under excluded `runtime/layout-backups`, and then install the
root import shim. DiamondHill's first migration preserved its old
source directory at `runtime/layout-backups/fireprot-before-python-layout`.

Source-path changes produce new Rust build identities. Published artifact
acceptance remains independent of original build paths: pinned bundles continue
to load without scanning old compiler inputs. Native source builds must still be
validated from their new paths. The [Rust installer](installation.md) now creates an independent Linux prefix
with pinned binaries, Python/kernel source and generated model profiles.
Single, batch and paired-benchmark entry points are Rust-owned. Explicit
[environment inventories](environment-inventories.md) cover declared library
roots; final deployment coverage must be checked against the actual model paths.

## Validation on DiamondHill (2026-10-04)

[Versioned evidence](../provenance/rust_layout_integration.json) records canonical
source hashes, all six cold-build receipts and four final model requests. A
previous directory-symlink prototype was replaced by the import shim to keep
historical path classification explicit; the final builds and requests below
were repeated after that change.

Checks pass 91 Rust tests, 247 CPU Python tests (110 GPU skips), 159 Protenix GPU
tests and 60 AtlasFold GPU tests. The latter two use freshly compiled libraries
from `kernels/amd`; the FP32 CK GEMM probe also passes its established 1e-4 bounds.
Clippy, formatting, shell syntax and synchronization dry-run checks pass. The
three layout contract cases check the path map and import roots from outside the
repository. All 19 historical tool edits match literal-path substitutions exactly.

The complete requests use existing pinned bundles through the new Python paths.
AtlasFold retains seed 102, four additional recycles, native precision, one sample
and 30/100 diffusion steps. Protenix retains Mini-default FP32, seed 101, four total
cycles, one step and one sample. All requests pass continuous PCI-based isolation.

| Model / route | Length | Native-reference correctness | Supervisor seconds / isolation samples |
| --- | ---: | --- | ---: |
| AtlasFold + AtlasLM-3B, fused HIP | 1000 | Four public arrays bitwise equal | 94.808 / 93 |
| AtlasFold + AtlasLM-3B, fused HIP | 1536 | Four public arrays bitwise equal; bounded native reference | 217.215 / 233 |
| Protenix Mini-default, FireProt | 1000 | Aligned RMSD 8.70362e-5 Angstrom; summary delta 3.81470e-6 | 44.970 / 42 |
| Protenix Mini-default, FireProt | 1536 | Aligned RMSD 1.32640e-4 Angstrom; summary delta 3.81470e-6 | 63.113 / 60 |

Protenix also passes the existing 1e-3 gates against preceding FireProt outputs;
raw confidence arrays remain diagnostic. AtlasFold records 528 fused attention
and 528 HIP multiplication calls with no native fallback. These concurrent
migration timings are not new E2E speedup measurements. OpenFold3 numerical parity,
other model variants, environment evidence and entry points remain unfinished
acceptance work. Later deterministic OpenFold3 parity and independent installation
results are described in [deterministic validation](openfold3-determinism.md) and
[installation](installation.md).
