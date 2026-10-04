# Rust migration priority and acceptance

## Required order

The user requested this ordering on 2026-10-04:

1. Finish the current OpenFold3 baseline and first operator-screening cycle,
   retain correctness/performance evidence, and commit the coherent result.
2. Rebuild FireProt's own framework in Rust before starting the next OpenFold3
   optimization cycle.
3. Resume OpenFold3 optimization through the migrated framework.

The migration is not itself a measured GPU speedup. Existing HIP/CK device
kernels remain appropriate GPU implementations. Upstream Protenix, AtlasFold,
and OpenFold3 are independent scientific implementations; their native Torch
execution must remain available as the reference during migration.

## Intended scope

Use a Cargo workspace for the maintained framework: CLI, typed model requests,
configuration, capability-based backend selection, job lifecycle, device
isolation, provenance, benchmark collection, and numerical validation. Expose
explicit interfaces to HIP/CK libraries and upstream model workers. Define a
versioned worker protocol rather than allowing Python helpers to retain hidden
ownership of framework policies. Native model execution that still requires
PyTorch must be identified explicitly, with its source and environment pinned.

Historical benchmark scripts and evidence remain reproducible archives; they
are not substitutes for migrating maintained framework entry points. New
framework features should be implemented in Rust after this cycle. Any remaining
Python component needs a recorded role and an explicit migration decision.

## Acceptance gates

- The Rust CLI can execute the supported model workflows and record equivalent
  input/model/weight/precision/random-state/cache provenance.
- Physical GPU mapping and continuous process isolation retain their current
  checks. Failed or missing model outputs cannot produce a successful report.
- Results retain coordinates, confidence arrays, native shape/dtype metadata,
  and comparison gates; no weakened gate or reduced inference work is allowed.
- Unsupported operator configurations retain native fallback, with the chosen
  backend and extension identity recorded.
- Validate migrated workflows against existing references before retiring old
  maintained entry points. Preserve historical evidence and upstream checkouts.
- Report migration coverage and remaining Python dependencies directly. A Rust
  launcher around otherwise unchanged framework logic does not meet this scope.

A detailed dependency and interface inventory will be written after the current
OpenFold3 cycle finishes and before implementation begins.


## Implementation progress (2026-10-04)

The immutable [implementation plan](plans/2026-10-04-gitnexus-plan-rust-framework-rebuild.md)
defines the full scope. The initial OpenFold3 cycle is closed with native long
baselines and an attention screen recorded; no candidate was promoted. Migration
is active and the completion gates above remain in force.

| Area | Current implementation | Remaining acceptance |
| --- | --- | --- |
| Request and worker contracts | Typed Rust model workloads, precision/cycle rules, artifact roles, protocol and lifecycle states | Additional model variants |
| Operator policy | Rust ordering, cache identity/invalidation, rejection reasons and fallback via the CPython bridge | Python tensor predicates and calls remain an explicit boundary |
| Process and device control | Rust preflight, fresh output directories, worker supervision, timeouts/cancellation, process cleanup, HIP-to-PCI-to-SMI mapping and continuous isolation | Complete environment/worker identity evidence |
| Source and resource provenance | Rust streaming hashes, pinned Git/exported sources, checkpoints, inputs and Protenix CCD/data resources | Complete adapter environment/resource coverage |
| Output validation | Rust sample/tensor acceptance, primitive NPY/NPZ, exact array comparison, alignment, confidence schemas/deltas and experimental coordinate masks | CIF/statistics workflow coverage |
| Protenix | Native and accelerated Mini-default FP32 N1000/N1536 under Rust supervision | Other variants before retiring the old entry point |
| AtlasFold + AtlasLM-3B | Native/chunked/HIP/fused HIP with dual weights and model-specific semantics | Final acceptance after layout migration |
| OpenFold3 / OpenBind-0 | Native N1000 inference executes under Rust supervision | Numerical migration parity remains unpassed; unchanged native runs also vary |
| Native builds | Rust dependency identities, directory inventories, cache locking/publication, guarded compiler lifecycle; all six maintained loaders connected | Complete model-environment/plugin inventories remain outside the build recipe |
| Precompiled artifacts | Rust publication and pinned acceptance for Python and C ABI bundles; all six loaders have explicit selectors | Packaging and final deployment acceptance |
| Repository layout | Root five-crate Cargo workspace and explicit model workers under `python/` | Move remaining tensor glue into `python/`, native sources into `kernels/amd`, checked historical path map |
| Maintained entry points | Rust CLI covers current request profiles | Replace remaining maintained scripts after parity; CI and packaging |

### Verified model scope

AtlasFold N1000 native and N1000/N1536 chunked/HIP/fused HIP routes pass exact
coordinates, pLDDT, PAE and pTM comparisons. N1536 uses the bounded native
reference. Model weights, additional recycles, diffusion steps, sample count and
precision boundaries are preserved. See [running](rust-running.md) and
`provenance/rust_atlasfold_accelerated.json`.

Protenix Mini-default FP32 native/accelerated N1000/N1536 requests pass existing
aligned-coordinate and summary-confidence gates. Full arrays retain their native
shape/dtype and finite-value checks; raw accelerated PAE/PDE arrays are not
bitwise identical to native. Four CCD/data resources are pinned in preflight.
See `provenance/rust_protenix_integration.json` and the later loader/publication
records linked from [kernel artifacts](kernel-artifacts.md).

OpenFold3 numerical parity has not passed. Native-repeat variation is recorded in
`provenance/rust_openfold3_integration.json`; it is not permission to widen a gate
or resume operator optimization. Final migration acceptance remains incomplete.

### Ownership and checks

Python remains at upstream model execution, tensor introspection/capability checks,
local operator calls and tensor serialization. Historical Python benchmarks remain
reproduction tools. Shared dispatch, builds, supervision and output acceptance are
Rust-owned. See [operator policy](rust-operators.md), [build services](kernel-builds.md),
[worker protocol](worker-protocol.md) and [validation](rust-validation.md).

```bash
cargo test --workspace --locked
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
PYTHONPATH=.:Protenix python -m pytest -q tests
```

GPU tests are explicit. Full-model acceptance must be repeated after source-path
changes and must record physical isolation, inputs, weights, precision, cycles,
steps, samples, seed and cache state. Concurrent migration timings do not establish
new E2E speedups. Finish layout, environment evidence, model parity, maintained
entry points and packaging before declaring this rebuild complete.
