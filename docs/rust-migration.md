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


## Implementation progress

The full implementation plan is
[the Rust framework rebuild plan](plans/2026-10-04-gitnexus-plan-rust-framework-rebuild.md).
The initial OpenFold3 cycle is closed with both long native baselines and the
first attention screen recorded. No candidate was promoted.

Implemented: root Cargo workspace and `fireprot-core` typed request contracts,
model-specific cycle semantics, explicit precision/backend checks, checkpoint
roles, source/artifact identity syntax checks, and an immutable validated-request
boundary. These checks do not yet verify files, hardware, or executable model
compatibility; those belong to the pending runtime layer.

Also implemented: versioned worker events, supervisor-owned lifecycle state, and
a Rust operator registry with deterministic selection, exact signature caching,
invalidation, rejection reasons and native fallback. These components are tested
offline; production Python workers and dispatch still await bridge integration.
See [worker protocol](worker-protocol.md).

Also implemented: Unix worker transport, bounded frames, timeout/cancellation,
process-group cleanup, isolated logs, fresh output directories, streaming SHA-256
and output containment checks. Real subprocess tests exercise failure paths.
The AMD device monitor now independently probes HIP identity, maps PCI to SMI,
and rejects invalid process observations. Model content validation remains an
injected Rust interface; its production implementation is pending. Source/input/
checkpoint preflight now gates the supervisor via `PreparedRequest`, with pinned
Git or exported-manifest source verification and canonical artifact identities.

Also implemented: `fireprot-validation` array identity, primitive NPY/NPZ
loading, Kabsch alignment, numeric confidence schema/deltas, and experimental
coordinate masking. NumPy fixtures pin interoperability and numerical behavior.
See [Rust validation](rust-validation.md). A model-output bundle now connects
required sample/tensor checks to the supervisor callback. The Rust CLI and
native OpenFold3 worker are implemented; one N1000 request completed with 297 isolation samples. Exact comparison
failed, and a fresh unchanged native control also showed repeat variation.
The numerical migration gate remains unpassed. See [Rust running](rust-running.md).

Pending: adapter resource/environment provenance, Rust operator-policy bridge,
CIF/statistics tooling, remaining model-worker integrations, kernel layout/build
migration, GPU parity, and replacement of maintained Python entry points. The
repository is in migration; the full framework is not yet rebuilt.

Local core checks:

```bash
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```
