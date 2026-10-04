# OpenFold3 deterministic migration validation

## Explicit execution policy

OpenFold3 requests accept an optional `workload.deterministic` boolean. Omission
preserves the upstream trainer default and the historical serialized workload;
`true` requests strict deterministic algorithms, and `false` explicitly disables
them. The worker passes the value to the upstream Lightning trainer and checks
Torch's effective state before each model forward. Strict requests fail if the
trainer did not enable deterministic algorithms or enabled warn-only behavior.
Unsupported deterministic operations must fail, without silently falling back.

This is an execution policy recorded in the Rust request and accepted-run archive.
It does not change weights, recycles, diffusion steps, sample count or precision.
It can change the reduction implementation, floating-point results and latency.
Compare like policies; old nondeterministic results are not relabeled as passing.

The explicit DiamondHill profile also sets `PYTHONHASHSEED=101`. This complements
the model's seed 101. The profile's environment is captured in `execution.json`.

```bash
target/release/fireprot infer \
  configs/requests/openfold3-n1000-deterministic.json \
  configs/diamondhill/openfold3-deterministic.json
```

Use a fresh request run ID and output directory for every repeat. This configuration
requires the existing pinned OpenBind-0 checkout, weight and ROCm environment.

## Native repeat evidence

The unchanged historical native harness was run with
`benchmarks/openfold3/deterministic.yml`, `PYTHONHASHSEED=101`, and separate physical
GCDs. Both N1000 runs completed with continuous process isolation passing. Rust's
`compare-npz` found exact equality in coordinates and all ten captured confidence
arrays. Model configuration hashes also match. The input uses FP32, three additional
recycles, 200 diffusion steps and one sample.

A separate synthetic replay of the pinned upstream
`aggregate_atom_feat_to_tokens` function used fixed FP32 `[1,10000,128]` features,
1,000 tokens and ten atoms per token. Twenty default-mode invocations produced
20 different output hashes (maximum difference `2.384185791015625e-7`). Twenty
strict-deterministic invocations produced one hash and zero difference. Isolation
passed. The function contains `scatter_add_` accumulation. This locates a source
of nondeterminism, without proving it is the only cause of complete-model drift.

The native comparison changed both the deterministic trainer setting and the
explicit Python hash seed relative to earlier runs. The evidence does not isolate
their individual full-model contributions. Two repeats of one input do not imply
cross-version or universal determinism, and the diagnostic timings do not establish
a speedup.

See [recorded conditions, artifact hashes and comparisons](../provenance/rust_openfold3_deterministic.json).
The earlier native-repeat differences remain recorded in
`provenance/rust_openfold3_integration.json`.

## Rust migration result

The Rust-supervised request completed on HIP 7 / SMI 5 / PCI `0000:93:00.0`,
with 321 passing isolation samples. Coordinates and all ten confidence arrays
match the second deterministic native run bit for bit. Normalization removes only
the native archive's leading batch/sample axes with `[0,0]`, without casts or
arithmetic. Both source and normalized archive hashes are recorded.

The successful forward also verifies that strict deterministic algorithms were
actually enabled and warn-only mode was disabled. The accepted receipt pins the
request, launch record, provenance, supervisor report and output artifacts. Rust
`collect-timings` reverified the receipt and all outputs after inference.

Observed diagnostic intervals were 249.957 seconds for the model, 275.831 seconds
for the worker request and 283.927 seconds for supervision. These are single-run
observations with distinct timing boundaries, not an E2E speedup measurement.
This passes the explicit deterministic N1000 configuration only. Packaging,
environment coverage and remaining workflow migration still need acceptance.

Validation: 109 Rust workspace tests, seven worker-policy tests, Rust 1.85 core
tests, formatting and Clippy passed. The new wire-format regression failed on the
previous implementation and passed after adding the optional policy. Legacy
requests retain their serialized workload when the new field is omitted.
