# Rust paired execution

`fireprot benchmark-pairs SPECIFICATION DESTINATION` owns paired execution in
Rust. The runtime validates and snapshots all controls, the shared batch lifecycle
applies the execution deadline/cancellation policy, and the CLI composes measured
inference with the existing numerical and comparison gates. Each job uses a fresh
model process. No Python experiment driver is required.

## Specification

The following is a shape example with placeholder paths and hashes:

```json
{
  "schema_version": 1,
  "execution": {
    "schema_version": 1,
    "timeout_seconds": 1800,
    "invocation_timeout_seconds": 600,
    "pairs": [{
      "first": "baseline",
      "baseline": {
        "request": {"path": "/new-baseline-request.json", "sha256": "REQUEST_SHA256"},
        "profile": {"path": "/profile.json", "sha256": "PROFILE_SHA256"}
      },
      "candidate": {
        "request": {"path": "/new-candidate-request.json", "sha256": "REQUEST_SHA256"},
        "profile": {"path": "/profile.json", "sha256": "PROFILE_SHA256"}
      }
    }]
  },
  "comparison": {
    "scope": "end_to_end",
    "cache_state": "published kernels; fresh processes; filesystem cache not flushed",
    "treatment_description": "Native versus published FireProt operators",
    "expected_changes": {
      "backend": {"baseline": "native", "candidate": "fireprot"}
    }
  }
}
```

Add the exact bridge/kernel/environment differences required by
[paired comparison](paired-comparisons.md); the abbreviated declaration above
is not sufficient for every model/profile. Use a new run ID and output directory
for every request. All relative paths retain caller-working-directory semantics.
The experiment destination must also be new and cannot alias a model output.

There must be 1–128 pairs. The first pair must run `baseline` first to establish
the declared comparison reference. Later pairs explicitly select `baseline` or
`candidate` first; alternating these orders gives AB/BA execution. The scheduler
does not infer or silently rebalance the supplied order. At least three pairs
are needed for the comparison tool's descriptive timing trend.

Before the first launch, Rust checks every request/profile pin and model profile
schema, unique run IDs/outputs, identical scientific settings across the experiment,
and stable backend/profile bytes within each arm. Model assets are verified in
each timed inference preflight. Verified control snapshots survive replacement or
deletion of the original control files; the original identities remain recorded.

## Gates and stopping

The first baseline is the reference anchor, after normal source/weight/device,
isolation and model-content acceptance. It is not an independently measured
experimental-coordinate reference. Every later run is compared with that anchor
before another run starts, including baseline-repeat consistency. Each completed
pair is also compared directly, and all pairs receive a final aggregate comparison.
The fixed model-specific gates from `compare-runs` remain unchanged.

Content, condition, numerical or journal-write failure stops the experiment.
Remaining jobs are explicitly skipped. SIGINT/SIGTERM and the shared execution
deadline cancel the active invocation cooperatively so its supervisor can clean
up separate worker groups. The per-invocation deadline remains independently
recorded. Neither deadline is a hard bound on synchronous I/O or cleanup time.
The batch execution deadline includes per-run/pair/final comparisons, starts after
control preparation, and excludes final benchmark-report writing.

## Records and timing boundaries

The destination contains:

- `execution-plan.json` and `comparison-settings.json`;
- pinned `request-NNN.json` and `profile-NNN.json` snapshots;
- `measurement-NNN/` with complete invocation records and logs;
- `reference-check-NNN.json` and `pair-check-NNN.json` for completed checks;
- `order.jsonl` with flushed/synchronized started, measured, verified or failed
  events, run/pair/arm identities and monotonic offsets;
- `comparison.json` when aggregate comparison completes;
- `benchmark.json`, binding the plan, settings, order journal, individual
  measurements/checks, batch outcome and accepted final-comparison pin.

The command prints the benchmark report's path/hash and outcome. Non-success
outcomes return nonzero and do not publish a final comparison in that outcome.
Individually accepted model archives remain available after experiment failure;
content acceptance alone does not imply numerical agreement with the reference.
An interruption can leave partial diagnostic files. Only the pinned successful
benchmark outcome establishes completion; the presence of a comparison file alone
does not. There is no implicit resume or skipped-run backfill.

Per-invocation E2E still measures the child `infer` command. Order offsets and
batch execution time include scheduling and comparison overhead and must not be
relabeled as inference E2E. Cache preparation remains an explicit caller statement;
the recorded execution order does not prove cold caches, randomized trials or
statistical significance. Hashes bind records to separately retained pins, not
to a cryptographic signer.

Individual model and invocation archives retain the relocation/replay support of
`collect-timings` and `compare-runs`. Benchmark report paths record their original
locations; moving an experiment requires supplying relocated archive pins when
replaying a comparison.

## Verified integration

[The DiamondHill record](../provenance/rust_paired_execution_integration.json)
contains a complete Rust-driven N1000 Protenix Mini-default experiment with
FP32, four cycles, one diffusion step, one sample and seed 101. The existing
native/published FireProt routes execute baseline/candidate/candidate/baseline
on independently mapped HIP 6 / PCI `0000:8e:00.0` / SMI 4. All four invocations,
three reference checks, two direct pair checks and the aggregate comparison pass.
The initial anchor also passes the existing native-reference gate with aligned
RMSD `9.23253e-5` Angstrom. All 12 journal events retain their expected order.

The recorded batch execution interval is 263.419 seconds, including scheduling
and correctness checks; this is not per-request E2E. With only two pairs, the
aggregate correctly reports `insufficient_repeats`. This validates orchestration
with unchanged kernels and does not replace the preceding three-pair timing
evidence or establish a new optimization.

All 91 audited Rust/build files match deployment. The full Rust suite passes
157 tests; the two runtime experiment tests, three CLI scheduler tests and five
interface tests also pass Rust 1.85. Workspace Clippy and formatting pass.
An initial concurrent-toolchain test run correctly rejected executable changes
when the other build replaced the shared CLI; the full suite passed when rerun
serially, with identity checks unchanged.
