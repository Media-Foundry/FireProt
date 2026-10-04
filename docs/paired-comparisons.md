# Paired run comparisons

`fireprot compare-runs SPECIFICATION` composes archived timing collection,
declared treatment checks, per-sample numerical comparisons and descriptive
performance ratios. The Rust CLI does not forward a Python benchmark command.

## Specification

```json
{
  "schema_version": 1,
  "treatment_description": "Native operators versus the published FireProt route",
  "expected_changes": {
    "backend": {"baseline": "native", "candidate": "fireprot"}
  },
  "baseline": {
    "schema_version": 1,
    "scope": "model",
    "cache_state": "published kernels; fresh process; filesystem cache not flushed",
    "runs": [{"path": "/baseline/accepted.json", "sha256": "RECEIPT_SHA256"}]
  },
  "candidate": {
    "schema_version": 1,
    "scope": "model",
    "cache_state": "published kernels; fresh process; filesystem cache not flushed",
    "runs": [{"path": "/candidate/accepted.json", "sha256": "RECEIPT_SHA256"}]
  }
}
```

This is a shape example, not a runnable experiment. Declare every observed
treatment difference, including a candidate-only `launch.inputs.rust-bridge`
identity where applicable. Missing/extra declarations and incorrect before/after
values fail with the observed change set. Each difference contains `baseline`
and `candidate`; JSON `null` represents an absent value. Accepted input identities
contain `name`, `canonical_path` and `sha256`.

Each arm uses the ordinary [collection format](run-archives.md). For E2E, both
arms use `scope: "end_to_end"` and corresponding `invocations` receipt pins.
Pair index zero compares the first baseline archive with the first candidate
archive, and so on. Counts must match; a run ID cannot appear twice across either
arm. Repeat consistency is checked independently within each arm.

## Conditions and treatment

Workload, model, precision, seed, input/checkpoint/resource pins, source,
physical device mapping, worker software versions, cache description and timing
scope must match. Model worker/executable identities, working directories and
arguments are also protected. Environment values outside `FIREPROT_*` cannot
differ, even when listed in the treatment declaration.

Permitted, explicitly declared treatment changes are:

- `backend`;
- `launch.environment.FIREPROT_*` values;
- `launch.inputs.rust-bridge` and `launch.inputs.amd-kernels/...` identities;
- `invocation.profile_sha256` for E2E profiles with declared operator changes.

These declarations do not bypass the numerical gates or authorize reducing
inference work. Full before/after values remain in the report. Input identities
are compared by stable name, so adding a bridge cannot shift array positions and
appear to change every other recorded input.

## Correctness and statistics

Every archive is hash/content/isolation checked. Every requested sample is
paired by sample ID. Native shape and dtype metadata must match. Schema-2 CIFs
must retain identical atom identities. AtlasFold and OpenFold3 require exact
array agreement. Protenix uses the existing fixed `1e-3` aligned-coordinate and
summary-confidence gates; input embeddings and atom/token metadata require exact
agreement. Its four full floating confidence arrays retain explicit diagnostic
differences without introducing an unsupported bitwise-equality gate.

A numerical failure gives a nonzero CLI status and `performance: null`. Valid
comparisons retain both arm summaries, the ratio of arm medians, ratio of arm
p95 values, every paired time ratio and their separate summary. No observation
is discarded. Ratio overflow or underflow is an error.

With fewer than three pairs, the trend is `insufficient_repeats`. Otherwise it
reports `faster_median_and_p95`, `slower_median_and_p95` or `mixed_or_unchanged`.
These are observed timing trends, not statistical significance, memory acceptance
or automatic operator promotion. With fewer than 20 observations, nearest-rank
p95 is the maximum.

Execution order and cache preparation remain caller declarations. The comparison
tool does not infer chronology from file names, prove balanced/randomized
execution, or eliminate thermal and system-load drift. Use explicit alternating
execution on the same physical device, validate each run, and retain the launch
order alongside the pinned measurement records.
[`benchmark-pairs`](paired-execution.md) executes pinned pairs and records that
order in Rust. Final cross-model benchmark acceptance remains migration work.

## N1000 integration evidence

[The integration record](../provenance/rust_paired_comparison_integration.json)
contains six fresh-process Protenix Mini-default FP32 requests on DiamondHill,
all with N1000, four cycles, one diffusion step, one sample and seed 101. The
same physical GCD is independently mapped as HIP 6 / PCI `0000:8e:00.0` / SMI 4.
The explicit serial order is native/FireProt, FireProt/native, native/FireProt.
Each run passes the established comparison against the existing native
reference before the next run starts; all three direct pairs also pass.

Both arms use the same published-operator execution profile and Rust executable.
The native worker retains its Torch route; the FireProt worker installs the
existing acceleration route. The only normalized condition differences are
backend selection and the candidate's pinned Rust bridge input. Source,
weights, model work, environment, device and cache declarations match.

| Scope | Native median | FireProt median | Median speedup | p95 speedup |
| --- | ---: | ---: | ---: | ---: |
| Model | 50.604 s | 23.460 s | 2.157x | 2.159x |
| Complete inference invocation | 87.618 s | 41.309 s | 2.121x | 2.122x |

All direct pairs pass unchanged `1e-3` gates. Maximum aligned coordinate RMSD is
`9.50052e-5` Angstrom and maximum summary-confidence difference is `7.62939e-6`.
Input/atom/token metadata compare exactly and CIF atom identities match. Full
floating confidence differences remain explicit diagnostics in the report.

These measurements validate the Rust comparison workflow with existing kernels;
they are not gains introduced by this commit. There are three pairs, no warmup
observations removed, and no filesystem cache flush. They do not replace warm
resident-model benchmarks or establish memory improvements or significance.
All 88 audited Rust/build files match the deployed source. The full Rust suite
passes 152 tests; the four paired-validation tests, CLI rejection integration
test and five interface tests also pass Rust 1.85. Workspace Clippy and formatting
pass. Rust execution orchestration is now provided by `benchmark-pairs`; final
cross-model acceptance remains open.
