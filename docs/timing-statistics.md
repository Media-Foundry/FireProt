# Repeat timing statistics

The Rust validation crate owns descriptive timing statistics. The CLI accepts
explicit observations and preserves every observation alongside its summary:

```bash
cargo run --locked -p fireprot-cli -- summarize-timings configs/timings/illustrative.json
```

This example contains **synthetic timings**, not model performance evidence.

## Input contract

Schema 1 contains an `observations` array. Each observation carries a complete
validated `request`, `scope`, nonempty `cache_state`, nonempty `environment` map,
and positive finite `seconds`. Run IDs must be distinct. Only `run_id` and
`output_dir` may vary between requests in one series; source, input, checkpoint
and resource pins, model workload, backend, seed and HIP ordinal must match.
Scope, cache description and environment must also match exactly.

The scope is one of:

| Value | Measurement boundary |
| --- | --- |
| `model` | Worker-reported model interval |
| `worker_request` | Worker request interval, including its initialization |
| `supervisor` | Rust process-spawn-to-final-report interval; excludes preflight |
| `end_to_end` | An explicitly externally measured complete invocation |

Callers must record the actual interval they measured. The command does not
infer a timing scope, cache state or environment from a filename. Cache-state
descriptions should name kernel build/load state, model residency, filesystem
cache preparation and any warmups removed before constructing the series.
Environment maps should retain software/library identities, physical GPU mapping
and isolation evidence references. Equality of these supplied strings does not
independently verify their contents.

## Statistics

The output reports count, minimum, median, p95, maximum and arithmetic mean.
Median averages the central pair for even counts. p95 uses the historical
nearest-rank definition `sorted[ceil(0.95 * count) - 1]`; with fewer than 20
observations it is the maximum. Scaled compensated summation avoids overflow
for finite positive observations. Empty, zero, negative or nonfinite timings
are rejected. The command never removes outliers or warmups automatically.

Different seeds or devices need separate series. Backend A/B comparison also
requires separate series and the established model-correctness and isolation
gates. This command produces descriptive statistics, not a speedup verdict or
proof of structural accuracy. It does not authenticate caller-supplied timings,
reopen weights, or revalidate prediction artifacts. For collection with artifact
and output-content verification, use [accepted run archives](run-archives.md).
Paired performance verdicts remain migration work.

Historical Python benchmark reports are preserved for reproduction. Their
statistical definition is covered by Rust fixtures, including odd/even medians,
the 19/20-observation p95 boundary and floating-point extremes.
