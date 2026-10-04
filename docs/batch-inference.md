# Pinned batch inference

`fireprot infer-batch MANIFEST` executes an ordered list of requests through the
same Rust preflight, worker supervision, output acceptance and archive sealing as
`fireprot infer`. The runtime owns sequencing and stop-on-failure behavior.
The CLI composes model execution and validates deployment-profile syntax.

A job names a request JSON and an execution-profile JSON, each with an explicit
SHA-256. All control files are read into validated snapshots before the first
model starts. Subsequent edits to those control files do not replace the loaded
job. Model assets and source are verified separately by each job's preflight.
Paths retain the single-request CLI convention: relative paths resolve against
the caller's working directory, not the manifest's directory.

```bash
# Run from the FireProt repository root on DiamondHill.
fireprot infer-batch configs/batches/protenix-smoke.json
```

The checked-in smoke manifest contains Mini-ESM, Mini-default and Tiny-default
on the existing 124-residue development case. Each job explicitly requests FP32,
four total cycles, one diffusion step, one sample and seed 101. Jobs run
sequentially on the GPU selected in each request. This is a load/output check,
not a performance or experimental-accuracy benchmark.

## Contract

- Schema 1, one to 256 jobs, and an 8 MiB bound for each control file.
- An explicit `timeout_seconds` from 1 to 604800. The smoke batch uses 600.
- Unique run IDs and fresh, distinct output directories; parent directories must
  already exist. Canonical parents prevent aliases from reusing a destination.
- Exact request/profile content hashes. After editing a request or profile,
  update its manifest hash explicitly.
- Every deployment profile must parse before any worker starts. Source/weight
  availability is checked per job, so a later asset failure can occur after an
  earlier job has completed.
- The first setup or model failure stops new launches. Later jobs are explicitly
  `skipped`; completed jobs and their accepted archives remain available.
- SIGINT/SIGTERM cancellation is shared with the active supervisor and persists
  across job boundaries. No next model is launched after cancellation.

The final JSON report names every planned job, its input hashes, output directory,
status and available supervisor report. A failed/cancelled batch exits nonzero.
The runtime deadline signals the same cancellation flag as SIGINT/SIGTERM;
`timed_out` is reported separately. It includes per-job preflight and execution
after control-file loading. Cancellation is cooperative during synchronous
filesystem work; GPU worker cleanup uses the existing supervisor. The report
records `timeout_seconds` and `execution_seconds`, not complete invocation E2E.
Per-job `supervisor.json` and `accepted.json` keep their existing semantics.
Redirect batch stdout to a new file to retain the aggregate report. Setup errors
that precede execution return an error without fabricating completed jobs.

This command does not parallelize model jobs, resume failed jobs, allocate a
cluster queue, or aggregate timings with unequal scientific settings. Use
`collect-timings` with accepted receipts and compatible conditions for timing
analysis. Historical Python benchmarks remain explicit reproduction tools.
