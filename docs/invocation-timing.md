# Complete invocation timing

Status: Rust CLI measurement and accepted-archive collection are connected.
[`compare-runs`](paired-comparisons.md) adds explicit paired conditions and
numerical gates; final cross-model benchmark acceptance remains migration work.

## Measure and collect

Create a specification with a new request/output directory and exact control-file
hashes. Paths use the caller's working directory, as with `infer`:

```json
{
  "schema_version": 1,
  "timeout_seconds": 600,
  "request": {"path": "/path/to/request.json", "sha256": "REQUEST_SHA256"},
  "profile": {"path": "/path/to/profile.json", "sha256": "PROFILE_SHA256"}
}
```

```bash
fireprot measure-inference specification.json /new/measurement-directory
```

The command writes immutable effective request/profile snapshots to the new
measurement directory, launches its own executable with `infer`, and records
stdout/stderr there. It prints the path and SHA-256 of `invocation.json` only
after successful child exit and archive/content/isolation verification. Preserve
that pin independently. Failed or cancelled invocations never publish this
record; post-launch failures retain `failure.json` where writing remains possible.
An existing measurement directory is rejected.

For `collect-timings`, use `scope: "end_to_end"`, the usual `runs` receipt pins,
and an `invocations` array of measurement-record pins in the same order. Both
lists must be nonempty and equal in length. Other timing scopes forbid invocation
pins. The collector rechecks control snapshots, request/executable identity,
successful model supervision, isolation and output contents. It rejects a
complete interval shorter than its supervisor interval. Repeated invocations
must retain the executable, caller working directory, effective environment,
profile bytes, timeout and all existing workload/device/cache conditions.

The model archive and measurement directory can both be moved. Supply their new
paths with unchanged hashes; original source, weights, executable and control
paths need not exist for replay. Records are not signatures or independent proof
against a writer who replaces both their contents and the trusted pins.

## Measurement boundary and lifecycle

`fireprot_runtime::invocation::measure` measures wall time from immediately before
spawning a trusted supervisor command through observing its exit. This includes
process startup, supervisor preflight, inference, output acceptance and process
shutdown. The monotonic clock has a 1 ms exit-poll interval plus scheduler delay.
Caller preparation and post-exit verification are outside this interval. In
particular, control loading, path normalization, snapshot writes and hashing of
the executable precede the timer. This can warm filesystem caches; do not label
these measurements as cold-cache runs without separately establishing that state.
All model source/weight/resource preflight, model work, sealing and shutdown in
the child `infer` command are inside the measured interval. This is the complete
child inference invocation, not the total wall time of `measure-inference`.

The child runs in a separate process group. Cancellation and deadline expiry send
SIGTERM to the supervisor, which must clean up its own worker groups before
exiting. The parent waits for this cooperative cleanup; a deadline is not a hard
upper bound. A hung or non-cooperative supervisor can therefore keep the parent
waiting. This API is not an arbitrary command runner. It does not provide cleanup
after parent SIGKILL, machine failure, or a child that abandons its own workers.

The returned measurement preserves the process exit status and any cancellation
or deadline reason. Status zero after cancellation is not success. Cancellation
before launch, invalid limits and spawn errors produce no measurement. Output
must go to inherited streams or files, not pipes whose contents nobody drains.
The primitive alone does not authenticate requests, executable contents or
predictions; the CLI composes those checks with the runtime archive services.

CPU tests exercise normal and failed exits, prelaunch rejection, cooperative
deadline cleanup, cancellation with zero exit status, and a Linux supervisor
reaping a worker in a separate process group. These are lifecycle tests, not GPU
performance evidence. CLI integration tests exercise repeat collection, relocated
archives, mismatched receipt pairs, modified controls, malformed timing records,
model-content failure and actual supervisor timeout.

## Long-model integration

The [integration record](../provenance/rust_invocation_timing_integration.json)
binds two sequential DiamondHill Protenix Mini-default N1000 requests using the
published FireProt operators, FP32, four cycles, one diffusion step, one sample
and seed 101. Both use independently mapped HIP 6 / PCI `0000:8e:00.0` / SMI 4.
They start fresh model processes; filesystem caches were not flushed and no
warmup observations were removed.

Complete inference invocation times are 42.219 and 41.491 seconds (median 41.855).
Both archives collect together under matching conditions. Raw-array and CIF
comparisons against the existing native reference pass the unchanged `1e-3`
coordinate/confidence gates: aligned RMSDs are `8.7071e-5` and `9.9357e-5`
Angstrom, with maximum confidence deltas `7.4506e-8` and `4.7684e-7`.
The deployed Rust/build input hashes match all 85 audited local files.

These two repeats verify the measurement workflow. They do not establish a new
speedup, a stable tail-latency estimate, or complete cross-model E2E acceptance.
The full Rust suite passes 147 tests; the 17 CLI collection/inference/interface
tests also pass Rust 1.85. Formatting and workspace Clippy pass.
