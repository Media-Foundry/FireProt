# Model worker protocol

Status: the Rust contracts and state machine are implemented in `fireprot-core`.
The Unix process transport and artifact hashing are implemented in
`fireprot-runtime`. The independent HIP/SMI device monitor is implemented. Rust content validation and Protenix, AtlasFold and native OpenFold3 workers are
connected. Full migration acceptance remains incomplete; see [current status](rust-migration.md).

## Ownership

The Rust supervisor owns job state and the success decision. A worker owns
upstream model execution and reports observations. A worker cannot report a
trusted isolation result or mark a run successful. Protocol frames must use a
channel separate from upstream model logs; the runtime enforces frame size, timeouts, exit handling and cancellation.

## Event envelope

```json
{
  "schema_version": 1,
  "run_id": "long-1536-seed102",
  "sequence": 0,
  "payload": {
    "event": "ready",
    "device": {
      "worker_pid": 1234,
      "logical_device": 0,
      "pci_address": "0000:93:00.0",
      "architecture": "gfx90a"
    }
  }
}
```

The run ID must match the validated request, sequence numbers start at zero and
increase by one, and unknown envelope/payload fields are rejected. Logical
HIP device zero inside the worker is not an SMI index. The runtime must match
the reported PCI identity independently and verify the child PID before it
authorizes model execution.

## Lifecycle

```mermaid
stateDiagram-v2
  [*] --> AwaitingReady
  AwaitingReady --> AwaitingStart: ready with expected PID and valid device syntax
  AwaitingStart --> Running: supervisor authorizes start
  Running --> Running: progress
  Running --> AwaitingVerification: nonempty output artifact list
  AwaitingVerification --> Succeeded: supervisor verifies all completion checks
  AwaitingReady --> Failed: error or cancellation
  AwaitingStart --> Failed: error or cancellation
  Running --> Failed: error or cancellation
  AwaitingVerification --> Failed: error or failed verification
```

Wrong version, run ID, sequence, ordering, identity or artifact declarations
permanently fail the session. A terminal session cannot be reopened. A clean
child exit without announced outputs fails. Output paths must be unique and
relative, and SHA-256 strings must be valid; these syntax checks do not replace
runtime containment, file-content, shape/dtype or numerical checks.

`CompletionChecks` is a Rust-only type, not deserializable from worker messages.
Success requires exit code zero, verification of every declared artifact, at
least one isolation sample, no unexpected GPU PIDs, and no monitoring errors.
The runtime supervisor constructs this evidence after its host monitor and
artifact/content checks pass.

## Operator policy

The Rust registry stores exact operation signatures, including shape, strides,
layout, dtype, architecture, mode, mask, aliasing intent and attributes. It
selects the highest-priority supported implementation, with stable name/version
tie ordering. Missing candidates or failed predicates yield native fallback;
predicate panics reject that candidate. Registration/removal clears decision
caches; explicit runtime-state clearing is required for external policy changes.

The registry is single-owner and requires mutable access for mutation/selection.
The in-process Rust bridge supplies synchronization and maps selected
implementation IDs to model-process GPU calls. It does not spawn a subprocess
or serialize tensors for each operator. Python retains tensor inspection and
callbacks; Rust owns shared selection policy.

## Unix transport and process supervision

`fireprot-runtime::supervisor::run` reserves a new output directory, writes the
validated request, and launches a command without a shell in a dedicated process
group. `FIREPROT_REQUEST` points to the request file, `FIREPROT_OUTPUT_DIR` is the
absolute output directory, and `FIREPROT_WORKER_SOCKET` points to a Unix socket
in a private temporary directory. Workers connect and send newline-terminated
event envelopes. Stdout/stderr go to separate log files.

After independent device verification and a successful initial isolation sample,
the supervisor replies `{"schema_version":1,"command":"start"}` followed by a
newline. Workers must wait for that command before loading/executing models.
The supervisor continues sampling through process exit. Frame size and per-tick
reads are bounded; timeout/cancellation stops the entire process group. The group
is also stopped before artifact verification after a normal parent exit.

Every declared output must resolve inside the output directory and match its
SHA-256. A required Rust content-validator callback must additionally verify the
model's required arrays and metadata. Process exit zero without outputs fails.
Post-launch failures are retained in `supervisor.json`; setup/spawn failures
return an I/O error and may leave the reserved directory for diagnosis. Existing
output directories are never reused.

This layer exposes an independent `DeviceMonitor` trait and an `AmdSmiMonitor`
implementation. Source/input/checkpoint preflight is mandatory at the runtime entry point.
Production model workers and Rust output validators are connected; the full
model acceptance matrix remains incomplete. CPU tests use a standard-library
Python fixture as a real child process and an injected host monitor; they do not
establish GPU isolation or full-model parity.

## Independent AMD device monitor

`AmdSmiMonitor` runs a separate HIP identity probe under the requested visibility
mask, compares its PCI address and architecture with the worker, and matches the
PCI address to `amd-smi list --json`. It never assumes HIP and SMI ordinals agree.
Each sample rechecks that mapping and parses `amd-smi process --json`. Only the
verified worker PID is allowed; descendants need a future explicit policy.
Missing devices/lists, duplicate identities/PIDs, unknown no-process messages,
command failures and malformed JSON all fail the run. The observed idle sentinel
`No running processes detected` is accepted only as the sole process entry.

Probe subprocesses have a timeout and output-size checks, separate stdout/stderr
and process-group cleanup. Monitoring is synchronous at present: the requested
interval is a minimum delay, and each probe may take up to its configured timeout.
This is infrastructure validation, not a measured model-overhead result.

Build the tiny native HIP identity helper on DiamondHill (the explicit RPATH is
required on the tested host):

```bash
/opt/rocm/bin/hipcc kernels/amd/tools/hip_device_probe.cpp \
  -Wl,-rpath,/opt/rocm/lib -o runtime/fireprot-hip-probe
/opt/rocm/bin/hipcc crates/fireprot-runtime/tests/fixtures/gpu_process.cpp \
  -Wl,-rpath,/opt/rocm/lib -o runtime/fireprot-gpu-fixture
FIREPROT_HIP_PROBE="$PWD/runtime/fireprot-hip-probe" \
FIREPROT_GPU_FIXTURE="$PWD/runtime/fireprot-gpu-fixture" \
  cargo test -p fireprot-runtime --test hardware_monitor -- --ignored --nocapture
```

The hardware test requires an idle authorized GPU. Its defaults are DiamondHill
HIP 7, PCI `0000:93:00.0`, gfx90a; override `FIREPROT_TEST_HIP_DEVICE` and
`FIREPROT_TEST_PCI` together for another device. It starts a real HIP allocation
process, proves that process is detected as external, then proves it is excluded
only when explicitly selected as the allowed worker. It cleans up the process.
CPU-only `cargo test --workspace` intentionally skips this hardware test.

## Provenance preflight

The supervisor accepts only a `PreparedRequest`, constructed by Rust `prepare`.
Preparation verifies the input and every checkpoint, records canonical paths,
SHA-256 and byte sizes, and normalizes request paths before the worker changes
its working directory. Each launched run retains `provenance.json`.

Two explicit source modes are supported:

- `GitCheckout`: the root must be the repository root at the requested commit
  with a clean tracked index/worktree. Every regular tracked file is additionally
  compared with its Git blob identity and executable mode, including files hidden
  from ordinary status by `assume-unchanged` or `skip-worktree`.
- `Exported`: a source manifest has its own explicitly supplied SHA-256. Its
  schema and commit must match the request, and every listed file is checked by
  SHA-256 and executable mode. Export the manifest from the verified local
  checkout; deploy it alongside synchronized source without copying Git metadata.

```bash
cargo run --example source_manifest -- Protenix COMMIT > source-manifest.json
cargo run --example preflight -- request.json
cargo run --example preflight -- request.json source-manifest.json MANIFEST_SHA256
```

Source manifests cover regular tracked files. Symlinks and gitlinks are rejected;
untracked/ignored resources and installed dependencies need explicit adapter
resource/environment provenance. A pinned manifest authenticates the caller's
selected file inventory; it does not infer whether that inventory is complete.
These checks observe filesystem contents, not an immutable snapshot: inputs,
weights and source must remain unchanged between preparation and execution.
Reprepare after any change. The exported mode never substitutes the remote
checkout's older Git HEAD for the verified synchronized source identity.

## Launch identity and persisted device mapping

The supervisor records and rechecks declared execution inputs, creates the worker
with a recorded explicit environment, and independently verifies `/proc/PID/exe`
before authorizing Start. Final reports retain the worker's announced identity,
`device_verified`, and AMD SMI's independent physical mapping. See
[execution provenance](execution-provenance.md) for exact scope and limits.
