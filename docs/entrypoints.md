# Maintained inference entry points

The maintained entry point is the Rust `fireprot` executable. Checkout shell
scripts only locate that executable and use `exec` to preserve arguments, exit
status and signal delivery. They contain no model settings or Python launch policy.

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
scripts/infer.sh configs/requests/protenix-mini-esm-n1000-native.json \
  configs/diamondhill/protenix-native.json
scripts/smoke.sh configs/batches/protenix-smoke.json
```

`FIREPROT_BIN=/absolute/path/to/fireprot` selects an installed executable. The
checkout default is `target/release/fireprot`. An unavailable binary returns
status 127 with build guidance. Help is available through either script with
`--help`, without model imports or weight loading.

`scripts/smoke.sh` with no arguments selects the checked-in smoke batch and runs
from the repository root. With explicit arguments, both wrappers retain the
caller's working directory. Requests and profiles keep the Rust CLI's positional
syntax; paths containing spaces remain single arguments.

## Migration from the old scripts

The former `scripts/infer.sh` accepted upstream Python options and supplied
Mini-ESM defaults. The current script accepts exactly `REQUEST PROFILE`.
Select model, seed, cycles, steps, precision, backend and physical HIP device in
the typed request. Select the Python environment and operator-library paths in
the execution profile. Legacy `FIREPROT_PYTHON`, `FIREPROT_BACKEND` and
`FIREPROT_BLAS_BACKEND` shell variables do not select these request fields.
Old `--input`/`--out_dir` command lines must be migrated to explicit requests.

The former `scripts/smoke.py` is retired. `scripts/smoke.sh` delegates the three
Protenix smoke workloads to [Rust batch inference](batch-inference.md). The batch
preserves the 124-residue case and C4/step1/sample1/FP32/seed101 settings, executes
jobs sequentially with independently monitored devices, and requests a 600-second
batch deadline. Successful jobs produce individual accepted archives; a failure,
cancellation or timeout makes the batch fail and prevents subsequent launches.

Original scripts are retained for historical reproduction:

- `.agents/legacy_protenix_infer.sh`: byte-identical original inference script.
- `.agents/legacy_smoke.py`: original smoke script with only its call to the
  archived inference script redirected.

[`entrypoint_path_map.json`](../provenance/entrypoint_path_map.json) records original
and archived hashes. Historical README/baseline commands refer to the old script
behavior. Use the archived path when reproducing those commands. Archived scripts
retain their historical checking limitations and do not establish current Rust
acceptance. New reports do not overwrite `provenance/smoke_report.json`.

## Supported scope

The current Protenix worker supports Mini-default, Mini-ESM and Tiny-default
v0.5.0 in FP32 for explicit sequence-only monomer requests. Tiny native and FireProt
are checked on the historical 124-residue smoke case; this does not extend its
performance or accuracy claims to long inputs. Mini-default and Mini-ESM retain
their separately documented long-input validation scope. AtlasFold and OpenFold3
use the same Rust inference/batch interface with their own request contracts.

Historical benchmark entry points remain reproducibility tools. Rust implements
[complete invocation timing](invocation-timing.md) and
[paired execution with correctness gates](paired-execution.md). Final migration
acceptance includes environment coverage and the updated installed workflows.

## Verified migration evidence

On DiamondHill the maintained smoke wrapper completed all three Rust-supervised
jobs in 147.302 seconds under its 600-second deadline. A separate invocation
through the maintained inference wrapper completed Tiny FireProt. All four runs
passed content acceptance, physical-device monitoring and accepted-archive replay.
Against an independent upstream Tiny reference, native/FireProt aligned RMSDs
were `7.7899e-5`/`9.0020e-5` Angstrom and both maximum summary-confidence deltas
were `9.5367e-7`, passing unchanged gates.

CPU verification passed 138 Rust tests and 270 Python tests (110 skips requiring
specific environments/hardware), plus nine batch/entrypoint tests on Rust 1.85.0.
A new wrapper-help test exposed argument matching that treated `--help` as a file;
that regression is fixed and all advertised commands now have help-path coverage.
See [`rust_entrypoint_integration.json`](../provenance/rust_entrypoint_integration.json)
for exact workloads, devices, hashes, numerical results and the recorded difference
between the GPU-tested CLI and its later help-only correction.
