# CLI composition and public interface

The CLI composes the framework library crates through separate modules:

| Module | Responsibility |
| --- | --- |
| `main.rs` | Process entry point, stderr errors and exit status |
| `args.rs` | Typed command parsing and command-specific help; no host services |
| `app.rs` | Command dispatch and JSON result presentation |
| `inference.rs` | Execution-profile loading, worker inputs/environment and device-monitor assembly |
| `collection.rs` | Accepted archive collection and timing-series composition |
| `support.rs` | Contextual JSON file errors, serialization and SIGINT/SIGTERM wiring |

Request validation remains in `fireprot-core`; supervision, isolation, source
preflight and native builds remain in `fireprot-runtime`; array validation and
statistics remain in `fireprot-validation`. The CLI does not duplicate those
policies. Model execution still uses the explicit upstream worker boundary.

## Help and errors

```bash
fireprot --help
fireprot help infer
fireprot infer --help
fireprot --version
```

Help and version queries exit successfully without loading weights, model
environments or GPU services. `-h` and `-V` are supported aliases. The version
comes from the Cargo package; exact build/source identities remain in execution
records. Version text alone is not a binary provenance identifier.

Existing positional command arguments and JSON result shapes are preserved.
Missing/extra arguments and unknown commands produce stderr guidance and exit
status 1. Invalid JSON errors name the input file. Arguments must be UTF-8;
unsupported byte sequences produce an error instead of a panic. Paths with
spaces are passed directly to filesystem/process APIs.

Successful data commands print JSON to stdout. `infer` also prints the terminal
JSON report for a supervised failure, then exits 1. `compare-npz` retains its
comparison JSON on numerical mismatch and exits 1. Setup errors before a report
exists do not fabricate a successful result. Signal handling and cancellation
semantics are unchanged.

## Verification

Rust integration tests exercise all command help forms, invalid argument counts,
non-UTF-8 arguments, all twelve request examples, exact NPZ comparison, timing
summaries and archive collection. A CPU-only synthetic worker drives the actual
`infer` binary through source preflight, explicit visibility, a fake independent
HIP/SMI mapping, output-content acceptance, receipt sealing and timing collection.
It also checks refusal to reuse an output directory and propagation of a worker
content failure to the CLI exit status.

The synthetic test does not run a model or establish GPU parity. Existing
DiamondHill evidence remains model-specific. CI discovers these Cargo tests
automatically. Reorganizing the CLI does not itself claim a performance gain or
complete the outstanding model/environment/packaging migration gates.

### Refactor acceptance (2026-10-04)

[Recorded evidence](../provenance/rust_cli_refactor.json) includes matching local
and DiamondHill source hashes, the remote CLI binary identity, and twelve
byte-identical archive-collection replays: all three timing scopes for Protenix,
OpenFold3 and both the individual and paired AtlasFold records from the preceding
archive-validation cycle. This replay checks preserved command output without
rerunning GPU models or claiming a new speedup.

The complete local Rust suite passes 108 tests. All nine CLI integration tests
pass under Rust 1.85 and on DiamondHill; formatting and Clippy pass. An initial
remote compile found two absent request-example files. Synchronizing those
tracked configurations resolved the setup issue before testing the new binary.

The `install SPECIFICATION DESTINATION` and
`verify-installation DIRECTORY MANIFEST_SHA256` commands compose Rust runtime
installation services. Publication, byte inventories and profile rewriting remain
in `fireprot-runtime::installation`; the CLI only parses arguments, reads JSON and
presents results. See [deployment](installation.md).

`capture-environment SPECIFICATION` delegates file/tree hashing to
`fireprot-runtime::environment_inventory` and emits the content manifest as JSON.
The execution profile references that manifest through a pinned artifact reference.
