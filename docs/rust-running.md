# Rust inference entry point

The maintained Rust entry point is under active migration. One N1000 OpenFold3
request has completed with the Rust supervisor, 297 isolation samples, and
validated output content. Exact agreement with the older native capture did not
pass; a fresh native-repeat control also differs from the older native capture. This route is not yet
promoted as migration-parity validated. The workers
implement native OpenFold3/OpenBind-0 and AtlasFold plus AtlasLM-3B monomer
inference. Protenix, accelerated backends and the in-process Rust operator bridge are not connected
to this entry point yet. Their existing workflows remain the migration references.

## Commands

```bash
cargo build --release -p fireprot-cli
target/release/fireprot inspect configs/requests/openfold3-n1000-native.json
target/release/fireprot infer configs/requests/openfold3-n1000-native.json \
  configs/diamondhill/openfold3-native.json
target/release/fireprot compare-npz reference.npz candidate.npz
```

The supplied execution profile uses existing DiamondHill paths and HIP device 7.
Run from the FireProt root on that host. Build the HIP identity probe following
[the worker protocol](worker-protocol.md#independent-amd-device-monitor).
Use a fresh output directory and run ID for every request; the supervisor refuses
to reuse an output directory. The request fixes source commit, input/checkpoint
paths/hashes, precision, additional recycle count, diffusion steps, sample count
and seed. The example retains 1000 residues, 3 additional recycles (4 trunk
passes), 200 diffusion steps, one sample, FP32, seed 101, and no MSA/template search.

Execution configuration names the Python virtual environment, model-worker
package, HIP identity probe, AMD SMI command and required environment paths.
Python executable paths preserve virtual-environment symlinks: resolving them
to the system interpreter would change the model dependencies. `PYTHONPATH` and
GPU visibility are set by Rust. SIGINT/SIGTERM request supervised cancellation.

`inspect` validates the request without opening weights or using a GPU.
`export-source ROOT COMMIT` prints a pinned source manifest for deployments that
synchronize files without transferring Git metadata. `compare-npz` performs
exact dtype/shape/value-byte comparison of every array key.

## Ownership and outputs

Rust performs source/input/weight preflight, launches the process, independently
checks HIP-to-SMI identity, continuously samples GPU processes, enforces protocol
ordering and verifies output hashes/content before reporting success.

`python/fireprot_workers` contains upstream model invocation, HIP identity
reporting, synchronized model timing and tensor serialization. It imports the
verified upstream checkout directly. It does not call the old benchmark harness,
choose isolation policy or decide whether the task passed. OpenFold3 parameters
are applied explicitly and checked against the instantiated model. Unsupported
model/backend combinations fail clearly.

Each run writes:

- `request.json` and `provenance.json`: resolved request and verified identities;
- `stdout.log` and `stderr.log`: model logs, separate from control messages;
- `predictions.json`: workload/seed, sample IDs, tensor descriptors and timing;
- `sample-N.npz`: one sample's coordinates and confidence arrays;
- `native/`: upstream output structures and associated files;
- `supervisor.json`: Rust's terminal decision, elapsed time, sample count and errors.

The model output gate requires every requested sample exactly once, verified
archive paths, matching descriptor names/shapes/storage dtypes, model-required
coordinate/confidence fields and finite numerical values. BF16 is serialized as
raw uint16 bits with explicit original dtype; BF16-to-FP32 conversion is rejected.
OpenFold3 batch/sample axes are split into individual sample archives. AtlasFold's
future worker must preserve its `[residues,14,3]` coordinate layout.

## Remaining migration gates

The CLI is not yet a replacement for every maintained workflow. Pending work
includes the other model workers and accelerated routes, Rust operator bridge,
resource/environment/build provenance, richer model schemas, CIF comparison,
statistics, deployment packaging, and full parity across the established length
buckets. The historical E2E speedups do not establish this entry point's overhead.

## First N1000 execution evidence

The first completed Rust request retains the native OpenBind-0 configuration:
model time 228.611 seconds, worker request time 254.505 seconds, and supervisor
wall time 262.461 seconds. The supervisor wall includes process launch/imports,
monitoring and output verification, but excludes prior Rust provenance preflight.
All output content checks and 297 isolation samples passed.

Exact output comparison with the older native capture failed. A fresh run of
the unchanged original harness also failed exact comparison with that capture:
maximum coordinate delta 9.74e-4 Angstrom, PAE 0.1032, PDE 0.2076. The instantiated
model configurations are equal. This demonstrates native repeat variation, but
is insufficient to establish migration equivalence. No tolerance was widened.
The exact migration gate remains unpassed; the route is experimental.
See [the complete comparison record](../provenance/rust_openfold3_integration.json).
These single requests are not a controlled E2E speedup measurement.

## AtlasFold with AtlasLM-3B

```bash
target/release/fireprot infer configs/requests/atlasfold-n1000-native.json \
  configs/diamondhill/atlasfold-native.json
```

The profile pins both original checkpoints and the independent AtlasFold source.
It preserves four additional recycles (five trunk passes), 30 diffusion steps,
one sample, seed 102, MLM probability 0.15, and the native BF16/FP32 boundaries.
`sampling_config` is passed explicitly; the worker checks recycle, step and
sample counts at the model boundary. It calls upstream `load_model` and `fold`
directly, with no benchmark-harness orchestration.

One N1000 request completed through the Rust supervisor with 216 isolation
samples. Its coordinates, pLDDT, PAE and pTM are byte-identical to the stored
seed102 native output, including shape and dtype. This verifies that exact
request only; native N1536 and accelerated AtlasFold routes are not migrated yet.
See [the request and complete comparison](../provenance/rust_atlasfold_integration.json).

AtlasFold's public API already exports FP32 NumPy arrays and a Python float pTM.
The worker preserves that native API. `atlasfold-model.json` separately records
internal output tensor dtypes, effective bucket, sampling settings and model
configuration; it is a worker diagnostic, not a supervisor-authenticated report.
