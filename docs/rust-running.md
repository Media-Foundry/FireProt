# Rust inference entry point

The maintained Rust entry point is under active migration. OpenFold3 N1000 with
explicit strict deterministic execution passes exact native-output parity; see
[policy and evidence](openfold3-determinism.md). The legacy default-mode route has
native-repeat variation and remains unpassed for numerical migration parity. Workers
implement native OpenFold3/OpenBind-0, AtlasFold plus AtlasLM-3B, and Protenix
Mini-default monomer inference. Protenix also has a FireProt backend connected to
the Rust operator bridge. AtlasFold chunked, HIP and fused HIP routes also use Rust dispatch and supervision; accelerated OpenFold3 remains pending.

## Commands

Use `fireprot --help`, `fireprot help COMMAND` or `fireprot COMMAND --help` to
inspect command arguments. See [CLI structure and result conventions](cli-design.md).

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
paths/hashes, named data resources, precision, cycle/recycle count, diffusion steps, sample count
and seed. The example retains 1000 residues, 3 additional recycles (4 trunk
passes), 200 diffusion steps, one sample, FP32, seed 101, and no MSA/template search.

Execution configuration names the Python virtual environment, model-worker
package, HIP identity probe, AMD SMI command and required environment paths.
Python executable paths preserve virtual-environment symlinks: resolving them
to the system interpreter would change the model dependencies. `PYTHONPATH` and
GPU visibility are set by Rust. The supervisor clears other inherited settings;
put backend/allocator/thread/cache options in the profile. See
[launch and device provenance](execution-provenance.md). SIGINT/SIGTERM request
supervised cancellation.

`inspect` validates the request without opening weights or using a GPU.
`export-source ROOT COMMIT` prints a pinned source manifest for deployments that
synchronize files without transferring Git metadata. `compare-npz` performs
exact dtype/shape/value-byte comparison of every array key.

## Ownership and outputs

Rust performs source/input/weight/resource preflight, launches the process, independently
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
- `execution.json`: effective launch environment, executable and declared worker/source identities;
- `stdout.log` and `stderr.log`: model logs, separate from control messages;
- `predictions.json`: workload/seed, sample IDs, tensor descriptors and timing;
- `sample-N.npz`: one sample's coordinates and confidence arrays;
- `native/`: upstream output structures and associated files;
- `supervisor.json`: Rust's terminal decision, elapsed time, isolation count, device verification/mapping, execution digest and errors.
- `accepted.json`: successful-run receipt binding the original metadata, final supervisor report and accepted worker outputs.

Use [archive collection](run-archives.md) to reverify completed runs and summarize
their recorded timing intervals. Keep receipt hashes separately from the archive.

The model output gate requires every requested sample exactly once, verified
archive paths, matching descriptor names/shapes/storage dtypes, model-required
coordinate/confidence fields and finite numerical values. BF16 is serialized as
raw uint16 bits with explicit original dtype; BF16-to-FP32 conversion is rejected.
OpenFold3 batch/sample axes are split into individual sample archives. AtlasFold
preserves its `[residues,14,3]` coordinate layout.

## Remaining migration gates

The CLI is not yet a replacement for every maintained workflow. Pending work
includes additional model variants and accelerated routes,
environment/build provenance, richer model schemas, CIF comparison,
paired performance verdicts, deployment packaging, and full parity across the established length
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
That legacy default-mode comparison remains unpassed. A later explicit deterministic
configuration passes exact parity, as described above.
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
request only. The accelerated-route checks below extend coverage to N1536.
See [the request and complete comparison](../provenance/rust_atlasfold_integration.json).

AtlasFold's public API already exports FP32 NumPy arrays and a Python float pTM.
The worker preserves that native API. `atlasfold-model.json` separately records
internal output tensor dtypes, effective bucket, sampling settings and model
configuration. Its file hash is verified by the supervisor; the diagnostic content
is not an independent scientific validation.

## Protenix Mini-default

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
target/release/fireprot infer configs/requests/protenix-n1000-native.json \
  configs/diamondhill/protenix-native.json
target/release/fireprot infer configs/requests/protenix-n1000-fireprot.json \
  configs/diamondhill/protenix-fireprot.json
```

Equivalent N1536 request examples are supplied. These requests preserve FP32,
four total trunk cycles, one diffusion step, one sample and seed 101. They use
hipBLAS, disable TF32, MSA/RNA-MSA/template searches, and keep upstream cache and
fusion settings enabled. The FireProt execution profile explicitly enables the
existing validated long-sequence paths. This is backend migration, not a new
kernel optimization. Requests outside the current Mini-default FP32,
sequence-only single-monomer contract fail clearly; ESM variants, modified
residues, external features and multimer workers remain pending.

The request's `resources` map pins all four upstream CCD/data assets, verified
with streaming SHA-256 before launch. The worker passes these exact paths to
upstream configuration before importing CCD. Missing assets never trigger a
download. `provenance/sources/protenix-bb3be993.json` pins all 248 tracked source
files without modifying the independent remote Git checkout.

Upstream `inference_jsons` still owns feature construction, seeding and native
model invocation. The thin worker checks the instantiated workload, captures
exactly one query, exports coordinates and complete unrounded full-data
confidence arrays, retains summary confidence, and requires all native CIFs.
It detects errors swallowed by the upstream runner and restores its capture
wrappers on exit. Sample IDs retain generation order rather than ranking order.
Rust requires pLDDT/PAE/PDE/contact arrays with matching atom/token shapes and the
requested cycle count in the confidence summary. The FireProt route also emits
`operator-runtime.json`, included in the supervisor's hash-verified artifacts.

`model_seconds` is synchronized wall time around upstream `predict`, including
its input transfer; `request_seconds` includes model imports, initialization,
features, prediction and serialization. Supervisor wall adds worker startup,
monitoring and output verification, excluding the earlier provenance preflight.
No historical timing boundary is silently substituted for these fields.

For migration checks against an archived native CIF, export its model-1
coordinates losslessly to NPZ, then run the Rust comparator:

```bash
cargo run --release -p fireprot-validation --example compare_protenix -- \
  reference.npz reference-confidence.json sample-0.npz candidate-confidence.json
```

This example preserves the existing 1e-3 Angstrom aligned-RMSD and 1e-3 numeric
summary-confidence gates. It does not assert bitwise equality or compare raw
confidence arrays that were absent from historical CIF/summary references.

### Protenix migration evidence

One request per route and length completed on DiamondHill HIP 7 with Rust PCI
mapping and continuous isolation. All six existing-gate comparisons pass: two
native versus archived native, two FireProt versus archived FireProt, and two
FireProt versus the new native requests.

| Length / route | Worker request | Synchronized predict | Supervisor wall, excluding preflight | Isolation samples |
| --- | ---: | ---: | ---: | ---: |
| 1000 native | 79.688 s | 50.759 s | 86.426 s | 97 |
| 1000 FireProt | 25.812 s | 15.289 s | 32.423 s | 35 |
| 1536 native | 176.818 s | 144.545 s | 184.354 s | 209 |
| 1536 FireProt | 43.851 s | 32.477 s | 50.796 s | 56 |

These are migration checks with fresh workers and warm filesystem/kernel caches,
not a repeated, interleaved speedup experiment. Exact hashes for the native worker
and subsequent accelerated-routing build are retained. The migrated FireProt
route still selects the measured CK attention and packed HIP multiplication,
fast model construction and unused-template omission.

The new complete confidence captures expose native/accelerated maximum absolute
PAE differences of 0.009623 / 0.011969 Angstrom and PDE differences of
0.039787 / 0.058372 Angstrom at N1000 / N1536. Exact-array comparison therefore
fails, as do exact coordinates. Integer atom/token metadata is identical. No raw
array tolerance was introduced after seeing these results. The established
coordinate/summary gate remains distinct: maximum paired aligned RMSD is
1.074e-4 Angstrom and maximum paired summary delta is 3.815e-6.

See [complete requests, build identities, outputs and comparisons](../provenance/rust_protenix_integration.json).


### AtlasFold accelerated routes

Build both CLI and bridge before using the supplied requests:

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
target/release/fireprot infer configs/requests/atlasfold-n1536-hip-fused.json \
  configs/diamondhill/atlasfold-native.json
```

Requests cover `chunked`, `hip` and `hip_fused` at N1000 and N1536. The JSON
backend uses an underscore; filenames and the adapter name use `hip-fused`.
N1000 pads to 1024 and retains 30 diffusion steps; N1536 retains 100 steps.
Both preserve four additional recycles, one sample, seed 102, original dual
weights and native precision. The N1536 reference uses bounded native math
because unchunked native attention exceeds available memory.

Rust owns route selection, cache decisions and counters. Python rechecks actual
mask values and module/tensor eligibility on every call. Existing attention and
multiplication arithmetic is unchanged. `operator-runtime.json` records the
Rust library hash, decisions, counts and extension identity and is included in
hash-verified output artifacts. Tensor arguments stay in process.

These checks use separate GCDs concurrently for correctness. Their timings are
diagnostic and do not establish a new E2E speedup or migration overhead.


All six route/length requests passed exact shape, dtype and value-byte comparison
for coordinates, pLDDT, PAE and pTM. Every request selected 528 attention calls;
fused HIP additionally selected 528 multiplication calls. No unsupported-call
native fallback occurred. All six supervisors passed continuous PCI-mapped GPU
isolation and output acceptance. Local validation passed 58 Rust tests and 201
Python tests (110 GPU cases skipped); the remote targeted GPU suite passed 62.
See [requests, identities and complete comparisons](../provenance/rust_atlasfold_accelerated.json).


### Published AtlasFold mask/bias library

`configs/diamondhill/atlasfold-published.json` selects a pinned prebuilt library
for AtlasFold `hip` requests. Rust checks artifact integrity and compatibility;
the loader does not enter a build path. N1000/N1536 public outputs pass exactly.
See [publication, acceptance and deployment scope](kernel-artifacts.md). Direct
build orchestration and other kernel recipes remain migration work.
