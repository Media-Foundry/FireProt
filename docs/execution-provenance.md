# Worker launch and device provenance

Rust records `execution.json` before spawning a worker. `supervisor.json` includes
its identity digest, the announced device, independent verification status and,
for the AMD SMI monitor, the resolved HIP/SMI/PCI/architecture mapping. A Ready
message alone never establishes a verified physical device or successful run.

## Explicit launch environment

The supervisor clears the inherited process environment and supplies exactly the
recorded map. The `explicit-v1` policy copies only these ambient host values when
present:

```text
PATH HOME USER LOGNAME TMPDIR TEMP TMP LANG LC_ALL LD_LIBRARY_PATH PYTHONHASHSEED
```

Execution-profile `environment` values then override those values. Rust finally
sets `HIP_VISIBLE_DEVICES`, `FIREPROT_REQUEST`, `FIREPROT_OUTPUT_DIR` and
`FIREPROT_WORKER_SOCKET`, and removes competing ROCr/CUDA/GPU ordinal selectors.
The CLI supplies `PYTHONPATH` and `PYTHONNOUSERSITE`. Backend switches, allocator
settings, thread counts and cache paths belong in the execution profile; setting
them only in the parent shell no longer changes inference silently.

The recorded map describes the environment at process creation. Model packages
may subsequently modify their own environment. Temporary socket/run paths make
separate request identities differ even when scientific inputs are the same.

## Explicit worker CPU affinity

On Linux, `FIREPROT_WORKER_CPU_AFFINITY` may be set in the profile environment
to `inherit` or a comma-separated CPU list such as `64-127,192-255`. Absence
preserves inherited affinity. Empty, malformed or out-of-range masks fail; an
explicit mask must be contained in the spawning thread's allowed CPU set.

Rust briefly applies the requested mask to the spawning thread so the worker
inherits it, then restores that thread before supervision proceeds. The guard
cannot move between threads and also restores on spawn failure. A restoration
failure after spawn follows the normal failed-run archive and child-group
cleanup path. The workspace's prohibition of unsafe code remains in effect.

The requested setting is part of the hashed launch environment. Before authorizing
Start, Rust reads the main worker's actual affinity and rejects a mismatch.
This is not a guarantee that every thread created by a model runtime retains
the mask forever. No memory-placement policy or system-wide CPU setting changes.
CPU lists are host-specific; the ESMC profile's list was measured on DiamondHill.

## Identity contents

- Absolute invocation path, preserving `venv/bin/python` spelling.
- Canonical executable path and SHA-256, exact argv and canonical working directory.
- Effective launch environment.
- Named content identities for explicitly declared files and source trees.
- A schema-versioned identity digest, computed over Rust compact JSON with its
  digest field empty; it is not the byte hash of the formatted JSON file.

The CLI declares its own executable, worker entry point, HIP probe, AMD SMI
entry point and root import shim. It includes `pyvenv.cfg` when present, and the
Rust bridge for accelerated routes. It inventories Python worker/tensor-boundary
sources and AMD kernel sources. These inventories identify the checkout's
available implementations; published kernel manifests separately identify the
compiled libraries actually loaded by model workers.

Rust rechecks declared inputs before authorizing Start and before accepting
predictions. Linux `/proc/PID/exe` independently confirms that the launched process
is the recorded native executable. PATH lookup occurs once and the resulting
absolute invocation path is used for both recording and launch. A source change,
retargeted input or mismatched executable prevents successful acceptance.

## Scope and remaining work

For an independently qualified monitor replacement, E2E comparison specifications
may set `monitor_treatment: true` and declare exact `launch.inputs.amd-smi`
identity changes. It defaults to false, cannot add/remove a monitor, and is
rejected outside E2E timing. Other protected programs and scientific conditions
remain immutable. Baseline-repeat checks retain default protection; all normal
archive/content checks still run. This comparison option does not qualify a
monitor implementation by itself. See the [native ESMC monitor](esmc-native-monitor.md).

This is launch provenance on a trusted host, not a hermetic process sandbox or a
complete Python package lock. It does not inventory all installed distributions,
late-loaded shared libraries, driver state or external caches. Dynamic dependencies
of the Python executable and AMD SMI's interpreter are not fully captured here.
Optional pinned [environment inventories](environment-inventories.md) now add
explicit package/import-root and runtime-library file trees to all three launch
verification points. Coverage still depends on declared roots; automatic loaded-library
and driver coverage remains unfinished. Python/HIP/Torch version strings in prediction
outputs remain observations of the model process.

`supervisor.elapsed_seconds` starts at process spawn and excludes source/weight
preflight and initial launch-identity capture. No E2E speedup follows from these
provenance changes. Pre-launch setup errors can leave a reserved output directory;
post-launch failures retain `supervisor.json` and logs. Existing run directories
are never reused.

## Validation on DiamondHill (2026-10-04)

[The integration record](../provenance/rust_execution_integration.json) contains
five complete requests, their effective launch maps, declared input hashes and
independently verified physical mappings. Every recorded input was rehashed again
when collecting the evidence. Rust tests pass 94 cases; Python tests pass 247
cases with 110 GPU-dependent skips. Clippy and formatting pass. Failure tests
cover source mutation before Start and before final acceptance, incorrect device
identity, invalid environments and executable mismatch; they retain the existing
protocol, cancellation and output-content gates.

| Model / route | Length | HIP / SMI | PCI | Supervisor seconds / isolation samples |
| --- | ---: | --- | --- | ---: |
| AtlasFold + AtlasLM-3B, fused HIP | 1000 | 0 / 2 | `0000:32:00.0` | 94.536 / 96 |
| AtlasFold + AtlasLM-3B, fused HIP | 1536 | 1 / 3 | `0000:35:00.0` | 216.256 / 232 |
| Protenix Mini-default FP32, FireProt | 1000 | 2 / 0 | `0000:11:00.0` | 42.704 / 41 |
| Protenix Mini-default FP32, FireProt | 1536 | 3 / 1 | `0000:14:00.0` | 62.450 / 62 |
| OpenFold3 / OpenBind-0, native | 1000 | 4 / 6 | `0000:ae:00.0` | 265.882 / 296 |

AtlasFold keeps seed 102, four additional recycles, one sample, native precision
and 30/100 diffusion steps; all four public output arrays remain bitwise equal to
the stored references. Protenix keeps seed 101, four total cycles, one step and
one sample. Native-reference aligned RMSD is 8.92590e-5 / 1.02133e-4 Angstrom and
summary-confidence max delta is 3.81470e-6 / 7.62939e-6 at N1000/N1536. Both pass
the existing 1e-3 gates against native and preceding FireProt references.

OpenFold3 keeps seed 101, three additional recycles, 200 diffusion steps, one
sample and the existing FP32 request. Execution, output-content and isolation
checks pass, but exact comparison with the previous Rust-native request fails.
Its numerical migration gate remains unpassed. No tolerance was widened and no
OpenFold3 optimization was promoted. An initial setup attempt found that its
execution profile had not been deployed; it failed before creating a run directory
or worker. The existing profile was synchronized before the successful launch.

These concurrent requests validate behavior and provenance, not an E2E speedup.

With `environment_inventory` configured, the manifest pin and its verified file
identities are included in the execution digest. Bytecode writing is disabled in
the worker. Existing records without an inventory retain their previous serialized
shape. Cancellation and timeout are rechecked after pre-Start validation and after
output validation, before authorizing work or accepting predictions. Synchronous
hashing can delay reaching those checkpoints; it is not an interruptible filesystem
snapshot or a concurrency sandbox.
