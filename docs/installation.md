# Install the Rust framework

FireProt can install its maintained CLI, compiler guard, HIP device probe, Rust
bridge, Python tensor/model boundary and AMD source kernels into a new Linux
prefix. Inference uses the installed files independently of the source/build tree.
The pinned upstream model checkout, model Python/ROCm environment, weights, inputs,
CCD resources and published kernel bundles remain external deployment inputs.

## Build and describe the inputs

```bash
cargo build --release -p fireprot-cli -p fireprot-runtime -p fireprot-bridge --locked
```

Build the HIP probe as described in [worker setup](worker-protocol.md). Create an
installation specification with the following schema, replacing every path and
SHA-256 with your actual deployment inputs. Use `sha256sum FILE` to obtain binary
digests. `profiles` contains the existing execution-profile objects, keyed by
installation-local names; examples are in `configs/diamondhill/`.

```json
{
  "schema_version": 1,
  "source_root": "/build/FireProt",
  "cli": {"path": "/build/FireProt/target/release/fireprot", "sha256": "REPLACE"},
  "bridge": {"path": "/build/FireProt/target/release/lib_fireprot_bridge.so", "sha256": "REPLACE"},
  "compiler_guard": {"path": "/build/FireProt/target/release/fireprot-compiler-guard", "sha256": "REPLACE"},
  "hip_probe": {"path": "/build/FireProt/runtime/fireprot-hip-probe", "sha256": "REPLACE"},
  "cache_root": "/var/tmp/fireprot-kernels",
  "profiles": {
    "openfold3": {
      "python": "/models/openfold3-venv/bin/python",
      "workers": "/build/FireProt/python",
      "hip_probe": "/build/FireProt/runtime/fireprot-hip-probe",
      "amd_smi": "/usr/bin/amd-smi",
      "source_manifest": null,
      "environment": {"OPENFOLD_CACHE": "/models/openfold3-cache", "PYTHONHASHSEED": "101"}
    }
  }
}
```

The installer preserves model-specific profile settings. It rewrites `workers`,
`hip_probe`, `FIREPROT_BRIDGE_LIBRARY` and `FIREPROT_BUILD_GUARD` to the installation,
and sets `FIREPROT_EXTENSION_CACHE` to the explicit external cache. It sets
`PYTHONDONTWRITEBYTECODE=1` so worker imports do not mutate the installation.
Model-environment availability and the complete execution-profile contract are
checked by `infer`; installation does not run or install model dependencies.

## Install, verify and run

```bash
target/release/fireprot install install.json /opt/fireprot/0.1.0 > installed.json
/opt/fireprot/0.1.0/bin/fireprot verify-installation /opt/fireprot/0.1.0 MANIFEST_SHA256
cd /tmp
/opt/fireprot/0.1.0/bin/fireprot infer /path/to/request.json /opt/fireprot/0.1.0/profiles/openfold3.json
```

Use the `manifest_sha256` returned in `installed.json`, retained independently.
The destination must be new and outside the source tree; existing directories
are never overwritten. A sibling staging directory is verified before Linux
`RENAME_NOREPLACE` publishes it atomically. An installation failure leaves no
published destination. Build inputs and binaries are copied and hashed; later
verification needs only the installed directory and pinned manifest digest.

```text
PREFIX/
  bin/fireprot
  bin/fireprot-compiler-guard
  bin/fireprot-hip-probe
  lib/fireprot/lib_fireprot_bridge.so
  lib/fireprot/python/fireprot/
  lib/fireprot/python/fireprot_workers/
  lib/fireprot/kernels/amd/
  lib/fireprot/fireprot/__init__.py
  profiles/NAME.json
  share/fireprot/{README.md,Cargo.toml,Cargo.lock}
  installation.json
```

The manifest records every installed file's SHA-256 and executable flag, framework
version and build-host OS/architecture. Verification rejects changed, missing or
extra files, symlinks and changed executable flags. It does not establish that
arbitrary caller-supplied binaries match source, prove compatible dynamic libraries,
or provide a cryptographic signature. Keep the build recipe and validation evidence
alongside the installation receipt. Full environment locking remains separate work.

Generated profiles contain absolute paths. To change prefix, install again at the
new destination; copying a prefix without updating profiles is not relocation.
The destination and cache parent must be writable by the installer/worker as
appropriate. Kernel compilation uses the external cache, leaving installed source
files untouched. No weights, predictions, upstream Git checkouts, build caches or
Git metadata are copied. Direct Python use must supply the installed Python path
and bridge/build/cache environment from the generated profile.

## Validation

The installation tests cover source-independent verification, preservation of model
profile settings, exclusion of runtime assets and bytecode, binary hash failures,
unsafe profile names, symlinks, extra/tampered files, changed execute bits and refusal
to replace an existing destination. Full-model installation results are recorded
separately in `provenance/rust_installation_integration.json` when available.

### DiamondHill installation acceptance (2026-10-05)

`/media/IntelSSD/fireprot-install-v1` contains 57 inventoried files. From `/tmp`,
the installed CLI completed five isolated model requests using the installed
worker/tensor boundary and existing external published kernel bundles:

| Model / route | Length | Native-reference gate | Isolation samples |
| --- | ---: | --- | ---: |
| AtlasFold + AtlasLM-3B / fused HIP | 1000 | All public arrays bitwise equal | 93 |
| AtlasFold + AtlasLM-3B / fused HIP | 1536 | All public arrays bitwise equal | 234 |
| Protenix Mini-default / FireProt | 1000 | Aligned RMSD 8.81228e-5 A; confidence delta 3.81470e-6 | 44 |
| Protenix Mini-default / FireProt | 1536 | Aligned RMSD 1.11646e-4 A; confidence delta 7.62939e-6 | 60 |
| OpenFold3 / strict deterministic native | 1000 | Coordinates and ten confidence arrays bitwise equal | 297 |

Rust reverified all five accepted archives. A separate installed-Python import
check verified the bridge and kernel-root locations. All six maintained loaders
then compiled and loaded from a fresh external cache using the installed compiler
guard and installed kernel sources. Their artifact hashes were rechecked; this
build/load check does not claim full-model numerical validation of those fresh
libraries. The model comparisons above used the existing pinned published bundles.
The installed inventory remained unchanged after all runs/builds.

112 Rust tests, three installation tests on Rust 1.85, formatting and Clippy pass.
The [evidence record](../provenance/rust_installation_integration.json) binds the
installation manifest, binary input pins, launch/archive receipts, references,
comparisons and cold-build artifacts. Concurrent timings are diagnostic only.
