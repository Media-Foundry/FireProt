# Pinned environment content inventories

The Rust CLI accepts an optional `environment_inventory` artifact reference in
an execution profile. The referenced JSON is produced by
`fireprot capture-environment SPECIFICATION`. Its SHA-256 is fixed in the profile;
changing the JSON or any inventoried content prevents successful execution.

```json
{
  "schema_version": 1,
  "files": [{"name": "python", "path": "/models/venv/bin/python"}],
  "trees": [
    {"name": "packages", "path": "/models/venv/lib", "suffixes": []},
    {"name": "base-python", "path": "/models/base-python/lib", "suffixes": []},
    {"name": "rocm", "path": "/opt/rocm/lib", "suffixes": []}
  ]
}
```

```bash
fireprot capture-environment environment-spec.json > environment.json
sha256sum environment.json
```

Add the resulting absolute path and digest to the execution profile:

```json
"environment_inventory": {
  "path": "/deployment/environment.json",
  "sha256": "REPLACE_WITH_THE_RECORDED_SHA256"
}
```

The spec lists absolute files and recursive trees. Empty suffix lists include all
files, including package metadata, `.pth`/editable-import hooks, bytecode, shared
libraries and package data. An explicit suffix filter narrows that scope. Input
names must yield unique identities; linked files/directories are resolved and
cycles rejected by the shared tree-inventory service. File additions/removals,
content changes and retargeted canonical paths fail comparison. Empty-directory
changes are not represented in the file inventory.

## Enforcement and ownership

Rust reads the exact manifest bytes whose digest it checks, then enumerates and
hashes the declared inputs. The execution record stores the manifest pin and the
verified file identities under `environment/` names. Revalidation occurs during
initial launch capture, before Start authorization and after the worker process
group has stopped, before output acceptance. A mismatch prevents an accepted run
receipt. Existing output, device and process-isolation checks remain mandatory.

The CLI forces `PYTHONDONTWRITEBYTECODE=1` for inventoried workers. Installed-profile
generation preserves the inventory reference. Omission retains legacy launch
behavior and omits the new execution-record field, preserving its old JSON shape.

The content-inventory implementation is Rust-owned. Model-specific Python code
continues to perform upstream/tensor work; no package-management policy moves into
the workers. Discovery of what must be inventoried remains explicit. A Python
interpreter's `sys.path`, `sys.prefix` and `sys.base_prefix` are useful discovery
observations, not proof that every future import or dynamic dependency is covered.

## Scope and cost

A virtual environment can import its base interpreter's standard library, system
site-packages, editable checkouts and `.pth`-injected paths. Inventory all relevant
real roots, and inspect import hooks that name additional source locations. A
missing ZIP import candidate can be covered by inventorying its parent directory.
Published kernels and model source/weight/resource pins retain their separate
checks.

This feature establishes content identity for the declared files; it does not
claim a hermetic environment or automatic discovery of late-loaded plugins,
operating-system libraries, drivers or external cache contents. An inventory may
be incomplete even when every declared byte matches. Complete model-environment
coverage still requires discovery and an explicit audit.

All unique canonical file bytes are rehashed on every verification. This has
measurable overhead and can produce large JSON records; aliased import roots
still retain every declared identity. Within one Unix capture, duplicate
canonical-file reads are avoided as described below. It is an explicit strict validation option, not a claimed
performance optimization. Keep manifest generation outside the measured inference
interval and report launch/preflight, model and supervisor times separately.

## DiamondHill validation (2026-10-05)

The three explicit `configs/diamondhill/*-inventoried.json` profiles pin snapshots
stored under `/media/IntelSSD/fireprot/runtime/environments/`. They cover the model
Python library trees, base interpreter library tree, ROCm library tree, interpreter
binary/config and additional real import roots observed during discovery. Protenix
had two injected project roots outside its environment; those roots are included.
Editable import hooks remain in the inventoried package trees; upstream model
source still has its separate source pin. See the recorded discovery/specifications
in [integration evidence](../provenance/rust_environment_inventory_integration.json).

| Model | File identities | Standalone capture seconds | N1000 model seconds | Supervisor seconds | Isolation samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| AtlasFold + AtlasLM-3B / fused HIP | 118,560 | 25.01 | 77.183 | 127.118 | 104 |
| Protenix Mini-default / FireProt | 102,190 | 17.26 | 24.679 | 70.513 | 43 |
| OpenFold3 / strict deterministic native | 134,776 | 18.31 | 236.261 | 305.453 | 305 |

These are diagnostic observations under shared host activity. Initial inventory
capture and source/weight preflight are outside supervisor timing; the supervisor
interval includes repeated inventory checks before Start and after worker exit.
No E2E speedup comparison is inferred. Multiple identities can refer to the same
canonical file; AtlasFold has 68,281 distinct canonical paths.

All three runs completed with verified physical devices and accepted archives.
AtlasFold and OpenFold3 match their matching native references bit for bit;
Protenix passes the existing 1e-3 coordinate/summary-confidence thresholds. Rust
`collect-timings` reverified the large execution records and all output artifacts.
The default profiles remain unchanged because strict scanning has nontrivial cost
and the complete runtime coverage audit is still unfinished.

116 Rust tests, inventory tests on Rust 1.85, formatting and Clippy pass. Tests
exercise package changes before Start and after predictions, additions to import
trees, manifest changes, alias retargeting and duplicate names. A separate regression
first reproduced cancellation during output validation incorrectly producing success,
then passed after the final acceptance checkpoint was added. The GPU runs preceded
that last cancellation/timeout guard change; the scientific and inventory paths
were unchanged, and CPU tests verify the new failure behavior.

## Scan-local canonical-file reuse

On Unix, each environment capture now hashes a canonical file once, retaining
all declared file/tree aliases in the returned inventory. A fresh capture at
every verification checkpoint rehashes the bytes. No digest is reused across
requests or Start/acceptance checks. Default build-tree callers keep their
existing per-entry hashing behavior.

Before reuse and at the end of a scan, Rust checks device/inode, size, mode,
mtime and ctime (including nanoseconds). Replacement or a metadata change during
capture rejects the scan. These guards do not create a filesystem snapshot. Metadata observations have
filesystem granularity limits and cannot establish immutability; inputs must
still remain unchanged during capture. Non-Unix builds retain
per-entry reads. The schema, aliases, canonical paths and digest bytes are
unchanged, and all three existing pinned DiamondHill inventories still match.

Sequential capture-only measurements use explicit before/after/after/before/
before/after order for AtlasFold, with no filesystem cache flush or discarded
warmup. Its three-run medians change from **15.855 s to 12.202 s (1.299x)**,
about 23% less scan time. Its 118,560 identities refer to 68,281 canonical files.
Protenix and OpenFold3 each have one before/after equality check (13.723/10.746 s
and 16.875/13.484 s); those single observations do not establish stable performance
ratios. Every emitted inventory is byte-identical to its existing pinned manifest.
These are inventory-command wall times, not inference or E2E speedups.

All 84 runtime tests, ten relevant Rust 1.85 tests, workspace Clippy and format
checks pass. Tests exercise one read per canonical file within a scan, fresh
reads on later scans, same-sized rewrites with restored mtime, inode replacement,
late changes, alias preservation and supervisor rejection of changed environments.
See `provenance/rust_environment_scan_dedup.json`.

## Loaded-library coverage audit in progress

Point-in-time process maps from two AtlasFold workers expose system libraries
and locale data outside the current environment roots. Device mappings require
separate hardware/driver evidence. The Rust bridge and published operator use
separate execution/kernel identities; absence from the environment inventory
alone does not imply that such an artifact is unverified.

`provenance/atlasfold_loaded_environment_audit.json` records the observed paths
and their classification. It is a discovery artifact, not complete process-
lifetime coverage. The remaining audit must bind the observed system dependencies
and loading paths, check the other model environments, and account for runtime
plugins and driver identity. The current strict profiles remain opt-in.

## Extended system inventories: N1000 acceptance

The specifications in `configs/environments/diamondhill-*-system-v2.json`
extend the original inventories with system library and locale directories,
loader configuration/cache, OS and kernel identity, and readable AMDGPU module
version/source/taint fields. OpenFold3 also includes its external static cache
configuration. The corresponding `configs/diamondhill/*-system-inventoried.json`
profiles pin newly captured manifests; the original profiles remain available.

Specifications may declare `absent_paths`. Rust checks these paths before and
after capture and rejects files, directories, and dangling symlinks at those
locations. Paths must be absolute, have an existing directory parent, and contain
no traversal components. Empty or omitted lists preserve the original schema-1
serialization. An older implementation rejects this extension as an unknown
field. The new profiles require `/etc/ld.so.preload` and
`/usr/local/lib/x86_64-linux-gnu` to remain absent.

All three extended manifests have been captured on DiamondHill. N1000 inference,
reference comparison and archive replay pass with these exact pins: AtlasFold's
four public arrays and OpenFold3's eleven arrays are bitwise equal to their
matching references. Protenix's aligned coordinate RMSD is 9.539e-5 Angstrom and
maximum summary-confidence delta is 7.451e-8, within the unchanged 1e-3 gates.
See `provenance/rust_system_environment_integration.json` for requests, receipts,
supervisor/device records and comparisons. Concurrent validation does not
establish E2E speedup. Observed library coverage and final installation acceptance
remain open; the SSH interruption prevented live map capture for these AtlasFold
and OpenFold3 workers.
The checks observe mutable filesystems; they do not provide an atomic snapshot,
prove every transient library load, or lock GPU firmware identity.

The current implementation passes all 172 Rust workspace tests on both the
local default toolchain and Rust 1.85. The absence tests cover files, directories, dangling links, invalid
paths, legacy serialization, and supervisor rejection before Start and after
worker output. These CPU results do not substitute for the pending GPU runs.
