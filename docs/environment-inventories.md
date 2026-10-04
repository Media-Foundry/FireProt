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

All bytes are rehashed on every verification. This has measurable overhead and
can produce large JSON records; shared libraries and aliased import roots can be
visited more than once. It is an explicit strict validation option, not a claimed
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
