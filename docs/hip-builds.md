# Direct HIP builds and C ABI loaders

## Maintained ownership

`fireprot-runtime::hip_build::HipBuildSpec` now builds the standalone CK Tile
attention and hipBLASLt BMM libraries. These loaders no longer own source/header
hashing, file locks, subprocess compilation or cache publication in Python.
The shared `build_process::CompilerCommand` also serves Torch helper builds, so
both paths use the same lifetime guard, cancellation checks, timeout and failure
log handling. Original compiler/linker flags and C ABI signatures are retained.

```text
fireprot build-hip SPECIFICATION_JSON CACHE_ROOT
```

The additive ABI-1 bridge function is `build_hip(specification_json, cache_root)`.
It releases the GIL during work and preserves Python signal exceptions. The
result contains `directory`, `cache_hit` and `receipt`. Build all executables:

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
```

## Recipe contract

`HipBuildSpec` contains `build` (schema-1 `BuildSpec`), named `compiler` and `guard`
file roles, ordered `sources`, and positive `timeout_ms` / `lock_timeout_ms`.
Input paths are absolute. Arguments are assembled without a shell in this order:

1. `build.arguments.compile` (required ordered flags).
2. `-I` for each ordered `build.arguments.include` path.
3. Staged source copies, in declared order.
4. `build.arguments.link` (required ordered flags).
5. `-o PACKAGE_DIGEST.so` in the staging directory.

Every include directory requires a full declared tree inventory. Sources are
copied and rehashed before compilation. Rust binds the roles, invocation path,
timeouts and direct-process version into `fireprot_hip_process` in the receipt.
`command.json` records the actual compiler argv. The compiler invocation preserves
its declared path while checking its canonical executable identity. Library
filenames use the full build digest. Only successful, verified output is published
into the transactional cache.

The guard accepts a program, its argv and a final status path. Its lifetime socket
is never inherited by the compiler. It remains the process-group leader until the
caller cleans up or disconnects, including after the compiler exits. Existing
Torch requests retain the same helper/recipe argv through this common service.

## Integrated libraries

CK attention still requires the measured CK version
`8d1ae90eff7d022f26019ec55b2ec6a7674b3112`. Its ordinary, exact-N1536 and
vector-bias-N1000 entry points retain their original ctypes signatures and
capability gates. The generic hipBLASLt wrapper retains its explicit linker
search path, rpath and `-lhipblaslt`; its linked library bytes are now hashed,
replacing the previous size/mtime identity. The optional N1536 hipBLASLt inference
route remains disabled by the existing default deployment profile.

The Python boundary describes compiler/header/library locations, checks the CK
version and imports the accepted library through ctypes. Rust scans dependencies,
hashes inputs, owns the process/cache and validates reuse. Both expose Rust
receipts through their existing extension status records and Protenix runtime
manifests. Missing tools, rejected receipts or load failures retain native fallback.

## Limits

The `hip-cabi-1` recipe inventories ROCm/Clang/system/GCC headers and declares
compiler tools, their load-time dependencies and explicitly linked libraries.
It does not fully inventory runtime plugins or hipBLASLt device-library data.
This is a trusted build-host recipe, not a hermetic environment or a security
boundary against concurrent writes by the same user. Individual hashing phases
are not interruptible; lock wait and compilation have separate ten-minute limits.
The inherited process-group containment limits in [the process guide](torch-build-process.md)
apply. On-disk compiler logs are not capped.

C ABI builds are cache artifacts, not published deployment bundles yet. They do
not declare a Python SOABI, so the current Python-extension publisher rejects
them. Extending pinned artifact acceptance to a typed C ABI manifest remains
separate work. Historical benchmark scripts retain their original build paths
for reproduction; the maintained inference loaders use Rust.

## Validation on DiamondHill (2026-10-04)

[Versioned evidence](../provenance/rust_hip_build_integration.json) records source
hashes, compact build receipts, complete requests, output hashes and comparisons.
Both C ABI libraries passed cold Rust builds and independent CLI cache reuse.
The test results were 90 Rust tests, 229 Python tests (110 GPU-dependent skips),
60 Torch/Atlas GPU tests, and 140 Protenix operator tests (two skips). One old
disable test initially failed because it intercepted the removed Python subprocess
interface; its build interception was updated, preserving the no-build assertion,
and the test then passed. Compiler lifetime, cancellation and failure tests remain
part of the Rust suite. Clippy and formatting checks pass.

Four complete requests ran concurrently on separate GCDs with continuous PCI-based
isolation monitoring. Their timing is diagnostic, not a new speedup comparison.

| Model / route | Length | Correctness against stored native reference | Supervisor seconds / isolation samples |
| --- | ---: | --- | ---: |
| AtlasFold + AtlasLM-3B, HIP mask/bias | 1000 | Coordinates, pLDDT, PAE and pTM bitwise equal | 143.968 / 152 |
| AtlasFold + AtlasLM-3B, HIP mask/bias | 1536 | Same four arrays bitwise equal; bounded native reference | 363.407 / 403 |
| Protenix Mini-default FP32, FireProt | 1000 | Aligned RMSD 9.24218e-5 Angstrom; summary max delta 3.81470e-6 | 56.240 / 55 |
| Protenix Mini-default FP32, FireProt | 1536 | Aligned RMSD 1.26343e-4 Angstrom; summary max delta 7.62939e-6 | 83.224 / 84 |

AtlasFold retains seed 102, four additional recycles, one sample, native precision,
and 30/100 diffusion steps at N1000/N1536. Both requests recorded 528 HIP attention
calls and zero native fallbacks. Protenix retains seed 101, four total cycles, one
diffusion step and one sample. Both Protenix requests also pass the established
1e-3 aligned-coordinate and summary-confidence gates against preceding FireProt
outputs. Raw Protenix confidence arrays remain diagnostic and are not bitwise
equal. Their CK build receipts use `hip-cabi-1`; hipBLASLt inference stays disabled.
