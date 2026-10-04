# Torch extension build processes

## Ownership

`fireprot-runtime::torch_build::TorchBuildSpec` compiles a declared extension
through a disposable Torch helper. Rust owns staging, source copies, recipe JSON,
process environment, timeout, process-group cleanup and cache publication. The
helper calls Torch's extension compiler; Python still discovers its environment's
Torch/ROCm locations. This is not a replacement for HIP/CK GPU arithmetic.

The CLI accepts:

```text
fireprot build-torch SPECIFICATION_JSON CACHE_ROOT
```

The CPython bridge provides `build_torch(specification_json, cache_root)` and
releases the GIL for the entire Rust operation. Its JSON result contains
`directory`, `cache_hit` and `receipt`, matching the existing cache service.
Rebuild the bridge before using this additive ABI-1 function.

## Specification

The outer JSON has these fields:

| Field | Meaning |
| --- | --- |
| `build` | Existing schema-1 `BuildSpec`, including files, inventories, versions, environment and compiler arguments |
| `python` | Name of the interpreter entry in `build.files` |
| `helper` | Name of the compiler helper entry in `build.files` |
| `sources` | Ordered, nonempty source input names from `build.files` |
| `timeout_ms` | Positive compiler-process timeout; excludes input hashing and cache-lock waits |

`build.arguments.cxx` and `.hip` are required ordered arrays. Rust reserves
`build.arguments.fireprot_torch_process` to bind the orchestration version,
selected roles, source order, timeout and interpreter invocation path into the
identity. Source basenames must be distinct and have supported C/C++/HIP suffixes.
Copied source hashes must match their input identities before compiler startup.
The helper receives the recipe filename as its only argument. Its recipe contains
`name`, `sources`, `build_directory`, `extra_cflags` and `extra_cuda_cflags`.

The interpreter path must be absolute. Rust preserves its spelling when invoking
it because canonicalizing a virtual-environment symlink can change Python's
package environment. The resolved executable bytes are still fingerprinted, and
the symlink target is checked before invocation. A full Python-environment lock
is a separate requirement; these checks do not establish one.

## Failure behavior and limits

The compiler inherits only declared environment values, has closed stdin and
writes both output streams to `compiler.log` in staging. A nonzero exit or timeout
returns the final 16 KiB of that log in the error. Staging is discarded on failure;
no successful cache receipt is created. A successful helper must create the exact
`PACKAGE_DIGEST.so`; missing, symlinked or corrupted artifacts fail existing cache
gates. Cache hits still verify inputs and artifact hashes.

The process group is killed and the direct child reaped on normal Rust cleanup,
including timeout and compiler failure. An uncatchable termination of the Rust
host cannot run this cleanup; process-group ownership across worker cancellation
still needs integration. Cache-lock waiting is currently unbounded. Logs are
bounded when included in errors, but their on-disk size is not capped. Helpers
and build specifications must be trusted, and dependency coverage remains the
responsibility of each recipe. Do not call this API from a cache callback that
already holds the same build-identity lock.

## AtlasFold integration

The opt-in `FIREPROT_RUST_BUILD=1` mask/bias path uses this service with recipe 4
and a ten-minute compiler timeout. The former Python compile callback is removed
from that path. The existing pinned published-artifact path takes precedence and
continues using its validated binary. Other HIP/CK recipes remain to be migrated.

CPU tests cover cache reuse, changed inputs, explicit environment, compiler
failure diagnostics, missing output, timeout, descendant cleanup and interpreter
symlink identity. The bridge test exercises concurrent calls and propagated
compiler errors. These fixtures use synthetic libraries; GPU and full-model
validation must separately establish the real extension's correctness.

### DiamondHill validation

The real gfx90a extension compiled successfully with 11 explicit inputs, 20,846
inventoried tree files and 47 dynamic-library inputs. The CLI independently
reconstructed the recipe and verified reuse of the same cache identity. Eleven
GPU tests pass. N1000 and N1536 full AtlasFold requests retain dual original
weights, seed 102, four additional recycles, one sample and their respective
30/100 diffusion steps. Coordinates, PAE, pLDDT and pTM are bitwise equal to the
stored native/bounded-native references. Both select 528 HIP attention calls and
zero unsupported native fallbacks; PCI isolation passes with 156/399 samples.

The local workspace passes 80 Rust tests and 212 Python tests (110 GPU tests
skipped), formatting and Clippy. The interpreter-symlink regression test was
observed to fail before its fix. See [process integration evidence](../provenance/rust_torch_process_integration.json)
for source/binary identities, deployment profile, compiler receipt/log identity,
requests and comparisons. Parallel model runs establish migration correctness;
their timings are not a new E2E speedup measurement.
