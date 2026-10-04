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
releases the GIL during Rust work, reacquiring it briefly to check Python signals. Its JSON result contains
`directory`, `cache_hit` and `receipt`, matching the existing cache service.
Build the CLI, bridge and private guard together:

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
```

The guard is `target/release/fireprot-compiler-guard`. The AtlasFold environment
probe accepts `FIREPROT_BUILD_GUARD` as an explicit override. The selected executable
and its load-time dependencies are fingerprinted; an absent guard rejects the
experimental build path. The bridge ABI remains 1.

## Specification

The outer JSON has these fields:

| Field | Meaning |
| --- | --- |
| `build` | Existing schema-1 `BuildSpec`, including files, inventories, versions, environment and compiler arguments |
| `python` | Name of the interpreter entry in `build.files` |
| `helper` | Name of the compiler helper entry in `build.files` |
| `guard` | Name of the Rust compiler-guard executable entry in `build.files` |
| `lock_timeout_ms` | Positive timeout for acquiring the build identity lock |
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

A dedicated Rust guard remains the compiler process-group leader until the
caller closes an exclusive lifetime socket. EOF makes the guard kill its whole
group, including when the caller is terminated without running destructors. The
guard reports compiler exit through an atomic status file and stays alive until
the caller finishes cleanup. This also prevents a completed leader's PID from
being reused while the caller still owns the group. Helpers must keep their
children in this group; deliberately detached subprocesses are outside the contract.
An independently killed or failed guard is not a substitute for OS cgroup containment.

Normal Rust cleanup kills the group and reaps its direct child. SIGKILL of the
caller can leave an ignored staging directory, but no successful receipt is
published. CLI SIGINT/SIGTERM and Python KeyboardInterrupt trigger cancellation;
the original Python exception propagates. Same-thread recursive build calls are
rejected before lock acquisition.

Cache locks are acquired with nonblocking attempts and 20 ms polling, checking
cancellation and the configured timeout. The compatibility callback API uses a
ten-minute lock timeout. Cancellation is checked between hashing/validation phases
and before atomic publication; a single hashing phase is not interruptible. The
compiler has a separate timeout. Logs are bounded when included in errors, but
their on-disk size is not capped. Helpers/specifications must be trusted, and each
recipe remains responsible for complete dependency coverage.

## AtlasFold integration

The opt-in `FIREPROT_RUST_BUILD=1` mask/bias path uses this service with recipe 5
and separate ten-minute compiler and lock-wait timeouts. The former Python compile callback is removed
from that path. The existing pinned published-artifact path takes precedence and
continues using its validated binary. Other HIP/CK recipes remain to be migrated.

CPU tests cover cache reuse, changed inputs, explicit environment, compiler
failure diagnostics, missing output, timeout, descendant cleanup and interpreter
symlink identity. The bridge test exercises concurrent calls and propagated
compiler errors. These fixtures use synthetic libraries; GPU and full-model
validation must separately establish the real extension's correctness.

### Historical recipe-4 validation

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

### Cancellation regression coverage

A subprocess test first reproduced the old behavior: killing the build host left
its compiler descendant alive long enough to write another file. With the guard,
the same scenario no longer writes that file or publishes a cache entry. Other
tests cover lock timeout/cancellation without disturbing its owner, cancellation
after compilation but before publication, compiler timeout, CLI SIGTERM, original
Python KeyboardInterrupt propagation, interrupted callback-cache waits, and
same-thread recursive build rejection. The fixture compiler uses no GPU.

### Recipe-5 deployment validation

DiamondHill cold-built the guarded gfx90a extension with 12 explicit inputs,
20,846 tree files and 48 dynamic inputs. The independent CLI cache check and
N1000/N1536 full AtlasFold requests reuse that same identity. Both model requests
preserve the recipe-4 scientific workload and pass bitwise comparison of all four
public arrays, with 528 HIP attention calls and no unsupported native fallback.
PCI isolation records 166/399 samples. Eleven targeted GPU tests pass; the final
local suite passes 85 Rust tests and 216 Python tests (110 GPU tests skipped),
formatting and Clippy. These parallel model timings are not an E2E speedup claim.
See [guard integration evidence](../provenance/rust_compiler_guard_integration.json)
for the final source/binary hashes, guard identity, compiler receipt, requests and
exact comparisons. Other HIP/CK loaders and repository layout migration remain
outstanding.
