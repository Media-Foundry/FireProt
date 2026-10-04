# Rust operator policy bridge

## Ownership

`fireprot-core::registry` owns candidate ordering, exact-signature caching,
registration invalidation, native fallback decisions, rejection reasons,
selection counters and decision history. `fireprot-bridge` exposes this policy
in process through the CPython stable C ABI (PyO3, Python 3.10 or newer).
There is no per-operation subprocess or GPU tensor serialization.

`fireprot/ops/registry.py` preserves the model-facing Python interface. It holds
Python implementation functions, constructs tensor signatures, and invokes the
selected function or the caller's native fallback. Existing Python capability
predicates still perform model/framework-specific checks. They receive an
equivalent immutable signature reconstructed from metadata; object identity is
not preserved. Porting these predicates and kernel build policy is separate work.

There is no second Python implementation of sorting, caching, counting or
fallback selection. Rust library loading is mandatory for this registry. Native
upstream inference paths that do not import the registry remain available.

## Build and load

Build on the deployment host before running a workflow that imports the registry:

```bash
cargo build --release -p fireprot-bridge --locked
PYTHONPATH=.:Protenix python -m pytest -q tests
```

The Linux source checkout defaults to `target/release/lib_fireprot_bridge.so`.
Set `FIREPROT_BRIDGE_LIBRARY` to an explicit library path for other deployment
layouts. A missing library or incompatible bridge ABI is an import error with
build instructions; inference never downloads packages or invokes Cargo.
Builds remain outside Git. Restart model workers after replacing a library.

The operation manifest retains `implementations`, `selection_counts` and
`decisions`, and adds `policy_engine` with the loaded library path, SHA-256 and
ABI version. Signature reports now include exact `strides`, also used in the
cache key. Manually created signatures may leave strides empty when unknown.
Decision arrays use Rust's deterministic field ordering, rather than Python
`repr` ordering. Consumers should identify a decision by its signature.

## Callback and state rules

- The Python shim serializes registry calls with an `RLock`, including callbacks
  that temporarily release the GIL. Registration cannot race a resolution.
- Capability predicates must not reenter or wait for another thread to access
  the same registry. PyO3 rejects reentrant state access; an ordinary predicate
  error rejects that candidate and records the reason.
- `KeyboardInterrupt`, `SystemExit` and other non-`Exception` control exceptions
  propagate without selecting another candidate, caching a fallback, or changing
  earlier counts and decisions.
- Callbacks participate in CPython cycle collection. Rust core closures retain
  weak references to callback slots; cycles through a Python owner are reclaimed.
- Registration and removal invalidate cached choices but retain audit history.
  `clear_runtime_state()` clears caches, counts and decisions, keeping candidates.
- Environment or capability policy changes require explicit cache invalidation.
  A returned implementation's execution errors propagate to the caller.

## Migration validation scope

The CPU suite covers original registry behavior, ties, cache invalidation, exact
strides, historical reports, concurrent calls, callback lifetime, reentrant
mutation, exception handling and invalid metadata. Rust workspace tests and
Clippy also run against the bridge crate.

DiamondHill validation uses the existing CK attention tests and complete
Protenix Mini-default N1000/N1536 requests, preserving FP32, four cycles, one
diffusion step, one sample and seed 101. It compares to stored pre-migration
FireProt and native Torch outputs using the unchanged 1e-3 aligned-coordinate
RMSD and confidence gates. These checks exercise Rust dispatch with existing
Python model glue; they do not establish a Rust Protenix worker integration.
Evidence and exact timings are recorded in `provenance/rust_operator_bridge.json`.
This migration makes no new GPU speedup claim.

The final monitored run records the following single-request results:

| Protenix length | Complete worker request | Model GPU time | Correct-device isolation samples | Output gates |
| --- | ---: | ---: | ---: | --- |
| 1000 | 27.627 s | 15.209 s | 23 | Pass against both stored references |
| 1536 | 46.052 s | 32.612 s | 38 | Pass against both stored references |

These are one fresh worker per length, with warm filesystem and compiled-kernel
caches. Worker timing excludes parent launch and preflight. The CPU suite passes
56 Rust tests and 186 Python tests; 110 Python GPU cases skip locally. The remote
targeted registry/CK suite passes 24 tests. The hardware-only Rust monitor test
remains opt-in and is not counted among these 56 tests.

The legacy Protenix harness assumes HIP and SMI ordinals match. The migration
validation explicitly probes HIP 7's PCI identity, resolves SMI 5, and adds that
SMI index to continuous monitoring, checking the mapping again after each run.
The extra visibility ordinal required by the old harness does not enable the
two-GCD path. The new Rust supervisor already performs PCI mapping internally;
the implemented Protenix worker removes this legacy harness dependency. Earlier
bridge trial timings without this correction are not accepted isolation evidence.


## AtlasFold dispatch

AtlasFold now uses a private instance of the shared Rust registry for chunked,
HIP and fused HIP routes. The former six Python counters are a compatibility
view of Rust selection counts. Eligible calls with unavailable extensions use
bounded native attention or original multiplication; unsupported calls use the
original upstream method.

Mask contents, module state and tensor eligibility are checked before every
resolution, including cache hits. Eligibility, extension availability, backend,
direction and actual tensor strides are included in signature metadata. No
mask value or tensor result is cached. CPU tests exercise same-shape mask
changes and extension availability changes; GPU tests exercise trained modules,
fallbacks, reinstall, stream behavior and precision boundaries.
