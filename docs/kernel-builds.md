# Native kernel build identity

## Migration status

`fireprot-runtime::build` implements the shared build identity primitive in Rust.
`fireprot-runtime::build_cache` adds transactional cache management on local Unix
filesystems. The existing Python loaders are not connected yet. Their historical cache names
and build behavior remain in use until each loader is migrated and GPU parity is
verified. The identity primitive does not compile or load an extension. The cache service
serializes a supplied build callback and publishes its declared library artifacts.

## Contract

`BuildSpec` schema 1 records a package, target architecture, ordered file inputs,
ordered arguments for each build stage, effective environment values and explicit
version metadata. Rust resolves each file to its canonical path and hashes its
contents. Inputs must include sources, transitive headers, compiler executables
and build helpers relevant to the recipe. A missing file, directory input,
duplicate input name, unsupported schema or invalid package/target fails.

Torch/HIP version strings alone do not identify a compiler. Adapters must supply
Python ABI, framework versions and recipe versions as well as actual compiler
and header inputs. Environment absence differs from an explicitly empty value.
Argument order and source order are significant. Canonical paths participate in
the digest because include resolution and generated binaries may depend on them.

The schema uses deterministic JSON field order, sorted map keys and a versioned
domain prefix. It does not concatenate unframed source/version strings. Cache
names use the complete SHA-256 digest. Changing source/header/tool bytes,
arguments, environment, versions, target or input order changes the identity.

Persisted `BuildIdentity` records retain all inputs. `verify()` reconstructs the
identity from current files and rejects modified receipts or changed/missing
inputs. `cache_name()` validates the receipt digest but does not rehash files;
callers must still verify inputs and built artifacts before reusing a cache.

## Remaining integration

A descriptor is only as complete as its recipe. This API cannot discover missing
transitive dependencies, establish deterministic compilation or prevent concurrent
file changes during a later compiler invocation. The integrated build service
must hold the cache lock, snapshot or revalidate its inputs around compilation,
verify the resulting library, and publish its receipt atomically. Interrupted
builds must never become reusable successes. PyTorch-specific compilation/loading
may remain a tensor ABI boundary; build policy and cache acceptance belong in Rust.

Next connect this contract to the bridge and migrate the HIP/CK loader recipes,
then rerun their operator and long-model correctness gates before moving kernels.


## Transactional cache service

`get_or_build` holds an OS advisory lock for the full cache transaction. Lock
files are persistent and never unlinked, so all waiters lock the same inode.
Before reuse, Rust rehashes declared inputs, checks receipt identity, and hashes
all declared artifacts. Corrupt or incomplete published entries return an error;
the service never silently deletes or overwrites them.

On a miss, the callback builds in a unique temporary directory under the cache
root. It must finish its compiler children before returning and return flat
library filenames. Empty/duplicate lists, missing files, symlink artifacts and
paths outside that directory fail. Rust rechecks input identity, syncs artifacts
and the receipt, then renames the directory atomically and syncs its parent.
The callback must not embed temporary build paths as runtime dependencies.

Errors drop the temporary directory. SIGKILL can leave an ignored staging
directory, but the kernel releases the lock and no completed receipt is published
at the final cache path. Tests kill a child while it holds the build lock and
verify that a fresh build succeeds instead of reusing its partial output. Other
tests cover concurrent callers, compiler failure, input changes during builds,
corrupt outputs, missing receipts and invalid artifact paths.

The cache assumes a user-owned local filesystem and cooperating writers. It is
not a security boundary against another process modifying the same user's files.
Before/after hashes detect persistent input changes, not changes restored while
the compiler is reading. Immutable input snapshots and complete recipe inputs
remain integration requirements. Loading is outside this service; only declared,
verified artifacts may be loaded, and intermediates must not be reused for a
future compilation. No HIP/CK loader uses this service yet.


## Python bridge

The explicitly built `_fireprot_bridge` library now exposes
`cached_build(specification_json, cache_root, callback)`. Rust parses the schema,
hashes files, resolves the cache name, owns the lock and decides reuse/publication.
The Python callback receives `(staging_directory, module_name)` and returns a
list of flat artifact filenames after its compiler processes finish. The result
is JSON containing `directory`, `cache_hit` and the complete `receipt`.

The bridge releases the GIL during file hashing, cache verification and lock
waiting. It reacquires the GIL only to execute the callback; Python subprocess
and compiler calls can release it normally. All callback exceptions, including
`KeyboardInterrupt` and `SystemExit`, propagate as the original Python exception
objects after Rust drops its staging directory and lock. Callbacks must not
reenter this API or wait for another caller of the same cache identity. Same-thread
reentry is rejected before acquiring another lock. Lock acquisition currently
waits for the active builder; timed/cancellable acquisition is not implemented.

This additive API retains bridge ABI version 1. Rebuild the bridge before using
it. Operator dispatch remains unchanged. No production HIP/CK recipe is migrated
by merely exposing this interface: dependency enumeration, immutable inputs,
Torch compilation/loading integration and fresh GPU parity are still required.

## First HIP integration candidate

`FIREPROT_RUST_BUILD=1` selects an experimental Rust cache recipe for AtlasFold's
mask/bias extension only. It is not the default or a completed loader migration.
The Python boundary discovers Torch/ROCm/Python header files, Torch shared
libraries, hipify helpers and selected compiler executables, and passes their
paths to Rust for content hashing. Rust owns cache acceptance and publication.
The recipe fixes gfx90a, preserves the existing optimization flags and copies
kernel sources into staging before compilation. A disposable Python process runs
Torch's extension compiler. The model process loads only the library returned by
the completed Rust transaction. Compilation failures retain native fallback.

The manifest records the full receipt and a dependency-coverage statement. System
compiler runtime/header dependencies and all compiler environment effects are
not yet closed; directory discovery is still Python glue. The candidate must
remain explicit until these gaps are resolved and long-model parity passes.
It must not be described as hermetic, promoted, or fully migrated. Other HIP/CK
loaders still use their preceding build paths.

### Candidate validation

DiamondHill compiled the gfx90a mask/bias extension through the Rust transaction
with 14,559 declared input files. Eleven targeted GPU tests passed. Two subsequent
fresh Rust-supervised AtlasFold requests hit that cache: N1000 (30 diffusion
steps) and N1536 (100 steps), both with AtlasLM-3B, four additional recycles,
one sample, seed 102 and native precision. All four public output arrays match
their stored native/bounded-native references exactly in shape, dtype and bytes.
Each request selected 528 HIP attention calls with zero unsupported-call fallback.
Continuous PCI-mapped isolation passed with 152 and 401 samples respectively.

The model checks ran concurrently on distinct GCDs. Their 138.150/356.267-second
supervisor wall times exclude preflight and do not establish a new speedup.
The generated library's ELF dynamic section contains no temporary build-directory
RPATH/RUNPATH. Local Python validation passed 209 tests with 110 GPU cases skipped.
Complete requests, comparisons, receipt/library identities and dependency limits
are recorded in [candidate evidence](../provenance/rust_mask_build_candidate.json).
Dependency-coverage gaps above still prevent promotion.

## Rust dependency directory inventories

The next experimental recipe replaces Python globbing with `BuildSpec.trees`.
Each tree declares a stable name, absolute root and optional filename suffixes.
Rust enumerates entries deterministically, follows directory symlinks, rejects
cycles/broken links/non-regular inputs, and hashes all selected files. A depth
limit prevents unbounded recursive traversal. The receipt retains tree declarations
and `tree_files` separately from explicitly listed files.

Verification repeats directory enumeration as well as file hashing. Added or
removed headers and changed symlink targets therefore invalidate the identity,
including mutations made while the compiler callback is running. Filtering is
explicit: the hipify Python tree includes `.py` files and excludes generated
bytecode. Existing schema-1 receipts without trees retain their cache identity;
a golden receipt test pins this backward compatibility.

The recipe now also declares `/usr/include`, `/usr/local/include`, `/usr/lib/gcc`
and the ROCm HIP runtime library, in addition to the preceding environment roots.
This is broader dependency coverage, not a claim of a hermetic toolchain:
complete dynamic compiler/runtime closure and all compiler environment effects
still require verification. The route remains opt-in.

The directory-inventory recipe passed a fresh HIP build (11 explicit inputs plus
20,846 tree files), eleven GPU tests and both N1000/N1536 complete-model checks.
All public output arrays remain byte-identical, each request records 528 HIP
attention calls, and both requests reuse the new Rust cache. PCI isolation passed
with 153/403 samples. These parallel migration requests are not a performance
comparison. The local suite passed 69 Rust tests and 210 Python tests (110 GPU
cases skipped). See [directory-inventory evidence](../provenance/rust_build_tree_integration.json).


## Dynamic toolchain dependencies

`BuildSpec.dynamic_inputs` optionally declares an absolute `ldd` inspector and
trusted host ELF objects. Rust records the inspector, objects and resolved
load-time libraries as `dynamic_files`. Verification re-runs dependency resolution
and hashes the resolved files. Runtime addresses are excluded from identity.
Missing libraries, malformed output, inspector errors and non-ELF inputs fail.
Static executables are unsupported by this initial inspector. Schema-1 receipts
without dynamic inputs retain their original identity.

The experimental HIP recipe declares Python, Ninja, the C++ driver, GCC's
`cc1plus`/`collect2`/linker, HIPCC, Clang, LLD, shell interpreters and the HIP runtime.
The compiler child receives only recorded environment values, a fixed gfx90a
target, deterministic Python hashing and no visible GPU. Model execution keeps
its separately supervised device assignment.

This records load-time dependencies of the declared objects. It does not discover
libraries opened later through runtime plugins or capture the complete upstream
model environment. Torch libraries remain directly fingerprinted by the directory
inventory. The route stays experimental until the remaining integration and
packaging gates pass. See the deployment boundary in `docs/architecture.md`:
explicit artifact building and lightweight artifact acceptance during inference
remain the target workflow.

The dynamic-dependency recipe passed a fresh HIP build, eleven GPU tests and
N1000/N1536 full AtlasFold checks. Its receipt contains 11 explicit files,
20,846 tree files and 47 dynamic files. Both model requests hit the cache,
selected 528 HIP attention calls and retained exact public-output bytes.
PCI isolation passed with 156/410 samples. Local validation passed 72 Rust tests
and 210 Python tests (110 GPU cases skipped). Parallel migration timings are
not a new E2E speedup comparison. See [dynamic build evidence](../provenance/rust_dynamic_build_integration.json).
