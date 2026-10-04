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
