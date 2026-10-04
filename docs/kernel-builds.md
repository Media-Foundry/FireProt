# Native kernel build identity

## Migration status

`fireprot-runtime::build` implements the shared build identity primitive in Rust.
The existing Python loaders are not connected yet. Their historical cache names
and build behavior remain in use until each loader is migrated and GPU parity is
verified. This primitive does not compile, lock, publish or load an extension.

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
