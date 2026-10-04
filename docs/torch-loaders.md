# Maintained Torch extension loaders

## Ownership and compatibility

Four Python extension boundaries now delegate native build policy to Rust:

| Loader | Extension | Preserved compile options |
| --- | --- | --- |
| `mask_bias` | BF16 sequential mask/bias addition | C++ `-O3`; HIP `-O3 --offload-arch=gfx90a` |
| `pair_fusions` | Pair softmax, transpose and QKV kernels | Same flags as mask/bias; four original translation units |
| `extension` | Protenix AMD kernels | C++ `-O3`; HIP `-O3`; explicit ROCm include directory |
| `ck_extension` | Torch CK GEMM wrapper | C++ `-O3 -DUSE_ROCM -DUSE_ROCM_CK_GEMM`; explicit ROCm include directory; no extra HIP flags |

All use the recorded gfx90a target, their original source files and the guarded
Rust cache transaction. The local recipe is `torch-hip-6`; the process contract
version is 3. Ordered include paths are passed to Torch through Rust and require
complete declared directory inventories. Include additions/deletions/content
changes invalidate the build identity.

The former Python source digests, cache-directory creation and direct extension
compiler calls are removed from these loaders. `FIREPROT_RUST_BUILD` is no longer
a selector: these maintained Torch loaders always use Rust build services. The
existing pinned mask/bias artifact path still takes precedence. Other loaders'
prebuilt-artifact selectors remain to be connected. On missing tools or rejected
builds, the existing native fallback remains available and records the error.
Disabling an extension or rejecting an unsupported Torch/HIP configuration still
happens before building. KeyboardInterrupt is not converted into native fallback.

Python retains Torch/environment inspection, tensor ABI import and model-specific
capability inspection. Rust owns content identity, header/library inventories,
cache locks, cancellation, process lifecycle and publication. Build receipts are
included in extension status and model runtime records. This is not a claim that
all framework Python has been retired or that all model variants are verified.

## Build and deploy

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge -p fireprot-runtime --locked
```

The bridge and `fireprot-compiler-guard` must be available in the deployment.
`FIREPROT_EXTENSION_CACHE` changes the cache root;
`FIREPROT_BUILD_GUARD` explicitly selects another built guard path. Neither
changes kernel math. Original optimization opt-ins, shape/stride/dtype gates,
model cycle semantics and native fallback policies remain model-specific.

The standalone C ABI CK Tile attention and hipBLASLt BMM loaders now use the
Rust [direct HIP service](hip-builds.md), sharing the process guard and cache.
Pinned C ABI bundles are now supported; see [artifact deployment](kernel-artifacts.md). The CK GEMM wrapper's build
migration does not promote its opt-in inference dispatch.

See [build identity](kernel-builds.md), [process lifetime](torch-build-process.md)
and [published artifacts](kernel-artifacts.md). Runtime artifacts and predictions
remain excluded from Git.

## Validation

All four libraries cold-built successfully on DiamondHill through Rust recipe 6.
The AtlasFold GPU suite passed 60 tests. The Protenix operator suite initially
passed 138 and skipped two; two other tests stopped at legacy assertions requiring
HIP device 6 or 7 because the authorized initial device was 2. Both passed
unchanged when rerun on device 6. The CK GEMM wrapper also agrees with Torch FP32
matmul for a seeded 128x256 by 256x128 probe at the predeclared 1e-4 tolerances.

Final Rust-supervised full-model checks preserve original weights, inputs, seeds,
precision and cycle/step/sample counts:

| Model / length | Gate | Result |
| --- | --- | --- |
| AtlasFold + AtlasLM-3B / N1000 | Four public arrays, exact | All bitwise equal; 528 fused attention and 528 HIP multiplication calls |
| AtlasFold + AtlasLM-3B / N1536 | Four public arrays, exact against bounded native | All bitwise equal; same route counts |
| Protenix Mini-default / N1000 | Existing 1e-3 aligned-coordinate and summary-confidence gates | Pass; aligned RMSD 9.683e-5 Å, maximum summary delta 3.815e-6 against stored native |
| Protenix Mini-default / N1536 | Same existing gates | Pass; aligned RMSD 1.361e-4 Å, maximum summary delta 3.815e-6 against stored native |

Protenix also passes against the preceding FireProt outputs. Its full raw
confidence arrays are retained as diagnostics and are not bitwise equal. The
first Protenix passes lacked generic AMD build receipts in their runtime records;
after adding status-only reporting, both lengths were rerun and the final records
confirm recipe-6 cache reuse. AtlasFold records its pair-fusion build receipt too.
All four final runs pass continuous PCI-based GPU isolation. These concurrent
runs establish migration correctness, not new E2E speedups or structural accuracy.

The final local suite passes 86 Rust and 226 Python tests, with 110 GPU tests
skipped locally; formatting and Clippy pass. Tests cover declared include trees,
header membership changes, retained compiler flags, native fallback, interrupt
propagation and manifest collection without triggering a build. See
[complete migration evidence](../provenance/rust_torch_loaders_integration.json)
for requests, references, source/binary hashes, build summaries, route records and
raw numerical diagnostics. Full receipts and predictions remain on DiamondHill.

## Precompiled deployment

All four maintained Torch loaders now accept pinned Python-extension bundles.
Their disable/HIP/version checks precede selection; selected bundles bypass
source/compiler discovery and never trigger a replacement build when rejected.
See [bundle selectors and deployment profiles](kernel-artifacts.md#complete-maintained-loader-coverage).
Native construction remains available through Rust when no bundle is selected.
