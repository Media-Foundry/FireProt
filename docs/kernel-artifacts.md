# Published kernel artifacts

## Status

Rust can publish verified cached Python extensions and C ABI libraries, then
accept pinned bundles without reading compiler inputs. AtlasFold mask/bias,
Protenix AMD operations, CK GEMM, CK attention, AtlasFold pair fusions and the
optional hipBLASLt BMM loader all have explicit published-artifact paths. Publication does not establish numerical quality;
operator and model validation are separate gates.

## Publish on the build host

```bash
cargo build --release -p fireprot-cli -p fireprot-bridge --locked
target/release/fireprot publish-kernel path/to/cache-entry path/to/new-bundle
```

Replace the paths with a successful Rust cache entry and a fresh destination.
On the trusted build host, publication invokes the recorded dependency inspector
and rechecks build inputs and artifact hashes,
copies the library, and atomically publishes:

```text
bundle/
  kernel.json          # Compatibility, library hash and build-receipt hash
  build-receipt.json   # Complete build identity and provenance
  PACKAGE_DIGEST.so    # Unchanged compiled library
```

The command returns the bundle path, `manifest_sha256` and compatibility metadata.
Record the digest in trusted deployment configuration. Existing destinations are
never overwritten. Libraries, caches and published bundles belong under excluded
`runtime/`, not in source Git.

## Verify a deployment

```bash
target/release/fireprot verify-kernel path/to/bundle MANIFEST_SHA256 compatibility.json
```

`MANIFEST_SHA256` is the digest recorded during trusted publication.
`compatibility.json` declares the expected loading environment, for example:

```json
{
  "package": "fireprot_mask_bias",
  "target": "gfx90a",
  "torch": "2.12.0a0+git78258b9",
  "hip": "7.14.60850",
  "python_soabi": "cpython-312-x86_64-linux-gnu"
}
```

Rust checks the pinned manifest, library and receipt hashes, exact compatibility,
flat non-symlink artifact paths and agreement between the manifest and receipt.
Acceptance does not run a compiler/inspector or open original source/header files.
The CLI checks the supplied expectations; the model-facing loader derives them
from the current Torch/HIP/Python environment and GPU architecture. Cross-host
portability still requires platform and numerical validation.

## AtlasFold loading

The DiamondHill profile `configs/diamondhill/atlasfold-published.json` pins the
existing mask/bias bundle. Use it with an AtlasFold `hip` request, a fresh run ID
and an unused output directory:

```bash
target/release/fireprot infer request.json configs/diamondhill/atlasfold-published.json
```

The profile sets `FIREPROT_MASK_BIAS_BUNDLE` and the mandatory
`FIREPROT_MASK_BIAS_BUNDLE_SHA256`. This choice precedes legacy and experimental
build paths. A rejected explicit bundle retains native fallback and an error
record; it does not initiate a replacement build. The existing disable switch
still disables the extension. Other AtlasFold backends use their existing loaders.

Python only inspects its runtime and loads the Rust-accepted library.
`operator-runtime.json` records `extension.build.mode = "published"`, the accepted
module/path, manifest pin, build identity and compatibility. Check route counts
when verifying that an inference actually exercised the requested HIP operator.

See [build services and historical migration evidence](kernel-builds.md) and
[model execution scope](rust-running.md).

## Historical Python bundle validation

The published DiamondHill mask/bias library is byte-identical to its previously
validated build output. Eleven GPU tests pass through the published loader, with
the build module absent from that test process. N1000 and N1536 full AtlasFold
requests also pass exact public-output comparison, select 528 HIP attention calls
each and record `mode = "published"`. PCI isolation passes with 150/395 samples.

The local suite passes 75 Rust tests and 211 Python tests (110 GPU cases skipped).
Rust tests cover missing build inputs after publication, wrong pins, compatibility
mismatch, corrupt libraries/receipts, inconsistent metadata, symlinks and existing
destinations. The real CLI acceptance check also succeeds with build tools absent
from `PATH`. These checks establish this deployment path; parallel model timings
are not a new E2E speedup claim. See [publication evidence](../provenance/rust_published_kernel_integration.json).

## C ABI bundles

C ABI publication uses schema 2. Compatibility contains `package`, `target`,
`torch`, `hip` and `c_abi`; it has no Python SOABI or module name. The typed Rust
compatibility enum rejects mixed Python/C fields, wrong ABI revisions and schema
mismatches. Schema-1 Python manifests retain their existing JSON representation.
The build receipt must declare `abi = "c"` and a nonempty `c_abi` revision.
Maintained revisions are `ck-attention-v1` and `lt-packed-bmm-v1`.

`publish-kernel` and `verify-kernel` use the same commands and integrity checks
for both types. A C ABI compatibility file looks like:

```json
{
  "package": "fireprot_ck_attention",
  "target": "gfx90a",
  "torch": "2.12.0a0+git78258b9",
  "hip": "7.14.60850",
  "c_abi": "ck-attention-v1"
}
```

Select a bundle with `FIREPROT_CK_ATTENTION_BUNDLE` and
`FIREPROT_CK_ATTENTION_BUNDLE_SHA256`, or `FIREPROT_LT_BMM_BUNDLE` and
`FIREPROT_LT_BMM_BUNDLE_SHA256`. The expected package/ABI is fixed by the loader;
Torch/HIP/architecture come from the actual tensor environment. Python loads the
Rust-accepted file through ctypes, with unchanged entry-point signatures.
Explicit bundles bypass CK header checks and compiler discovery; the pinned build
receipt identifies the original CK installation. Rejected explicit bundles keep
native fallback and never initiate a replacement build. The CK disable switch
still takes precedence. Selecting the hipBLASLt bundle does not enable its
experimental inference route.

Bundles contain one library, not a complete ROCm environment. Dynamic HIP/BLAS
libraries and their data must still be installed in compatible locations; ABI
metadata and hashes do not establish cross-host or cross-version portability.
The recipe's runtime-plugin/device-data inventory limits still apply. See
[direct HIP builds](hip-builds.md).

## C ABI validation on DiamondHill (2026-10-04)

[Versioned evidence](../provenance/rust_published_cabi_integration.json) includes
both publication manifests, source/library hashes, CLI acceptance without build
tools in `PATH`, loaded artifacts and complete requests. The deployment profile
is `configs/diamondhill/protenix-published.json`.

Five C ABI GPU tests pass with the compiler-descriptor module unavailable:
ordinary/exact1536/vector1000 CK attention, full long attention, and hipBLASLt
stream/workspace behavior. Eleven legacy schema-1 AtlasFold GPU tests pass with
its compiler helper unavailable. CPU checks pass 91 Rust and 232 Python tests
(110 GPU-dependent skips), including real compiled C library publication/loading,
missing build inputs, mixed ABI metadata and incorrect ABI/schema rejection.
Clippy and formatting pass.

Two complete Mini-default FP32 requests retain seed 101, four total cycles, one
diffusion step and one sample. Both record CK `mode = "published"` and pass the
existing 1e-3 coordinate/summary-confidence gates against stored native and prior
FireProt references. Native-reference results are:

| Length | Aligned RMSD (Angstrom) | Summary max absolute delta | Supervisor seconds / isolation samples |
| ---: | ---: | ---: | ---: |
| 1000 | 9.14347e-5 | 2.98023e-8 | 49.451 / 53 |
| 1536 | 1.35987e-4 | 2.98023e-8 | 75.827 / 83 |

The runs used separate GCDs concurrently. Their times establish no new E2E
speedup. Raw Protenix arrays remain diagnostic and are not bitwise identical.
hipBLASLt model dispatch remains disabled; its bundle is operator-tested only.

## Complete maintained loader coverage

All six maintained loaders can select pinned bundles before compiler discovery.
The same Rust acceptance service checks both Python and C ABI artifacts.

| Loader package | Environment prefix | ABI |
| --- | --- | --- |
| `fireprot_amd_ops` | `FIREPROT_AMD_OPS` | Python / schema 1 |
| `fireprot_ck_ops` | `FIREPROT_CK_OPS` | Python / schema 1 |
| `fireprot_pair_fusions` | `FIREPROT_PAIR_FUSIONS` | Python / schema 1 |
| `fireprot_mask_bias` | `FIREPROT_MASK_BIAS` | Python / schema 1 |
| `fireprot_ck_attention` | `FIREPROT_CK_ATTENTION` | C / schema 2 |
| `fireprot_lt_bmm` | `FIREPROT_LT_BMM` | C / schema 2 |

For each prefix, set both `_BUNDLE` and `_BUNDLE_SHA256`. An explicit selection
bypasses source existence checks, `torch.utils.cpp_extension` imports and native
build descriptors. Disable switches and model/operator capability checks still
apply; pair softmax retains its validated Torch/HIP version gate. A rejected bundle
records an error and keeps native fallback. Without a bundle, the existing Rust
build/cache route remains available.

`configs/diamondhill/protenix-all-published.json` selects all four Protenix
libraries while preserving the default disabled CK GEMM/hipBLASLt model routes.
`configs/diamondhill/atlasfold-all-published.json` selects mask/bias and pair
fusions; the request's backend still determines which operators are used.
These configurations pin DiamondHill artifacts and are not portable installers.


### Complete-loader validation (2026-10-04)

[The integration record](../provenance/rust_all_published_integration.json) pins
all three newly published Torch libraries and both complete deployment profiles.
244 CPU Python tests pass (110 GPU cases skipped). GPU checks pass 159 Protenix
and 60 AtlasFold tests with the Torch native-build module unavailable. A separate
FP32 CK GEMM probe passes against Torch matmul at rtol/atol 1e-4; it does not
promote CK GEMM model dispatch. Rust implementation bytes are unchanged from the
preceding 91-test validation.

Complete requests use original weights and fixed semantics: AtlasFold seed 102,
four additional recycles, native precision, one sample and 30/100 diffusion steps;
Protenix Mini-default FP32 seed 101, four total cycles, one step and one sample.

| Model / route | Length | Native-reference result | Supervisor seconds / isolation samples |
| --- | ---: | --- | ---: |
| AtlasFold + AtlasLM-3B, fused HIP | 1000 | Four public arrays bitwise equal | 95.386 / 98 |
| AtlasFold + AtlasLM-3B, fused HIP | 1536 | Four public arrays bitwise equal, bounded native reference | 217.008 / 237 |
| Protenix Mini-default, FireProt | 1000 | Aligned RMSD 8.65243e-5 Angstrom; summary delta 7.62939e-6 | 44.215 / 41 |
| Protenix Mini-default, FireProt | 1536 | Aligned RMSD 1.25892e-4 Angstrom; summary delta 3.81470e-6 | 61.783 / 60 |

AtlasFold records 528 fused-attention calls, 528 HIP multiplication calls and zero
native fallbacks. Protenix records published AMD and CK-attention libraries;
CK GEMM and hipBLASLt stay unloaded in model inference. It also passes the existing
1e-3 gates against preceding FireProt outputs. These concurrently run migration
checks are not E2E speedup measurements. Raw Protenix arrays remain diagnostic.
