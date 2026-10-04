# Published kernel artifacts

## Status

Rust can publish a verified cached Python extension and accept a pinned bundle
without reading its compiler inputs. AtlasFold's mask/bias loader has an explicit
published-artifact path. Direct build orchestration, other HIP/CK recipes and
C ABI bundles remain pending. Publication does not establish numerical quality;
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

## Validation evidence

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
