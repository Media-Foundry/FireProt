# FireProt

**Protein structure inference for AMD GPUs, focused on long sequences.**
FireProt combines model-specific adapters with qualified HIP/CK kernels. Measured
long-protein workloads currently include 1,000 and 1,536 residues on gfx90a MI250.

The maintained entry point is the **Rust CLI**. Rust owns request validation,
operator selection, builds/caches, process and GPU supervision, numerical acceptance,
and paired benchmarks. Upstream models and tensor execution use Python/PyTorch;
GPU acceleration comes from kernels and execution changes, not the language switch.
See the [Rust acceptance audit](docs/rust-acceptance.md).

## Model support

| Model | Connected paths | Validated scope |
| --- | --- | --- |
| Protenix Mini-default v0.5.0 | `native`, `fireprot` | FP32 N1000/N1536; coordinate/confidence regression gates |
| Protenix Mini-ESM and Tiny-default v0.5.0 | `native`, `fireprot` | Mini-ESM N1000 with ESM2-3B; Tiny L124; model-specific gates |
| AtlasFold 260703 **with AtlasLM-3B** | Native, chunked, HIP and fused HIP | N1000/N1536; exact public-output bytes; N1536 reference uses chunked native math |
| OpenFold3 / OpenBind-0 | `native`, opt-in `hip_fused` | Strict deterministic FP32 N1000/N1536; all 11 arrays bitwise equal |
| Protenix Base-default v0.5.0 | `native`, experimental `fireprot` | Strict N600 initialization: exact outputs; N1000 CK/HIP meets user 0.1 Å aligned-RMSD criterion, legacy confidence gate fails ([status](docs/protenix-base.md)) |
| OpenFold2, ESMFold and other models | Planned | No verified integration or speedup claim |

Protenix and OpenFold3 are separate implementations. Precision, preprocessing,
recycles, diffusion steps, samples and language-model work remain model-specific.
The latest OpenFold3 workspace/cache/triangle combination is an **experiment with
frozen deployment artifacts**, not automatically enabled by the maintained profile.

## Build

```bash
cargo build --release -p fireprot-cli -p fireprot-runtime -p fireprot-bridge --locked
target/release/fireprot --help
target/release/fireprot inspect configs/requests/openfold3-n1536-hip-fused.json
```

`inspect` is CPU-only and does not open weights. Inference additionally requires:

- The request's pinned upstream source, checkpoints, input and data resources.
- A model-specific Python environment with compatible ROCm PyTorch dependencies.
- ROCm, AMD SMI and the [HIP device identity probe](docs/worker-protocol.md#independent-amd-device-monitor).
- The profile's published kernel bundles or supported build environment.
- A matching [environment inventory](docs/environment-inventories.md), when pinned.

On a prepared ROCm host, build the device probe with:

```bash
mkdir -p runtime
/opt/rocm/bin/hipcc kernels/amd/tools/hip_device_probe.cpp \
  -Wl,-rpath,/opt/rocm/lib -o runtime/fireprot-hip-probe
```

The checked-in profiles describe DiamondHill, whose workspace is
`/media/IntelSSD/fireprot`. They are deployment examples with absolute paths,
not a portable dependency installer. Update paths and verify artifact/inventory
pins for your host; stale inventories are rejected. See
[installation](docs/installation.md), [checkpoint locations](docs/diamondhill-checkpoints.md)
and [kernel artifacts](docs/kernel-artifacts.md).

## Run inference

The interface is:

```text
fireprot infer REQUEST.json PROFILE.json
```

**The request selects the model, backend and workload.** It records `run_id`,
`output_dir`, `hip_device`, input/checkpoint/source pins, precision, cycles or
recycles, diffusion steps, samples and seed. **The profile selects the execution
environment**: Python, workers, device probe, AMD SMI, environment settings and
optional inventory. Rust sets GPU visibility and resolves HIP-to-SMI identity by
PCI address; HIP and SMI ordinals are not interchangeable.

Copy an example request and set a **new `run_id` and unused `output_dir`** before
each run. Change input paths/hashes and workload fields explicitly when needed.
Validate the copy, then execute it with a prepared matching profile:

```bash
cp configs/requests/openfold3-n1536-hip-fused.json /tmp/my-openfold3.json
# Edit /tmp/my-openfold3.json: fresh run_id/output_dir and your pinned input.
target/release/fireprot inspect /tmp/my-openfold3.json
target/release/fireprot infer /tmp/my-openfold3.json \
  configs/diamondhill/openfold3-hip-fused.json
```

Other starting points:

| Workflow | Request to copy | DiamondHill profile |
| --- | --- | --- |
| Protenix Mini N1000 | `configs/requests/protenix-n1000-fireprot.json` | `configs/diamondhill/protenix-fireprot.json` |
| Protenix Mini N1536 | `configs/requests/protenix-n1536-fireprot.json` | `configs/diamondhill/protenix-fireprot.json` |
| AtlasFold + AtlasLM-3B N1536 | `configs/requests/atlasfold-n1536-hip-fused.json` | `configs/diamondhill/atlasfold-all-published.json` |
| OpenFold3 strict N1000 | `configs/requests/openfold3-n1000-hip-fused.json` | `configs/diamondhill/openfold3-hip-fused.json` |
| OpenFold3 strict N1536 | `configs/requests/openfold3-n1536-hip-fused.json` | `configs/diamondhill/openfold3-hip-fused.json` |

The Mini examples explicitly use **C4/D1/S1**; they are not the upstream Mini
D5 default. Review the workload before using an example as a scientific preset.
Mini-ESM needs its ESM2-3B assets as well; see [Mini-ESM setup](docs/protenix-mini-esm.md).

`scripts/infer.sh REQUEST PROFILE` delegates to Rust `infer`.
`scripts/smoke.sh [MANIFEST]` delegates to Rust `infer-batch`; its default is the
three-model Protenix smoke manifest. See [batch inference](docs/batch-inference.md).

## Outputs and measurement

A successful request writes `accepted.json`, `supervisor.json`, provenance and
execution records, `predictions.json`, `sample-N.npz`, and native structures under
`native/`. Failures do not produce successful acceptance receipts. Details:
[commands and outputs](docs/rust-running.md), [numerical/CIF validation](docs/cif-validation.md).

```bash
target/release/fireprot compare-npz reference.npz candidate.npz
target/release/fireprot help measure-inference
target/release/fireprot help benchmark-pairs
```

Use `measure-inference SPECIFICATION DESTINATION` for complete Rust invocation
timing, and `benchmark-pairs SPECIFICATION DESTINATION` for declared, pinned
comparisons with per-run gates. Specifications and reproduction procedures are in
[invocation timing](docs/invocation-timing.md) and [paired execution](docs/paired-execution.md).
Model forward time, worker request time and complete invocation E2E are different
boundaries; `predictions.json` alone is not complete E2E evidence.

## Measured speedups

These are versioned comparisons, not a uniform benchmark of the current CLI.
Ratios are reference time divided by optimized time. All workloads retain the
same scientific settings within their comparison; filesystem/kernel caches are
warm or unflushed as specified in the linked evidence.

### Direct native comparisons

| Model / length | Reference → optimized | Speedup | Timing boundary and evidence |
| --- | ---: | ---: | --- |
| Protenix Base, 600 aa, initialization only | 179.350 → 127.047 s | **1.412x** | Complete Rust invocation; one pair, strict deterministic C10/D200/S1; all 9 arrays exact ([details](docs/protenix-base.md#fast-construction-qualification)) |
| Protenix Mini, 1000 aa | 82.074 → 27.949 s | **2.937x** | Historical worker request; three pairs [¹](docs/protenix-native-requests.md) |
| Protenix Mini, 1536 aa | 177.345 → 45.596 s | **3.889x** | Historical worker request; three pairs [¹](docs/protenix-native-requests.md) |
| AtlasFold + AtlasLM-3B, 1000 aa | 184.581 → 84.174 s | **2.193x** | Earlier packaged revision; one request pair [²](docs/atlasfold-performance.md#packaged-bias-reuse-and-batched-transpose) |
| AtlasFold + AtlasLM-3B, 1536 aa | 553.359 → 203.910 s | **2.714x** | Chunked native math; medians of sequential three-run sets [²](docs/atlasfold-performance.md#final-qkv-request-repetitions) |
| OpenFold3, 1536 aa, HIP softmax experiment | 956.112 → 694.973 s | **1.376x** | Complete Rust invocation; three pairs [³](docs/openfold3-performance.md#exact-fp32-hip-softmax-three-pair-n1536-result) |
| OpenFold3, 1536 aa, combined reuse experiment | 949.999 → 633.712 s | **1.499x** | Complete Rust invocation; three pairs, before later increments [³](docs/openfold3-performance.md#combined-reuse-direct-original-torch-n1536-e2e-result) |

1. **Protenix:** Mini-default v0.5.0 FP32, **C4/D1/S1**, seed101, no MSA/templates.
   Reference is the same checkout's native Torch route, including a shared bond-mask
   change, not an unmodified upstream checkout. Worker timing excludes parent
   preflight and part of startup. Historical HIP/SMI monitor mapping limits the
   isolation evidence; these are not newly verified Rust E2E numbers. Outputs pass
   the established coordinate/confidence tolerances, not bitwise equality.
2. **AtlasFold:** Includes AtlasLM-3B on every required pass; native BF16/FP32,
   R4/five trunk passes, D30 at N1000 or D100 at N1536, S1, seed102.
   Request timing includes loading/features/prediction/serialization, excluding
   imports, hashing and preflight. Unchunked native N1536 OOMs. Public outputs
   match byte-for-byte. N1000 has no new repeated direct comparison after QKV packing.
3. **OpenFold3:** OpenBind-0 strict deterministic FP32 with memory filling,
   R3/four trunk passes, D200/S1, seed101. All 11 arrays are bitwise equal;
   comparisons also check CIF identities. These experiments do not mean every
   optimization is enabled by `configs/diamondhill/openfold3-hip-fused.json`.

### Experimental Base N1000 cycle

With fast initialization in both arms, native GPU computation takes **357.921 s**
and full CK/HIP takes **146.035 / 146.199 s** complete invocation time: approximately
**2.45x**, one baseline and two candidate observations. The candidate repeats match
all nine arrays bitwise. Native-to-candidate aligned RMSD is **0.054688 Å**, within
the user-authorized 0.1 Å coordinate criterion; the old confidence gate still fails.
This is numerical regression, not experimental structure accuracy. Base FP32,
C10/D200/S1, seed101, N1000, strict determinism, HIP7, unflushed caches.
See [scope, profiles and separate raw-native baseline](docs/protenix-base.md#full-ckhip-cycle-and-revised-coordinate-assessment-2026-10-07).

### Later OpenFold3 increments — separate baselines

| N1536 experiment | Previous → candidate E2E median | Incremental speedup |
| --- | ---: | ---: |
| D16 coverage vs combined reuse | 628.689 → 540.848 s | **1.1624x** |
| Retained workspace/mask handling vs D16/cache combination | 540.302 → 449.723 s | **1.2014x** |
| Triangle multiplication copies + attention K packing vs workspace | 452.471 → 416.226 s | **1.0871x** |

Each row has three pairs under the same strict R3/D200/S1 contract. The latest
triangle comparison also reduces model forward time **356.595 → 321.699 s
(1.1085x)**; all six outputs match all 11 arrays and full CIF bytes against the
native, D16 and workspace anchors. N1000 fallback passes separately.
See [complete evidence and reproduction scope](docs/openfold3-performance.md).

**Do not multiply these incremental ratios into an original-Torch speedup.**
A complete three-pair direct original-Torch comparison of the latest combination
is not available; 1.499x above applies to its earlier measured combination.

The final additional attention cycle rejected both exact candidates: native BMM
into reused scores plus HIP softmax made complete attention approximately 11%
slower, and double-LDS buffering made it approximately 1.3% slower. The validated
triangle-memory combination is retained; neither rejected candidate is promoted.

C = Protenix cycles; R = additional recycles; D = diffusion steps; S = samples.
Long inputs and the eight short development cases are regression/performance
workloads, not an independent experimental structure-accuracy test set.

## Repository layout and documentation

| Path | Responsibility |
| --- | --- |
| `crates/fireprot-{core,runtime,validation,cli,bridge}` | Rust contracts, execution, acceptance, CLI and CPython bridge |
| `python/fireprot_workers`, `python/fireprot` | Upstream model invocation and tensor adapters |
| `kernels/amd` | Maintained HIP/CK operators |
| `experiments`, `provenance` | Experimental implementations and measured evidence |
| `configs`, `tests`, `docs`, `.agents` | Workloads, validation, guides and project history |
| `Protenix/` | Independent upstream Git checkout, not a top-level submodule |
| `runtime/`, `outputs/` | Local assets/build caches and predictions; excluded from Git |

- [Architecture](docs/architecture.md) and [Rust acceptance](docs/rust-acceptance.md)
- [Installation](docs/installation.md), [checkpoint inventory](docs/diamondhill-checkpoints.md), [entry points](docs/entrypoints.md)
- [Protenix measurements](docs/protenix-native-requests.md), [AtlasFold measurements](docs/atlasfold-performance.md), [OpenFold3 measurements](docs/openfold3-performance.md)
- [Environment pins](docs/environment-inventories.md), [execution provenance](docs/execution-provenance.md), [run archives](docs/run-archives.md)
- [Original Protenix baseline notes](.agents/protenix_baseline.md) (Chinese, retained unchanged)

Weights and predictions are not committed or transferred by the source-sync
script. Review its destination and dry-run changes before synchronization.

## License

FireProt is a multi-license distribution. Original FireProt material is
**GPL-2.0-only with a limited OpenFold 3 exception**. OpenFold 3-derived files
identified in [THIRD_PARTY.md](THIRD_PARTY.md) remain **Apache-2.0**. The exception
permits combination and distribution with OpenFold 3; it is not a general
exception for Apache-2.0 code.

**Advanced Micro Devices, Inc. and its directly or indirectly controlled
subsidiaries** also receive an Apache-2.0 option for original FireProt material.
This grants no new rights over third-party code or model assets. Apache-2.0
downstream redistribution rights apply to copies lawfully distributed under
that option. Earlier Apache-2.0 grants are not revoked.

See [LICENSE](LICENSE) for the exact grants and complete license texts, and
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution terms. Cargo uses `license-file`
because this custom arrangement is not a standard SPDX exception; the repository
does not declare an unrestricted `GPL-2.0-only OR Apache-2.0` choice.
