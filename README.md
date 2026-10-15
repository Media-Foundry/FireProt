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
| Protenix Base-default v0.5.0 | `native`, opt-in `fireprot` | Strict FP32 N1000 C10/D200/S1: maintained composition passes user 0.1 Å aligned-RMSD and summary-confidence criteria; legacy 0.001 Å coordinate gate fails ([status](docs/protenix-base.md)) |
| Biohub ESMFold2, including ESMC | `native`, experimental opt-in `fireprot` | N600/N1000 native precision, loops3/schedule50/S1; all 12 arrays and complete CIF bytes exact; N1000 three-pair E2E validated ([status](docs/esmfold2.md)) |
| OpenFold2, original ESMFold v1 and other models | Planned | No verified integration or speedup claim |

Protenix and OpenFold3 are separate implementations. Precision, preprocessing,
recycles, diffusion steps, samples and language-model work remain model-specific.
The latest OpenFold3 composition is an **experiment with
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
| Protenix Base N1000, opt-in composed route | `configs/requests/protenix-base-n1000-fireprot.json` | `configs/diamondhill/protenix-base-n1000-optimized.json` |
| Protenix Base, earlier published route | `configs/requests/protenix-base-n1000-fireprot.json` | `configs/diamondhill/protenix-base-published.json` |
| Protenix Mini N1000 | `configs/requests/protenix-n1000-fireprot.json` | `configs/diamondhill/protenix-fireprot.json` |
| Protenix Mini N1536 | `configs/requests/protenix-n1536-fireprot.json` | `configs/diamondhill/protenix-fireprot.json` |
| AtlasFold + AtlasLM-3B N1536 | `configs/requests/atlasfold-n1536-hip-fused.json` | `configs/diamondhill/atlasfold-all-published.json` |
| OpenFold3 strict N1000 | `configs/requests/openfold3-n1000-hip-fused.json` | `configs/diamondhill/openfold3-hip-fused.json` |
| OpenFold3 strict N1536 | `configs/requests/openfold3-n1536-hip-fused.json` | `configs/diamondhill/openfold3-hip-fused.json` |
| Biohub ESMFold2 + ESMC N1000, experimental opt-in | `configs/requests/esmfold2-n1000-fireprot.json` | `configs/diamondhill/esmfold2-experimental.json` |

The opt-in Base profile requires **N1000, FP32 C10/D200/S1** and uses maintained
workers plus the published bundles and buffered bridge installed on DiamondHill.
Both umem and 5ZMW outputs match their frozen optimized references bitwise
through the Rust-selected maintained context. The latest three matched native
comparisons, including unused-template omission, measure **381.968 -> 110.025 s
complete invocation medians (3.472x, 71.20% less time)** on the umem N1000 input,
with shared fast initialization. Model-forward medians are 357.795 -> 86.894 s
(4.118x). Native-to-optimized outputs meet the explicit 0.1 Angstrom aligned-RMSD
and 0.001 summary-confidence criteria; cross-arm arrays are not bitwise equal,
and the unchanged legacy 0.001 Angstrom coordinate gate still fails.
The separate three-pair template experiment measures **111.936 -> 110.248 s
(1.0153x incremental)** with identical outputs. The latest direct result is
measured independently; incremental ratios are not multiplied.
See [Base entry and qualification](docs/protenix-base.md#maintained-n1000-optimization-entry).

HIP Graph is **not enabled** in this profile. Three matched comparisons against
maintained eager execution found no E2E gain (112.761 s eager versus 113.375 s
Graph medians), despite bitwise-identical outputs. See the
[Graph decision](docs/protenix-base.md#graph-versus-maintained-eager-three-pair-decision).

The Mini examples explicitly use **C4/D1/S1**; they are not the upstream Mini
D5 default. Review the workload before using an example as a scientific preset.
Mini-ESM needs its ESM2-3B assets as well; see [Mini-ESM setup](docs/protenix-mini-esm.md).

The ESMFold2 examples use **loops3/four passes, requested sampling50/S1** in both
arms, whereas the pinned upstream `fold()` defaults to loops20/sampling200.
They include bundled ESMC and the native query-only MSA path. Use the frozen
experimental CLI/profile together; see [ESMFold2 entry and scope](docs/esmfold2.md#run-the-opt-in-pipeline).

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

### Latest direct native comparison per model and length

ESMFold2 evidence added on **2026-10-15**; other rows retain the historical
measurements reviewed on 2026-10-10. Base, OpenFold3 and ESMFold2 measure the complete
Rust invocation. Mini and AtlasFold retain their narrower historical request
timers; their ratios are not complete Rust E2E measurements. An existing result
is not a fresh benchmark of today's checkout, and ratios across models do not
rank their absolute performance or prediction quality.

| Model / length | Reference → optimized | Speedup | Timing boundary and evidence |
| --- | ---: | ---: | --- |
| Protenix Base, 1000 aa, latest maintained composition | 381.968 → 110.025 s | **3.472x** | Complete Rust invocation; three AB/BA/AB pairs, strict FP32 C10/D200/S1, shared fast initialization, unused-template omission; explicit aligned-coordinate/summary-confidence policy ([details](docs/protenix-base.md#latest-maintained-direct-native-result)) |
| Protenix Mini, 1000 aa | 82.074 → 27.949 s | **2.937x** | Historical worker request; three pairs [¹](docs/protenix-native-requests.md) |
| Protenix Mini, 1536 aa | 177.345 → 45.596 s | **3.889x** | Historical worker request; three pairs [¹](docs/protenix-native-requests.md) |
| AtlasFold + AtlasLM-3B, 1000 aa | 184.581 → 84.174 s | **2.193x** | Earlier packaged revision; one request pair [²](docs/atlasfold-performance.md#packaged-bias-reuse-and-batched-transpose) |
| AtlasFold + AtlasLM-3B, 1536 aa | 553.359 → 203.910 s | **2.714x** | Chunked native math; medians of sequential three-run sets [²](docs/atlasfold-performance.md#final-qkv-request-repetitions) |
| OpenFold3, 1536 aa, latest experimental composition | 950.891 → 331.970 s | **2.864x** | Complete Rust invocation; three fresh AB/BA/AB pairs, all 11 arrays and full CIF bytes exact [³](docs/openfold3-performance.md#latest-composition-direct-native-e2e-result) |
| Protenix Mini-ESM + ESM2-3B, 1000 aa | Not established | — | Numerical integration validated; no controlled paired speedup ([scope](docs/protenix-mini-esm.md#n1000-numerical-checks)) |
| Protenix Tiny, 124 aa | Not established | — | Loading/numerical checks do not establish paired E2E performance |
| Biohub ESMFold2 + ESMC, 1000 aa | 379.298 → 189.072 s | **2.006x** | Complete Rust invocation; three AB/BA/AB pairs, native BF16/FP32, loops3/four passes, requested schedule50/S1 (34 denoiser calls), all 12 arrays and full CIF exact ([details](docs/esmfold2.md#n1000-direct-native-e2e-result)) |
| OpenFold2 / original ESMFold v1 | Not measured | — | Adapters remain planned |

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

Earlier direct results remain available as history: Base N600 initialization
only [1.412x](docs/protenix-base.md#fast-construction-qualification), OpenFold3
softmax only [1.376x](docs/openfold3-performance.md#exact-fp32-hip-softmax-three-pair-n1536-result),
and OpenFold3 combined reuse
[1.499x](docs/openfold3-performance.md#combined-reuse-direct-original-torch-n1536-e2e-result).
They do not describe the latest compositions above. AtlasFold N1000 retains its
earlier direct comparison because later QKV packaging has no new matched native
measurement; do not divide observations from different experiments.

### Where further optimization is justified

This is an evidence-based work priority, not a forecast of additional speedup.

| Model | Remaining opportunity and next step |
| --- | --- |
| **ESMFold2 + ESMC — current focus** | Native versus opt-in packing reuse now gives **2.006x complete N1000 invocation** and **3.289x forward** with exact outputs. Profile the optimized route before testing fused HIP cast/layout packing; no additional gain is established. |
| **OpenFold3 — current cycle closed** | Final QKV LDS padding trials gave no consistent gain. Non-forward attribution is recorded; a thread-cap comparison was rejected by the protected environment gate. The accepted E2E result remains **2.864x**. [Final trials](docs/openfold3-performance.md#final-bounded-trials-and-transition-to-esmfold2-2026-10-15). |
| **Mini-ESM — exploratory profiling** | ESM2-3B loading/features execute outside the folding-forward timer. Establish a serial N1000 complete-invocation baseline and stage split before selecting kernels; its remaining headroom is unknown. |
| **Mini-default — remeasure before porting** | Historical N1000 request/model medians are **27.949 / 15.391 s**. Measure the current Rust entry and check whether later Base/OpenFold3 ideas apply to its actual shapes and C4/D1 workload. No transferred gain is established. |
| **AtlasFold + AtlasLM-3B — reopen only with a new hotspot** | The previous exact-operator cycle reached a practical plateau. Later layout/LM trials gave negligible gains or failed equality; preserve every stochastic LM pass. A fresh stage profile is needed before further tuning. |
| **Protenix Base — retain the qualified route** | Sampled triangle BMMs reach about **99% MFMA activity**; Graph and gate-fusion follow-ups did not improve E2E. The N1000 cycle and batching investigation are closed at their measured scope. |
| **Tiny / OpenFold2 / original ESMFold v1** | Tiny is outside the long-sequence priority; the planned adapters need integration and baselines before an optimization ranking. |

See [optimization priorities and evidence](docs/performance-priorities.md).
The latest narrow-projection screen is an operator experiment only; it adds
no measured E2E gain to the OpenFold3 **2.864x** result.

### Experimental Base N1000 cycle

With fast initialization in both arms, native GPU computation takes **357.921 s**
and full CK/HIP takes **146.035 / 146.199 s** complete invocation time: approximately
**2.45x**, one baseline and two candidate observations. The candidate repeats match
all nine arrays bitwise. Native-to-candidate aligned RMSD is **0.054688 Å**, within
the user-authorized 0.1 Å coordinate criterion; the old confidence gate still fails.
This is numerical regression, not experimental structure accuracy. Base FP32,
C10/D200/S1, seed101, N1000, strict determinism, HIP7, unflushed caches.
See [scope, profiles and separate raw-native baseline](docs/protenix-base.md#full-ckhip-cycle-and-revised-coordinate-assessment-2026-10-07).

The historical 357.921 s control used FireProt dispatch with native Torch triangle
operators and already omitted the all-valid mask's zero bias. It is not the same
execution path as `backend=native`. The new maintained direct comparison uses
the latter with shared fast initialization; its latest **3.472x** result is listed above.
See the [baseline route audit](docs/protenix-base.md#strict-n1000-attention-isolation).

The published-artifact route retains the identical HIP/CK binaries and avoids
build-input scans during inference. **Three AB/BA/AB pairs** measure complete
invocation medians **146.146 -> 133.437 s (1.095x incremental, 8.70% less time)**.
All six outputs match all nine arrays bitwise. This is a loading-path gain over
full CK/HIP, separate from the earlier native-GPU baseline. The experimental
profile above pins both bundles. See
[published Base deployment](docs/protenix-base.md#published-artifact-loading).

Two later Base experiments use separate matched baselines, each with three pairs:

| Increment | Complete invocation medians | Incremental speedup |
| --- | ---: | ---: |
| Scoped Torch hipBLASLt triangle contractions | 133.856 -> 131.123 s | **1.0208x** |
| Buffered Rust parsing of published kernel receipts | 131.745 -> 122.459 s | **1.0758x** |

The Lt experiment meets the user coordinate criterion while its legacy comparison
still fails; the buffered-reading comparison preserves all nine arrays bitwise
and passes unchanged gates. These use frozen experimental workers/bridges, beyond
the published profile above. Do not multiply their ratios into an original-Torch
claim. See [Lt evidence](docs/protenix-base.md#trained-lt-cost-diagnosis-and-torch-blas-reuse)
and [receipt parsing](docs/protenix-base.md#receipt-parsing-diagnosis).


Static diffusion bias reuse was subsequently qualified in the experimental Base
worker. With global bias caching already enabled in both arms, **three local-bias
cache pairs** measure **120.025 -> 113.318 s** complete invocation medians
(**1.0592x incremental**, 5.59% less time). All six outputs match all nine arrays
bitwise and unchanged Rust gates pass. This frozen worker is not yet the published
profile's default. See [cache results](docs/protenix-base.md#static-diffusion-bias-audit).


### Later OpenFold3 increments — separate baselines

| N1536 experiment | Previous → candidate E2E median | Incremental speedup |
| --- | ---: | ---: |
| D16 coverage vs combined reuse | 628.689 → 540.848 s | **1.1624x** |
| Retained workspace/mask handling vs D16/cache combination | 540.302 → 449.723 s | **1.2014x** |
| Triangle multiplication copies + attention K packing vs workspace | 452.471 → 416.226 s | **1.0871x** |
| Identical template-slot reuse vs triangle-memory combination | 419.239 → 387.985 s | **1.0806x** |
| Exact HIP C128 LayerNorm vs template-reuse combination | 391.162 → 364.159 s | **1.0742x** |
| Fused triangle gates/packing vs canonical-layout HIP normalization combination | 352.949 → 348.258 s | **1.0135x** |
| Isolated single-item data preparation vs gate/packing combination | 349.971 → 341.782 s | **1.0240x** |
| QK-softmax-PV fusion with PV-unroll4 vs isolated-data combination | 345.753 → 330.999 s | **1.0446x** |
| HIP-packed global diffusion bias + grouped softmax vs unroll4 | 330.161 → 322.417 s | **1.0240x** |

Each row has three pairs under the same strict R3/D200/S1 contract. The
triangle comparison also reduces model forward time **356.595 → 321.699 s
(1.1085x)**; all six outputs match all 11 arrays and full CIF bytes against the
native, D16 and workspace anchors. N1000 fallback passes separately.
The later template comparison reduces model forward **323.661 → 291.027 s
(1.1121x)**; all six outputs match all 11 arrays and full CIF bytes against the
qualified triangle-memory anchor. Its scope is N1536 and it remains experimental.
See [complete evidence and reproduction scope](docs/openfold3-performance.md).

A subsequent global-bias cache experiment measures329.492 ->320.447 s median
(1.0282x), but one of three pairs regresses; it is **not promoted**. A separate
HIP bias transpose plus grouped softmax screen reduces synthetic global attention
from3.650 to1.470 ms with exact results. Full-model qualification and all six paired runs preserve all 11 arrays and
complete CIF bytes. Three pairs measure **330.161 ->322.417 s E2E medians
(1.0240x incremental, 2.35% less time)**; every pair improves. Retain this addition
in the experimental composition. The attention ratio itself is not an E2E claim.
See [diffusion experiment status](experiments/openfold3_diffusion/README.md#hip-packing-screen).

**Do not multiply these incremental ratios into an original-Torch speedup.**
The latest composition now has its own three-pair direct comparison: **950.891 →
331.970 s complete invocation medians (2.864x, 65.09% less time)**. Model forward
medians are **855.152 → 237.199 s (3.605x)**. Candidate observations span
321.836–374.020 s; all are retained. Both arms use the common FireProt launcher,
and the native route preserves upstream data-module defaults. This composition
remains available through the frozen experimental entry, pending maintained integration.

An earlier additional attention cycle rejected both exact candidates: native BMM
into reused scores plus HIP softmax made complete attention approximately 11%
slower, and double-LDS buffering made it approximately 1.3% slower. The validated
triangle-memory combination is retained; neither rejected candidate is promoted.

A later exact QK–softmax–PV fusion also failed its E2E gate: three pairs gave
**341.626 ->342.072 s (0.9987x)** despite a 3.3–3.5% module replay gain. All six
outputs match all 11 arrays and full CIF bytes. At that stage the isolated-data combination
was retained; the initial index32 fusion was not promoted. Later unroll4 and
global-softmax additions are measured separately in the table above. See the
[paired decision](docs/openfold3-performance.md#qkv-index32-three-pair-n1536-e2e-rejection).

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
