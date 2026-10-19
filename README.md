# FireProt

**Protein structure inference for AMD GPUs, focused on long sequences.**
FireProt combines model-specific adapters with qualified HIP/CK kernels. Measured
long-protein workloads currently include 1,000 and 1,536 residues on gfx90a MI250.
Separate gfx1100 W7900 qualifications are reported below.

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
| Protenix Mini N1000 | `configs/requests/protenix-n1000-fireprot.json` | `configs/diamondhill/protenix-all-published.json` |
| Protenix Mini N1536 | `configs/requests/protenix-n1536-fireprot.json` | `configs/diamondhill/protenix-all-published.json` |
| AtlasFold + AtlasLM-3B N1536 | `configs/requests/atlasfold-n1536-hip-fused.json` | `configs/diamondhill/atlasfold-all-published.json` |
| OpenFold3 strict N1000 | `configs/requests/openfold3-n1000-hip-fused.json` | `configs/diamondhill/openfold3-hip-fused.json` |
| OpenFold3 strict N1536 | `configs/requests/openfold3-n1536-hip-fused.json` | `configs/diamondhill/openfold3-hip-fused.json` |
| Biohub ESMFold2 + ESMC N1000, experimental opt-in | `configs/requests/esmfold2-n1000-fireprot.json` | `configs/diamondhill/esmfold2-experimental.json` |

The opt-in Base profile requires **N1000, FP32 C10/D200/S1** and uses maintained
workers plus the qualified bundles and buffered bridge installed on DiamondHill.
It includes shared fast initialization and omission of unused template work.
Native-to-optimized outputs meet the explicit **0.1 Å aligned-RMSD** and
**0.001 summary-confidence** criteria. Cross-arm arrays are not bitwise equal;
the unchanged legacy 0.001 Å coordinate gate still fails. Use the
[latest complete-call table](#latest-e2e-refresh--2026-10-16) for measured speedups.
See [Base entry and qualification](docs/protenix-base.md#maintained-n1000-optimization-entry).

HIP Graph is not enabled: three matched comparisons found no E2E gain despite
bitwise-identical outputs. See the [Graph decision](docs/protenix-base.md#graph-versus-maintained-eager-three-pair-decision).

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

## Standalone protein language models

Use `fireprot esm REQUEST PROFILE` for ESMC/ESM2/ESM3 sequence inference and
`fireprot atlaslm REQUEST PROFILE` for AtlasLM. These commands export embeddings
and available sequence logits independently of folding. ESM3 currently uses its
sequence-only path. See [commands and precision](docs/protein-language-models.md).

For real GPU batches, provide an explicit `workload.batch` and pinned sequence
manifest. Rust controls length grouping and microbatch limits; the worker loads
the model once. All four native entries pass equal-length B4, mixed-length B4
and resident B1 checks. Compare optimized outputs with the official implementation
using **the same batch grouping**; singleton-to-batch rounding is a separate
measurement. See [batch inputs and examples](docs/plm-batch.md).

Fresh direct-native E2E results are in the [latest table](#latest-e2e-refresh--2026-10-16).
They include loading, full checkpoint hashes and all output exports. Native
precision is preserved; ESM2 remains FP32. These are standalone pLM results,
not folding speedups. Warm-forward results exclude loading/export and must not
be substituted for complete invocation results.

The retained ESM routes combine qualified HIP operations with checkpoint placement
and Rust host scheduling improvements. The detailed experiments remain available:

- [ESM operators and direct-native experiments](docs/esm-optimization.md).
- [ESMC pinned loading](docs/esmc-loading.md), [startup](docs/esmc-startup.md),
  [host scheduling](docs/esmc-host-scheduling.md) and
  [combined monitor](docs/esmc-snapshot-monitor.md).
- Experimental resident-batch [HIP pointwise](docs/esmc-pointwise.md) and
  [FFN composition](docs/esmc-resident-gemm.md) entries.

The latest CK tests did not pass the required exact-output acceptance. The
published ESMC route retains hipBLASLt. B16 grouping and fixed FFN algorithms
also delivered only marginal complete-call gains with mixed paired trends;
they remain experimental. See [CK diagnostics](docs/esm-optimization.md#esmc-ck-follow-up-rejected-candidates),
[larger batches](docs/esmc-large-batch-followup.md) and [B16 FFN results](docs/esmc-b16-ffn.md).

For context, an earlier three-pair AtlasLM study compared four **already optimized**
singleton calls (58.967 s total) with one optimized B4 call (15.204 s): **3.879x
aggregate throughput**. Each arm matched its own official grouping. That result
measures amortized loading across four sequences; it is not a single-sequence
latency improvement or a folding speedup.

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

## Latest E2E refresh — 2026-10-16

The new DiamondHill refresh uses complete Rust invocations, fresh workers,
three AB/BA/AB pairs per workload, full checkpoint verification and all outputs.
Only HIP6 / PCI `0000:8e:00.0` is used. DiamondHill-sourced asset transfers are
paused during timing; a later independent local-to-Precision relay uses neither
DiamondHill storage nor its CPU.
Filesystem/GPU caches are unflushed. Scientific settings remain model-specific.

| Standalone pLM, N1000 × 4 / B4 | Native median | Optimized median | E2E speedup |
| --- | ---: | ---: | ---: |
| ESMC bundled 6B | 24.089 s | 12.267 s | **1.964x** |
| ESM2-3B | 35.482 s | 15.489 s | **2.291x** |
| ESM3-open | 11.830 s | 11.067 s | **1.069x** |

A separate **AtlasLM singleton N1000, seed102** comparison measures
**14.839 -> 13.118 s (1.131x)**. Its initial B4 label was incorrect; the pinned
request has no batch field. A fresh AtlasLM B4 comparison is pending.

All 24 pLM invocations and 78 prediction archives pass independent rehashing
and byte equality against the same-group native reference, including all
requested embeddings/logits. Native precision is unchanged (ESM2 remains FP32).
These fresh ESM B4 and AtlasLM singleton measurements supersede their respective
historical E2E values; the historical AtlasLM B4 result remains separate. Their
warm-forward measurements remain separate historical observations.
[Raw timings, pins and independent checks](provenance/e2e_refresh_plm_20261016.json).

The following folding rows have completed three pairs and independent evidence
verification. Remaining connected models are still running; their historical
results are listed separately below.

| Folding model / workload | Native median | Optimized median | E2E speedup | Acceptance |
| --- | ---: | ---: | ---: | --- |
| Protenix Mini N1000, FP32 C4/D1/S1 | 71.042 s | 33.122 s | **2.145x** | Existing coordinate/summary-confidence gates; not bitwise equal |
| Protenix Mini N1536, FP32 C4/D1/S1 | 167.120 s | 51.180 s | **3.265x** | Existing coordinate/summary-confidence gates; not bitwise equal |
| ESMFold2 + ESMC N1000, native precision, loops3/sampling50/S1 | 368.783 s | 178.666 s | **2.064x** | All arrays and complete CIF bytes exact |

[Workloads, native-reference scope and acceptance](docs/e2e-refresh-20261016.md);
[folding timings and numerical diagnostics](provenance/e2e_refresh_folding_20261016.json).
The first OpenFold3 attempt rejected a stale environment inventory before model
execution; a retry retains full inventory coverage and identical numerical gates.
Precision is a separate gfx1100 deployment; DiamondHill speedups are not
attributed to it.

### Precision gfx1100 results — separate host

Precision uses W7900 / gfx1100 with Torch 2.13/ROCm 10. The following are
separate three-pair studies; each row keeps its own control and revision.

| Workload / comparison | Control median | Candidate median | E2E speedup |
| --- | ---: | ---: | ---: |
| ESM3-open B4/N1000, native → frameless sequence route | 12.436 s | 10.953 s | **1.135x** |
| ESM2-3B B4/N1000, native → exact-loading route | 38.908 s | 22.431 s | **1.735x** |
| OpenFold3 N1000, native → wave32 softmax | 322.388 s | 285.978 s | **1.127x** |
| OpenFold3 N1000, softmax → added pair/atom/template reuse | 283.614 s | 239.335 s | **1.185x incremental** |
| OpenFold3 N1000, reuse composition → added HIP layout copies | 238.182 s | 231.214 s | **1.030x incremental** |
| OpenFold3 N1000, copy composition → private QK-score reuse | 231.101 s | 223.261 s | **1.035x incremental** |

All listed studies preserve same-host native output bytes. OpenFold3 also
preserves all 11 arrays and complete CIF bytes, using strict FP32/R3/D200/S1.
Its optimized workers remain frozen experiments. Later ESM2 GEMM/loading and
ESM3 host-path increments have separate evidence; these rows do not claim a
fresh direct-native comparison of every subsequent composition. Never multiply
ratios across studies or transfer DiamondHill results to this host.
The initial ESM3-only native baseline (11.885 s) remains a separate earlier study.
[Precision entries and detailed evidence](docs/precision-setup.md);
[current gfx11 handoff](GFX11_HANDOFF.md).

## Measured speedups

These are versioned comparisons, not a uniform benchmark of the current CLI.
Ratios are reference time divided by optimized time. All workloads retain the
same scientific settings within their comparison; filesystem/kernel caches are
warm or unflushed as specified in the linked evidence.

### Historical direct comparisons — earlier boundaries and revisions

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
| **Standalone ESM/AtlasLM** | Fresh ESM B4 complete-call medians are 11–15 s, while reported model forwards take 1.3–1.9 s. Loading, verification, startup and export dominate this small batch's remaining latency. Consider measured residency/host costs before more kernel tuning; exact CK trials remain unqualified. |
| **ESMFold2 + ESMC** | Fresh E2E is 178.666 s versus an 81.682 s model forward. Accepted nested timers leave about 20.4 s elsewhere inside the worker request and 77.0 s outside it, similar in both arms. Profile verification/startup/acceptance before selecting more GPU kernels; these boundaries do not identify individual stages. [Timing evidence](docs/e2e-refresh-20261016.md#esmfold2-remaining-time-outside-the-model-forward). |
| **OpenFold3** | The preceding QKV LDS-padding trials found no consistent E2E gain; the fresh strict-FP32 run is pending. Preserve the qualified composition until profiling identifies a new dominant cost. [Final trials](docs/openfold3-performance.md#final-bounded-trials-and-transition-to-esmfold2-2026-10-15). |
| **Mini-ESM** | The fresh three-pair result is 118.396 → 83.048 s (1.426x), with folding-forward medians 47.848 → 11.549 s. Prioritize profiling ESM2 feature preparation/loading and the remaining invocation work; the complete-call difference is not an isolated LM timer. |
| **Mini-default** | Fresh N1000 optimized E2E/forward medians are 33.122/14.757 s; N1536 medians are 51.180/31.931 s. About 18–19 s remains outside the reported forward boundary. Any transferred optimization needs validation for its C4/D1 shapes. |
| **AtlasFold + AtlasLM-3B** | The previous exact-operator cycle reached small returns; later layout/LM trials gave negligible gains or failed equality. Preserve every stochastic LM pass and reprofile the refreshed route before further tuning. |
| **Protenix Base** | Sampled triangle BMMs reached about 99% MFMA activity. Graph and gate-fusion trials did not improve E2E; the user closed this optimization direction. Retain the qualified route. |
| **Tiny / OpenFold2 / original ESMFold v1** | Tiny remains outside the long-sequence optimization priority. OpenFold2 and original ESMFold v1 need integration and baselines before ranking. |

See [optimization priorities and evidence](docs/performance-priorities.md).
The latest narrow-projection screen is an operator experiment only; it adds
no measured E2E gain to the OpenFold3 **2.864x** result.

### Historical optimization experiments

Detailed experiments retain their original revisions, controls and timing
boundaries in [Protenix Base](docs/protenix-base.md),
[OpenFold3](docs/openfold3-performance.md), [AtlasFold](docs/atlasfold-performance.md),
[Mini](docs/protenix-native-requests.md) and [ESMFold2](docs/esmfold2.md).
Their incremental gains must not be multiplied or substituted for a fresh
complete invocation comparison. The [Base reference-route audit](docs/protenix-base.md#strict-n1000-attention-isolation)
explains why its older 357.921 s control differs from `backend=native`.

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
