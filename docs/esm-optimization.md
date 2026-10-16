# Standalone ESM optimization

This cycle targets ESMC bundled 6B, ESM2-3B, and ESM3-open sequence inference on
gfx90a. It does not change folding-model speedup claims. The numerical reference
is the official implementation with **the same batch grouping**, padding, native
precision, seed, and output heads. All retained output arrays must match bitwise.
Differences between singleton and batch execution remain separately documented in
[pLM batching](plm-batch.md).

## Implementation

- **ESMC:** HIP RoPE and SwiGLU preserve BF16 intermediate rounding. The loader
  reads each checkpoint block on GPU, converts to native BF16, and applies the
  upstream `published_to_native` row-concatenation mapping. Temporary projection
  tensors are released per block. Strict state loading, complete shard hashing,
  the native tokenizer, attention, LayerNorm, GEMMs, and output are retained.
- **ESM2:** HIP GELU fuses the existing FP32 arithmetic, including `erf`, without
  changing the expression or enabling FMA contraction. The previously qualified
  omission of discarded attention means remains. FP16 checkpoint tensors are
  transferred before promotion to **native FP32 inference**. This is a loading
  change, not mixed-precision inference.
- **ESM3:** HIP RoPE/SwiGLU plus a guarded sequence-only geometry simplification.
  Native geometric attention masks its intermediate output to positive zero when
  every structure frame is invalid. FireProt retains the native output projection
  of that zero tensor, including bias and autocast semantics. Any valid frame,
  training, enabled gradients, or disabled zero-mask policy uses native geometry
  in the helper. The qualified accelerated entry remains sequence-only and
  requires its declared operator coverage. Structure-conditioned inference is
  outside this entry's contract. The official factory and all output heads remain.

Rust owns request admission, complete asset verification, operator selection,
supervision, exact-array acceptance, and paired E2E measurement. Tensor execution
remains in the upstream PyTorch models. The new library is
`kernels/amd/fireprot_esm.hip`, built by the Rust HIP build service with
`-ffp-contract=off`. `hip_esm` is an explicitly pinned request resource; old
requests keep their previously qualified routes. ESM3's FireProt backend requires
this resource. The qualified scope is N1000, B1/B2/B4, native precision, gfx90a,
Torch `2.12.0a0+git78258b9`, HIP `7.14.60850`.

## Profiling and warm forward

N1000 profiles found ESM3's sequence-only geometric distance norm alone took
31.6 ms at B1, with an intermediate shape `[1,256,1002,1002,3]`. This calculation
preceded the unconditional zero mask for absent frames. ESMC and ESM2 were
dominated by their projection GEMMs, with additional activation and layout work.
ATen rows and their device kernel rows overlap and must not be summed together.

The maintained operator scope passed B1/B2/B4 real-weight output checks. Warm
measurements include scope entry/exit, use two warmups per arm per pair, and retain
nine observations per arm over AB/BA/AB. They exclude loading, tokenization,
export, and Rust invocation overhead; they are **not E2E speedups**.

| Model | B1 warm forward speedup | B2 | B4 |
| --- | ---: | ---: | ---: |
| ESMC bundled 6B | 1.150x | 1.138x | 1.113x |
| ESM2-3B | 1.065x | 1.068x | 1.068x |
| ESM3-open | 2.220x | 2.282x | 2.395x |

These are direct comparisons with native arithmetic. They are not incremental
ratios over the previous HIP implementation. Distinct deterministic rotations of
the development N1000 sequence are used; these are not an independent biological
test set.

A separate B4 AB/BA/AB comparison against the **previous qualified FireProt
operators** measures ESMC **574.333 → 568.985 ms (1.0094x)** and ESM2
**876.000 → 838.984 ms (1.0441x)**, with every array also checked against native.
ESMC's added activation benefit is small; its new complete-invocation gain mainly
comes from checkpoint loading. These increments are not multiplied into the
direct native ratios.

## Loading experiments

These fresh-process diagnostics exclude imports and CUDA initialization. Each
candidate matches the official N1000 singleton arrays exactly. Timings are single
observations used to select candidates for repeated complete-invocation tests.

| Model / route | Loader time | Peak allocated GPU memory |
| --- | ---: | ---: |
| ESMC original CPU layout conversion | 10.967 s | 12.795 GB |
| ESMC complete FP32 GPU layout conversion, rejected for excess temporary memory | 6.043 s | 43.626 GB |
| ESMC BF16 GPU layout conversion | 5.487 s | 21.640 GB |
| ESMC per-block BF16 GPU layout conversion, selected | 5.450 s | 12.890 GB |
| ESM2 existing meta loader, CPU FP32 promotion | 7.408 s | 11.508 GB |
| ESM2 combined device/dtype transfer | 6.781 s | 11.508 GB |
| ESM2 GPU FP32 promotion, selected | 5.343 s | 11.400 GB |

Memory values are decimal GB, not GiB. ESMC still reads the original checkpoints
and performs the original exact weight rearrangement on every invocation. There
is no repacked checkpoint, integrity cache, or output cache.

## Rejected GEMM alternatives

Changing the installed BLAS preference to the ROCm backend selected through
PyTorch's `cublas` spelling changes final embeddings for all three models; ESM2
and ESM3 logits also change. These candidates fail the approved bitwise gate and
are rejected. Explicitly selecting `cublaslt` reproduces the existing default and
gives approximately 1.000x. No GEMM-backend change is promoted.

## Complete invocation and reproduction

Final measurements use fresh processes on HIP6 / PCI `0000:8e:00.0`, seed101,
N1000, native precision, identical B4 grouping, unflushed caches, and three
AB/BA/AB pairs. ESMC's four-thread full-file SHA verification is shared by both
arms. All requested outputs are exported and accepted. A separate ESM3 workload
uses 64 inputs in 16 B4 groups with one resident model load per invocation.

The frozen candidate is
`/media/IntelSSD/fireprot/outputs/experiments/esm-speed-20261016/qualified-v2`.
Use its Rust executable together with the published exact-route request/profile
examples; do not substitute an older installed binary. Measurement scripts and
raw diagnostic paths are retained under `experiments/plm` and the experiment
directory.
Versioned evidence: [operator/loading qualification](../provenance/esm_optimization_diagnostics_20261016.json)
and [complete invocation results](../provenance/esm_optimization_e2e_20261016.json).
The collector independently rehashed all 24 accepted invocations and 456 output
archives, checked HIP6 PCI identity, and compared every retained array byte.

| Model / workload | Native E2E median | Candidate E2E median | Speedup |
| --- | ---: | ---: | ---: |
| ESMC, four N1000 inputs / B4 | 23.795089486 s | 19.916732481 s | **1.194728579x** |
| ESM2, four N1000 inputs / B4 | 38.461337132 s | 16.463957907 s | **2.336093019x** |
| ESM3, four N1000 inputs / B4 | 12.383835161 s | 11.461301233 s | **1.080491203x** |
| ESM3, 64 N1000 inputs / sixteen B4 groups | 20.211988712 s | 15.490191705 s | **1.304824956x** |

All three pairs improve for each model. ESMC's baseline already includes the
previous parallel hashing improvement. ESM2's direct ratio includes its previously
qualified meta construction, the new transfer order, and exact operators.
These are freshly measured direct ratios, not products of incremental speedups.
The 64-input result includes one model load per arm and all 64 output archives.
It measures aggregate work on a resident model, not single-sequence latency.

Model-forward medians within these fresh invocations are ESMC
1.476 → 1.321 s, ESM2 2.151 → 2.053 s, ESM3 B4 2.326 → 1.539 s, and
ESM3 resident64 8.474 → 4.079 s (sum over sixteen groups). First-use overhead is
included here, unlike the warm-forward table. No timing boundary is substituted
for another when calculating a ratio.

### Run the explicit route

Copy an example and choose a new `run_id` and unused `output_dir` before each
invocation. All examples select HIP6 and N1000 native precision.

| Model | Request | Profile |
| --- | --- | --- |
| ESMC, four inputs / B4 | `configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json` | `configs/diamondhill/plm-esmc-fold2-6b-exact.json` |
| ESM2, four inputs / B4 | `configs/requests/plm-esm2-3b-b4-n1000-exact.json` | `configs/diamondhill/plm-esm2-3b-exact.json` |
| ESM3, four inputs / B4 | `configs/requests/plm-esm3-open-b4-n1000-exact.json` | `configs/diamondhill/plm-esm3-open-exact.json` |
| ESM3, 64 inputs / sixteen B4 groups | `configs/requests/plm-esm3-open-resident64-n1000-exact.json` | `configs/diamondhill/plm-esm3-open-exact.json` |

```bash
cd /media/IntelSSD/fireprot
cp configs/requests/plm-esm3-open-b4-n1000-exact.json /tmp/my-esm3.json
# Edit run_id and output_dir in /tmp/my-esm3.json before each invocation.
outputs/experiments/esm-speed-20261016/qualified-v2/target/release/fireprot \
  esm /tmp/my-esm3.json configs/diamondhill/plm-esm3-open-exact.json
```
