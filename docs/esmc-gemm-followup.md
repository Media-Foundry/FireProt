# ESMC GEMM and HIP graph follow-up

This experiment continues the [ESMC CK investigation](esm-optimization.md#esmc-ck-follow-up-rejected-candidates).
It uses HIP6 / PCI `0000:8e:00.0`, the pinned Biohub ESMC 6B checkpoint,
N1000 inputs, seed101 and native BF16. Acceptance remains raw-byte equality
against the official implementation with identical batch grouping. No checkpoint,
layer count, output head or precision is changed.

## Accumulation order investigation

The first-block FFN up/gate projection was checked with eight combinations of
BLAS backend and operand layout, followed by 48 permutations of the low three
K-index bits in the direct CK kernel. Two transposed CK layouts are unsupported.
No CK layout or permutation restores exact native output. The original K order
has the fewest differing elements among the tested permutations (2,185).
CK's configured FP32-to-BF16 conversion uses round-to-nearest-even; this experiment
does not establish the precise source of the remaining accumulation differences.

## Frozen native algorithms

Torch TunableOp was screened using actual first-block activations. This Torch
version refuses zero tolerances, so its internal screen uses `atol=rtol=1e-45`,
below the smallest BF16 spacing. Independent byte comparisons, including signed
zero, remain the acceptance gate. Every measured complete forward passes that gate.

| Warm forward | Current FireProt | Frozen TunableOp | Incremental speedup |
| --- | ---: | ---: | ---: |
| B1 | 170.914 ms | 163.358 ms | 1.046x |
| B4 | 578.038 ms | 544.235 ms | 1.062x |

Each row uses three AB/BA/AB pairs and nine measurements per arm. Offline tuning
and checkpoint loading are excluded. A fresh-worker diagnostic exposes a large
initialization cost: model time is 1.358 s without TunableOp and 5.456 s with it.
All four exported archives match exactly. That paired run stopped because its
profile-hash treatment declaration was incomplete, so it is **not** a qualified
E2E comparison or a statistical speedup result. The reproduction recipe now
declares both the environment switch and the profile hash. This route is not promoted.

## Direct HIP/C++ FFN projection

`experiments/plm/esmc_lt_fixed.hip` calls the selected hipBLASLt algorithm directly,
avoiding TunableOp's enumeration of algorithms. It replaces only the FFN up/gate
projection. The qualified screening IDs are 4335 for B1 and 4139 for B4, bound to
hipBLASLt version 100401 / revision `8d1ae90e`. Plan creation checks algorithm
support and workspace requirements. The Python tensor scope retains native
LayerNorm, SwiGLU rounding, down projection, attention and residual computation.

| Warm forward | Current FireProt | Direct FFN | Incremental speedup |
| --- | ---: | ---: | ---: |
| B1 | 170.800 ms | 165.901 ms | 1.030x |
| B4 | 575.346 ms | 546.701 ms | 1.052x |

All 80 replaced projections were compared individually before timing; final
tokens and embeddings match bitwise on every timed forward. Timings use three
AB/BA/AB pairs, two warmups per arm and three observations per arm per pair.
The first diagnostic's plan-creation field also includes previously queued native
work and is not a valid isolated initialization measurement. The reproduction
script now synchronizes before that timer; the forward timers already synchronized.

The complete-invocation experiment compares current FireProt with the direct
FFN candidate using the same worker, original resources and pinned shared library.
Only `FIREPROT_ESMC_FIXED_GEMM` and the declared profile hash differ. Rust retains
full preflight, GPU supervision, archive acceptance and paired comparisons.

| Complete invocation workload | Current FireProt median | Direct FFN median | Incremental ratio |
| --- | ---: | ---: | ---: |
| Four N1000 inputs, one B4 group | 20.830 s | 20.638 s | 1.009x |
| 64 N1000 inputs, sixteen B4 groups | 31.199 s | 30.435 s | 1.025x |

All three pairs pass exact-array acceptance for each workload. An independent
collector rehashes all accepted artifacts and compares every array in **408
archives across 12 invocations**, including HIP6 identity and per-group operator
coverage. These performance observations are mixed: one pair is slower in each
workload, and resident64's candidate maximum is worse (31.797 versus 31.268 s).
Do not describe this as a consistent E2E improvement or multiply these ratios by
the earlier direct-native ratio.

Model-forward medians within the complete calls are 1.343 -> 1.479 s for one B4
group and 9.996 -> 9.618 s for sixteen groups. First-use costs outweigh the warm
kernel saving in the single-group model boundary. The small single-group E2E
median difference therefore does not establish a robust model speedup. The
candidate remains experimental; default dispatch and the published 1.195x
direct-native B4 E2E result remain unchanged.

Full evidence: [ESMC GEMM follow-up](../provenance/esmc_gemm_followup_20261016.json).

## HIP graph

The original forward cannot be captured because `bool_mask.all()` requires a
host decision. For these explicitly unpadded inputs, the experiment verifies the
mask outside capture and uses the same embedding/transformer branch with no
attention mask, intermediate outputs or SAE computation. The graph also passes
byte equality after replacing its token buffer with different inputs.

| Batch | Eager warm forward | Graph replay | Speedup | Extra warmup + capture |
| --- | ---: | ---: | ---: | ---: |
| B1 | 170.849 ms | 168.835 ms | 1.012x | 0.448 s |
| B4 | 580.373 ms | 578.002 ms | 1.004x | 1.216 s |

These replay-only gains exclude capture costs and output export. They are too
small to justify enabling graph capture for the measured short-lived requests.
No graph route is promoted.

## Reproduction and retained artifacts

Remote experiment root:
`/media/IntelSSD/fireprot/outputs/experiments/esmc-gemm-order-20261016`.
The original production deployment remains unchanged. Experimental scripts live
under `experiments/plm`; the direct FFN frozen deployment is
`lt-validation/qualified-fixed` under the remote experiment root.

The raw order, tuning, graph, direct-kernel and complete-invocation reports retain
their separate timing boundaries. Early setup failures are retained: unsupported
CK layout, unavailable tuning APIs, graph mask synchronization, incomplete frozen
deployment copies, checkpoint-resource separation and the missing profile-hash
declaration. None is counted as a successful performance observation.

To reproduce the direct route on the pinned DiamondHill installation, copy a
candidate request/profile from `lt-validation/esmc-fixed-4-pairs` or
`lt-validation/esmc-fixed-64-pairs`, change `run_id` and `output_dir` to unused
values, and run the `lt-validation/qualified-fixed/target/release/fireprot esm`
command with those files. This uses the experimental deployment, not the default
top-level binary. `prepare_esmc_fixed_pairs.py` creates a complete isolated worker
snapshot; `collect_esmc_fixed_pairs.py` rechecks all accepted evidence. Offline
builds and tuning are excluded from E2E, while normal preflight, library/plan
loading, forward computation and requested output exports are included.

Further work should target checkpoint placement and startup overhead as well as
steady-state GEMM scheduling. The current loader copies FP32 checkpoint tensors
to the GPU before BF16 conversion. Streaming CPU BF16 conversion and pinned
asynchronous transfer are candidates, not established improvements.
