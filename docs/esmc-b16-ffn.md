# ESMC B16 fixed FFN qualification

This experiment tunes the largest remaining GPU category, GEMM, while preserving
native BF16 output bytes. It targets the FFN up/gate projection at N1000/B16:
16,032 token rows, 2,560 input channels and 13,824 output channels. It does not
change attention, model depth, precision, sequence grouping or output selection.

## Offline selection and native implementation

A real first-block activation is screened with Torch TunableOp. Algorithm
`Gemm_Hipblaslt_3903` passes independent byte equality, including signed zero.
The internal TunableOp tolerance of `1e-45` is not the acceptance criterion.
Offline tuning takes 260.337 seconds and is excluded from inference timings.
The tuning budget is 10 milliseconds per solution, not 10 seconds.

A Rust-built HIP C ABI library fixes algorithm 3903, rejects unsupported rows,
checks the installed hipBLASLt version/revision and algorithm support, and uses
FP32 accumulation with BF16 inputs/outputs. It requires no extra workspace for
this qualified signature. The plan is created lazily at the first matching FFN,
after native QKV has initialized the library; creation takes about 2.1 ms.
Subsequent groups reuse the plan. There is no online tuning or silent fallback.

The direct path passes all 80 projection comparisons and complete model-output
checks. Six unsupported row counts are rejected before launching a kernel.

| Warm measurement | Reference | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| First-block projection, TunableOp screen | 9.450707 ms | 8.980955 ms | 1.0523x |
| Full B16 forward, TunableOp screen | 2.083235 s | 2.052286 s | 1.0151x |
| Full B16 forward, fixed HIP library | 2.084192 s | 2.050774 s | 1.0163x |

Each forward screen uses three AB/BA/AB pairs, two warmups per arm per pair and
three observations per arm per pair. Every timed output is checked. Weights
are resident, caches unflushed; loading, exports and offline compilation are
outside these warm measurements. Do not combine these ratios.

## Complete invocation and correctness

Both E2E arms use the same frozen workers and Rust CLI, same assets, same B16
grouping, native BF16, seed 101 and HIP6 / PCI `0000:8e:00.0`. Both retain HIP
RoPE/pointwise operations, pinned loading, import overlap, eight full-file hash
threads, local CPU affinity and snapshot monitoring. Only the B16 FFN flag
and its profile hash differ. Each invocation starts a fresh worker, loads once,
checks original checkpoints and exports every prediction. Caches are unflushed.

The 16- and 64-input manifests contain distinct N1000 development sequences,
not an independent biological test set. Official native B16 inference on all
64 sequences supplies the strict reference; its first 16 outputs also match
the preceding frozen B16 reference. Rust seals the measured calls and compares
all arrays; an independent collector rehashes accepted artifacts and checks
all output bytes, grouping, device identity and per-group operator execution.
No cross-grouping equality or folding-quality claim is made.

| Workload | Reference E2E median | Fixed-FFN E2E median | Ratio |
| --- | ---: | ---: | ---: |
| 16 inputs, one B16 group | 14.044837 s | 14.015487 s | **1.00209x** |
| 64 inputs, four B16 groups | 21.865148 s | 21.776128 s | **1.00409x** |

There are three AB/BA/AB pairs per workload. Paired ratios are
1.008265 / 1.001541 / 1.000284 for 16 inputs, and
0.999203 / 1.006966 / 1.013532 for 64 inputs. The latter includes one regression.
Forward medians within these fresh invocations are 2.763674 -> 2.731094 s and
9.029481 -> 8.877749 s respectively; these include first-use costs and are not
the separately measured warm-forward boundary.

All 12 invocations and **480 prediction archives** pass independent checks
against the **64 official reference archives**. Each candidate group executes
80 fixed FFN calls; the plan is created only in the first group. Every group
also retains 80 RoPE, 80 SwiGLU and 160 residual calls.

**Decision: retain as an experiment.** The E2E gain is small and the resident
paired trend is mixed. No default profile, model code or maintained executable
changes. The published native-to-optimized B4 E2E remains **1.978x**. These
incremental B16 ratios must not be multiplied into it.

## CK interpretation

This result uses hipBLASLt, not CK. Earlier installed CK GEMM increased B4 warm
forward time by about 7.3% and failed exact-output acceptance. All 48 separately
tested direct CK tile/projection combinations also failed byte equality, although
some FFN up/gate kernels were faster in isolation. See the
[CK experiment](esm-optimization.md#esmc-ck-follow-up-rejected-candidates).
Those failures concern the tested shapes and tiles, not all possible CK kernels.
Replacing the backend wholesale is currently unsupported by the evidence.

## Reproduction and evidence

Remote root: `/media/IntelSSD/fireprot/outputs/experiments/esmc-b16-ffn-20261016`.
It retains the offline screen, generated HIP source, build receipt, library,
frozen deployment, official reference, both paired runs and measured scripts.
The frozen CLI in `qualified-b16-ffn/target/release/fireprot` is used for both arms.
For a fresh run, use a new output root and fresh request/output identifiers.
The maintained scripts under `experiments/plm` are:

- `screen_esmc_b16_ffn_tunable.py`: offline selection and all-layer validation.
- `build_esmc_b16_lt.py`: Rust build of the fixed algorithm.
- `esmc_b16_lt_scope.py`: lazy projection scope used by the frozen worker.
- `screen_esmc_b16_direct.py`: direct-library parity and warm measurements.
- `prepare_esmc_b16_ffn_pairs.py`: frozen deployment, native reference and E2E pairs.
- `collect_esmc_b16_ffn_pairs.py`: independent archive and execution validation.

See the [pinned evidence](../provenance/esmc_b16_ffn_20261016.json).
