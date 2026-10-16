# ESMC exact HIP pointwise follow-up

This experiment investigates remaining ESMC operator costs after the
[1.911x direct-native E2E result](esmc-host-scheduling.md). It does not replace
the published profile. It uses handwritten HIP with explicit BF16 rounding;
no new CK GEMM or attention implementation is admitted.

## Why this direction

The preceding native B4 profiler attributes approximately 459.5 ms to the four
GEMMs per block across 80 blocks, 48.6 ms to attention, 23.4 ms to LayerNorm,
6.6 ms to LayerNorm-input copies, and 13.8 ms to residual addition/division.
These are diagnostic GPU times, not complete-invocation timing, and the rows
must not be added again to their underlying kernel events.

Earlier CK experiments failed strict projection equality and did not beat the
qualified native GEMM path consistently. See [CK results](esm-optimization.md#esmc-ck-follow-up-rejected-candidates)
and [accumulation-order experiments](esmc-gemm-followup.md). This follow-up
therefore tests work that can preserve the native accumulation order:

- Fuse residual scaling and addition while retaining the intermediate BF16
  rounding before addition. B4 selects four BF16 values per thread.
- Vectorize the existing exact HIP SwiGLU to eight BF16 values per thread,
  retaining the BF16 rounding between SiLU and multiplication.

The compatibility library keeps the original qualified RoPE/GELU implementation
and ABI. Attention and all GEMMs remain unchanged. The library is built through
Rust's dependency-pinned HIP build service. Rust also owns the E2E execution,
physical-device verification, archive acceptance and paired comparisons.

## Operator and warm-forward screen

Conditions: bundled ESMC 6B, N1000, seed101, native BF16, HIP6 / PCI
`0000:8e:00.0`, B1 or B4 with identical grouping. Weights and operators are
resident; caches are not flushed. Each forward comparison uses three AB/BA/AB
pairs with two warmups and three observations per arm per pair. Loading,
checkpoint verification and output export are excluded from this screen.

| Candidate | Batch | Current FireProt forward | Candidate forward | Incremental speedup |
| --- | ---: | ---: | ---: | ---: |
| Residual fusion, vector2 | 1 | 171.607 ms | 169.801 ms | 1.0106x |
| SwiGLU, vector8 | 1 | 171.292 ms | 170.052 ms | 1.0073x |
| Residual fusion, vector4 | 4 | 576.031 ms | 569.964 ms | 1.0106x |
| SwiGLU, vector8 | 4 | 574.835 ms | 566.942 ms | 1.0139x |

The B4 residual operator improves 1.742x and the B4 SwiGLU operator improves
1.656x. Their individual warm-forward ratios must not be multiplied. All 160
residual outputs and all 80 SwiGLU activations are byte-checked in each batch
screen. Every timed final token/embedding output matches native execution with
the same grouping, including signed-zero representation.

## Complete-invocation composition

Both arms use the current pinned loader, import overlap, eight full-file hash
workers and Rust CPU placement. The treatment combines residual vector4 with
SwiGLU vector8. Three separately measured AB/BA/AB pairs per workload include
preflight, process startup, shared-library loading, inference and all exports.
Processes are fresh; filesystem/GPU caches are not flushed. Native BF16,
seed101, HIP6, N1000 and B4 grouping are unchanged.

| Workload | Current FireProt E2E | Pointwise E2E | Incremental ratio | Decision |
| --- | ---: | ---: | ---: | --- |
| Four inputs, one B4 group | 12.619 s | 12.644 s | 0.9980x | Keep default unchanged |
| 64 inputs, sixteen B4 groups | 23.737 s | 23.059 s | 1.0294x | Experimental resident-batch opt-in |

The resident64 paired ratios are 1.02958x, 1.02993x and 1.02546x; all three
candidate calls are faster. The candidate maximum is 23.064 s, versus the
baseline maximum of 23.754 s. Model-forward medians are 9.940 -> 9.735 s
(1.0211x), so the full 0.677 s E2E difference must not be attributed solely to
GPU arithmetic. The one-group model-forward boundary improves 1.298 -> 1.245 s
(1.0421x), while its E2E result remains mixed and unchanged.

Rust exact-array gates pass for all 12 measured invocations / 408 prediction
archives. An independent collector rehashes those archives plus 68 reference
archives, comparing every array to native execution with identical grouping.
The 64-input reference is an additional official native run, not an extra
performance pair. All 476 archives pass; original checkpoint checks and the
physical GPU identity remain verified. No numerical tolerance is relaxed.

The published direct-native B4 E2E result remains **1.911x**. The new 1.0294x
is incremental over current FireProt for a different, resident64 workload;
do not multiply the ratios or claim a new direct-native resident64 ratio.

### Experimental resident64 entry

The checked-in profile preserves the measured profile bytes. Copy the request
and choose a new run ID and unused output directory before every execution:

```bash
cd /media/IntelSSD/fireprot
cp configs/requests/plm-esmc-fold2-6b-resident64-pointwise.json /tmp/esmc-resident64.json
# Edit run_id and output_dir in the copied request.
outputs/experiments/esmc-pointwise-20261016/validation/qualified-pointwise/target/release/fireprot \
  esm /tmp/esmc-resident64.json configs/diamondhill/plm-esmc-fold2-6b-pointwise.json
```

This is a DiamondHill deployment example with pinned experiment assets, including
the 64-sequence input manifest and native library. Use the frozen worker and CLI
together. `FIREPROT_ESMC_POINTWISE=0` selects the measured control in that same
deployment. The general public `plm-esmc-fold2-6b-exact.json` remains unchanged.

Full evidence: [pointwise build, screen and E2E qualification](../provenance/esmc_pointwise_20261016.json).
The subsequent [lazy FFN composition](esmc-resident-gemm.md) measures a further
1.0111x for resident64 and exposes a separate experimental request/profile.
The original pointwise entry above remains unchanged for reproduction.

## Retained setup failures

The first screening attempt replaced the FFN wrapper after entering the existing
HIP scope, causing its independent SwiGLU coverage check to fail. That run is
retained and is not an accepted complete screen. The corrected path uses a
compatible shared-library ABI, preserving all 80 RoPE and 80 SwiGLU counters.

Two compatibility-library build attempts were also rejected before GPU use:
an undeclared include-search tree, then an unstaged quoted header. The successful
build declares a separate complete header tree. No build guard was bypassed.

## Reproduction

Remote root: `/media/IntelSSD/fireprot/outputs/experiments/esmc-pointwise-20261016`.
The successful library is recorded by `build-receipt-v4.json`; its original HIP
source, compatibility source and qualified header are all pinned by the build.
`screen-v2/report.json` retains the complete screen. Original failed logs remain
alongside it. Current local screening code corresponds to remote
`screen_esmc_pointwise_v2.py`; the scope later adds only the E2E opt-in wrapper.
The measured screen's original scope bytes remain under the experiment root.

The E2E deployment is `validation/qualified-pointwise`. Its request/profile
pairs are in `validation/b4-pairs` and `validation/b64-pairs`.
`prepare_esmc_pointwise_pairs.py` freezes a new copy of the host-scheduling
deployment and executes both workloads sequentially; use a fresh work directory.
`collect_esmc_pointwise_pairs.py` independently rehashes accepted archives and
compares every output to an official same-batch native reference.

No top-level model source, existing default profile, checkpoint or maintained
binary is modified by the experiment. The separate opt-in request/profile above
exposes only the measured frozen deployment.
