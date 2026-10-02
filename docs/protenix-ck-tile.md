# CK Tile triangle attention on MI250

Status: promoted for the exact FP32 gfx90a N1000 implicit-mask bucket,
2026-10-03. Candidate: `af43d8e`; default promotion: `8fbdddc`.
The preceding [generic CK tuning](protenix-attention-tuning.md) remains a
separate experiment. This implementation uses CK Tile's QRKSVS pipeline.
The [evidence record](../provenance/protenix_ck_tile_8fbdddc.json) preserves
build identities, screening, both model orders, numerical checks and the
final default verification.

## Implementation

The selected kernel uses an M128/N64 query/key tile, four wave64 warps,
GEMM0 K16, GEMM1 K64, and FP32 inputs, intermediate probabilities,
accumulators, and output. Q stays in registers; K and V use shared memory.
K16 satisfies the installed pipeline's fixed two-stage GEMM0 prefetch
requirement for head dimension 32. A first K32 instantiation failed that
compile-time requirement and was corrected before measurement.

The installed CK headers report version 1.2.0, commit
`8d1ae90eff7d022f26019ec55b2ec6a7674b3112`. The backend requires this measured
version and records the complete CK Tile `.hpp` tree, C++ source, compiler,
build command, and library digests. Compilation is lazy and protected by a
process lock; complete libraries replace a temporary file atomically.
Neither installed headers nor the independent Protenix checkout is edited.

Dispatch requires logical Q/K/V `[1000,4,1000,32]`, strides
`[128000,32,128,1]`, aligned FP32 pointers on one gfx90a device, inference
mode, and one triangle bias `[1,4,1000,1000]`. The bias is packed before
launch. Output has contiguous head-last storage. Other signatures and build
failures continue through SDPA or native fallback. The current HIP stream
and tensor's device are used explicitly.

## Operator screening

Diagnostic screening used seed 419, FP32 with TF32 disabled, observed QKV
strides, five warmups, fifteen repetitions, and physical GPU 7. These
diagnostics used prepacked bias and a preallocated CK output; the formal
comparison below includes both costs.

| CK Tile configuration | Diagnostic median |
| --- | ---: |
| M64/N64, 2 waves | 34.49 ms |
| M64/N64, 4 waves | 34.83 ms |
| M128/N64, 4 waves | **32.68 ms** |
| M64/N64, fast exp2 | 39.02 ms |

The separate smaller-tile screen measured M64/N32 at 34.60 ms, M32/N32 at
40.37 ms, and M32/N64 at 38.01 ms. The selected configuration keeps the
default exp2 policy. Compiler resource remarks for M128/N64 reported 154
VGPRs, 9,216 LDS bytes per block, three waves per SIMD, and zero spills or
scratch. These are compiler diagnostics, not a hardware-counter attribution
of the entire speedup.

The formal operator benchmark used the actual backend, seed 419, five
warmups and twenty repetitions, with bias packing and output allocation:

| Operator | Median | p95 | Peak allocated in probe |
| --- | ---: | ---: | ---: |
| Direct SDPA | 36.361 ms | 37.944 ms | 3,105,554,432 B |
| CK Tile | **32.711 ms** | **32.888 ms** | **2,593,554,432 B** |
| Direct SDPA repeated last | 36.349 ms | 36.491 ms | 3,105,554,432 B |

The maximum absolute output difference was `3.93e-6`, passing the unchanged
`atol=rtol=2e-5` operator gate. Probe peaks include inputs and the reference
output; they are not full-model memory measurements. All reported runs
passed preflight and continuous GPU PID isolation checks.

## Model comparison

The workload is Protenix Mini-default v0.5.0, the retained real 1,000-residue
input, FP32, four cycles, one diffusion step, one sample, seed 101, hipBLAS,
TF32 disabled, and no MSA/templates. The independent Protenix integration
is `77218da6f5ccf8562a76577dd568869785e6b45b`. Both sides use the same
candidate code, with CK selected by `FIREPROT_ENABLE_CK_ATTENTION=1` or
disabled by `=0`. Each process uses one warmup; model GPU times exclude
loading, preprocessing and serialization.

| Order on GPU 6 | Repetitions per side | SDPA median | CK median | Speedup | SDPA p95 | CK p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDPA then CK | 3 | 13.319 s | 12.581 s | 1.059x | 13.605 s | 12.582 s |
| CK then SDPA | 5 | **13.120 s** | **12.580 s** | **1.043x** | **13.179 s** | **13.103 s** |

The first SDPA side fluctuated, and a separate numerical run used GPU 7
during part of that first comparison. The reverse order kept GPU 7 idle and
is the conservative primary incremental result. Five-sample p95 is the
maximum of those samples; CK's 13.103-second slow sample is retained.
This establishes a measured median improvement for this bucket, not a
precise production tail-latency estimate.

In the reverse comparison, triangle attention improved from **6.712 to
6.183 seconds**, and Pairformer output from **12.096 to 11.596 seconds**.
CK attention was selected 864 times across one warmup and five measured
calls. Peak model allocated memory was identical at **10,037,256,704 bytes**;
maximum reserved memory was identical at **14,294,188,032 bytes**. The full
six-call benchmark worker improved from 130.610 to 128.674 seconds (1.015x).
That worker includes startup and a warmup and is not single-request latency.

Final default verification at `8fbdddc`, without an opt-in, measured **12.584 s
median** and **12.586 s p95** over three calls after one warmup. It selected CK
attention 576 times, with attention at 6.181 s and Pairformer output at
11.586 s. The four-call benchmark worker took 96.303 s. Relative to the
historical matching native median of 45.350 s, this is approximately 3.604x
overall model acceleration; the controlled incremental result remains the
1.043x reverse-order comparison above.

The dedicated native-agreement run passed the existing `1e-3` gates over
7,979 atoms: aligned coordinate RMSD `9.37e-5 A`, maximum coordinate delta
`9.765625e-4 A`, and maximum confidence delta `3.81e-6`. This performance
input has no experimental-coordinate mapping. Recycle count, diffusion
steps, precision and random state were unchanged.

## Controls and reproduction

```bash
# Formal operator comparison; compilation is excluded from timing.
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python -m benchmarks.protenix.benchmark_attention \
  --include-ck-tile --output outputs/benchmarks/ck-tile-attention.json

# Default exact-bucket CK model path.
HIP_VISIBLE_DEVICES=6 PYTHONPATH=.:Protenix \
FIREPROT_PROTENIX_COMMIT=77218da6f5ccf8562a76577dd568869785e6b45b \
python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250-long.json \
  --backend fireprot --blas-backend hipblas
```

Set `FIREPROT_DISABLE_CK_ATTENTION=1` to compare against direct SDPA.
`FIREPROT_ENABLE_CK_ATTENTION=0` remains supported for the original A/B
commands. `FIREPROT_DISABLE_LONG_DIRECT_ATTENTION=1` disables both direct
implementations and restores chunking; the overall long-attention disable
still disables every long implementation. Unsupported CK versions or builds
fall back without changing native defaults outside FireProt.

Final tests cover capability rejection, build disable/failure, actual registry
selection, nondefault HIP streams, the N1000 tail, all-negative-infinity bias
rows, a large positive bias, contiguous output, and input preservation.
Local: 99 passed / 13 skipped. GPU 7: 111 passed / 1 skipped.

## Remaining work

Only this exact shape, dtype, layout, GPU architecture and installed CK
version are promoted. The next priority is real inputs longer than 1,000
residues, with workspace and numerical gates for each added bucket, followed
by the remaining long-sequence projections and triangle multiplication.
Base-model and multi-input quality/performance coverage remain unproven.
