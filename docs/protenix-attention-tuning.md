# Long triangle attention: SDPA and CK

Status: measured on 2026-10-03. Direct SDPA is promoted for the exact unmasked
1,000-token FP32 gfx90a bucket. The modified CK kernel remains a benchmark
candidate. [Machine-readable evidence](../provenance/protenix_amd_attention_2ad81b6.json)
contains timings, source digests, build commands, isolation checks, and numerical results.

## What SDPA actually runs

The remaining triangle-attention stage took 7.085 seconds of the preceding
13.501-second Mini-default model median. A fresh profile attributed 5.333
seconds to AOTriton's `attn_fwd` kernels. PyTorch's efficient SDPA route is
already a fused attention implementation. Selecting CK as the process-wide
BLAS backend does not replace this AOTriton kernel.

Both the tested CK operator and AOTriton fuse QK, softmax, and PV without
materializing the complete attention matrix. Their internal work differs:

- The installed AOTriton forward source loads Q before the key-block loop.
  The original CK GEMM pipeline reads Q again for each key block.
- AOTriton accumulates an unnormalized output numerator and divides once at
  the end. The original CK recurrence normalizes the output at each block.
- Their thread layouts and memory traffic differ. An actual rocprof dispatch
  reported 128 threads and zero static LDS bytes for this AOTriton kernel,
  versus 256 threads and 16,896 LDS bytes for the original CK M64/N64 kernel.
  AOTriton also reported 80 scratch bytes; zero LDS alone does not establish
  an efficiency advantage or imply zero memory overhead.

The source inspected on DiamondHill is AOTriton 0.11.2, commit
`dd1b68b604b5258ee7a9f7b66ad95e7a82c18065`, in the PyTorch build's
`aotriton_runtime/tritonsrc/fwd_kernel.py` and `fwd_kernel_inner.py`.
The installed CK headers report 1.2.0, commit
`8d1ae90eff7d022f26019ec55b2ec6a7674b3112`. These are distinct from the
separate CK source checkout on the machine. Source hashes and dispatch
resource fields are preserved in the evidence record. Reducing non-matmul
work and shared-memory communication is also a documented principle of
[FlashAttention-2](https://arxiv.org/abs/2307.08691); its published speedups
are not FireProt measurements.

## CK changes and measured result

The isolated [header patch](../patches/ck/README.md) caches Q's transfer
registers when GEMM0 has a single K=32 stage and defers output normalization.
The probe also tunes GEMM1 K and the M/N tiles. It retains FP32 data and
accumulation; no reduced precision or recycle reduction is involved.

The canonical comparison uses logical Q/K/V `[1000,4,1000,32]` with actual
Protenix head-last storage, additive triangle bias `[1,4,1000,1000]`, seed
419, TF32 disabled, five warmups, twenty measured repetitions, and GPU 7.
All entries include bias packing and output allocation. Compilation is
excluded. Every candidate passed `atol=rtol=2e-5` against chunked SDPA.

| Implementation | Median ms | p95 ms | Maximum absolute delta |
| --- | ---: | ---: | ---: |
| Previous chunked SDPA | 37.699 | 37.859 | 0 |
| Direct SDPA | 36.021 | 36.138 | 0 |
| Original CK M64/N64 | 45.641 | 45.785 | 3.10e-6 |
| CK cached Q, M64/N64 | 37.121 | 37.144 | 2.86e-6 |
| CK cached Q, M64/N32 | 36.139 | 36.278 | 2.62e-6 |
| CK cached Q, M128/N32 | 37.096 | 37.233 | 2.62e-6 |
| Direct SDPA, repeated last | 35.997 | 36.132 | 0 |

The best CK candidate is **1.263x faster than the original CK** and remains
about 0.3–0.4% slower than direct SDPA. This is near parity, not a demonstrated
replacement speedup. CK's probe peak allocation was 2,593,554,432 bytes;
direct SDPA's was 3,105,554,432 bytes, including inputs and the retained
reference. This does not establish a whole-model memory reduction.

Diagnostic ablations with prepacked bias and preallocated CK output found:
deferred normalization alone changed roughly 45.43 to 44.20 ms; changing
GEMM1 K alone changed 45.44 to 45.25 ms; the combined Q-cache variant reached
37.04 ms. Persistent Q in LDS reached 37.28 ms and fast-math on the cached-Q
variant reached 37.40 ms, so neither was retained. MFMA-32 and larger tiles
were also slower in the original matrix. These separate diagnostics support
the data-reuse hypothesis but do not isolate an exact additive contribution
for every change. Their absolute times should not be mixed with the canonical
table because allocation and bias-packing policies differ.

## Promoted model change

For the exact unmasked long bucket, one SDPA call now emits a head-last
contiguous result. This removes outer chunks of 64, their output assignments,
and the subsequent layout copy. Explicit pair masks retain the preceding
path. `FIREPROT_DISABLE_LONG_DIRECT_ATTENTION=1` restores chunked SDPA.

Same-source A/B measurements used Mini-default v0.5.0, the retained real
1,000-residue input, FP32, C4/S1, one sample, seed 101, hipBLAS, no MSA or
templates, one warmup per process, warm measured calls, and physical GPU 6.

| Run order | Measured repetitions per side | Chunked model median | Direct model median | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Chunked then direct | 3 | 13.496 s | 13.129 s | 1.028x |
| Direct then chunked | 5 | 13.482 s | 13.136 s | 1.026x |

The final default-path verification at `5236d0c` measured **13.124 s median**
and **13.143 s p95**, with 576 direct-attention selections across one warmup
and three measured calls. Attention took 6.726 s; triangle multiplication
took 3.917 s. Maximum model allocated memory remained 10,037,256,704 bytes.
The operator's larger local workspace did not raise this workload's model
peak. The complete four-call benchmark worker took 98.528 s; it is not a
single-request cold-start latency. Relative to the preceding native median
of 45.350 s, the final model is approximately 3.456x faster; the incremental
A/B above is the controlled evidence for this change.

The dedicated same-seed output comparison covered 7,979 atoms and passed the
unchanged `1e-3` aligned-coordinate-RMSD and confidence gates: aligned RMSD
`1.0293e-4 A`, maximum coordinate delta `9.765625e-4 A`, and maximum confidence
delta `9.54e-7`. This input has no experimental-coordinate mapping. These are
native-agreement checks, not structural accuracy measurements. CK has not
passed a model-coordinate gate and is not used in these model results.

## Reproduce and interpret provenance

```bash
# Exact operator comparison; the patch is applied only inside runtime/.
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python -m benchmarks.protenix.benchmark_attention \
  --include-ck --ck-set tuned --output outputs/benchmarks/attention-tuned.json

# Run each mode sequentially on GPU 6; use separate output directories.
HIP_VISIBLE_DEVICES=6 PYTHONPATH=.:Protenix \
FIREPROT_PROTENIX_COMMIT=77218da6f5ccf8562a76577dd568869785e6b45b \
python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250-long.json \
  --backend fireprot --blas-backend hipblas
# Prefix FIREPROT_DISABLE_LONG_DIRECT_ATTENTION=1 for the preceding path.
```

Source-label correction: the verified independent Protenix integration commit
is `77218da6f5ccf8562a76577dd568869785e6b45b`. Earlier records had the correct
short prefix but an incorrect full suffix. The final default manifest also
inherited remote upstream Git metadata because its override was omitted.
An audit of all 248 tracked files found identical local and deployed tree
digest `8e62964d09093524c0bb19955c30cd43a5d38dccdb96d45d74197e29551a8c78`.
Historical raw labels are retained, with the correction in the new evidence
record. Synchronization deliberately excludes Git metadata.

Final FireProt tests: local 97 passed / 12 skipped; GPU 7 with HIP tests enabled
108 passed / 1 skipped. An accidentally unscoped pytest command collected
upstream Protenix tests too and exposed unrelated PDBe-network and missing
LAPACK failures; these are recorded in provenance rather than hidden.

## Next work

The next CK experiment should reduce repeated LDS transfers and barriers,
evaluate a suitable FP32 CK Tile QRKSVS pipeline, and examine exp2/softmax
scheduling. That pipeline was inspected but has not yet been compiled or
measured. A replacement needs a stable operator win and full-model numerical,
p95, and memory gates. Larger than 1,000 tokens, Base weights, and additional
inputs still require their own measurements; these exact-bucket results do
not establish their performance or workspace safety.
