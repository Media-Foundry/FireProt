# N1000 CK bias vectorization

Status: promoted for the exact validated N1000 bucket after model validation.

The installed CK QRKSVS pipeline sets `kAlignmentBias=1` whenever K
sequence padding is enabled. N1000 requires tail masking, but its bias
row stride and logical K length both divide four. A derived pipeline can
retain the padded Q/K semantics while advertising four-element alignment
for the bias tensor view. This does not change sequence length, FP32
arithmetic, masks, recycle count, or diffusion steps.

The controlled probe used gfx90a physical GPU 6, idle sibling GPU 7,
FP32/TF32 disabled, seed 907, QKV `[1000,4,1000,32]` with head-last
storage, five warmup groups and twenty measured groups. Each group
contains eight calls; timings include bias packing and output allocation,
and exclude compilation.

| Implementation | Median per call |
| --- | ---: |
| Current padded CK, before | 32.682 ms |
| Vector bias, M128/N64/W4 | 20.711 ms |
| Vector bias, M128/N32/W4 | 20.058 ms |
| Current padded CK, after | 32.675 ms |

The N64 comparison isolates the alignment override at the original tile
shape. The N32 candidate is about 1.63x faster than the current kernel.
Maximum absolute delta against SDPA is 6.68e-6. Full outputs, fully masked
rows, extreme bias, and output-buffer sentinels pass. Both continuous GPU
process monitors pass. This run does not time SDPA, so it must not be
presented as a fresh same-run SDPA performance comparison.

[Compact probe evidence](../provenance/protenix_ck_vector1000_probe_1cb0d35.json)
records builds, source hashes and measurements. The raw report remains at
`outputs/benchmarks/attention-vector-bias-1cb0d35.json` on DiamondHill.

## Integration

`FIREPROT_ENABLE_CK_VECTOR_BIAS1000=1` selects a separate C ABI and
M128/N32/W4 padded kernel for the existing exact N1000 capability bucket.
This path is now the default after model A/B validation. Set the flag to
`0` to restore the previous CK kernel. N1536 retains its separate exact-sequence specialization.
The loader fingerprints all build inputs and reports the experimental
flag in its manifest. A binding/build failure retains SDPA fallback.

The ABI rejects unsupported dimensions and null or unaligned pointers
before launch. The wrapper repairs a contiguous bias with a misaligned
storage offset by cloning it. Focused GPU tests cover that input, the
last query/key tile, input preservation, queued calls, non-default streams,
and guards before and after the output buffer.

## Same-source model validation

Candidate `41238d5` ran on GPU 7 with GPU 6 idle and continuously monitored.
Both sides used Mini-default v0.5.0, the same real 1000-residue input, FP32,
TF32 disabled, four cycles, one diffusion step, one sample, seed 101,
hipBLAS, no MSA/templates, and expandable allocator segments. The candidate
ran first, followed by the previous CK path, with one warmup and three
measured calls per process. Both runs selected CK attention 576 times
across warmup and measurements. Both monitor pairs passed.

| Metric | Previous CK | Vector-bias CK |
| --- | ---: | ---: |
| Warm model median | 12.615 s | 10.745 s |
| Warm model p95 | 12.646 s | 10.822 s |
| Triangle attention median | 6.197 s | 4.326 s |
| Peak allocated memory | 10,011,042,304 B | 10,011,042,304 B |
| Complete benchmark worker wall time | 97.120 s | 89.936 s |

The warm model improves **1.174x**, saving about 1.870 seconds per call;
triangle attention improves **1.433x**. Warm model timings exclude loading,
preprocessing, serialization and first compilation. The worker wall-time
comparison includes one warmup and three measured predictions; it is not a
single-request cold-start measurement.

All three output pairs pass the unchanged 1e-3 aligned-coordinate RMSD and
confidence gates over 7,979 atoms. Maximum aligned RMSD is 9.23e-5 Angstrom
and maximum confidence delta is 5.25e-6. This establishes agreement with
the previous CK baseline on one input, not experimental structure accuracy.

[Model evidence](../provenance/protenix_ck_vector1000_model_41238d5.json)
contains source/build hashes, timing distributions, selection counts,
monitors and output comparisons. The promoted C++ source is identical to
the model-tested source; promotion changes only the Python flag default.
The SDPA fallback remains available with `FIREPROT_DISABLE_CK_ATTENTION=1`.
