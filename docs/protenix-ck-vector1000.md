# N1000 CK bias vectorization

Status: experimental integration; model validation pending.

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
The default remains the previous validated CK kernel until model A/B
validation. N1536 retains its separate exact-sequence specialization.
The loader fingerprints all build inputs and reports the experimental
flag in its manifest. A binding/build failure retains SDPA fallback.

The ABI rejects unsupported dimensions and null or unaligned pointers
before launch. The wrapper repairs a contiguous bias with a misaligned
storage offset by cloning it. Focused GPU tests cover that input, the
last query/key tile, input preservation, queued calls, non-default streams,
and guards before and after the output buffer.
