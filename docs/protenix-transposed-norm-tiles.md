# Transposed LayerNorm scheduling and subgroup experiment

## Operator result

This experiment keeps the per-row Welford arithmetic and changes which pair
rows a block visits. A first scan uses one row per wave64 with square pair
tiles. A second scan uses two independent 32-lane groups per wave64, each
reading four channels per lane. The redundant first shuffle that combines
32 empty lanes with the populated half-wave is omitted; the remaining
within-row reduction order is retained.

The best mapping depends on length. N1000 favors a linear traversal with four
physical waves (eight rows) per block. N1536 favors a 32x32 pair tile with eight
physical waves; this improves locality across both transposed axes.

| N | Current production before / after | Selected candidate | Ratio vs after |
| --- | ---: | ---: | ---: |
| 1000 | 1.864 / 1.866 ms | 0.791 ms | 2.360x |
| 1536 | 8.525 / 8.524 ms | 2.249 ms | 3.790x |

The preceding wave64 tiled scan reaches 4.235 ms at N1536 with a 16x16 tile
and four waves, versus its 8.530 ms linear control. The subgroup scan's N1536
linear variants are slower than production, so the 32-lane grouping must be
combined with the measured 2D mapping at this length. N1000 2D tiles are slower
than its subgroup linear traversal. No universal tile is selected.

All 32 validation comparisons per scan pass with zero measured maximum
absolute delta against Torch: two lengths, eight variants, normal and stress
inputs. Stress rows include constants 0 and +/-1000, `1+arange(128)*1e-6` and
`1000+arange(128)*0.001`. Output guard regions and tail handling pass. Peak
allocations match the current production path. Timing includes allocation and
excludes compilation. Both scans use GPU 7 serially, with continuous isolation
checks on GPUs 6 and 7; every monitor passes. FP32, TF32 off, seed 1043, five
warmups, twenty measured GPU-event samples, non-default stream, warm caches.
These are random affine-parameter operator replays, not model predictions;
checkpoint/cycles/steps/sample count do not apply.

Base source is `9c9dd1d`. In the raw probe reports, the `waves` field in
validation rows is a **selector**, not always a wave count: 4/8 select linear
traversal, while `tile*10+waves` selects a tiled variant (e.g. 328 means tile32,
eight physical waves). Timing rows use the corresponding `variant` name.

## Opt-in integration

`FIREPROT_ENABLE_TRANSPOSED_NORM_SUBGROUP=1` selects the new mapping inside the
existing transposed LayerNorm capability checks. Pair-layout and transposed
LayerNorm flags remain required. The old kernel stays available for same-source
A/B; the new flag defaults off while model validation is pending. Counts
separately identify `amd-gfx90a-transposed-layernorm-subgroup`.

The C++ binding keeps the previous four-argument API, with an optional fifth
boolean selecting the subgroup implementation. Shape, stride, dtype, device,
epsilon, alignment and inference guards are unchanged. Production tests cover
both mappings, both lengths, a non-default stream, stress inputs, input
preservation, dispatch and fallback. An operator speedup is not a model or
end-to-end speedup claim.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_transposed_norm_tiles.py \
  --output outputs/benchmarks/transposed-norm-tiles.json
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_transposed_norm_subgroup.py \
  --output outputs/benchmarks/transposed-norm-subgroup.json
```
