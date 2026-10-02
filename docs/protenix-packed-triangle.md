# Packed triangle multiplication on MI250

Status: measured on the 1,000-token Mini-default FP32 C4/S1 workload.

The packed path combines two handwritten HIP layout kernels with a library BMM. It is enabled for the exact `[1000,1000,128]` contiguous, implicit-mask, gfx90a inference signature after numerical and end-to-end validation. Other shapes and explicit masks retain the existing capability fallback. `FIREPROT_DISABLE_LONG_TRIANGLE_PACKED=1` selects the previous chunked triangle-gate implementation for diagnosis.

## Why the layout changes matter

The preceding path spent 11.324 seconds in triangle multiplication per model call. Its observed copy operations at channel-first chunk shapes consumed several seconds, while its two dominant chunked BMM signatures consumed about 1.35 seconds in aggregate. Optimizing the contraction library alone would leave the repeated normalizations and layout traffic in place.

The new flow is:

1. Normalize the original pair tensor once and retain it for both projections and the output gate.
2. Project the two branches with their existing linear layers.
3. Fuse `sigmoid(gate) * projection` with a tiled conversion from `[N,N,128]` to contiguous `[128,N,N]`. Both reads and writes are coalesced through a padded LDS tile.
4. Compute `a @ b.transpose(-1,-2)` for outgoing updates or `a.transpose(-1,-2) @ b` for incoming updates. These are BLAS-compatible matrix views, so the contraction does not need a channel-layout copy.
5. Fuse the output layout conversion and LayerNorm. A wave64 handles each 128-value row; variance uses squared deviations from the mean to avoid cancellation on large constant offsets.
6. Apply the existing output projection, fused sigmoid gate, and requested in-place add or overwrite.

The full intermediate projections consume more temporary memory than the old chunked algorithm. On this exact 1,000-token workload, the measured whole-model peak allocated memory stayed at 10,037,256,704 bytes. This observation does not justify using the full path at arbitrary long lengths. A future larger bucket may need chunked packing or an explicit workspace policy.

## Incremental comparison

The two runs used the same candidate source, physical GPU 6, hipBLAS, input, weights, FP32, four recycles, one diffusion step, one sample, seed 101, and warm in-process caches. Each used one warmup and three measurements. Both continuous process monitors passed.

| Metric | Previous chunked path | Packed path | Speedup |
| --- | ---: | ---: | ---: |
| Model median | 20.865 s | 13.495 s | 1.546x |
| Triangle multiplication median | 11.324 s | 3.919 s | 2.889x |

The default-enabled source subsequently measured a 13.501-second model median and selected the packed implementation 576 times across warmup and measured calls. The implementation preserves the requested four recycles.

The final default-path comparison to native Protenix covered 7,979 atoms. Aligned coordinate RMSD was `9.46e-5 A`, maximum coordinate delta was `9.77e-4 A`, and maximum numeric confidence delta was `7.63e-6`; all values were finite and confidence schemas matched. The established aligned-RMSD and confidence gates of `1e-3` passed. The workload has no experimental-coordinate mapping, so this validates native agreement.

## CK and BLAS evaluation

The complete preceding profile yielded 62 FP32 BLAS shape signatures. The measurement runner exercised auto, hipBLAS, hipBLASLt, process-wide CK, and the direct ATen CK extension in separate monitored processes. CK supported 48 signatures and direct CK supported 31. Major FP32 projection GEMMs were slower with the measured CK routes than with hipBLAS. The library comparison is recorded in `provenance/protenix_amd_long1000_blas_a8bbfef.json`.

Those isolated measurements replay contiguous operands with observed dimensions. They do not replay original strides or prove which internal kernel served a process-wide preference. Full-model comparisons remain the promotion gate.

With the packed path, hipBLASLt measured a 13.431-second model median versus 13.495 seconds with hipBLAS, with approximately 78 MB more peak allocated memory. Three samples and a 0.5% difference do not establish a robust default-policy improvement. The reported primary configuration retains hipBLAS. The new fused packing kernels are handwritten HIP; they are not claimed as CK library kernels.

## Reproduction and tests

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix LAYERNORM_TYPE=torch \
  python -m benchmarks.protenix.benchmark_triangle \
  --blas-backend hipblas --output outputs/benchmarks/packed-triangle-operator.json

HIP_VISIBLE_DEVICES=6 python benchmarks/protenix/profile_inference.py \
  --config benchmarks/protenix/configs/mi250-long.json \
  --backend fireprot --blas-backend hipblas
```

The operator benchmark uses seeded nonzero random parameters and records source hashes, per-call timings, peak memory, and continuous GPU process isolation. Model validation uses the real checkpoint. GPU tests cover both triangle directions, add and overwrite aliasing, partial tiles, nondefault streams, constant rows, and large means. Capability tests cover explicit masks, unsupported lengths, dtype, architecture, layout, autograd/training mode, and unavailable extensions.

The remaining largest measured stage is triangle attention. The next optimization should profile its SDPA, projection, and layout costs on this new execution mix and extend the packed operator only after testing additional real long-sequence buckets.
