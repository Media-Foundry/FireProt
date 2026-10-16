# ESMC bounded pinned checkpoint loading

The standalone ESMC 6B exact profile now enables CPU BF16 conversion directly
into two reusable pinned block buffers, followed by asynchronous HIP transfers.
The original checkpoints and complete-file SHA-256 preflight remain unchanged.
This change applies to standalone ESMC, not the ESMFold2 folding adapter.

## Implementation and correctness

The checkpoint contains 1,042 ESMC FP32 tensors, totaling 25,381,120,000 bytes.
The existing loader transfers these FP32 tensors before casting on the GPU.
The new loader casts each tensor directly into a pinned BF16 buffer and transfers
12,690,560,000 bytes of parameter payload. It retains the upstream
`published_to_native` mapping and strict state assignment.

Two staging slots alternate across blocks. A HIP event must complete before a
slot is overwritten. The loader synchronizes before releasing its staging
buffers or closing the source files, including its exception path. Only two
live block buffers are retained; the host allocator may cache released pages.
No converted checkpoint is written or reused between invocations.

All **802 native parameter tensors / 6,345,280,000 parameters** match the current
GPU-loaded model byte for byte. Native B4 tokens and embeddings also match.
The final E2E collector independently checks all artifacts and compares every
array in **48 archives from 12 fresh-worker invocations**, across both native
and current-FireProt reference groups. Every comparison is exact.

## Loader diagnostics

These are single fresh-process observations, not E2E ratios. The timer covers
`load_model()` after Torch/CUDA initialization, including lazy model imports
inside that function. File caches were not flushed. Each trial checks its output.

| Loading path | Load time | Peak GPU allocation |
| --- | ---: | ---: |
| Existing FP32 H2D, GPU BF16 conversion | 5.958 s | 12.890 GB |
| CPU BF16 conversion, synchronous H2D | 6.045 s | 12.906 GB |
| CPU BF16 conversion, additional pinned copy, async H2D | 5.708 s | 12.906 GB |
| Direct conversion into reusable pinned slots, async H2D | **4.347 s** | **12.906 GB** |

GPU sizes are decimal GB. Peak process RSS increases from 26,101,744 KiB in the
GPU-loading pilot to 26,718,644 KiB in the selected pilot. These RSS observations
include checkpoint mappings and allocator caches, not just staging buffers.

## Complete invocation results

Both comparisons use N1000 x4, identical B4 grouping, seed101, native BF16,
HIP6 / PCI `0000:8e:00.0`, unflushed caches, and three AB/BA/AB pairs. Rust includes
full resource verification, model startup, forward execution and all requested
exports. The four-thread full-file hash setting is shared by both arms.

| Reference | Baseline median | Candidate median | Speedup | Individual paired ratios |
| --- | ---: | ---: | ---: | --- |
| Previous exact FireProt, GPU loading | 20.893729 s | 18.734573 s | **1.115250x** | 1.1156x, 1.1152x, 1.1234x |
| Original native implementation | 24.343565 s | 17.839632 s | **1.364578x** | 1.3352x, 1.3807x, 1.3420x |

All six pairs improve and pass exact-array acceptance. These are two separately
scheduled comparisons, so their candidate medians differ. Compute each ratio
from its own matched measurements; do not multiply this incremental result by
an older speedup. The direct-native result supersedes the earlier 1.195x ESMC
B4 E2E result. It does not change singleton, folding or other-model claims.

Model-forward medians are 1.331 -> 1.370 s in the incremental comparison and
1.483 -> 1.320 s in the direct-native comparison. The loader adds no new attention
or GEMM optimization. The prior exact HIP operators remain enabled in FireProt;
the experimental fixed-GEMM and graph routes are not included.

Evidence: [loading qualification and E2E](../provenance/esmc_loading_20261016.json).

## Entry point and fallback

The existing request remains valid:
`configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json`.
Its profile, `configs/diamondhill/plm-esmc-fold2-6b-exact.json`, now sets
`FIREPROT_ESMC_CPU_LOADING=1` and points to the measured frozen deployment:

```text
/media/IntelSSD/fireprot/outputs/experiments/esmc-load-20261016/validation/qualified-loading
```

On DiamondHill:

```bash
cd /media/IntelSSD/fireprot
cp configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json /tmp/esmc-request.json
# Set a new run_id and an unused output_dir in /tmp/esmc-request.json.
outputs/experiments/esmc-load-20261016/validation/qualified-loading/target/release/fireprot \
  esm /tmp/esmc-request.json configs/diamondhill/plm-esmc-fold2-6b-exact.json
```

For the old loader, set `FIREPROT_ESMC_CPU_LOADING=0` in a copied profile. Missing
the flag also retains the GPU loader; values other than `0` or `1` are rejected.
The historical frozen profile is retained as
`configs/diamondhill/plm-esmc-fold2-6b-gpu-loading.json`.

Maintained code is in `python/fireprot_workers/plm_loading.py` and
`python/fireprot_workers/esmc_cpu_loading.py`. Their bytes match the measured
deployment. Rust remains responsible for requests, asset verification, device
supervision and acceptance. Reproduction scripts and raw reports are retained
under `experiments/plm` and the remote `outputs/experiments/esmc-load-20261016`.
