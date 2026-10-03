# Fused QKV projection experiment

## Scope and implementation

This experiment targets QKV preparation in long Protenix triangle attention.
The baseline performs three FP32 hipBLAS projections and a separate Q scaling
operation. The handwritten gfx90a HIP candidate shares input tiles across three
MFMA accumulations, applies Q scaling, and writes the existing CK-compatible
head-last storage directly. It is a benchmark candidate, not a production path.

The selected tile is R128/C64/K16 with 16 waves. Logical Q/K/V shapes are
`[B,4,N,32]`, each with strides `[N*128,32,128,1]`. N1000 uses B1000;
N1536 uses B512, matching one actual outer chunk. This is not the entire
1536-residue triangle-attention stage.

The C ABI is intentionally a private probe: it accepts only the two measured
row counts and non-null pointers. Production tensor, device, architecture,
autograd, aliasing, and parameter guards still need to be implemented before
integration. The kernel computes FP32 MFMA and preserves the requested model
cycle count by making no model execution changes.

## Measurements

Base source: `b1fd5f5`. MI250 gfx90a physical GPU 7, with GPUs 6 and 7
continuously monitored; all four reports pass isolation. FP32, TF32 disabled,
hipBLAS process policy, seed 1019, normal random inputs, normal(0,0.1) weights,
five warmups and twenty synchronized GPU-event measurements per implementation.
Compilation is excluded; allocation and Q scaling are included. No model
checkpoint, recycle cycles, diffusion steps, or prediction samples are involved.

### QKV preparation

Values below come from the wide tile sweep; controls bracket the candidates.

| B / N | Native before / after (ms) | R128/C64/K16/W16 (ms) | Candidate p95 (ms) |
| --- | ---: | ---: | ---: |
| 1000 / 1000 | 4.475 / 4.473 | 3.652 | 3.897 |
| 512 / 1536 | 3.532 / 3.534 | 2.838 | 2.861 |

All MFMA outputs pass `atol=rtol=2e-5`, with maximum absolute delta
`5.24521e-6`. Output guard regions pass. Peak allocated memory matches the
baseline in each replay. The smaller R64/C64/K16/W4 candidate measures
3.762 ms and 3.026 ms respectively; its N1000 p95 is steadier.

A single concatenated GEMM without layout conversion measures 4.333 ms and
3.410 ms in the initial experiment, but produces CK-incompatible row strides.
Adding a fused HIP split increases latency to 5.920 ms and 4.675 ms and adds
an intermediate allocation. This route is not selected. Cached weight
concatenation is outside its timing, which favors that rejected route.

### Complete attention chunk

This includes QKV preparation, the current CK attention kernel and bias packing,
and native output gating/projection. It excludes the enclosing triangle input
LayerNorm and bias projection. Both sides use the same CK implementation.

| B / N | Native preparation before / after (ms) | Fused preparation (ms) | Speedup versus after | Maximum output delta |
| --- | ---: | ---: | ---: | ---: |
| 1000 / 1000 | 28.838 / 28.773 | 28.122 | 1.023x | 1.25170e-6 |
| 512 / 1536 | 29.627 / 29.637 | 28.946 | 1.024x | 1.37091e-6 |

Peak allocated memory is unchanged at 4,658,078,720 bytes and 3,695,706,112
bytes respectively in this replay. These are probe allocations, not model
memory measurements. All complete outputs pass the same tolerance.

## Reproduce

Run serially on an idle authorized GPU:

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_qkv_projection.py --hip-mfma --wide-sweep \
  --output outputs/benchmarks/qkv-mfma-wide.json

HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_qkv_projection.py --hip-mfma --full-attention \
  --output outputs/benchmarks/qkv-full-attention.json
```

[Evidence](../provenance/protenix_qkv_probe_b1fd5f5.json) retains source and
binary hashes, compiler/runtime identities, per-call samples, numerical checks,
and device monitoring records for all four experiments.

## Integration gate

Next, add an inference-only self-attention preparation hook with exact shape,
layout, weight, dtype, device, architecture, and scaling checks and native
fallback. Validate a non-default stream and unsupported inputs, then run
same-source Mini-default C4/S1 model A/B comparisons at both lengths with
dispatch counts and the locked coordinate/confidence gates. No new end-to-end
speedup is established by this experiment, and runtime defaults are unchanged.

## Opt-in integration candidate

Candidate `6ffcc776d282a1370ef60b40db13b2fc61c29ad5` adds
`FIREPROT_ENABLE_QKV_PROJECTION=1`. Set it before installing the Protenix
runtime hooks. It remains disabled by default pending model validation.
The N1536 shape also requires `FIREPROT_ENABLE_LONG1536=1`.

The hook requires identical data pointers, shapes, strides, devices and dtypes
for query and key/value input, exact measured
input shapes, contiguous aligned FP32 tensors on gfx90a, inference mode, four
heads of width 32, and Q scaling enabled. Autocast, biased projections,
OpenfoldLinear precision overrides, custom linear subclasses, and local
projection hooks use native preparation. The C++ binding independently checks
shape, dtype, alignment, device, architecture, and disabled gradient recording,
allocates distinct output storage, and launches on the input device's current
HIP stream. Output Q/K/V share one allocation but occupy disjoint regions.

Runtime manifests expose `hooks.qkv_projection` with `enabled` and counters
`n1000`, `n1536`, and `fallback`. Unsupported attention calls retain native
preparation. Extension build failure also falls back.

Local regression: 133 passed, 45 HIP tests skipped. GPU 7 validation:
12 passed across QKV and existing dual-projection tests, with passing isolation
monitors on GPUs 6 and 7. QKV tests cover both exact shapes, non-default streams,
repeated calls, zero and positive/negative large input rows, input preservation,
output strides, unsupported weights, scaling, training, autocast, precision,
local hooks, bias, and unavailable-extension fallback. Full-model timing and
prediction agreement are separate gates.

The first model attempt at `6ffcc77` selected zero fused QKV calls: upstream
chunking independently creates equivalent query and key/value views. The
initial Python object-identity guard rejected all 2,592 calls in N1536. That
attempt is not a fused-path performance result. The view regression test fails
on the initial implementation; the corrected guard accepts identical tensor
regions while rejecting distinct cross-attention storage.
