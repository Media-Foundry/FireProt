# Fused QKV projection experiment

## Scope and implementation

This work targets QKV preparation in long Protenix triangle attention.
The baseline performs three FP32 hipBLAS projections and a separate Q scaling
operation. The handwritten gfx90a HIP candidate shares input tiles across three
MFMA accumulations, applies Q scaling, and writes the existing CK-compatible
head-last storage directly. The integrated path is now enabled by default for
the validated shapes after the model comparison below. The original standalone
C ABI remains a benchmark probe.

The selected tile is R128/C64/K16 with 16 waves. Logical Q/K/V shapes are
`[B,4,N,32]`, each with strides `[N*128,32,128,1]`. N1000 uses B1000;
N1536 uses B512, matching one actual outer chunk. This is not the entire
1536-residue triangle-attention stage.

The C ABI is intentionally a private probe: it accepts only the two measured
row counts and non-null pointers. Production tensor, device, architecture,
autograd, aliasing, and parameter guards belong to the integrated binding and
dispatch layer described below. The kernel computes FP32 MFMA and preserves the requested model
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

The original probe required an inference-only self-attention preparation hook
with exact shape, layout, weight, dtype, device, architecture, and scaling checks
and native fallback. Non-default streams and unsupported inputs must be checked
before same-source Mini-default C4/S1 model A/B comparisons at both lengths,
including dispatch counts and the locked coordinate/confidence gates. The
standalone experiment established no end-to-end speedup; the integration and
model evidence below complete these gates for the two measured shapes.

## Opt-in integration candidate

Candidate `6ffcc776d282a1370ef60b40db13b2fc61c29ad5` adds
`FIREPROT_ENABLE_QKV_PROJECTION=1`. Set it before installing the Protenix
runtime hooks. It was disabled by default while awaiting model validation;
the subsequent promotion defaults to enabled. Set the flag to `0` to disable it.
The N1536 shape also requires `FIREPROT_ENABLE_LONG1536=1`.

The hook requires identical data pointers, shapes, strides, devices and dtypes
for query and key/value input, exact measured
input shapes, contiguous aligned FP32 tensors on gfx90a, inference mode, four
heads of width 32, and Q scaling enabled. Non-FP32 autocast, biased projections,
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

The subsequent `0aa10b8` attempt selected 144 calls but rejected the other
2,448 because the FP32 trunk runs inside a FP32 autocast context. A single-call
diagnostic confirmed the remaining shape/layout/weight checks match. The
corrected capability check permits FP32 autocast while retaining BF16/FP16
fallback. The regression now runs within the model's FP32 autocast context;
it fails before this correction. These incomplete-selection attempts are not
used to claim a fused-QKV model speedup.

## Same-source model comparison

Candidate `c4afab80c2f6434b5886dcb0b1ff74b0591bd84c`, Mini-default v0.5.0,
FP32 with TF32 disabled, four cycles, one diffusion step, one sample, seed 101,
no MSA/template search, hipBLAS, expandable allocator segments, one warmup and
five measured predictions per side. GPU 7 executes inference and GPU 6 remains
idle; continuous monitors pass for both. N1536 runs enabled then disabled;
N1000 runs disabled then enabled. Only the QKV flag changes between each pair.

| Length | Warm model disabled / enabled (s) | Speedup | Model p95 disabled / enabled (s) | Attention disabled / enabled (s) |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 9.919 / 9.745 | 1.018x | 9.944 / 9.810 | 4.374 / 4.194 |
| 1536 | 28.519 / 28.127 | 1.014x | 28.600 / 28.406 | 13.628 / 13.280 |

Peak allocated memory is unchanged: 10,011,042,304 bytes at N1000 and
22,721,185,792 bytes at N1536. The enabled runs select 864 and 2,592 QKV calls
respectively across warmup plus measurements, with zero fallbacks. All ten
prediction pairs pass the locked `1e-3` aligned-coordinate-RMSD and confidence
gates. Maximum aligned RMSD is 0.00010215 A (N1000) and 0.00010146 A (N1536);
maximum confidence delta is 2.67029e-5 and 1.52588e-5 respectively.

Warm model timing excludes loading, preprocessing, serialization and compilation.
Complete worker time is 112.684 to 111.176 s (N1000) and 245.183 to 243.772 s
(N1536); each worker includes one warmup and five predictions, not one cold
request. These are incremental comparisons against the existing FireProt
baseline, not new original-Torch comparisons. N1536 model sample ranges overlap;
the reported ratio is the five-run median, while all enabled attention-stage
samples are below all disabled attention-stage samples.

`FIREPROT_ENABLE_QKV_PROJECTION=0` restores native QKV preparation. N1536
continues to require its existing length opt-in. The promotion changes Python
defaults only; the HIP source and compiled binary remain those measured here.
See [model evidence](../provenance/protenix_qkv_model_c4afab8.json) and
[integration regressions](../provenance/protenix_qkv_integration_validation.json).
