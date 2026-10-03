# Transposed pair LayerNorm

## Previously uncovered layout

Transposing the two sequence axes of a contiguous `[N,N,128]` pair tensor gives
strides `[128,N*128,1]`. Existing FireProt LayerNorm supports contiguous long
inputs and a different short channel-first layout, so this layout falls back
to Torch. The new HIP candidate directly reads transposed rows and writes
contiguous normalized output, avoiding an intermediate contiguous input.

Actual Pairformer execution requires an additional integration change:
`PairformerBlock` configures both triangle-attention modules as starting-node
modules and explicitly makes a contiguous transpose before and after its
ending-node call. Therefore the unmodified model never exposes this strided
layout to the new LayerNorm dispatcher. The first model candidate completed
with **zero selections** and was rejected before proceeding to its baseline.
That run is not evidence of model acceleration. A subsequent diagnostic
confirmed contiguous C128 inputs throughout the observed model calls.

The kernel uses float4 input groups and a wave64 Welford reduction, including
empty-lane counts. It follows the arithmetic structure inspected in PyTorch's
[official LayerNorm source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu),
but acceptance is based on the installed Torch binary's numerical tests.
Four and eight waves per block were measured; the integration uses eight.

## Operator evidence

Base `6cf8e2e`, MI250 gfx90a GPU 7, GPUs 6 and 7 continuously monitored and
isolated, FP32, TF32 off, seed 1043, five warmups, twenty GPU-event samples,
non-default stream, warm cache, allocation included and compilation excluded.
This is an operator replay with random affine parameters; checkpoint, cycle,
diffusion step and sample settings do not apply.

| N | Torch before / after | HIP eight waves | Speedup vs after |
| --- | ---: | ---: | ---: |
| 1000 | 5.802 / 5.805 ms | 1.887 ms | 3.076x |
| 1536 | 20.154 / 20.161 ms | 8.536 ms | 2.362x |

Both wave counts have zero measured maximum absolute error against Torch on
normal random tensors and on constant/low-variance stress rows. Validation
uses unchanged `atol=rtol=2e-5`; output guards pass. Stress values include
1000, 0, -1000, `1 + arange(128)*1e-6` and `1000 + arange(128)*0.001`.
Whole-probe peak allocations fall from 1,544,001,536 to 1,024,001,024 bytes at
N1000 and from 3,642,754,048 to 2,415,920,128 bytes at N1536. These include
input and affine tensors and are not model peak-memory claims.

## Opt-in integration

`FIREPROT_ENABLE_TRANSPOSED_LAYERNORM=1` enables the candidate through the
existing LayerNorm dispatcher. The default is off; N1536 also requires the
existing long-bucket enablement. Python checks exact shape, strides, FP32,
gfx90a, inference mode, affine tensor properties, epsilon and alignment.
The C++ binding independently checks tensor and device constraints and uses
the current HIP stream. Unsupported calls and unavailable builds retain the
existing dispatch/fallback behavior. Existing extension source hashing covers
the new kernel and binding.

To expose the layout in the actual Pairformer, independent Protenix commit
`fb1f39a` adds optional transpose dispatch only in the `inplace_safe` FireProt
branch. `FIREPROT_ENABLE_PAIR_LAYOUT=1` together with the LayerNorm flag allows
exact long FP32 gfx90a inference tensors to use views instead of the two
contiguous copies. The second transpose restores the original contiguous
layout before pair transition. Training, autograd, unsupported shapes and
layouts, disabled flags or unavailable extensions retain contiguous copies.
GPU tests verify aliasing, original-stride restoration and exact residual
update equivalence. The standalone norm flag alone is insufficient to
accelerate the existing Pairformer.

The constructor optimization initially pinned only the preceding model source
tree and fell back on this commit. Its allowlist now also includes `fb1f39a`
after AST verification that only `PairformerBlock.forward` changed and a real
checkpoint test showing identical loaded parameters and Python/NumPy/Torch
CPU/GPU RNG states. It remains opt-in. Construction is disabled on both sides
of the layout model experiment. See the
[compatibility evidence](../provenance/protenix_construction_layout_08666dc.json).

Model numerical agreement and end-to-end improvement remain separate gates.
The operator result alone does not establish a model speedup.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_transposed_norm.py \
  --output outputs/benchmarks/transposed-norm.json
```

## Coupled layout model comparison

The corrected pair-layout candidate is top-level commit `08666dc` with
independent Protenix `fb1f39a`. Runs record base `ae9eca5` plus exact uncommitted
candidate file hashes; those runtime files were then committed unchanged.
Mini-default v0.5.0, FP32, TF32 off, hipBLAS, C4/S1, one sample, seed 101,
no MSA/templates, expandable allocator, one warmup and three measured
predictions. GPU 7 executes serially and GPU 6 is also monitored; every
isolation check passes. N1536 runs enabled then disabled, N1000 reverses order.
The two flags are switched together. Fast construction is disabled on both
sides. These are warm-model measurements, not fresh-worker single requests.

| N | Disabled median | Enabled median | Ratio | Allocated peak disabled / enabled |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 9.734 s | 9.710 s | 1.0025x | 10.011 / 9.100 GB |
| 1536 | 28.090 s | 27.719 s | 1.0134x | 22.721 / 20.631 GB |

N1000 timing ranges overlap: disabled 9.730–9.749 s, enabled 9.671–9.734 s.
This is not a demonstrated stable latency improvement. N1536 ranges are
28.051–28.105 s and 27.706–27.788 s. Each enabled process selects the new
LayerNorm 288 times across four predictions; both disabled processes select
it zero times. All six coordinate/confidence comparisons pass the unchanged
0.001 Angstrom aligned-RMSD and 0.001 confidence gates.

The complete worker (construction, preprocessing, warmup plus three measured
predictions and output writing) takes 85.371 → 85.548 s at N1000 and
173.293 → 171.726 s at N1536. Thus the observed worker ratio is 0.998x and
1.009x respectively. Do not substitute the isolated operator ratio for these
request-level figures or multiply it by earlier model speedups.

N1536 triangle-attention timing itself rises from 13.279 to 13.709 s because
the candidate changes the normalization arithmetic/path within that scope;
removing the two copies outside that scope yields the net model improvement.
The strongest shared result is approximately 9% lower peak allocation.
Both flags remain opt-in pending further N1536 memory-access tuning and
broader model coverage. [Model evidence](../provenance/protenix_pair_layout_model_ae9eca5.json)
retains the exact sources, script, dispatch counts, individual samples,
validation and isolation records.
