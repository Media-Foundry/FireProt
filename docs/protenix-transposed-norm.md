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

The constructor optimization currently pins the preceding model source tree;
on the new Protenix commit it falls back to native construction. Construction
is disabled on both sides of this layout experiment.

Model numerical agreement and end-to-end improvement remain separate gates.
The operator result alone does not establish a model speedup.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_transposed_norm.py \
  --output outputs/benchmarks/transposed-norm.json
```
