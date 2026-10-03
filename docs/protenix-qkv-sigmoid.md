# QKV sigmoid epilogue experiment

## Motivation and scope

The previous four-projection QKVG experiment failed complete-attention
numerical checks. This experiment keeps the native hipBLAS gate projection
and fuses only sigmoid into the handwritten FP32 QKV MFMA kernel. The reference
uses the promoted QKV kernel, CK attention and native output gating/projection.
No production dispatch changes are made.

Two private implementations were measured:

- A scalar sigmoid epilogue writes Q/K/V/gate to one allocation.
- A vector epilogue applies sigmoid in place to independently owned gate logits
  after QKV accumulation. Only channel tile zero processes each row's gate;
  float4 loads/stores cover all 128 channels without overlapping writes.

Both retain the native gate bias and preserve QKV arithmetic. The gate logits
are a newly allocated projection result, not an input or model parameter.
The C ABIs are benchmark-only and validate pointers and the two row counts;
production tensor/autograd/architecture capability checks are not implemented.

## Protocol

Base `f21e6f7`; exact script/kernel hashes are in the reports. GPU 7 executes
serially, with continuous isolation checks on GPUs 6 and 7. MI250 gfx90a,
FP32, TF32 off, hipBLAS, seed 1027, five warmups and twenty GPU-event samples.
Normal inputs and normal(0,0.1) weights; first rows are 0, +100 and -100.
Non-default stream, output guard regions, exact CK-compatible QKV strides,
and unchanged `atol=rtol=2e-5` checks are used. All device monitors pass.
Compilation is excluded; allocations and native gate projection are included.
Cycles, diffusion steps, checkpoints and prediction samples do not apply to
this operator replay.

## Result: not integrated

All preparations and complete attention outputs have zero measured maximum
absolute difference from their controls. This resolves the prior QKVG
numerical rejection on these inputs but gives no performance improvement.

Final vector result (R128/C64/K16/W16):

| B / N | Preparation baseline before / after | Candidate | Complete baseline before / after | Candidate |
| --- | ---: | ---: | ---: | ---: |
| 1000 / 1000 | 5.617 / 5.650 ms | 5.752 ms | 28.069 / 28.016 ms | 28.121 ms |
| 512 / 1536 | 4.414 / 4.379 ms | 4.515 ms | 28.947 / 28.922 ms | 29.050 ms |

The R64/C64/K16/W8 candidate is also slower. The scalar packed variant takes
28.238 and 29.427 ms for complete attention. None is selected for model tests,
and there is no new model or end-to-end speedup claim.

Initial candidate functions retained local Q/K/V references through output
projection. The final vector benchmark explicitly releases those references
after CK attention, matching the reference lifetime. This reduces measured
memory but does not reverse the latency result. Peak allocated memory still
increases from 5,170,659,328 to 5,682,659,328 bytes at N1000 and from
4,136,304,640 to 4,538,957,824 bytes at B512/N1536: the additional amount equals
one FP32 gate tensor computed before attention. These are whole-probe peaks,
including reference tensors, not model peak-memory estimates. Gate views in
the packed scalar implementation additionally keep their shared QKV allocation
alive. All three reports are retained, including the initial lifetime issue.

The experiment shows that eliminating a sigmoid launch alone does not justify
moving gating ahead of CK attention. Further fusion should be evaluated for
large-tensor traffic and lifetime effects, with complete-operation timing.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_qkv_sigmoid.py \
  --output outputs/benchmarks/qkv-sigmoid.json

HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_qkv_sigmoid_vector.py \
  --output outputs/benchmarks/qkv-sigmoid-vector.json
```

[Evidence](../provenance/protenix_qkv_sigmoid_f21e6f7.json) retains compiler,
binary and source hashes, individual timing samples, strides, maximum errors,
peak allocations and continuous isolation records for all three runs.
