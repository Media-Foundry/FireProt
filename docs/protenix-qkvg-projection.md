# QKV and gate fusion probe

This private HIP experiment adds the attention gate projection and sigmoid to
the validated three-projection MFMA kernel. It shares input tiles across Q, K,
V and gate, writes CK-compatible QKV strides, and produces a contiguous gate.
The reference is current fused QKV plus hipBLAS gate projection and Torch
sigmoid. Complete attention also includes CK attention and output projection.
The reference computes its gate after attention, matching the existing model.

## Protocol

Base source `b74161cb162b61e8cb4e10e9337801d326fcc462`, MI250 gfx90a GPU 7,
FP32, TF32 disabled, hipBLAS, seed 1027, five warmups, twenty measured calls.
GPUs 6 and 7 pass continuous isolation checks. Random normal inputs and
normal(0,0.1) weights are used; the first three input rows are overwritten with
0, +100 and -100 for the numerical stress check. There is no model checkpoint,
recycle cycle, diffusion step or prediction sample in this operator replay.
Allocation is included and compilation excluded.

## Result: not accepted for integration

All five tiles pass preparation checks and output guard regions, but all fail
the complete-attention `atol=rtol=2e-5` numerical gate at both lengths. The
N1000 result has five violating elements out of 128,000,000; B512/N1536 has
three out of 100,663,296. The greatest absolute difference among the reported
violating elements is about 2.93e-5 and 2.97e-5 respectively. Preparation
maximum deltas are 2.68e-6 and 1.31e-6. Small intermediate errors therefore
need assessment at the complete operation boundary.

Rejected complete-attention candidates are not timed. The report returns
`completed_with_rejections`, records every failure, and proceeds to the next
configuration without changing the tolerance. There is no model integration
or additional end-to-end speedup claim.

Preparation medians from the final run:

| R / C / K / waves | B1000 / N1000 | B512 / N1536 |
| --- | ---: | ---: |
| Reference before | 5.672 ms | 4.425 ms |
| 128 / 64 / 16 / 16 | 5.981 ms | 6.066 ms |
| 128 / 64 / 16 / 8 | 5.861 ms | 5.188 ms |
| 64 / 64 / 16 / 4 | 6.632 ms | 5.272 ms |
| 64 / 128 / 16 / 8 | 8.705 ms | 6.794 ms |
| 64 / 64 / 16 / 8 | 5.176 ms | 4.194 ms |
| Reference after | 5.656 ms | 4.429 ms |

The smaller eight-wave tile improves preparation timing, but this does not
override the failed complete-operation numerical gate. The fusion has no
default or experimental model dispatch entry.

## Measurement housekeeping

The initial run stopped on the first numerical rejection. A subsequent scan
recorded all failures, but failed-assertion traceback cycles retained tensors
and distorted later peak-memory readings. The final script collects cycles
between candidates and shapes. Before/after complete-attention control peaks
then agree at 5,170,659,328 bytes (N1000) and 4,136,304,640 bytes (B512/N1536).
These include probe references and are not model-memory measurements. All
three reports are retained to make the correction explicit.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python \
  benchmarks/protenix/benchmark_qkvg_projection.py \
  --output outputs/benchmarks/qkvg-probe.json
```

[Evidence](../provenance/protenix_qkvg_probe_b74161c.json) includes source and
binary hashes, compiler/runtime identities, per-call samples, numerical
rejections, memory and isolation reports.
