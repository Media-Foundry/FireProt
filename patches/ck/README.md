# CK attention experiment

`attention-cache-q-fp32.patch` is an experimental derivative of the AMD
Composable Kernel MIT-licensed header. See [LICENSE](LICENSE).

The benchmark applies it to a private copy under `runtime/ck_attention_probes`.
It never edits installed ROCm headers. The original header must have SHA256
`03661397734c09cc5abcd0c2587623a145b6bf95cfae5cd5ece3203ba0406a43`.
The measured installation reports CK 1.2.0, commit
`8d1ae90eff7d022f26019ec55b2ec6a7674b3112`.

The patch retains an unnormalized output numerator until the final key block
and caches the Q transfer registers for the single-stage K=32 path. It is
only built by the private FP32, head-dimension-32, unmasked, inference probe;
it is not a general CK patch or a production operator registration. The C ABI
accepts only the measured batch=sequence=1000 bucket. Masked rows, arbitrary
dimensions, other dtypes, training, and model-level numerical equivalence
have not been validated for this candidate.

```bash
HIP_VISIBLE_DEVICES=7 PYTHONPATH=.:Protenix python -m benchmarks.protenix.benchmark_attention \
  --include-ck --ck-set tuned --output outputs/benchmarks/attention-tuned.json
```

The benchmark includes bias packing and output allocation, verifies each
candidate against SDPA, records source/header/library digests, and checks
GPU process isolation continuously. `--ck-set all` adds the original MFMA
and tile matrix. CK remains outside model dispatch until repeated end-to-end
performance and model numerical gates justify a replacement.
