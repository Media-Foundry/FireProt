# Protenix integration patches

These patches reproduce the FireProt integration commits in the independent `Protenix/` checkout. They are exported from branch `fireprot/protenix-amd-acceleration` and apply after upstream Protenix v1.1.0 commit `d3b4db6a121dd4584edd85e93744239325a2b72e`.

```bash
cd Protenix
git checkout d3b4db6a121dd4584edd85e93744239325a2b72e
git am ../patches/protenix/0001-*.patch \
  ../patches/protenix/0002-*.patch \
  ../patches/protenix/0003-*.patch \
  ../patches/protenix/0004-*.patch \
  ../patches/protenix/0005-*.patch
```

| Patch | SHA-256 | Purpose |
| --- | --- | --- |
| `0001-feat-add-FireProt-triangle-multiplication-dispatch.patch` | `78b09c161fe97f1dba9aa80153e463048c67f0f219d56e961bcb1a6007d05703` | Add the capability-gated triangle multiplication dispatch choice |
| `0002-feat-add-FireProt-triangle-attention-dispatch.patch` | `cd271e7ce9046bebd99d6214d9283cee521365a3c0a4e846657eb28c3d00f2ad` | Add the capability-gated triangle attention dispatch choice |
| `0003-perf-skip-unused-triangle-std-in-fp32.patch` | `83bfa9a201085e4090366978e06bd7ad2c7659f294644bcdbab795e81164c312` | Skip triangle-projection standard deviations whose values are used only by the FP16 overflow guard |
| `0004-feat-expose-triangle-inference-fusion-callbacks.patch` | `0fdbd483d691fbe5a978347ab36d437e768801d1416e2cb2122af37f2880ffb7` | Expose optional projection and output-gate fusion callbacks in the chunked inference path while preserving the native default |
| `0005-perf-omit-implicit-all-ones-triangle-mask.patch` | `0577b14c1157b7b8ca295142873b7921724f32ea435d29beb78b56315f95eb05` | Let the FireProt attention path omit an implicit all-ones mask and its zero additive bias |

The dispatch patches do not change upstream kernel defaults. Native Protenix continues to select its existing kernels unless `fireprot` is requested explicitly. The FP32 reduction patch removes unused work without changing the requested dtype, operation order of retained tensors, or output contract.
