# Protenix integration patches

These patches reproduce the FireProt integration commits in the independent `Protenix/` checkout. They are exported from branch `fireprot/protenix-amd-acceleration` and apply after upstream Protenix v1.1.0 commit `d3b4db6a121dd4584edd85e93744239325a2b72e`.

```bash
cd Protenix
git checkout d3b4db6a121dd4584edd85e93744239325a2b72e
git am ../patches/protenix/0001-*.patch ../patches/protenix/0002-*.patch
```

| Patch | SHA-256 | Purpose |
| --- | --- | --- |
| `0001-feat-add-FireProt-triangle-multiplication-dispatch.patch` | `78b09c161fe97f1dba9aa80153e463048c67f0f219d56e961bcb1a6007d05703` | Add the capability-gated triangle multiplication dispatch choice |
| `0002-feat-add-FireProt-triangle-attention-dispatch.patch` | `cd271e7ce9046bebd99d6214d9283cee521365a3c0a4e846657eb28c3d00f2ad` | Add the capability-gated triangle attention dispatch choice |

The patches do not change upstream defaults. Native Protenix continues to select its existing kernels unless `fireprot` is requested explicitly.
