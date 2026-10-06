# Exact triangle memory experiments

These isolated HIP experiments target OpenFold3 / OpenBind-0 on gfx90a.
They are not maintained model dispatch and do not establish an E2E speedup.

`copy.cpp` transposes FP32 storage as uint32 bits. `copy_runtime.py` qualifies
measured N1536/C128/chunk256 layouts and falls back before launch otherwise.
`replay.py` verifies the upstream file hash, extracts its Apache-2.0 method,
and substitutes only copies and explicit native operand packing. GEMM shapes,
LayerNorm, sigmoid, residual order, and chunking remain native.

The first trained incoming/outgoing replay passed all bits and measured
238.182 → 160.799 ms (1.481x) and 239.371 → 167.584 ms (1.428x).
See `provenance/openfold3_triangle_copy_operator_v1.json`. Full-model acceptance
and paired E2E are separate gates.

`attention_copy.cpp` independently packs head-last keys into the contiguous
layout consumed by the existing QK kernel. Its screen and replay are separate
from multiplication. The attention profile identifies QK+softmax and PV as
larger remaining costs; a packing win alone cannot resolve them.

Run in the pinned DiamondHill OpenFold3 interpreter, HIP_VISIBLE_DEVICES=7,
PYTHONDONTWRITEBYTECODE=1, after confirming no competing GPU jobs. Rust owns
native builds through `_native_hip.load`, full model supervision, and public
output acceptance. Standalone scripts provide experimental tensor replay only.

```sh
python build.py /tmp/new-triangle-build
python screen.py /tmp/new-triangle-build/build.json /tmp/copy-screen.json
python replay.py /tmp/new-triangle-build/build.json CAPTURE_DIRECTORY /tmp/replay.json
python attention_screen.py /tmp/new-key-build
python attention_replay.py /tmp/new-key-build/build.json ATTENTION_CAPTURES CONTROLS /tmp/key-replay.json
```

Screens keep strict deterministic FP32 and uninitialized-memory filling.
Synthetic tests include NaN/signed-zero bits, offset destination canaries,
nondefault streams and native fallback. Trained replay uses balanced order
and excludes the first call per arm. Every promoted change must additionally
match all 11 arrays and full CIF bytes under N1536 R3/D200/S1 seed101, retain
N1000 fallback, and pass isolated paired complete-request measurement.
