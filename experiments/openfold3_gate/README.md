# OpenFold3 exact FP32 triangle projection gate fusion

Experimental gfx90a kernel for the qualified N1536 OpenBind-0 worker. This is
not enabled by maintained inference profiles. Model recycles, diffusion steps,
samples and projection GEMMs are unchanged.

`gate.cpp` includes two kernels:

- `fireprot_gate` fuses sigmoid, projection multiplication and mask multiplication
  into one contiguous-output kernel.
- `fireprot_gate_pack` additionally writes channel-first physical storage, using
  a 32-row by 128-channel shared-memory transpose with a padded pitch of 132.

Both retain the native sigmoid expression and multiplication order. The packed
result is exposed as a view with the original logical `[1, rows, columns, 128]`
shape. `runtime.py` provides the ctypes tensor boundary. `build.py` uses the Rust
HIP build service. `model_scope.py` is deployed into the frozen experimental
worker's `gate_experiment` package beside its qualified `triangle_memory`
package; it requires that deployment and its pinned library. It is not a
standalone importable adapter. It retains the existing exact-shape capability
predicate and native fallback.

## Operator qualification

On HIP device7, strict FP32, seed101:

- Nine numerical cases pass bitwise, including small tail dimensions, real
  `[1,256,1536,128]` and `[1,1536,256,128]` shapes, noncontiguous masks, signed
  zeros, subnormals, infinities and NaNs.
- Both trained incoming/outgoing triangle multiplication captures pass all
  forty complete-module calls with the packed candidate and baseline.
- Baseline includes canonical-output HIP normalization and qualified triangle
  copies. Two warmups and eight balanced samples are taken per arm/capture,
  with unflushed caches and correctness checks outside the timed interval.

| Full triangle multiplication | Baseline | Packed gate candidate |
| --- | ---: | ---: |
| Incoming | 126.356 ms | 116.203 ms |
| Outgoing | 133.070 ms | 120.994 ms |

These are operator measurements, not E2E speedups. Full-model strict N1536 R3/D200/S1, seed101, HIP7 qualification passes
all 11 arrays and complete CIF bytes against the canonical-normalization anchor.
There are 430 accepted triangle modules and 5,160 fused calls. One observation
measures 346.785 s complete invocation and 252.692 s model forward; a paired E2E
measurement is pending. See [qualification](../../provenance/openfold3_gate_pack_qualification.json). The simpler contiguous gate also passed but
had smaller operator gains (126.251 → 120.539 ms; 133.044 → 127.438 ms in its
separate comparison).

The qualified packed library SHA-256 is
`b7a980bf469a0f13581985033f904541cae5783eab01d4579054588e75c3614c`.
Raw builds, numerical screens and operator replays are in DiamondHill's
`outputs/experiments/openfold3-gate-pack-20261007` and the
`openfold3-qk-layout-gate-followups-20261007.tar.gz` archive (SHA-256
`9aed59a6554343c2244946dffb45a5359536d2ffdbf2b56376725c4c1b858903`).
The later model sources and outputs are in
`openfold3-gate-model-20261007.tar.gz`, SHA-256
`097f3f18e8e2aa83efead3f15fbf3538b7b7cb61805bc2676f28ce4862cde2f8`.
