# Exact duplicate-template block reuse

Status: full N1536 model and guard checks passed; paired E2E measurement pending.
This is an experimental execution optimization, not a scientific setting change.

## Contract

`template_reuse.py` wraps the individual blocks in OpenFold3's template stack.
For the measured `[1,4,1536,1536,64]` FP32 input, it compares every template
slot and corresponding mask as `int32` bits. Equality includes signed-zero and
NaN payload distinctions. Eligible calls require inference mode, strict
deterministic execution, native attention flags and in-place operation.
Each native block processes its first slot, then copies those output bytes
to the other three slots. The original input storage remains the returned
storage. Distinct or unsupported calls invoke the unchanged native block.

The stack's final normalization and the embedder's four-slot sum/division are
unchanged. No cache spans blocks, recycling passes, or requests. The source pin
is checked before installing hooks, and hooks are restored on context exit.
The optimization does not infer equality from a no-template configuration.

## Qualification

- OpenBind-0, N1536, strict FP32 with uninitialized-memory filling enabled,
  R3/four trunk passes, D200/S1, seed101, HIP7/PCI `0000:93:00.0`.
- Upstream OpenFold3 commit `7de748b7bc93adb5af0a4032d5e9f208b6e1325f`;
  template module SHA256
  `f75d982e7c94462a0276b22c860c69676baab003556d2c665a5ab9db7bbb9d63`.
- All 11 arrays match the qualified triangle-memory reference bitwise, and
  full CIF bytes match. The scope selected nine calls and reused 27 slots.
- Inclusive template events: 49.415 seconds before, 14.192 seconds with reuse.
  Instrumentation and changes in other stage timings prevent an E2E claim.
- GPU guard screen checks successful reuse and aliasing, signed-zero and mask
  differences, training/non-inplace fallback, and method restoration.

Run the guard screen with the pinned Torch environment and no competing GPU job:

```bash
HIP_VISIBLE_DEVICES=7 /path/to/openfold3/python \
  experiments/triangle_memory/template_reuse_screen.py /fresh/screen-output
```

The full-model frozen deployment is on DiamondHill at
`/media/IntelSSD/fireprot/outputs/experiments/openfold3-template-reuse-20261007`.
It retains the source, request/profile, Rust invocation specification, driver,
all frozen-file hashes, diagnostic outputs and terminal receipt. Use a fresh
request/output directory for a replay; never overwrite the qualified run.
The tensor scope is entered around the original `forward_with_backend` call,
alongside the accepted pair/atom cache, D16, workspace and triangle-copy scopes.
Rust owns model invocation, GPU isolation and output acceptance.

## Paired timing

The fresh timing deployment is
`/media/IntelSSD/fireprot/outputs/experiments/openfold3-template-pairs-20261007`.
It uses one shared frozen worker, no model profiler, and only changes
`FIREPROT_TEMPLATE_REUSE=0/1` plus the corresponding profile digest. Rust
`benchmark-pairs` executes AB/BA/AB with fresh processes and unflushed caches.
All six runs must pass the original strict array/CIF gates. The reference arm
already includes the accepted triangle-memory combination; this measures an
incremental improvement, not a direct original-Torch ratio.

Do not overlap the live timing job with another GPU experiment or heavy remote
build. The maintained profiles do not enable this experiment.
