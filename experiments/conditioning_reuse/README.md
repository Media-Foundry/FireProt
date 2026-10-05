# OpenFold3 conditioning reuse diagnostic

Status: CPU-checked diagnostic; a first N1536 full-model run is in progress.
No model validation result, caching implementation or speedup is claimed. See
[`docs/openfold3-conditioning-reuse.md`](../../docs/openfold3-conditioning-reuse.md)
for the source dependency audit and acceptance requirements.

`probe.py` leaves upstream return values and arithmetic unchanged. For each
sampling invocation it records conditioning calls, requested chunk sizes, actual
`_forward` calls including tuner attempts, input layouts and completion status.
For calls 0, 1, 99 and 199 it additionally compares full pair output bytes with
the first sampled output, records the noise level and CPU/device RNG invariance,
and checks pair bytes again after diffusion consumers finish.

Snapshots are copied to CPU. A sampled GPU pair output is retained until its
diffusion call exits, then released. This can affect memory peaks and later
chunk tuning. These runs are diagnostic and must never supply E2E performance
claims. Observed equality at four calls is not proof across all steps; pass
`sampled_calls=range(200)` for a later exhaustive D200 observation, with
substantial host-transfer overhead. Input layouts alone do not prove input
immutability. Failed consumer calls leave the post-consumer verdict absent.

All method overrides and retained snapshots are scoped and released on exceptions.
CPU tests detect time-dependent signed-zero changes, downstream in-place mutation,
RNG consumption, per-invocation reset and exceptional-exit restoration:

```bash
python -m pytest -q experiments/conditioning_reuse/test_probe.py
```

The tests use small stand-in modules. They validate diagnostic behavior, not
upstream OpenFold3 compatibility, GPU arithmetic or whole-model correctness.

## Isolated worker preparation

The original HIP softmax paired benchmark has completed successfully.
Copy the maintained worker/shim/kernel tree to a fresh experiment directory,
apply `worker-overlay.patch` with `git apply -p0`, and copy `probe.py` to
`python/fireprot_workers/conditioning_probe.py` in that copy. Use a new Rust
execution profile whose worker source identity covers all these files.
The transport patch includes `conditioning-audit.json` in accepted artifacts.
Use unchanged scientific requests and strict environment inventories. Do not
patch an active validation's worker tree, CLI, bridge or environment.

The first diagnostic uses `/tmp/fireprot-conditioning-probe-v1`, request
`conditioning-probe-v1-n1536`, HIP5/PCIb3/SMI7 and unchanged FP32/R3/D200/S1/
seed101. It runs concurrently with prepared-runtime N1536 numerical validation
on HIP7; neither run supplies a new E2E speedup claim. Its output and 11-array
native comparison are still pending.

Run first with the native backend and compare all 11 full model arrays against
the strict native reference. Only accepted diagnostic runs can inform the
subsequent cache trial. Runtime or numerical failure must remain in the evidence.
