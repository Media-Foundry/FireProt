# Steady diffusion-call diagnostic

Status: prepared and queued; no diagnostic result or optimization claim yet.
Historical post-normalization profiling recorded about44.35 s in the complete
sampling module, but its operator trace covered the trunk. This diagnostic
collects the missing steady denoiser detail on the qualified isolated-data
baseline, independently of unfinished QKV speedup claims.

`probe.py` wraps `model.sample_diffusion.diffusion_module.forward` only within an
explicit context. The upstream source digest must match the audited diffusion
module. All200 denoiser calls retain the qualified baseline inputs and arithmetic. CUDA/HIP
events measure each call; only call5, after four normal calls, receives Torch
CPU/GPU profiling with input shapes. Immediate child modules receive named ranges
and shape/stride/dtype metadata for that call. No tensor values are copied out
for metadata. Instance methods are restored on normal exit or failure.

Outputs are `diffusion-profile.json` and `diffusion-call5.chrome.json`. The
profile includes operator self/inclusive CPU/device statistics, seven immediate
child ranges and metadata, per-call events and the trace digest. The whole-model
run must still match all11 arrays and full CIF bytes; unchanged runner settings,
normalization/gate counts and200 denoiser calls are checked independently.

The trace excludes outer random augmentation, noise generation and scheduler
arithmetic. Its timings contain instrumentation overhead, nested times overlap,
and shapes alone do not prove CU saturation or a memory/computation bound.
Neither this run's invocation time nor its model time is an E2E speedup result.
It does not enable graph capture, lower precision or change diffusion steps.

DiamondHill coordinator:
`outputs/experiments/openfold3-diffusion-probe-queue-20261010/run_after_unroll8.py`.
It waits for the existing unroll8 replay to finish successfully, verifies the
selected physical GPU is idle, prepares a new frozen diagnostic root, and uses
Rust `measure-inference` for environment checks, execution and acceptance.
Physical scope remains HIP7/PCI0000:93:00.0/SMI5. No profiler overlaps paired E2E
or the queued attention replay. The model runs from
`outputs/experiments/openfold3-diffusion-profile-20261010` once prepared; output
is `outputs/openfold3-diffusion-profile-n1536-20261010`.
