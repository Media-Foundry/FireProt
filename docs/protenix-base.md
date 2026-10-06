# Protenix Base integration

Base-default v0.5.0 uses the existing Rust request and supervised Protenix worker
boundary. Its checkpoint passes strict CPU loading: 368,088,191 parameters,
4,092 FP32 tensors, no missing or unexpected keys. See
[checkpoint evidence](../provenance/protenix_base_checkpoint.json).

## Initial workload

The N1000 and N1536 requests explicitly select FP32, C10/D200/S1, seed 101,
sequence-only monomers without MSA or templates, on HIP device 7. Base's upstream
default sample count is **five**; these single-sample experiments are not the full
upstream default workload. Keep all scientific settings identical within each
native/FireProt comparison. Mini C4/D1 speedups do not establish Base performance.

```bash
target/release/fireprot inspect configs/requests/protenix-base-n1000-native.json
target/release/fireprot infer configs/requests/protenix-base-n1000-native.json \
  configs/diamondhill/protenix-base-fireprot.json
target/release/fireprot infer configs/requests/protenix-base-n1000-fireprot.json \
  configs/diamondhill/protenix-base-fireprot.json
```

Copy requests with fresh run IDs and output paths before repeating them. The
N1536 equivalents use `protenix-base-n1536-{native,fireprot}.json`.
The same profile can be used for both arms: the native backend does not install
FireProt hooks. Only HIP devices 6 and 7 are authorized for these experiments.

## Transplant boundaries

Existing HIP/CK triangle multiplication, triangle attention, LayerNorm, transition
and layout paths retain their exact shape/dtype/stride/device capability guards.
Unsupported signatures retain native execution. Their selection on Base must be
verified in `operator-runtime.json`, and model outputs must pass the existing
Protenix coordinate/confidence gates before reporting performance gains.

Fast model construction and skipped template preparation remain disabled in the
Base profile; their current model guards only qualify Mini. Base's deeper stack
and full diffusion workload require independent output validation. GPU numerical
and performance qualification is in progress; no Base speedup is claimed yet.
