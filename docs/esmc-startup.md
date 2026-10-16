# ESMC import overlap

The standalone ESMC profile now imports the unchanged upstream module definitions
after emitting `ready`, while Rust verifies the worker and device. Model
construction, checkpoint loading and inference still wait for the supervisor's
valid `start` command. Imports execute synchronously in the worker; no Python
import thread or background model execution is introduced.

The upstream source pin is checked before importing ESMC. Complete checkpoint
hashing, execution identity checks, GPU isolation and output acceptance remain
enabled. The qualified pinned loader and exact HIP RoPE/SwiGLU operators are
unchanged. No new CK, attention or GEMM implementation is included.

## Startup diagnostic

A separate instrumented invocation retained Python import traces and explicit
timestamps. It passed byte equality against native output. These are individual
diagnostic observations, not paired performance results:

| Boundary | Seconds |
| --- | ---: |
| Torch import | 1.649 |
| CUDA/HIP initialization | 0.147 |
| Ready emission to receipt of start | 1.178 |
| ESMC module import, cumulative import trace | 1.752 |
| Checkpoint conversion, packing and placement | 1.290 |
| Output announcement and hashing | 0.017 |

The diagnostic complete invocation took 18.283 s, with 11.112 s under supervisor
timing. The listed components do not cover all boundaries. In particular, do not
attribute the unmeasured remainder entirely to Python teardown or preflight.
Import tracing was disabled for both paired comparisons below.

## Complete invocation measurements

Both comparisons use N1000 x4/B4, seed101, native BF16 and HIP6 / PCI
`0000:8e:00.0`. Each uses three AB/BA/AB pairs, fresh workers, unflushed caches,
full original checkpoint hashes and all requested exports. No transformed
checkpoint or prediction cache is used.

| Reference | Baseline median | Candidate median | Speedup | Paired ratios |
| --- | ---: | ---: | ---: | --- |
| Previous FireProt pinned loader | 18.173892 s | 16.997369 s | **1.069218x** | 1.0873x, 1.0488x, 1.1094x |
| Original native implementation | 24.353151 s | 16.913512 s | **1.439864x** | 1.4312x, 1.3948x, 1.4430x |

Every pair improves. All **12 invocations / 48 output archives** were rehashed
and compared independently, including comparison to retained official same-batch
output; tokens and embeddings remain byte-identical. Physical device identity,
batch grouping and 160 HIP operator calls per forward were also verified.

The direct-native result supersedes the previous 1.365x standalone B4 claim.
The two comparisons were scheduled separately; do not mix their medians or
multiply this incremental ratio by an older ratio. No singleton, resident-batch
or ESMFold2 folding result is inferred.

Within-invocation forward medians are 1.394 -> 1.373 s for the incremental
comparison and 1.519 -> 1.369 s for the native comparison. The improvement is in
startup scheduling. The candidate's `request_seconds` starts after the relocated
imports and is therefore unsuitable for estimating this E2E improvement.

## Entry point and fallback

Use the existing request `configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json`
with `configs/diamondhill/plm-esmc-fold2-6b-exact.json`. Copy the request and set a
fresh `run_id` and unused `output_dir` first. On DiamondHill:

```bash
cd /media/IntelSSD/fireprot
outputs/experiments/esmc-startup-20261016/native/qualified-import-overlap/target/release/fireprot \
  esm /tmp/esmc-request.json configs/diamondhill/plm-esmc-fold2-6b-exact.json
```

The profile sets `FIREPROT_ESMC_IMPORT_OVERLAP=1` and points to that frozen
deployment. `0` or absence preserves sequential startup; other values fail.
Enabling it for a model other than standalone ESMC also fails. The preceding
published profile remains at
`configs/diamondhill/plm-esmc-fold2-6b-pinned-loading.json`.

Maintained worker sources and Rust binaries were checked against the measured
deployment's file hashes. Six CPU tests cover readiness ordering, invalid/missing
authorization, import failures, source pins, model scope and defaults; the 13
existing OpenFold3 worker tests also pass after adapting the ready test double.

Evidence: [sealed reports and file hashes](../provenance/esmc_startup_20261016.json).
Reproduction scripts are `experiments/plm/profile_esmc_startup.py`,
`prepare_esmc_startup_pairs.py` and `collect_esmc_startup_pairs.py`.
Raw artifacts are retained under
`/media/IntelSSD/fireprot/outputs/experiments/esmc-startup-20261016`.
