# ROCm operator coverage experiment

This experiment investigates the ideas described in Protenix issue 354. Its
unverified MI300 speedup is not evidence for FireProt or DiamondHill MI250.
No code or patch from the issue author is used.

`torch_probe.py` is a diagnostic Torch boundary, not an inference launcher.
It records inclusive GPU stage events and uses the installed Torch profiler to
observe trunk operators grouped by input shapes. Rust `fireprot infer` retains
request validation, source/weight/environment verification, physical GPU
monitoring, worker supervision and output acceptance. Numerical comparison with
the original references remains required after instrumentation.

## Experiment deployment

The first deployment is `/tmp/fireprot-coverage-v1` on DiamondHill. It contains
copies of the maintained `python/`, root import shim and kernel sources. Apply
`worker-overlay.patch` with `git apply -p0` to that copy and place `torch_probe.py` alongside the
copied workers. The two model-boundary changes enter the probe after model
construction. The transport change includes `operator-profile.json` among the
Rust-verified output artifacts. Published kernel bundles and the existing Rust
bridge remain explicitly pinned in execution profiles.

The first setup omitted the root shim/kernel tree and Rust rejected both requests
before model execution. Corrected runs use fresh `coverage-v2-*` run IDs, with
full request/profile JSON retained in the deployment directory. No shared model
environment or upstream checkout was changed.

Persistent copies of both experiment directories are stored on DiamondHill under
`outputs/experiments/rocm-coverage-20261005/` as compressed record bundles.
Their paths, sizes and SHA-256 hashes are retained in both provenance JSON files.
Accepted predictions and raw profiler outputs remain in their original `outputs/`
run directories. Bundle paths describe records, not model weights.

## Interpretation limits

- Stage intervals are inclusive and nested; do not sum parents and children.
- Instrumentation adds overhead; these runs cannot establish E2E speedup.
- The initial detailed profiler scope is `get_pairformer_output` for Protenix
  and `run_trunk` for OpenFold3. Diffusion/confidence have stage timing only.
- CPU ATen attribution and GPU kernel events overlap. Never add both tables to
  estimate total GPU time. Shapes do not establish tensor strides or eligibility.
- Protenix runtime dispatch records provide independent selection/fallback
  evidence. GPU profiler event counts may be incomplete when ROCTracer warns.
- The initial Protenix N1000 trace reports a duplicate ROCTracer flow start;
  retain that warning and avoid treating kernel counts as exhaustive.
- The original cycle, step, sample, precision and seed settings are retained.
  Any later approximation policy requires separate authorization/validation.

## First measured coverage

Evidence: `provenance/rocm_coverage_firstpass.json`. Protenix Mini-default uses
FP32/C4/D1/S1/seed101. OpenFold3 OpenBind-0 uses strict deterministic
FP32/R3/D200/S1/seed101. R3 retains four trunk passes. All four instrumented
requests passed their existing numerical gates and archive replay.

| Trunk ATen self-device attribution | Protenix N1000 | Protenix N1536 | OpenFold3 N1000 | OpenFold3 N1536 |
| --- | ---: | ---: | ---: | ---: |
| Native LayerNorm | 17.325 ms | 25.955 ms | 15.576 s | 36.962 s |
| Copy | 12.718 ms | 342.870 ms | 23.414 s | 65.353 s |

These are diagnostic attribution totals, not clean model timings. Protenix
already selects its maintained HIP/CK operators; its registry selection counts
match the previous accepted runs. Remaining native LayerNorm is a low priority
there. OpenFold3's materialized triangle attention scores expose larger targets:
the two observed chunk sizes attribute 34.728 seconds to mask additions and
56.457 seconds to pair-bias additions. These values are not additive with nested
stage or GPU kernel tables, and do not predict a realized E2E gain.

OpenFold3 N1536 confirms the same priority: the dominant `[128,4,1536,1536]`
score shape attributes 286.262 seconds to pair-bias additions and 128.383 seconds
to mask additions. Its native LayerNorm totals 36.962 seconds and copies total
65.353 seconds. The instrumented request passes all 11 arrays bitwise against
the strict N1536 baseline and passes archive replay. It also reports duplicate
ROCTracer flow starts and asynchronous signal waits, so direct GPU kernel counts
must not be treated as exhaustive or free of duplicates.

Neither tested model environment has an installed Triton module. OpenFold3
explicitly disables Triton/DeepSpeed/cueq. The issue's claimed Triton improvement
does not establish a usable backend in these environments. Absence of the Python
Triton module does not imply absence of Torch's separately supplied ROCm SDPA
backend; Protenix already has validated CK and SDPA routes.

## Verified-zero-mask candidate

`zero_mask.py` is an experimental tensor-boundary intervention. Apply
`zero-mask-worker-overlay.patch` with `git apply -p0` to a fresh production worker copy, add this
helper alongside it, and keep the root import shim and kernels in the deployment.
The actual deployment is `/tmp/fireprot-zero-mask-v1` on DiamondHill. The same
copied worker serves both arms; `FIREPROT_SKIP_ZERO_MASK=0` disables the candidate
and `=1` enables it. Rust owns invocation, GPU monitoring, output acceptance and
paired benchmarking. No maintained model source or production operator is changed.

The candidate checks each eligible FP32 H4/D32 N1000/N1536 mask's values before
omitting its addition. Nonzero and nonfinite masks are retained. There is no
pointer-cache assumption. QK multiplication, remaining bias additions, native
softmax and PV multiplication are unchanged. The per-call scalar check introduces
a synchronization cost; only complete invocation measurement can establish value.
Removing a zero addition is not a universal IEEE bitwise guarantee (signed zero
and exceptional values require care), so promotion remains limited by actual
model and operator comparisons.

The first N1000 full-model candidate skipped 1,702 additions and matched all
11 output arrays bitwise against the independent strict native reference.
`test_zero_mask.py` passed on HIP5/PCI b3: eight signed-zero/nonzero/partial-mask
and precision combinations, plus unsupported-shape fallback and restoration.
Its initial pytest invocation failed because pytest is absent in that environment;
the standalone entry point passed without installing dependencies.

Both lengths pass all 11 arrays exactly. The N1536 candidate skips 5,596 mask
additions. The N1000 AB/BA/AB benchmark finishes successfully, including checks
against the initial baseline anchor, every pair, and the aggregate comparison.
Evidence: `provenance/openfold3_zero_mask_experiment.json`.

| OpenFold3 workload | Native model | Candidate model | Model ratio | Native E2E | Candidate E2E | E2E ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N1000, three pairs, arm medians | 238.060 s | 211.670 s | 1.125x | 332.802 s | 305.387 s | 1.090x |
| N1536, one pair | 854.413 s | 775.981 s | 1.101x | 952.871 s | 880.036 s | 1.083x |

N1000 individual E2E ratios are 1.077x, 1.145x and 1.090x; no run is discarded.
Its native model times range from 236.889 to 256.143 seconds. The N1536 trend
remains `insufficient_repeats`. These are shared-host observations on fixed inputs,
not a generalized speedup or structural-quality claim. E2E includes the complete
Rust child invocation, including preflight and output acceptance.

The separate N1536 operator diagnostic is complete and excluded from timing ratios.
The N1000 timing experiment uses fresh model processes, unflushed filesystem
cache and strict environment inventories. It overlaps the N1536 baseline and
subsequent candidate and native profiler run on a different GCD of the same host. The immutable benchmark
specification originally anticipated only the baseline overlap; the deployment
evidence records this additional overlap explicitly. GPU isolation does not imply
a globally idle host. No production route is promoted by these experimental results.

## Follow-up priorities

The user directs further work toward handwritten HIP and CK-style operators.
Do not pursue Triton enablement. Existing Triton availability observations are
historical environment facts, not an implementation workstream. Prefer measured
kernel choices; no general claim that HIP always beats other implementations is
established by this experiment.

1. The zero-mask candidate passes the paired N1000 E2E gate and the N1536
   exact-output check. Before a maintained route, add complete broadcast/shape
   guards, exception behavior and per-adapter ownership, then repeat long-input
   measurements in a separate window with no other FireProt GPU jobs. The current
   prototype is limited to the validated requests, not arbitrary caller inputs.
2. Investigate pair-bias/softmax fusion on the materialized score tensor.
   A HIP implementation must preserve the installed native softmax's reduction
   and exponential behavior to meet the current OpenFold3 bitwise gate.
   A generic CK/SDPA substitution already failed that gate in the previous
   operator screen; faster isolated timing does not justify promotion.
   The maintained `kernels/amd/fireprot_pair_softmax.cu` offers an existing
   BF16 AtlasFold example of matching Torch's wave64 reduction. OpenFold3 observes
   `softmax_warp_forward<float,...,10,...,64>` at N1000 and the corresponding
   exponent-11 variant at N1536. Its FP32 dtype, exact sequence tails and
   pair-bias strides require a
   separate implementation and validation. Do not widen the existing BF16
   loader's capability checks without that evidence.
3. Capture actual strides and allocation sites for pair LayerNorm and triangle
   packing. The current N1000 trace attributes 3.578 seconds of native LayerNorm
   to `[1,1000,256,128]`, 3.181 seconds to `[1,1000,1000,128]`, and 2.563 seconds
   to `[1,1000,244,128]`. Copy activity is substantial in the corresponding
   256/244 chunks and channel-first projections. Shape similarity alone does
   not establish compatibility with Protenix's packed HIP kernels.
4. Inspect diffusion's timestep-independent pair conditioning for safe request-
   local reuse. Prove dependencies, mutation and aliasing rules before caching;
   an unchanged data pointer is insufficient. This has no measured candidate yet.

Keep operator numerical checks, complete-model checks and performance gates
separate. No recycle/step reduction or relaxed correctness threshold is part
of these proposals. Both lengths now have trunk operator traces; diffusion and
confidence still have stage timing only, not exhaustive per-operator coverage.
