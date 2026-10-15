# PairFormer-only precision diagnostics

Status (2026-10-15): full-model diagnostics finished. Unprotected FP16 produced
nonfinite activations in the first PairFormer cycle; BF16 completed but changed
the aligned structure substantially. Neither candidate is promoted. No
mixed-precision speedup or quality acceptance is established. The user requested **reporting deviations
before choosing a numerical threshold**; the Base 0.1 Angstrom policy is not
automatically applied to OpenFold3.

## Official configuration and tested scope

At the pinned OpenFold3 commit `7de748b7bc93adb5af0a4032d5e9f208b6e1325f`:

- [Training recipe](https://github.com/aqlaboratory/openfold-3/blob/7de748b7bc93adb5af0a4032d5e9f208b6e1325f/examples/training_yamls/initial_training.yml)
  selects `bf16-mixed`; the three fine-tuning recipes also select it.
- [Inference configuration](https://github.com/aqlaboratory/openfold-3/blob/7de748b7bc93adb5af0a4032d5e9f208b6e1325f/openfold3/entry_points/validator.py)
  defaults to `32-true`. An FP16 output-file dtype is not FP16 model execution.
- [Model boundaries](https://github.com/aqlaboratory/openfold-3/blob/7de748b7bc93adb5af0a4032d5e9f208b6e1325f/openfold3/projects/of3_all_atom/model.py)
  return FP32 trunk representations and explicitly select FP32 autocast for
  diffusion rollout. The deployed source bytes are separately pinned by the scope.

This experiment autocasts only `model.pairformer_stack` within each recycle.
Input/recycle embeddings, template/MSA processing, diffusion and confidence stay
FP32; master parameters remain FP32. PairFormer outputs return to FP32 before
the next stage. Recycles, diffusion steps, samples and random state are unchanged.
BF16 and FP16 are independent candidates. This restricted experiment is not a
claim to reproduce the complete official BF16 training configuration.

## Guards and observations

`scope.py` records effective precision, per-cycle completion and activation ranges
by dtype, aggregate per-module ranges, and actual Torch matmul backend flags.
Diagnostic hooks stop on nonfinite module
outputs. FP16 has a finite range limit of 65504, so projection outputs, triangle
updates, attention and normalization boundaries are checked; no values are
clipped to make a failure pass. Functional intermediates inside a module are not
all individually inspected, so finite monitored outputs do not prove that every
temporary remained finite.

FP32-only experimental triangle gate fusion is bypassed during a low-precision
PairFormer call, then restored. Remaining dispatch guards retain their dtype
fallbacks. Diffusion input boundaries and leaf-module outputs reject non-FP32
floating GPU tensors. Confidence entry likewise requires FP32. All hooks and
temporary replacements are restored on success or failure.

The frozen base request retains its FP32 anchor declaration. An explicit,
hash-pinned `FIREPROT_PRECISION_POLICY` file declares the scoped override;
`precision-policy.json` and `pairformer-precision.json` are included in sealed
outputs. Results must be described using this **effective stage policy**, never
as pure FP32 inference. The maintained request schema/default profile is not
extended by this diagnostic.

## Execution and acceptance boundaries

`prepare.py` copies the qualified HIP-packing worker into an isolated tree and
creates fresh controls for FP32, BF16 and FP16. `run_diagnostic.py` uses Rust
`measure-inference` for preflight, physical-device supervision and serialization
checks. The FP32 diagnostic must reproduce all eleven reference arrays and full
CIF bytes before lower-precision trials begin. An accepted mixed-precision
archive establishes content validity, not agreement with the FP32 reference.

The queue uses HIP7 only (PCI `0000:93:00.0`, SMI5), N1536, R3/four trunk passes,
D200/S1 and seed101 with strict determinism, memory filling, fresh processes and
unflushed caches. It waits for the existing FP32 paired cycle to finish and checks
that the executing GCD is idle before each diagnostic. Instrumented model times
are not E2E speedup measurements. No mixed-precision timing verdict is issued.

`deviation.rs` uses maintained Rust NPZ/CIF parsing and coordinate metrics. It
checks atom identities and CIF/NPZ association, reports all-atom and C-alpha
alignment, globally aligned residue deviations, all C-alpha pair-distance changes,
and confidence-array differences. Proper Kabsch rotations exclude reflections.
It deliberately emits no quality threshold or pass verdict. This compares
predictions, not experimental structure truth.

CPU checks:

```bash
HIP_VISIBLE_DEVICES='' python -m unittest experiments.openfold3_precision.test_scope -v
```

Four tests cover overflow rejection, scope restoration, diffusion isolation and
unchanged cycle/step counts using toy CPU modules. They mock autocast and source
pins; real GPU arithmetic and deployed-source compatibility require the queued
model runs. Two Rust tests verify rigid-motion removal and retention of local
distortion/reflection differences. The standalone Rust binary links the current
workspace's `fireprot-validation`, `nalgebra` and `serde_json` release artifacts;
its build commands and artifact digests are retained with experiment evidence.

Current remote queue: `outputs/experiments/openfold3-precision-recovery1-queue-20261010`.
Completed model root: `outputs/experiments/openfold3-precision-diagnostic-recovery1-20261010`.
The initial waiting queue was explicitly stopped before model preparation or GPU
execution to add per-cycle range reporting. Its frozen controls remain intact.

## Collected results and next experiment

[Results and artifact pins](../../provenance/openfold3_precision_results_20261015.json)
cover N1536, R3/four trunk passes, D200/S1, seed101, HIP7. The FP32 diagnostic
reproduces all eleven reference arrays exactly. BF16 completes all four cycles
and 200 FP32 diffusion calls with finite monitored outputs. FP16 stops in cycle
one at `blocks.0.pair_stack.tri_mul_out.layer_norm_out` with NaN outputs.
This locates the first observed failure; contraction overflow upstream of the
normalization remains a hypothesis until its intermediate tensors are inspected.

BF16 versus the FP32 diagnostic, using raw NPZ coordinates and matching CIF atom
identities:

| Metric | Value |
| --- | ---: |
| Raw all-atom RMSD | 5.29650 Angstrom |
| Proper-rotation/translation aligned all-atom RMSD | 5.29304 Angstrom |
| Independently aligned C-alpha RMSD | 5.23828 Angstrom |
| Aligned atom displacement p95 / maximum | 11.04323 / 40.54562 Angstrom |
| C-alpha pair-distance absolute change median / p95 | 1.05840 / 7.66723 Angstrom |
| Atom pLDDT absolute change mean / maximum | 0.46462 / 8.93342 |

Rigid-motion compensation does not remove these differences. Mean atom pLDDT
is 37.89925 for FP32 and 37.91813 for BF16, so this is a low-confidence prediction;
neither prediction is experimental ground truth. No general accuracy conclusion
or numerical acceptance threshold follows from this single case. Instrumented
diagnostic latency is deliberately excluded from speedup claims.

The next bounded precision experiment should inspect the first triangle
contraction before normalization, then evaluate FP16 projections with FP32
contraction and normalization. Preserve FP32 residual storage and every
diffusion operation. Do not clamp values or reduce cycles/steps to hide failure.
First establish finite intermediate values and report final aligned/local and
confidence deviations; only an accepted candidate should proceed to uninstrumented
paired E2E measurements.

A conservative first protected candidate is now running: **complete native
triangle multiplicative updates remain FP32**, with FP16 autocast for the rest
of PairFormer. This is broader protection than contraction alone. The hashed
policy declares `triangle_precision: float32`; every triangle module must run
under protection in every trunk pass. Extra upstream chunk-tuner calls are
recorded per module and cycle. Eight CPU scope tests cover the original gates,
protected dispatch, missing coverage, failure restoration and chunk-tuner warmup.
The fixed-total check was too strict: upstream `_prep_blocks` executes block zero
while tuning chunk size. Its regression test fails before the correction and
passes afterward. The FP16 candidate using that check was explicitly cancelled
through its own Rust inference supervisor; its completed FP32 anchor remains
valid. The recovery rechecks that anchor and runs only the corrected candidate.
[Preparation and control pins](../../provenance/openfold3_precision_triangle_fp32_preparation_20261015.json).
No protected-candidate GPU numerical result is established yet. HIP7 remains
the only authorized device; all these runs are instrumented diagnostics.
