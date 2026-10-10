# PairFormer-only precision diagnostics

Status: CPU lifecycle tests pass; full-model diagnostics are queued after the
current FP32 narrow-projection paired experiment. No mixed-precision speedup or
quality acceptance is established. The user requested **reporting deviations
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
Future model root: `outputs/experiments/openfold3-precision-diagnostic-recovery1-20261010`.
The initial waiting queue was explicitly stopped before model preparation or GPU
execution to add per-cycle range reporting. Its frozen controls remain intact.
