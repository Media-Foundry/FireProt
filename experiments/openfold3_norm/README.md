# OpenFold3 C128 Welford experiment

Status: operator and full N1536 model correctness passed; paired E2E measurement
is running. No maintained execution profile enables these candidates.

The installed DiamondHill Torch reports commit
`78258b9de6f4e44e7b8df165d4651bf2e14acf62`. Its matching source checkout is
`/home/pc/pytorch`, with an unmodified
`aten/src/ATen/native/cuda/layer_norm_kernel.cu`.
The aligned FP32 C128 route uses a 256-thread block for each row. Four successive
values enter each participating lane's Welford calculation, followed by wave
and inter-wave reductions. Protenix's existing sum/squared-sum kernels do not
preserve this evaluation order and are not substitutes for the strict OpenFold3
array gate.

`norm.cpp` is a standalone HIP candidate with either four wave64 rows or eight
32-lane rows per block. It explicitly retains merges with the native tree's
empty lanes and waves. It uses no approximation mode or changed dtype. Matching
that intended tree does not establish numerical equality: compiler contraction,
reciprocal and epilogue behavior still require actual array checks.
The PyTorch evaluation-order attribution uses the retained
[PyTorch notice](../../licenses/PyTorch-BSD.txt).

## Build and screen

Build through the Rust-owned HIP service with the repository Python path and
bridge configured for the selected installation:

```bash
HIP_VISIBLE_DEVICES='' PYTHONPATH=/path/to/fireprot/python \
  FIREPROT_RUST_BRIDGE=/path/to/lib_fireprot_bridge.so \
  python experiments/openfold3_norm/build.py /fresh/build-directory

HIP_VISIBLE_DEVICES=7 python experiments/openfold3_norm/screen.py \
  /fresh/build-directory/build.json /fresh/screen-directory \
  /media/IntelSSD/fireprot/outputs/triatt-profile-n1536-v1
```

Do not run a remote build or screen while the template E2E pairs are active.
The local compile used the available HIP 7.14 compiler with GPU visibility
empty; its receipt is `/tmp/fireprot-of3-norm128-local-build-v1/build.json`.
This does not qualify the library against DiamondHill's installed Torch.

## Qualified operator screen

The third frozen candidate, built through the Rust HIP service on DiamondHill,
passes all 24 input/width combinations, including both pinned trained inputs.
Outputs, means and reciprocal standard deviations are bitwise equal to native
Torch. Large-shape outputs are also checked before timing. FP32, deterministic
algorithms and uninitialized-memory filling remain enabled.

| Rows / C128 | Torch | Wave64 candidate | 32-lane candidate |
| --- | ---: | ---: | ---: |
| 393,216 | 2.125 ms | 1.008 ms | **0.464 ms** |
| 2,359,296 | 12.635 ms | 5.955 ms | **2.706 ms** |

These are complete allocation-inclusive operator medians over eight observations
per variant with alternating order. They are not model or E2E speedups.
Evidence: `provenance/openfold3_norm_operator_screen.json` at the repository root.

The first two candidates failed the bias-only output check by up to
4.768e-7 while statistics and other cases matched. A pointwise diagnostic showed
native Torch matched fused `addcmul`, while the candidate matched separate
multiply/add. Explicit `fmaf` in that branch fixes the discrepancy; the rejected
screens remain in the evidence. The first shell wrapper also had an EXIT-trap
quoting error; its completed screen still records no passing variant and no
timing. Subsequent wrappers use direct subprocess arguments and durable JSON
status records.

`model_scope.py` is an experimental full-model hook restricted to the two measured
row counts, C128, the pinned upstream normalization source, FP32 and strict
inference. It preserves the native contiguous-input conversion and falls back
for unmatched calls.

The complete OpenBind-0 N1536 R3/D200/S1 seed101 qualification passes all 11
arrays bitwise and full CIF bytes against the accepted template-reuse output.
It selects 11,202 normalization calls, with 7,846 native fallbacks and 4,730
contiguous-input copies. Its single model-forward observation is 260.607 seconds;
the 355.960-second full invocation is a qualification observation, not a paired
speedup. Evidence: `provenance/openfold3_norm_model_n1536.json`.

The uninstrumented AB/BA/AB comparison completed at
`/media/IntelSSD/fireprot/outputs/experiments/openfold3-norm-pairs-20261007`.
All six runs pass all 11 array bits and full CIF bytes. E2E medians are
391.162 -> 364.159 seconds (1.0742x incremental); model medians are
287.825 -> 260.434 seconds (1.1052x). Both arms retain template reuse and the
prior qualified combination; only `FIREPROT_NORM128` and the profile digest differ.
Evidence: `provenance/openfold3_norm_e2e_n1536.json`. Maintained-profile promotion
remains pending.

The screen compares output, mean and reciprocal standard deviation **bitwise**
against native Torch. Inputs cover row tails, random values, plain/scale/bias
variants, constant and low-variance rows, large values, signed zero and retained
trained triangle-attention inputs. When a capture directory is supplied, both
files in `captures.json` must match their recorded sizes and SHA256 hashes before
GPU execution. An empty, incomplete or corrupt directory is an error. Omitting
the directory explicitly gives synthetic-only coverage, recorded in the report.
`python experiments/openfold3_norm/test_captures.py` checks the capture guards
without using a GPU. Only variants passing the numerical checks are timed.
Timing includes ordinary strict output allocation and alternates call order.
Neither operator correctness nor an operator speedup establishes model parity
or E2E benefit. Full-model gates and a fresh paired comparison remain required
before any model integration is accepted.

The subsequent [direct strided-input experiment](STRIDED.md) removes input copies
for three measured layouts. Operator and full-model qualification pass; its
paired E2E result remains pending.
