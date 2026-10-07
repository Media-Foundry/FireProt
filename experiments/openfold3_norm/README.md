# OpenFold3 C128 Welford experiment

Status: local HIP compilation passed; GPU correctness and speed are **unverified**.
No model dispatch or maintained execution profile uses these candidates.

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
