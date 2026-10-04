# AMD native kernels

This directory contains the maintained HIP/CK sources. Rust build services own
input identities, cache transactions, compiler lifetime and artifact acceptance.
Python tensor loaders describe the local Torch/ROCm boundary and invoke those
services. See [build ownership](../../docs/kernel-builds.md) and
[published deployment](../../docs/kernel-artifacts.md).

The current validated target is gfx90a. Shape, stride, precision, inference-mode
and model compatibility checks live in `python/fireprot`; these files do not
imply support for every protein length or GPU. Experimental variants remain in
`benchmarks/`, outside maintained dispatch unless separately promoted.

Keep weights, compiled libraries, caches and predictions outside this directory.
Native source changes require operator correctness and relevant full-model gates.
