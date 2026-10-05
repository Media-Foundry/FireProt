"""Prepare the renamed C ABI artifact through Rust; no GPU execution."""
import json
from pathlib import Path
import sys

import torch

from fireprot._paths import KERNEL_ROOT
from fireprot.backends.amd import softmax_fp32
from fireprot.backends.amd._native_hip import load


def main(destination):
    if str(torch.__version__) != softmax_fp32.TORCH_VERSION or torch.version.hip != softmax_fp32.HIP_VERSION:
        raise RuntimeError("Build with the validated OpenFold3 Torch/HIP environment")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    _, receipt = load(
        softmax_fp32.PACKAGE, KERNEL_ROOT / "fireprot_softmax_fp32.cpp", Path("/opt/rocm"),
        extra_files=[__file__, softmax_fp32.__file__], versions={"c_abi": softmax_fp32.C_ABI})
    (destination / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
