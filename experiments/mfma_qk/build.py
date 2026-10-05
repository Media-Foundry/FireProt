"""Prepare an experimental C ABI library through Rust after E2E timing ends."""
import json
from pathlib import Path
import sys

import torch

from fireprot.backends.amd import softmax_fp32
from fireprot.backends.amd._native_hip import load


def main(destination):
    if (str(torch.__version__) != softmax_fp32.TORCH_VERSION
            or torch.version.hip != softmax_fp32.HIP_VERSION):
        raise RuntimeError('Use the validated OpenFold3 Torch/HIP environment')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    _, receipt = load(
        'fireprot_experimental_qk_mfma16', Path(__file__).with_name('qk_probe.cpp'),
        Path('/opt/rocm'), extra_files=[__file__],
        versions={'c_abi': 'experimental-qk-mfma16-f32-v1'})
    (destination / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
