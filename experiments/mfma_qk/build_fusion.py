"""Build the unvalidated packed QK/softmax experiment through Rust."""
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
        'fireprot_experimental_qk_softmax', Path(__file__).with_name('fused_qk_softmax.cpp'),
        Path('/opt/rocm'), extra_files=[__file__, softmax_fp32.__file__],
        versions={'c_abi': 'experimental-qk-softmax-n1536-f32-v1'})
    (destination / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
