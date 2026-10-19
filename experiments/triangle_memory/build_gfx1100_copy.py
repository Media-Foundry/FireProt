"""Reproduce the isolated gfx1100 bit-copy build through Rust, with GPUs hidden."""
import json
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
from fireprot.backends.amd._native_hip import load

destination = Path(sys.argv[1]).resolve()
destination.mkdir(parents=True, exist_ok=False)
_, receipt = load('fireprot_triangle_bitcopy_gfx1100_v1',
    Path(__file__).with_name('copy.cpp'), Path('/opt/rocm'), target='gfx1100',
    extra_files=[__file__], versions={'c_abi': 'triangle-bitcopy-v1'})
(destination/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(receipt['library'])
