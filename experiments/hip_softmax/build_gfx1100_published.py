"""Describe a strict gfx1100 C ABI build; Rust owns compiler execution."""
import json
from pathlib import Path
import sys
from fireprot._paths import KERNEL_ROOT
from fireprot.backends.amd._native_hip import load

destination = Path(sys.argv[1]).resolve()
destination.mkdir(parents=True, exist_ok=False)
_, receipt = load('fireprot_softmax_gfx1100',
    KERNEL_ROOT / 'fireprot_softmax_gfx1100.hip', Path('/opt/rocm'), target='gfx1100',
    compile_flags=['-fno-fast-math', '-ffp-contract=off'], extra_files=[__file__],
    versions={'c_abi': 'softmax-gfx1100-v1'})
(destination / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(receipt['directory'])
