"""Build the separate private-score ABI through Rust with GPUs hidden."""
import json
import os
from pathlib import Path
import sys

os.environ['HIP_VISIBLE_DEVICES'] = ''
from fireprot.backends.amd._native_hip import load

destination = Path(sys.argv[1]).resolve()
destination.mkdir(parents=True, exist_ok=False)
_, receipt = load('fireprot_softmax_gfx1100_owned_v1',
    Path(__file__).with_name('gfx1100_owned.hip'), Path('/opt/rocm'), target='gfx1100',
    compile_flags=['-fno-fast-math', '-ffp-contract=off'],
    extra_files=[__file__], versions={'c_abi': 'gfx1100-owned-softmax-v1'})
(destination/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(receipt['library'])
