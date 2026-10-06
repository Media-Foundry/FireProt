"""Build the isolated bit-copy kernel through the maintained Rust service."""
import json
from pathlib import Path
import sys
from fireprot.backends.amd._native_hip import load


def main(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    _, receipt = load('fireprot_triangle_bitcopy_v1',
        Path(__file__).with_name('copy.cpp'), Path('/opt/rocm'),
        extra_files=[__file__], versions={'c_abi': 'triangle-bitcopy-v1'})
    (destination/'build.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main(sys.argv[1])
