"""Build the isolated C128 Welford candidate through Rust; no GPU execution."""
import json
from pathlib import Path
import sys
from fireprot.backends.amd._native_hip import load


def main(destination):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    _, receipt = load('fireprot_of3_welford128_strided_v1', Path(__file__).with_name('norm_strided.cpp'),
        Path('/opt/rocm'), extra_files=[__file__], versions={'c_abi': 'of3-norm128-strided-v1'})
    (out / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
