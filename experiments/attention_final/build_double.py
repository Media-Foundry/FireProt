"""Rust-owned build of the unpromoted double-LDS candidate."""
import json
from pathlib import Path
import sys
from fireprot.backends.amd._native_hip import load


def main(destination):
    out=Path(destination);out.mkdir(exist_ok=False)
    _,receipt=load('fireprot_experimental_qk_softmax',
        Path(__file__).with_name('qk_double_buffer.cpp'),Path('/opt/rocm'),
        extra_files=[__file__],versions={'c_abi':'experimental-qk-softmax-n1536-f32-v1'})
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main(sys.argv[1])
