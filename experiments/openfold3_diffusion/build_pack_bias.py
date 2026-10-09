"""Build the isolated bias packer through the Rust compiler/cache service."""
import json
from pathlib import Path
import sys
from fireprot.backends.amd._native_hip import load

if __name__ == "__main__":
    output = Path(sys.argv[1])
    output.mkdir(exist_ok=False)
    _, receipt = load("fireprot_of3_bias_pack_v1", Path(__file__).with_name("pack_bias.cpp"),
                      Path("/opt/rocm"), extra_files=[__file__],
                      versions={"c_abi": "of3-bias-pack-h16-f32-v1"})
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
