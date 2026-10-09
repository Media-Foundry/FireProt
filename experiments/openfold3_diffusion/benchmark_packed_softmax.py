"""Run the seven-case attention gate with a separately pinned HIP bias packer."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import benchmark_global_softmax as benchmark
from packed_global_softmax import PackedGlobalSoftmaxDispatcher


if __name__ == "__main__":
    destination, receipt = Path(sys.argv[1]), Path(sys.argv[2])
    tile_rows = int(sys.argv[3])
    instances = []

    def factory(fallback, kernel):
        dispatch = PackedGlobalSoftmaxDispatcher(fallback, kernel, receipt, tile_rows)
        instances.append(dispatch)
        return dispatch

    benchmark.GlobalSoftmaxDispatcher = factory
    benchmark.run(destination)
    # Bit-copy checks also cover signed zero, infinities and arbitrary NaN payloads.
    with torch.inference_mode():
        bits = torch.randint(-(2**31), 2**31 - 1, (1, 1, 1536, 1536, 16),
                             device="cuda", dtype=torch.int32)
        bits.flatten()[:8] = torch.tensor([0, -2147483648, 2139095040, -8388608,
                                          2143289345, -4194303, 1, -2147483647],
                                         device="cuda", dtype=torch.int32)
        value = bits.view(torch.float32).permute(0, 1, 4, 2, 3)
        packed = instances[0].pack_bias(value)
        torch.cuda.synchronize()
        assert benchmark.same(packed, value)
    # Keep the original screen immutable; completion requires this extra receipt.
    report = dict(arbitrary_bits_exact=True, tile_rows=tile_rows,
                  screen_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  dispatch=instances[0].manifest(),
                  sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in (Path(__file__), Path(__file__).with_name("packed_global_softmax.py"),
                                     Path(__file__).with_name("pack_bias.cpp"))])
    with destination.with_suffix(".packing.json").open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
