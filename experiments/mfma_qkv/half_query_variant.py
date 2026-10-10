"""Generate an eight-query fused-QKV trial from the qualified unroll4 source.

This trades redundant MFMA rows and more QK barriers for smaller blocks and
twice the active-wave fraction during PV. It is a hypothesis, not a speedup.
Every retained output keeps the original ascending MFMA and softmax order.
"""
import hashlib
from pathlib import Path
import sys

BASE_SHA256 = "dd7bbb785dae1afed98ad6fd357fa8e6822f0e49310a87530a296ee7c0d9ff9f"


def generate(source, destination):
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != BASE_SHA256:
        raise ValueError("Expected the qualified unroll4 source")
    code = data.decode()

    def replace(old, new):
        nonlocal code
        if code.count(old) != 1:
            raise ValueError(f"Ambiguous source anchor: {old}")
        code = code.replace(old, new)

    replace("__launch_bounds__(1024, 1)", "__launch_bounds__(512, 1)")
    replace("Fixed 1024x1x1", "Fixed 512x1x1")
    replace("blockIdx.y * 96 + blockIdx.x", "blockIdx.y * 192 + blockIdx.x")
    replace("(linear % 3072) / 8", "(linear % 6144) / 8")
    replace("(reuse ? head_query % 96 : blockIdx.x) * 16",
            "(reuse ? head_query % 192 : blockIdx.x) * 8")
    replace("((linear / 3072) * 8 + linear % 8) * 4 + head_query / 96",
            "((linear / 6144) * 8 + linear % 8) * 4 + head_query / 192")
    replace("float stage[16 * 256]", "float stage[8 * 256]")
    replace("query_base + coordinate) * 32 + k",
            "query_base + (coordinate & 7)) * 32 + k")
    # Only the QK exchange changes to twelve 128-column bands. PV still stages
    # six 256-column bands, retaining its exact original reduction sequence.
    start = code.index("    // Compute and transpose one 256-column band")
    end = code.index("    {\n        // Delay input additions", start)
    qk = code[start:end]
    qk = qk.replace("one 256-column band", "one 128-column band")
    qk = qk.replace("all six old accumulator tiles", "all twelve accumulator tiles")
    qk = qk.replace("band < 6", "band < 12").replace("band * 256", "band * 128")
    qk = qk.replace("*reinterpret_cast<acc4*>(&stage[destination]) = packed;",
                    "if (coordinate < 8)\n            *reinterpret_cast<acc4*>(&stage[destination]) = packed;")
    qk = qk.replace("j < 4", "j < 2").replace("values[band * 4 + j]", "values[band * 2 + j]")
    qk = qk.replace("band != 5", "band != 11")
    code = code[:start] + qk + code[end:]
    replace("bv[slot] = stage[stage_index(coordinate, local_k)];",
            "bv[slot] = stage[stage_index(coordinate & 7, local_k)];")
    replace("        if (wave < 2) {\n            #pragma unroll\n            for (int e",
            "        if (wave < 2 && coordinate < 8) {\n            #pragma unroll\n            for (int e")
    replace("dim3(96, batch * 4), dim3(1024)", "dim3(192, batch * 4), dim3(512)")
    code = "// Experimental eight-query fused QKV; requires independent GPU gates.\n" + code
    with Path(destination).open("x") as output:
        output.write(code)
    return hashlib.sha256(code.encode()).hexdigest()


if __name__ == "__main__":
    print(generate(*sys.argv[1:]))
