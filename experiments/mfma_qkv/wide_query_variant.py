"""Generate a 32-query QKV tile with shared K loads and unchanged reductions.

The 1,024-thread block retains two softmax rows per wave. QK reuses each K
fragment across two independent accumulators; four waves perform PV instead
of two. Extra register/LDS pressure requires independent exact GPU gates and
trained module timing. The qualified 16-query implementation is unchanged.
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
    for old, new in (
        ("blockIdx.y * 96", "blockIdx.y * 48"),
        ("linear % 3072", "linear % 1536"),
        ("linear / 3072", "linear / 1536"),
        ("head_query % 96 : blockIdx.x) * 16", "head_query % 48 : blockIdx.x) * 32"),
        ("head_query / 96", "head_query / 48"),
        ("stage[16 * 256]", "stage[32 * 256]"),
        ("float q_values[8];", "float q_values[16];"),
        ("float values[32];", "float values[64];"),
        ("t < 32; ++t)\n        values[t] =", "t < 64; ++t)\n        values[t] ="),
        ("dim3(96, batch * 4)", "dim3(48, batch * 4)"),
    ):
        assert code.count(old) == 1, old
        code = code.replace(old, new)
    qload = "        q_values[fragment] = query[input_base + uint32_t(query_base + coordinate) * 32 + k];"
    assert code.count(qload) == 1
    code = code.replace(qload, qload + "\n        q_values[8 + fragment] = query[input_base + uint32_t(query_base + 16 + coordinate) * 32 + k];")
    accumulator = "        acc4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};"
    assert code.count(accumulator) == 1
    code = code.replace(accumulator, accumulator + "\n        acc4 upper = {0.0f, 0.0f, 0.0f, 0.0f};")
    mfma = "                    prefetched[slot], q_values[group * 4 + slot], accumulator, 0, 0, 0);"
    assert code.count(mfma) == 1
    code = code.replace(mfma, mfma + "\n                upper = __builtin_amdgcn_mfma_f32_16x16x4f32(\n                    prefetched[slot], q_values[8 + group * 4 + slot], upper, 0, 0, 0);")
    start = code.index("        const acc4 swapped_pairs")
    end = code.index("        __syncthreads();", start)
    pack = code[start:end]
    upper = pack.replace("accumulator", "upper").replace("swapped_pairs", "upper_swapped").replace("packed", "upper_packed").replace("destination", "upper_destination").replace("stage_index(coordinate,", "stage_index(coordinate + 16,")
    code = code[:end] + upper + code[end:]
    assignment = "            values[band * 4 + j] = value;"
    assert code.count(assignment) == 1
    code = code.replace(assignment, assignment + "\n            values[32 + band * 4 + j] = stage[stage_index(wave + 16, lane + j * 64)];")
    start = code.index("        // Delay input additions")
    end = code.index("        // All QK stage readers", start)
    softmax = code[start:end].replace("values[", "row_values[")
    softmax = softmax.replace("bias[bias_base +", "bias[bias_base + query_half * 16 * bq +")
    softmax = softmax.replace("probability_output[row_base +", "probability_output[row_base + query_half * 16 * N +")
    code = code[:start] + """        #pragma unroll
        for (int query_half = 0; query_half < 2; ++query_half) {
            float* row_values = values + query_half * 32;
""" + softmax + "        }\n" + code[end:]
    old = """            for (int j = 0; j < 4; ++j)
                stage[stage_index(wave, lane + j * 64)] = values[band * 4 + j];"""
    new = """            for (int j = 0; j < 4; ++j) {
                stage[stage_index(wave, lane + j * 64)] = values[band * 4 + j];
                stage[stage_index(wave + 16, lane + j * 64)] = values[32 + band * 4 + j];
            }"""
    assert code.count(old) == 1
    code = code.replace(old, new)
    assert code.count("if (wave < 2)") == 2
    code = code.replace("if (wave < 2)", "if (wave < 4)")
    code = code.replace("const int feature_base = wave * 16;", "const int feature_base = (wave % 2) * 16;")
    code = code.replace("stage[stage_index(coordinate, local_k)]", "stage[stage_index(coordinate + (wave / 2) * 16, local_k)]")
    code = code.replace("output[(matrix * N + query_base + coordinate) * 32", "output[(matrix * N + query_base + (wave / 2) * 16 + coordinate) * 32")
    code = code.replace("+ wave * 16 + k_lane * 4 + e]", "+ (wave % 2) * 16 + k_lane * 4 + e]")
    code = "// Experimental 32-query QKV tile; requires independent GPU qualification.\n" + code
    with Path(destination).open("x") as output:
        output.write(code)
    return hashlib.sha256(code.encode()).hexdigest()


if __name__ == "__main__":
    print(generate(*sys.argv[1:]))
