"""Generate a512-thread,16-query fusion trial without redundant MFMA rows.

Each wave retains two softmax rows; QK exchanges twelve128-column bands.
PV retains its exact original arithmetic order. Increased registers and barriers
must be measured against the potential scheduling gain. No default is changed.
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
    assert code.count("__launch_bounds__(1024, 1)") == 1
    code = code.replace("__launch_bounds__(1024, 1)", "__launch_bounds__(512, 1)")
    code = code.replace("Fixed 1024x1x1", "Fixed 512x1x1")
    code = code.replace("float values[32];", "float values[64];")
    code = code.replace("for (int t = 0; t < 32; ++t)\n        values[t] =",
                        "for (int t = 0; t < 64; ++t)\n        values[t] =")
    start = code.index("    // Compute and transpose one 256-column band")
    end = code.index("    {\n        // Delay input additions", start)
    qk = code[start:end]
    qk = qk.replace("one 256-column band", "one 128-column band")
    qk = qk.replace("all six old accumulator tiles", "all twelve accumulator tiles")
    qk = qk.replace("band < 6", "band < 12").replace("band * 256", "band * 128")
    qk = qk.replace("j < 4", "j < 2").replace("values[band * 4 + j] = value;",
        "values[band * 2 + j] = value;\n            values[32 + band * 2 + j] = stage[stage_index(wave + 8, lane + j * 64)];")
    qk = qk.replace("band != 5", "band != 11")
    code = code[:start] + qk + code[end:]
    start = code.index("        // Delay input additions")
    end = code.index("        // All QK stage readers", start)
    softmax = code[start:end].replace("values[", "row_values[")
    softmax = softmax.replace("bias[bias_base +", "bias[bias_base + query_half * 8 * bq +")
    softmax = softmax.replace("probability_output[row_base +", "probability_output[row_base + query_half * 8 * N +")
    code = code[:start] + """        #pragma unroll
        for (int query_half = 0; query_half < 2; ++query_half) {
            float* row_values = values + query_half * 32;
""" + softmax + "        }\n" + code[end:]
    old = """            for (int j = 0; j < 4; ++j)
                stage[stage_index(wave, lane + j * 64)] = values[band * 4 + j];"""
    new = """            for (int j = 0; j < 4; ++j) {
                stage[stage_index(wave, lane + j * 64)] = values[band * 4 + j];
                stage[stage_index(wave + 8, lane + j * 64)] = values[32 + band * 4 + j];
            }"""
    assert code.count(old) == 1
    code = code.replace(old, new)
    assert code.count("dim3(96, batch * 4), dim3(1024)") == 1
    code = code.replace("dim3(96, batch * 4), dim3(1024)", "dim3(96, batch * 4), dim3(512)")
    code = "// Experimental two-row-per-wave fused QKV; requires independent GPU gates.\n" + code
    with Path(destination).open("x") as output:
        output.write(code)
    return hashlib.sha256(code.encode()).hexdigest()


if __name__ == "__main__":
    print(generate(*sys.argv[1:]))
