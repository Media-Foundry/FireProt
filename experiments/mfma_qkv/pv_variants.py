"""Reproduce bounded PV scheduling variants of the exact index32 candidate.

Only loop unrolling and load prefetch grouping change. The ascending MFMA
fragment sequence is unchanged. GPU numerical/timing gates are still required.
"""
from pathlib import Path
import hashlib
import json
import sys

BASE_SHA256 = '12467c8bb72ab7786039a864bc463a4cac5750ad0ba9fc8778eb1b0dd7bc8a24'

def generate(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != BASE_SHA256:
        raise ValueError('Expected the frozen qualified index32 source')
    baseline = data.decode()
    destination.mkdir(exist_ok=False)
    loop = '#pragma unroll 1\n                for (int group = 0; group < 16; ++group)'
    assert baseline.count(loop) == 1
    variants = {
        'unroll4': baseline.replace(loop, loop.replace('unroll 1', 'unroll 4')),
        'unroll16': baseline.replace(loop, loop.replace('unroll 1', 'unroll 16')),
    }
    start = baseline.index('                    float av[4], bv[4];')
    end = baseline.index('\n                }', start)
    segment = baseline[start:end]
    assert segment.count('__builtin_amdgcn_mfma_f32_16x16x4f32') == 1
    replacement = '''                    #pragma unroll
                    for (int slot = 0; slot < 4; ++slot) {
                        const int fragment = group * 4 + slot;
                        const int local_k = (fragment / 2) * 8 + 2 * k_lane + (fragment % 2);
                        const float av = value_matrix[(matrix * N + band * 256 + local_k) * 32
                                                      + feature_base + coordinate];
                        const float bv = stage[stage_index(coordinate, local_k)];
                        pv_accumulator = __builtin_amdgcn_mfma_f32_16x16x4f32(
                            av, bv, pv_accumulator, 0, 0, 0);
                    }'''
    variants['prefetch1'] = baseline[:start] + replacement + baseline[end:]
    records = []
    for name, code in variants.items():
        path = destination / (name + '.cpp')
        path.write_text(code)
        records.append(dict(name=name, source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(baseline_sha256=BASE_SHA256, variants=records,
                  scope='Source generation only; no new numerical or speedup claim')
    (destination / 'sources.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

if __name__ == '__main__':
    print(json.dumps(generate(*sys.argv[1:]), indent=2))
