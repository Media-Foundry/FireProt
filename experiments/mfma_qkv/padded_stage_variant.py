"""Generate bounded LDS row-padding trials from the qualified FP32 QKV source.

Arithmetic, reduction order, block mapping and synchronization are unchanged.
Only physical shared-memory row pitch changes; numerical and trained-module
gates remain mandatory. Padding is a scheduling hypothesis, not a speedup claim.
"""
import hashlib
import json
from pathlib import Path
import sys

BASE_SHA256 = 'dd7bbb785dae1afed98ad6fd357fa8e6822f0e49310a87530a296ee7c0d9ff9f'


def generate(source, destination):
    data = Path(source).read_bytes()
    if hashlib.sha256(data).hexdigest() != BASE_SHA256:
        raise ValueError('Expected the qualified unroll4 source')
    baseline = data.decode()
    root = Path(destination)
    root.mkdir(exist_ok=False)
    records = []
    for pitch in (264, 272):
        # Every logical value has one physical slot. Four-dword stores stay
        # aligned, and no used address enters the padding of another row.
        addresses = {q * pitch + (k ^ ((q & 3) | ((q & 12) << 1)))
                     for q in range(16) for k in range(256)}
        assert len(addresses) == 16 * 256 and max(addresses) < 16 * pitch
        assert pitch % 4 == 0
        code = baseline
        for old, new in (
            ('return query * 256 +', f'return query * {pitch} +'),
            ('float stage[16 * 256];', f'float stage[16 * {pitch}];'),
        ):
            assert code.count(old) == 1, old
            code = code.replace(old, new)
        path = root / f'pitch{pitch}.cpp'
        path.write_text(code)
        records.append(dict(pitch=pitch, shared_bytes=16 * pitch * 4,
                            path=str(path), sha256=hashlib.sha256(code.encode()).hexdigest()))
    result = dict(base_sha256=BASE_SHA256, variants=records,
                  scope='Source generation and CPU address checks only; no GPU or speedup verdict.')
    (root / 'sources.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(generate(*sys.argv[1:]), indent=2))
