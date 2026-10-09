"""Check the address reconstruction from the SHA-pinned native PV main loop.

This is an integer address audit, not floating-point or GPU qualification.
Formula anchors: 0x44f234..0x44f300, stores 0x44f61c..0x44f644,
reads and MFMA operands 0x44f69c..0x44f8ac in the archived PV excerpt.
"""
from pathlib import Path
import hashlib
import json
import sys


def main(excerpt, destination):
    excerpt, destination = Path(excerpt), Path(destination)
    repository = Path(__file__).resolve().parents[2]
    evidence = json.loads((repository / 'provenance/openfold3_native_mfma_review.json').read_text())
    digest = hashlib.sha256(excerpt.read_bytes()).hexdigest()
    assert digest == evidence['pv']['disassembly_sha256']
    a, b = {}, {}
    for thread in range(256):
        wave, lane = divmod(thread, 64)
        klane, coordinate = divmod(lane, 16)
        for delta_k in (0, 4):
            for component in (0, 1):
                k = wave * 8 + klane + delta_k
                feature = 2 * coordinate + component
                address = 4 * (k * 32 + feature)
                assert address not in a
                a[address] = (k, feature)
        for delta_q in (0, 4, 8, 12):
            for component in (0, 1):
                query = wave * 16 + klane + delta_q
                k = 2 * coordinate + component
                address = 4096 + 136 * query + 4 * k
                assert address not in b
                b[address] = (k, query)
    assert len(a) == 32 * 32 and len(b) == 64 * 32
    offsets_a = (0, 128, 1024, 1152, 2048, 2176, 3072, 3200)
    offsets_b = (0, 4, 32, 36, 64, 68, 96, 100)
    checks = 0
    for thread in range(256):
        wave, lane = divmod(thread, 64)
        klane, coordinate = divmod(lane, 16)
        read_a = 4 * (coordinate + 64 * klane + 16 * (wave & 1))
        read_b = 4096 + 136 * (coordinate + 16 * (wave // 2)) + 8 * klane
        for accumulator_group in (0, 1):
            for fragment, (oa, ob) in enumerate(zip(offsets_a, offsets_b)):
                ak, feature = a[read_a + oa]
                bk, query = b[read_b + ob + accumulator_group * 4352]
                expected_k = (fragment // 2) * 8 + 2 * klane + fragment % 2
                assert ak == bk == expected_k
                assert feature == coordinate + 16 * (wave & 1)
                assert query == coordinate + 16 * (wave // 2) + 32 * accumulator_group
                checks += 1
    report = dict(
        scope='Static integer LDS address reconstruction only; no GPU arithmetic proof',
        excerpt=dict(path=str(excerpt), sha256=digest),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        operand_address_checks=checks,
        depth_u=32,
        per_instruction_k_lanes=[[(f // 2) * 8 + 2 * lane + f % 2 for lane in range(4)] for f in range(8)],
        hypothesis='Even K then odd K within each group of eight',
        limits=['Main-loop address mapping does not prove complete launch argument identities.',
                'Instruction arithmetic and output bits still require the queued GPU comparisons.'])
    with destination.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(*sys.argv[1:])
