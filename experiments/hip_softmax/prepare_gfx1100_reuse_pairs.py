"""Prepare Rust paired execution after all three gfx1100 reuse gates pass."""
import argparse
import hashlib
import json
from pathlib import Path


def pin(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def prepare(root):
    root = Path(root).resolve()
    audit = read(root / 'audit-cycle3.json')
    assert audit['all_passed'] and len(audit['records']) == 3
    assert {r['mode'] for r in audit['records']} == {1, 2, 3}
    for item in read(root / 'preparation.json')['files']:
        assert pin(Path(item['path']))['sha256'] == item['sha256'], item['path']
    profiles = {arm: pin(root / f'profile-{mode}.json')
                for arm, mode in [('baseline', 0), ('candidate', 3)]}
    template = read(root / 'request-0.json')
    pairs = []
    for index in range(3):
        pair = dict(first='candidate' if index % 2 else 'baseline')
        for arm in ('baseline', 'candidate'):
            request = dict(template, run_id=f'{root.name}-paired-{arm}-{index}',
                           output_dir=str(root / f'prediction-paired-{arm}-{index}'))
            path = root / f'request-paired-{arm}-{index}.json'
            write(path, request)
            pair[arm] = dict(request=pin(path), profile=profiles[arm])
        pairs.append(pair)
    write(root / 'pairs.json', dict(schema_version=1,
        execution=dict(schema_version=1, timeout_seconds=10800,
                       invocation_timeout_seconds=1800, pairs=pairs),
        comparison=dict(scope='end_to_end',
            cache_state='Fresh processes; filesystem/GPU caches unflushed; same prebuilt wave32 library; no asset transfers; clocks not fixed',
            treatment_description='Previous gfx1100 softmax versus plus rollout-local pair/atom and exact duplicate-template reuse; strict FP32 N1000 R3/D200/S1',
            expected_changes={
                'launch.environment.FIREPROT_GFX1100_REUSE_MODE': dict(baseline='0', candidate='3'),
                'invocation.profile_sha256': {arm: p['sha256'] for arm, p in profiles.items()}})))
    write(root / 'paired-preparation.json', dict(schema_version=1,
        cycles_audit=pin(root / 'audit-cycle3.json'),
        frozen_worker_preparation=pin(root / 'preparation.json'),
        specification=pin(root / 'pairs.json'), preparer=pin(Path(__file__)),
        boundary='Incremental comparison against previous softmax, not direct native speedup'))
    print(json.dumps(pin(root / 'pairs.json'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    prepare(parser.parse_args().root)
