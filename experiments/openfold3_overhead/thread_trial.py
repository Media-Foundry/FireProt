"""Prepare a bounded three-pair CPU thread-cap trial; HIP arithmetic unchanged."""
import hashlib
import json
from pathlib import Path
import sys


def prepare(root):
    root = Path(root).absolute()
    source = root.parent / 'openfold3-hip-bias-pack-model-20261010'
    anchor = root.parent / 'openfold3-overhead-phase-queue-20261015/driver-status.json'
    result = json.loads(anchor.read_text())
    assert result['exit_code'] == 0 and result['all11_exact'] and result['full_cif_exact']
    root.mkdir(exist_ok=False)

    def save(name, value):
        p = root / name
        p.write_text(json.dumps(value, indent=2) + '\n')
        return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    profiles = {}
    for arm in ('baseline', 'candidate'):
        profile = json.loads((source / 'profile.json').read_text())
        profile['environment_inventory'] = dict(
            path=str(root.parent / 'openfold3-environment-audit-20261015/current.json'),
            sha256='14c60e2a88af9c7149b739e0af142154c8c1faf01096e014a7d04f1b26d3214f')
        assert not {'OMP_NUM_THREADS', 'MKL_NUM_THREADS'} & profile['environment'].keys()
        if arm == 'candidate':
            profile['environment'].update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        profiles[arm] = save(arm + '-profile.json', profile)
    pairs = []
    for i, first in enumerate(('baseline', 'candidate', 'baseline')):
        pair = dict(first=first)
        for arm in profiles:
            request = json.loads((source / 'request.json').read_text())
            name = f'of3-thread8-20261015-{arm}-{i}'
            request.update(run_id=name, output_dir=f'/media/IntelSSD/fireprot/outputs/{name}')
            assert request['hip_device'] == 7 and not Path(request['output_dir']).exists()
            pair[arm] = dict(request=save(name + '.json', request), profile=profiles[arm])
        pairs.append(pair)
    expected = {'invocation.profile_sha256': {a: p['sha256'] for a, p in profiles.items()}}
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        expected['launch.environment.' + name] = dict(baseline=None, candidate='8')
    save('pairs.json', dict(schema_version=1,
        execution=dict(schema_version=1, timeout_seconds=7200, invocation_timeout_seconds=1200, pairs=pairs),
        comparison=dict(scope='end_to_end',
            cache_state='Fresh processes; unflushed caches; HIP7 only; AB/BA/AB; all observations retained.',
            treatment_description='Qualified strict FP32 N1536 R3 D200 S1 seed101 route: inherited CPU threads versus OMP/MKL=8. No operator, inventory, cycle or precision changes.',
            expected_changes=expected)))
    save('preparation.json', dict(source=str(source), phase_anchor=str(anchor),
        reference=json.loads((source / 'collected.json').read_text()),
        cli=json.loads((source / 'preparation.json').read_text())['cli'],
        scope='Final bounded OpenFold3 overhead trial before ESMFold integration.'))


if __name__ == '__main__':
    prepare(*sys.argv[1:])
