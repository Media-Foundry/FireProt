"""Refresh the reviewed inventory equally in both arms, retaining the failed run."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    return pin(path)


root = Path(sys.argv[1])
parent = int(sys.argv[2])
proc = Path('/proc') / str(parent)
assert b'/tmp/fireprot-e2e-refresh-run.py' in (proc / 'cmdline').read_bytes()
assert (proc / 'stat').read_text().split(') ', 1)[1].split()[0] == 'T'
try:
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().split(') ', 1)[1].split()
        except FileNotFoundError:
            continue
        assert int(fields[1]) != parent or fields[0] == 'Z', 'Active benchmark child'
    audit = root / 'openfold3-environment-refresh'
    diff = json.loads((audit / 'diff.json').read_text())
    inventory = pin(audit / 'current.json')
    assert inventory == diff['new_inventory']
    assert diff['same_specification'] and diff['old_files'] == diff['new_files'] == 157886
    assert len(diff['added']) == len(diff['removed']) == 14
    assert len(diff['changed']) == 58
    # Reviewed categories: host OS libraries/loader and the unused top-level bridge.
    for item in diff['added'] + diff['removed']:
        assert item['name'].startswith(('host/system-libraries/', 'host/system-libraries-alias/'))
    for item in diff['changed']:
        assert item['name'].startswith(('host/system-libraries/', 'host/system-libraries-alias/')) or item['name'] in ('host/loader-cache', 'experimental/rust-bridge')
    old = root / 'openfold3-n1536'
    case = root / 'openfold3-n1536-refreshed-env'
    case.mkdir(exist_ok=False)
    spec = json.loads((old / 'spec.json').read_text())
    profiles = {}
    for arm in ('baseline', 'candidate'):
        profile = json.loads((old / f'{arm}-profile.json').read_text())
        assert profile['environment_inventory'] == diff['old_inventory']
        profile['environment_inventory'] = inventory
        profiles[arm] = write(case / f'{arm}-profile.json', profile)
    for i, pair in enumerate(spec['execution']['pairs']):
        for arm in profiles:
            request = json.loads((old / f'{arm}-{i}.json').read_text())
            assert request['hip_device'] == 6
            request.update(run_id=f'e2e-refresh-20261016-openfold3-env-{arm}-{i}', output_dir=str(case / f'{arm}-{i}'))
            pair[arm] = dict(request=write(case / f'{arm}-{i}.json', request), profile=profiles[arm])
    spec['comparison']['expected_changes']['invocation.profile_sha256'] = {a:p['sha256'] for a,p in profiles.items()}
    spec['comparison']['treatment_description'] += ' Both arms repin the same fully audited environment after host updates; unchanged coverage and numerical gates.'
    specpin = write(case / 'spec.json', spec)
    write(case / 'preparation.json', dict(original_spec=pin(old/'spec.json'), audit=pin(audit/'diff.json'), inventory=inventory, preparer=pin(Path(__file__)), review='All changes are host OS libraries/loader and the unused top-level bridge; model environments and qualified operator bundles are unchanged.'))
    planpath = root / 'plan.json'
    plan = json.loads(planpath.read_text())
    assert all(c['name'] != case.name for c in plan['cases'])
    plan['cases'].append(dict(name=case.name, spec=specpin, replaces_failed='openfold3-n1536', policy='Unchanged Rust gates plus independent exact-output verification'))
    write(root / 'plan-before-openfold3-retry.json', json.loads(planpath.read_text()))
    planpath.write_text(json.dumps(plan, indent=2)+'\n')
    assert pin(Path(plan['cli']['path'])) == plan['cli']
    env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
    for key in ('ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
        env.pop(key, None)
    print('Starting OpenFold3 with refreshed full inventory', flush=True)
    with (case / 'driver.log').open('x') as log:
        result = subprocess.run([plan['cli']['path'], 'benchmark-pairs', str(case/'spec.json'), str(case/'measurements')], cwd='/media/IntelSSD/fireprot', env=env, stdout=log, stderr=subprocess.STDOUT)
    write(case / 'driver-status.json', dict(exit_code=result.returncode))
    print('OpenFold3 retry exit', result.returncode, flush=True)
finally:
    if proc.exists():
        os.kill(parent, signal.SIGCONT)
        print('Main folding queue resumed', flush=True)
