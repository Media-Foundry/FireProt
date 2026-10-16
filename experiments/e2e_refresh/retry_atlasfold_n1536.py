"""Correct the bridge declaration and queue fresh AtlasFold pairs after Base/B4."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    return pin(path)


root = Path(sys.argv[1])
old = root / 'atlasfold-n1536'
spec = json.loads((old / 'spec.json').read_text())
bridges = {}
for arm in ('baseline', 'candidate'):
    execution = json.loads((old / f'{arm}-0/execution.json').read_text())
    bridges[arm] = next(x for x in execution['inputs'] if x['name'] == 'rust-bridge')
assert bridges['baseline'] == bridges['candidate']
bridge = bridges['baseline']
assert pin(Path(bridge['canonical_path']))['sha256'] == bridge['sha256']
declaration = spec['comparison']['expected_changes'].pop('launch.inputs.rust-bridge')
assert declaration == dict(baseline=None, candidate=bridge)
assert spec['comparison']['expected_changes'] == {'backend': {'baseline': 'chunked', 'candidate': 'hip_fused'}}
case = root / 'atlasfold-n1536-declaration-retry'
case.mkdir(exist_ok=False)
for arm in ('baseline', 'candidate'):
    shutil.copy2(old / f'{arm}-profile.json', case / f'{arm}-profile.json')
for i, pair in enumerate(spec['execution']['pairs']):
    for arm in ('baseline', 'candidate'):
        request = json.loads((old / f'{arm}-{i}.json').read_text())
        assert request['hip_device'] == 6 and request['seed'] == 102
        request.update(run_id=f'e2e-refresh-20261016-atlasfold1536-retry-{arm}-{i}', output_dir=str(case / f'{arm}-{i}'))
        pair[arm] = dict(request=write(case / f'{arm}-{i}.json', request), profile=pin(case / f'{arm}-profile.json'))
spec['comparison']['treatment_description'] += ' Corrected declaration: the chunked reference already loads the identical Rust policy bridge. All numerical and protected-condition checks remain unchanged.'
specpin = write(case / 'spec.json', spec)
write(case / 'preparation.json', dict(failed_spec=pin(old / 'spec.json'), actual_execution={a:pin(old / f'{a}-0/execution.json') for a in bridges}, common_bridge=bridge, preparer=pin(Path(__file__)), correction='Remove an incorrectly declared bridge difference; do not change either model route or its numerical gates. Rerun all three pairs into fresh outputs.'))
planpath = root / 'plan.json'
plan = json.loads(planpath.read_text())
assert all(c['name'] != case.name for c in plan['cases'])
write(root / 'plan-before-atlasfold-retry.json', plan)
plan['cases'].append(dict(name=case.name, spec=specpin, replaces_failed='atlasfold-n1536', policy='Unchanged strict numerical and protected-condition gates'))
planpath.write_text(json.dumps(plan, indent=2)+'\n')
print('Prepared corrected AtlasFold N1536; waiting for Base driver completion (its script also queues AtlasLM B4)', flush=True)
base_status = root / 'base-n1000/driver-status.json'
while not base_status.exists():
    time.sleep(2)
print('Base driver completed:', base_status.read_text().strip(), flush=True)
assert pin(Path(plan['cli']['path'])) == plan['cli']
env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
for key in ('ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
    env.pop(key, None)
print('Starting corrected AtlasFold N1536 three-pair study', flush=True)
with (case / 'driver.log').open('x') as log:
    result = subprocess.run([plan['cli']['path'], 'benchmark-pairs', str(case / 'spec.json'), str(case / 'measurements')], cwd='/media/IntelSSD/fireprot', env=env, stdout=log, stderr=subprocess.STDOUT)
write(case / 'driver-status.json', dict(exit_code=result.returncode))
print('AtlasFold retry finished:', result.returncode, flush=True)
