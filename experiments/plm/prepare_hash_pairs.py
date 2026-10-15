"""Prepare and execute the ESMC parallel-verification E2E comparison.

Run in a fresh frozen source/build snapshot on DiamondHill after building the
CLI and bridge. The native/candidate pair shares source, weights and all outputs.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
base = Path('/media/IntelSSD/fireprot')
exe = root/'target/release/fireprot'
bridge = root/'target/release/lib_fireprot_bridge.so'

def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')
    return pin(path)

profiles = {}
for arm in ['baseline', 'candidate']:
    profile = json.loads((base/'configs/diamondhill/plm-esmc-fold2-6b.json').read_text())
    profile['workers'] = str(root/'python')
    profile['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(bridge)
    if arm == 'candidate':
        profile['environment']['FIREPROT_PREFLIGHT_HASH_THREADS'] = '4'
    profiles[arm] = write(root/f'{arm}-profile.json', profile)

pairs = []
for index in range(3):
    pair = dict(first='candidate' if index == 1 else 'baseline')
    for arm in ['baseline', 'candidate']:
        request = json.loads((base/'configs/requests/plm-esmc-fold2-6b-n1000-fireprot.json').read_text())
        request['backend'] = 'native' if arm == 'baseline' else 'fireprot'
        request['hip_device'] = 6
        request['run_id'] = f'esmc-hash-{arm}-{index+1}'
        request['output_dir'] = str(root/request['run_id'])
        pair[arm] = dict(request=write(root/f'{request["run_id"]}.json', request), profile=profiles[arm])
    pairs.append(pair)
spec = dict(schema_version=1,
    execution=dict(schema_version=1, timeout_seconds=1800, invocation_timeout_seconds=300, pairs=pairs),
    comparison=dict(scope='end_to_end',
        cache_state='Fresh processes; filesystem/GPU library caches unflushed; prior N1000 qualification; no output feature cache.',
        treatment_description='Native serial full-asset SHA-256 verification versus four-thread full-asset verification plus qualified HIP RoPE. No hashes or model outputs omitted.',
        expected_changes={
            'backend': dict(baseline='native', candidate='fireprot'),
            'launch.inputs.rust-bridge': dict(baseline=None, candidate=dict(name='rust-bridge', canonical_path=str(bridge), sha256=pin(bridge)['sha256'])),
            'launch.environment.FIREPROT_PREFLIGHT_HASH_THREADS': dict(baseline=None, candidate='4'),
            'invocation.profile_sha256': {arm: profiles[arm]['sha256'] for arm in profiles},
        }))
write(root/'esmc-fold2-6b-pairs.json', spec)
env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
for name in ['CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']:
    env.pop(name, None)
with (root/'pairs.log').open('x') as log:
    subprocess.run([str(exe), 'benchmark-pairs', str(root/'esmc-fold2-6b-pairs.json'),
                    str(root/'esmc-fold2-6b-measurements')], cwd=root, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
