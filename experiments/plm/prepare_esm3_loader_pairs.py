"""Measure the official ESM3 factory against the existing local mmap loader.

Both arms use native Torch math, BF16, all heads, and complete Rust acceptance.
Only the offline loading route differs. Run in a fresh frozen deployment.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
base = Path('/media/IntelSSD/fireprot')

def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

profiles = {}
for arm, flag in [('baseline', '1'), ('candidate', '0')]:
    profile = json.loads((base/'configs/diamondhill/plm-esm3-open.json').read_text())
    profile['workers'] = str(root/'python')
    profile['environment']['FIREPROT_ESM3_REFERENCE_FACTORY'] = flag
    profiles[arm] = write(root/f'{arm}-profile.json', profile)
pairs = []
for index in range(3):
    pair = dict(first='candidate' if index == 1 else 'baseline')
    for arm in ['baseline', 'candidate']:
        request = json.loads((base/'configs/requests/plm-esm3-open-n1000-native.json').read_text())
        request['hip_device'] = 6
        request['run_id'] = f'esm3-loader-{arm}-{index+1}'
        request['output_dir'] = str(root/request['run_id'])
        pair[arm] = dict(request=write(root/f'{request["run_id"]}.json', request), profile=profiles[arm])
    pairs.append(pair)
spec = dict(schema_version=1,
    execution=dict(schema_version=1, timeout_seconds=1200, invocation_timeout_seconds=300, pairs=pairs),
    comparison=dict(scope='end_to_end',
        cache_state='Fresh processes; filesystem/GPU library caches unflushed; prior N1000 qualification; no output feature cache.',
        treatment_description='Official ESM3.from_pretrained factory with data_root routed to the pinned local checkpoint versus existing FireProt strict mmap/assign loader. Native Torch BF16 math and all heads retained.',
        expected_changes={
            'launch.environment.FIREPROT_ESM3_REFERENCE_FACTORY': dict(baseline='1', candidate='0'),
            'invocation.profile_sha256': {arm: profiles[arm]['sha256'] for arm in profiles},
        }))
write(root/'esm3-open-pairs.json', spec)
env = dict(os.environ, HIP_VISIBLE_DEVICES='6')
for name in ['CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']:
    env.pop(name, None)
with (root/'pairs.log').open('x') as log:
    subprocess.run([str(root/'target/release/fireprot'), 'benchmark-pairs',
                    str(root/'esm3-open-pairs.json'), str(root/'esm3-open-measurements')],
                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
