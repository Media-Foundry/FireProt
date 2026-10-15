"""Formal N1000 AtlasLM loading + HIP RoPE/SwiGLU E2E comparison.

Use a new frozen worker/kernel/CLI/bridge directory; all timing, device isolation,
artifact verification and pairwise numerical gates are performed by Rust.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
build = json.loads(Path(sys.argv[2]).read_text())
base = Path('/media/IntelSSD/fireprot')
exe = root/'target/release/fireprot'
bridge = root/'target/release/lib_fireprot_bridge.so'
library = Path(build['directory'])/next(iter(build['receipt']['artifacts']))

def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n')
    return pin(path)

assert pin(library)['sha256'] == build['receipt']['artifacts'][library.name]
profile = json.loads((base/'configs/diamondhill/plm-atlaslm-3b.json').read_text())
profile['workers'] = str(root/'python')
profile['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(bridge)
profile_pin = write(root/'profile.json', profile)
pairs = []
for index in range(3):
    pair = dict(first='candidate' if index == 1 else 'baseline')
    for arm in ['baseline','candidate']:
        r = json.loads((base/'configs/requests/plm-atlaslm-3b-n1000-fireprot.json').read_text())
        r['backend'] = 'native' if arm == 'baseline' else 'fireprot'
        r['hip_device'] = 6
        r['resources']['hip_swiglu'] = pin(library)
        r['run_id'] = f'atlaslm-compose-{arm}-{index+1}'
        r['output_dir'] = str(root/r['run_id'])
        pair[arm] = dict(request=write(root/f'{r["run_id"]}.json',r), profile=profile_pin)
    pairs.append(pair)
spec = dict(schema_version=1,
    execution=dict(schema_version=1,timeout_seconds=1200,invocation_timeout_seconds=300,pairs=pairs),
    comparison=dict(scope='end_to_end',
        cache_state='Fresh processes; filesystem/GPU library caches unflushed; prior N1000 qualification; no output feature cache.',
        treatment_description='Native AtlasLM loader/operators versus eager GPU checkpoint restoration, shared HIP RoPE preparation, and exact BF16 HIP SwiGLU; native masks and GEMMs retained.',
        expected_changes={
            'backend':dict(baseline='native',candidate='fireprot'),
            'launch.inputs.rust-bridge':dict(baseline=None,candidate=dict(name='rust-bridge',canonical_path=str(bridge),sha256=pin(bridge)['sha256'])),
        }))
write(root/'atlaslm-3b-pairs.json',spec)
env = dict(os.environ,HIP_VISIBLE_DEVICES='6')
for name in ['CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']: env.pop(name,None)
with (root/'pairs.log').open('x') as log:
    subprocess.run([str(exe),'benchmark-pairs',str(root/'atlaslm-3b-pairs.json'),str(root/'atlaslm-3b-measurements')],
                   cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
