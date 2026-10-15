"""Matched native versus opt-in complete invocation for identical N1000 B4."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/media/IntelSSD/fireprot')
experiment = Path(sys.argv[1]).resolve()
deployment = experiment/'qualified-v3'
cli = deployment/'target/release/fireprot'
build = json.loads((experiment/'batch-build-receipt.json').read_text())
library = Path(build['directory'])/next(iter(build['receipt']['artifacts']))

def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path, value):
    with path.open('x') as handle: json.dump(value,handle,indent=2);handle.write('\n')
    return pin(path)

for name, base_profile in [('atlaslm-3b','plm-atlaslm-3b-optimized'),('esmc-fold2-6b','plm-esmc-fold2-6b-optimized'),('esm2-3b','plm-esm2-3b')]:
    work = experiment/f'{name}-pairs'; work.mkdir(exist_ok=False)
    profile = json.loads((root/f'configs/diamondhill/{base_profile}.json').read_text())
    profile['workers'] = str(deployment/'python')
    profile['environment']['FIREPROT_BRIDGE_LIBRARY'] = str(deployment/'target/release/lib_fireprot_bridge.so')
    profile_pin = write(work/'profile.json',profile)
    case = experiment/'entry-qualification-v2'/f'{name}-1000-1000-1000-1000-b4-request.json'
    base_request = json.loads(case.read_text())
    if name != 'esm2-3b': base_request['resources']['hip_batch'] = pin(library)
    pairs=[]
    for index in range(3):
        pair=dict(first='candidate' if index==1 else 'baseline')
        for arm in ('baseline','candidate'):
            r=dict(base_request,backend='native' if arm=='baseline' else 'fireprot',
                   run_id=f'{name}-b4-{arm}-{index+1}',output_dir=str(work/f'{arm}-{index+1}'))
            pair[arm]=dict(request=write(work/f'{arm}-{index+1}.json',r),profile=profile_pin)
        pairs.append(pair)
    spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=1800,invocation_timeout_seconds=300,pairs=pairs),
        comparison=dict(scope='end_to_end',cache_state='Fresh processes; unflushed filesystem/GPU caches; prior B4 qualification; no output feature cache.',
            treatment_description='Identical N1000 B4 grouping: native pLM versus exact HIP batch operators and qualified checkpoint loading. ESMC full-file parallel hashing is shared.',
            expected_changes={'backend':dict(baseline='native',candidate='fireprot')}))
    write(work/'spec.json',spec)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'): env.pop(key,None)
    with (work/'pairs.log').open('x') as log:
        subprocess.run([str(cli),'benchmark-pairs',str(work/'spec.json'),str(work/'measurements')],
                       cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'B4 E2E pairs complete',flush=True)
