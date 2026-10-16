"""Complete Rust invocation AB/BA/AB comparisons for exact ESM candidates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/media/IntelSSD/fireprot')
work=root/'outputs/experiments/esm-speed-20261016'
deployment=Path(sys.argv[1]).resolve()
models=sys.argv[2:] or ['esm3-open','esmc-fold2-6b','esm2-3b','esm3-open-resident64']
build=json.loads((work/'trials-v2/build-receipt.json').read_text())
library=Path(build['directory'])/next(iter(build['receipt']['artifacts']))

def pin(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    return pin(path)

for label in models:
    name=label.removesuffix('-resident64')
    count=64 if label.endswith('-resident64') else 4
    case=work/f'{label}-pairs';case.mkdir(exist_ok=False)
    profile=json.loads((root/f'configs/diamondhill/plm-{name}-batch.json').read_text())
    profile['workers']=str(deployment/'python')
    profile['environment']['FIREPROT_BRIDGE_LIBRARY']=str(deployment/'target/release/lib_fireprot_bridge.so')
    profile_pin=write(case/'profile.json',profile)
    request=json.loads((root/f'configs/requests/plm-{name}-b4-n1000-native.json').read_text())
    request['resources']['hip_esm']=pin(library)
    if count==64:
        original=json.loads(Path(request['input']['path']).read_text())['sequences'][0]['sequence']
        sequences=[dict(id=f'item-{i}',sequence=original[(i*137)%1000:]+original[:(i*137)%1000]) for i in range(count)]
        request['input']=write(case/'input.json',dict(schema_version=1,sequences=sequences))
        request['workload']['batch']['items']=count
    pairs=[]
    for index in range(3):
        pair=dict(first='candidate' if index==1 else 'baseline')
        for arm in ('baseline','candidate'):
            value=dict(request,backend='native' if arm=='baseline' else 'fireprot',hip_device=6,
                       run_id=f'{label}-exact-{arm}-{index+1}',output_dir=str(case/f'{arm}-{index+1}'))
            pair[arm]=dict(request=write(case/f'{arm}-{index+1}.json',value),profile=profile_pin)
        pairs.append(pair)
    spec=dict(schema_version=1,execution=dict(schema_version=1,timeout_seconds=3600,invocation_timeout_seconds=600,pairs=pairs),
        comparison=dict(scope='end_to_end',cache_state='Fresh processes, published HIP; unflushed filesystem/GPU caches; no output cache.',
            treatment_description=f'Official same-grouping native {name} versus qualified ESM HIP/sequence-only geometry simplification and checkpoint loading. N1000 x {count}, B4, seed101, native precision, all heads retained.',
            expected_changes={'backend':dict(baseline='native',candidate='fireprot')}))
    write(case/'spec.json',spec)
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(key,None)
    with (case/'pairs.log').open('x') as log:
        subprocess.run([str(deployment/'target/release/fireprot'),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                       cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(label,'paired invocation complete',flush=True)
