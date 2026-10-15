"""Build the pinned batch HIP library and run sequential real-weight screens."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/media/IntelSSD/fireprot')
experiment = Path(sys.argv[1]).resolve()
deployment = experiment/'qualified-v3'
spec = json.loads((root/'outputs/experiments/atlaslm-speed-20261016/build-spec.json').read_text())
spec['build']['package'] = 'fireprot_plm_batch'
spec['build']['versions']['recipe'] = 'plm-batch-bf16-round-v1'
for file in spec['build']['files']:
    if file['name'] == 'source': file['path'] = str(deployment/'kernels/amd/fireprot_plm_batch.hip')
spec_path = experiment/'batch-build-spec.json'
with spec_path.open('x') as handle: json.dump(spec,handle,indent=2)
receipt_path = experiment/'batch-build-receipt.json'
with receipt_path.open('x') as handle:
    subprocess.run([str(deployment/'target/release/fireprot'),'build-hip',str(spec_path),str(experiment/'kernel-cache')],
                   cwd=root,stdout=handle,check=True)
receipt = json.loads(receipt_path.read_text())
library = Path(receipt['directory'])/next(iter(receipt['receipt']['artifacts']))
for name, profile in [('atlaslm-3b','plm-atlaslm-3b-optimized'),('esmc-fold2-6b','plm-esmc-fold2-6b-optimized'),('esm2-3b','plm-esm2-3b')]:
    cfg=json.loads((root/f'configs/diamondhill/{profile}.json').read_text())
    request=root/f'configs/requests/plm-{name}-n1000-native.json'
    r=json.loads(request.read_text())
    env=dict(os.environ,**cfg['environment'])
    env.update(HIP_VISIBLE_DEVICES='6',FIREPROT_BRIDGE_LIBRARY=str(deployment/'target/release/lib_fireprot_bridge.so'),
               PYTHONPATH=str(deployment/'python')+(':'+r['source']['root']+'/src:'+r['source']['root'] if name!='esm2-3b' else ''))
    for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'): env.pop(key,None)
    with (experiment/f'{name}-ops.log').open('x') as log:
        subprocess.run([cfg['python'],str(experiment/'batch_ops_screen.py'),str(request),str(library),str(experiment/f'{name}-ops')],
                       cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'batch operator screen passed',flush=True)
