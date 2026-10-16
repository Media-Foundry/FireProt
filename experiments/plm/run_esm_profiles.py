"""Run ESM profiling sequentially on HIP6 using frozen qualified loaders."""
import json
import os
from pathlib import Path
import subprocess

root = Path('/media/IntelSSD/fireprot')
work = root/'outputs/experiments/esm-speed-20261016'
for name in ('esm3-open', 'esmc-fold2-6b', 'esm2-3b'):
    cfg = json.loads((root/f'configs/diamondhill/plm-{name}-batch.json').read_text())
    request = root/f'configs/requests/plm-{name}-n1000-native.json'
    req = json.loads(request.read_text())
    env = dict(os.environ, **cfg['environment'])
    env.update(HIP_VISIBLE_DEVICES='6', PYTHONPATH=cfg['workers']+
               (':'+req['source']['root'] if name!='esm2-3b' else ''))
    for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'): env.pop(key,None)
    with (work/f'{name}-profile.log').open('x') as log:
        subprocess.run([cfg['python'],str(work/'profile_esm.py'),str(request),str(work/f'{name}-profile')],
                       cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(name,'profile complete',flush=True)
