"""Sequential maintained-scope and BLAS screens; HIP6 only."""
import json
import os
from pathlib import Path
import subprocess

root=Path('/media/IntelSSD/fireprot')
work=root/'outputs/experiments/esm-speed-20261016'
deployment=work/'qualified-v1'
receipt=json.loads((work/'trials-v2/build-receipt.json').read_text())
library=Path(receipt['directory'])/next(iter(receipt['receipt']['artifacts']))
for name in ('esm3-open','esmc-fold2-6b','esm2-3b'):
    cfg=json.loads((root/f'configs/diamondhill/plm-{name}-batch.json').read_text())
    request=root/f'configs/requests/plm-{name}-n1000-native.json'
    req=json.loads(request.read_text())
    env=dict(os.environ,**cfg['environment'])
    env.update(HIP_VISIBLE_DEVICES='6',FIREPROT_BRIDGE_LIBRARY=str(deployment/'target/release/lib_fireprot_bridge.so'),
        PYTHONPATH=str(deployment/'python')+(':'+req['source']['root'] if name!='esm2-3b' else ''))
    for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(key,None)
    if name=='esm3-open':
        subprocess.run([cfg['python'],str(deployment/'tests/adapters/test_esm_frameless.py')],env=env,cwd=deployment,check=True)
    for kind,script,args in (
        ('packaged','screen_esm_packaged.py',[str(request),str(library),str(work/f'{name}-packaged')]),
        ('blas','screen_esm_blas.py',[str(request),str(work/f'{name}-blas')])):
        with (work/f'{name}-{kind}.log').open('x') as log:
            subprocess.run([cfg['python'],str(work/script),*args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name,kind,'complete',flush=True)
