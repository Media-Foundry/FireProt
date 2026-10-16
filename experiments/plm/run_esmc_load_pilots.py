"""Run serial fresh-process loader diagnostics on HIP6 after exact qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys


root=Path('/media/IntelSSD/fireprot')
work=Path(sys.argv[1]).resolve()
qualification=json.loads((work/'qualify/report.json').read_text())
assert len(qualification['cases'])==2
assert all(not c['differing_parameters'] and all(c['output_exact'].values()) for c in qualification['cases'])
profile=json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-gpu-loading.json').read_text())
request=root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json'
req=json.loads(request.read_text())
env=dict(os.environ,**profile['environment'])
env.update(HIP_VISIBLE_DEVICES='6',PYTHONPATH=profile['workers']+':'+req['source']['root'])
for key in ('CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):env.pop(key,None)
for mode in ('gpu_bf16','cpu_bf16','pinned_bf16'):
    with (work/f'pilot-{mode}.log').open('x') as log:
        subprocess.run([profile['python'],str(work/'screen_esmc_cpu_loading.py'),str(request),mode,
                        str(work/f'pilot-{mode}'),str(work/'qualify/reference.npz')],
                       env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(json.loads((work/f'pilot-{mode}/report.json').read_text())),flush=True)
