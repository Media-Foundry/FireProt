"""Run prepared snapshot comparisons only after real isolation qualification."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def check(pin):
    assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']


def main():
    root=Path('/media/IntelSSD/fireprot');work=Path(sys.argv[1]).resolve()
    assert json.loads((work/'qualification.json').read_text())['passed']
    assert json.loads((work/'isolation-check/report.json').read_text())['passed']
    pins=[json.loads((work/f'{name}.json').read_text()) for name in ('runtime-library','benchmark-cli')]
    pins.extend(json.loads((work/'binaries.json').read_text()).values())
    env=dict(os.environ,HIP_VISIBLE_DEVICES='6')
    for key in ('ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'):env.pop(key,None)
    for count in (4,64):
        for pin in pins:check(pin)
        case=work/f'b{count}-pairs'
        with (case/'pairs.log').open('x') as log:
            subprocess.run([str(work/'fireprot-bench'),'benchmark-pairs',str(case/'spec.json'),str(case/'measurements')],
                           cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for pin in pins:check(pin)
        print(case.name,'complete',flush=True)


if __name__=='__main__':main()
