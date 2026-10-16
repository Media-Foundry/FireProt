"""Build the stage-timing HIP library with Rust, then run isolated HIP6 probes."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path('/media/IntelSSD/fireprot')
work = Path(sys.argv[1]).resolve()
deployment = root/'outputs/experiments/esmc-startup-20261016/native/qualified-import-overlap'
spec = json.loads((root/'outputs/experiments/esmc-gemm-order-20261016/lt-build-spec.json').read_text())
spec['build']['package'] = 'esmc_lt_serialized'
spec['build']['versions']['recipe'] = 'esmc-lt-serialized-stage-v1'
for item in spec['build']['files']:
    if item['name'] == 'source':
        item['path'] = str(work/'esmc_lt_serialized.hip')
if len(sys.argv) > 2:
    previous = Path(sys.argv[2]).resolve()
    assert (previous/'esmc_lt_serialized.hip').read_bytes() == (work/'esmc_lt_serialized.hip').read_bytes()
    for name in ('build-spec.json', 'build-receipt.json'):
        assert not (work/name).exists()
        shutil.copy2(previous/name, work/name)
else:
    with (work/'build-spec.json').open('x') as handle:
        json.dump(spec, handle, indent=2)
    with (work/'build-receipt.json').open('x') as handle:
        subprocess.run([str(deployment/'target/release/fireprot'), 'build-hip', str(work/'build-spec.json'), str(work/'kernel-cache')],
                       cwd=root, stdout=handle, check=True)
receipt = json.loads((work/'build-receipt.json').read_text())
library = Path(receipt['directory'])/next(iter(receipt['receipt']['artifacts']))
assert hashlib.sha256(library.read_bytes()).hexdigest() == receipt['receipt']['artifacts'][library.name]
profile = json.loads((root/'configs/diamondhill/plm-esmc-fold2-6b-startup.json').read_text())
request_path = root/'configs/requests/plm-esmc-fold2-6b-b4-n1000-exact.json'
request = json.loads(request_path.read_text())
env = dict(os.environ, **profile['environment'], HIP_VISIBLE_DEVICES='6',
           PYTHONPATH=f"{deployment/'python'}:{request['source']['root']}")
for name in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
    env.pop(name, None)
for mode in ('index-1', 'restore-1', 'index-2', 'restore-2'):
    command = [profile['python'], str(work/'probe_esmc_lt_startup.py'), str(request_path), str(library), str(work/mode)]
    if mode.startswith('restore'):
        command.append(str(work/'index-1/algorithm.bin'))
    with (work/f'{mode}.log').open('x') as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((work/mode/'report.json').read_text())
    if report['status'] == 'rejected':
        assert mode.startswith('restore') and report['return_code'] == -9
    print(mode, report['status'], flush=True)
