"""Synthetic CLI transport fixture; no Torch, model, HIP or scientific inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket

assert os.environ['HIP_VISIBLE_DEVICES'] == '7'
assert 'ROCR_VISIBLE_DEVICES' not in os.environ
assert 'CUDA_VISIBLE_DEVICES' not in os.environ
assert 'GPU_DEVICE_ORDINAL' not in os.environ
assert os.environ['PYTHONNOUSERSITE'] == '1'
request = json.loads(Path(os.environ['FIREPROT_REQUEST']).read_text())
root = Path(os.environ['FIREPROT_OUTPUT_DIR'])
sock = socket.socket(socket.AF_UNIX)
sock.connect(os.environ['FIREPROT_WORKER_SOCKET'])

def emit(sequence, payload):
    sock.sendall((json.dumps(dict(schema_version=1, run_id=request['run_id'],
                                 sequence=sequence, payload=payload)) + '\n').encode())

emit(0, dict(event='ready', device=dict(worker_pid=os.getpid(), logical_device=0,
                                      pci_address='0000:93:00.0', architecture='gfx90a')))
assert json.loads(sock.makefile().readline())['command'] == 'start'
fixture = Path(os.environ['FIREPROT_TEST_FIXTURE'])
bundle = json.loads((fixture / 'predictions.json').read_text())
bundle.update(run_id=request['run_id'], seed=request['seed'], workload=request['workload'])
if os.environ.get('FIREPROT_TEST_INVALID') == '1':
    bundle['seed'] += 1
(root / 'predictions.json').write_text(json.dumps(bundle))
shutil.copyfile(fixture / 'sample-0.npz', root / 'sample-0.npz')
artifacts = [dict(path=name, sha256=hashlib.sha256((root / name).read_bytes()).hexdigest())
             for name in ['predictions.json', 'sample-0.npz']]
emit(1, dict(event='outputs', artifacts=artifacts))
