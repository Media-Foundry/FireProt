"""CPU-only protocol fixture; never imports a model or GPU library."""
import hashlib
import json
import os
import socket
import sys
import time

mode = sys.argv[1]
request = json.load(open(os.environ['FIREPROT_REQUEST']))
if mode == 'exit':
    sys.exit(0)
if mode == 'timeout':
    time.sleep(30)
sock = socket.socket(socket.AF_UNIX)
sock.connect(os.environ['FIREPROT_WORKER_SOCKET'])

def emit(seq, payload):
    sock.sendall((json.dumps({'schema_version': 1, 'run_id': request['run_id'],
                             'sequence': seq, 'payload': payload}) + '\n').encode())

if mode == 'malformed':
    sock.sendall(b'not json\n')
    sys.exit(0)
if mode == 'oversize':
    sock.sendall(b'x' * 9000)
    time.sleep(30)
if mode == 'partial':
    sock.sendall(b'{')
    sys.exit(0)
emit(0, {'event': 'ready', 'device': {'worker_pid': os.getpid(), 'logical_device': 0,
      'pci_address': '0000:93:00.0', 'architecture': 'gfx90a'}})
assert json.loads(sock.makefile().readline())['command'] == 'start'
print('model logs remain separate from protocol', flush=True)
assert os.environ['HIP_VISIBLE_DEVICES'] == '7'
assert 'ROCR_VISIBLE_DEVICES' not in os.environ
if mode == 'failure':
    emit(1, {'event': 'failed', 'message': 'model error'})
    sys.exit(0)
if mode == 'wait':
    time.sleep(30)
if mode == 'descendant':
    pid = os.fork()
    if pid == 0:
        time.sleep(1)
        open(os.path.join(os.environ['FIREPROT_OUTPUT_DIR'], 'escaped'), 'w').write('bad')
        sys.exit(0)
    time.sleep(30)
output = os.path.join(os.environ['FIREPROT_OUTPUT_DIR'], 'prediction.bin')
if mode == 'escape':
    os.symlink('/etc/hostname', output)
else:
    open(output, 'wb').write(b'prediction')
emit(2 if mode == 'sequence' else 1,
     {'event': 'outputs', 'artifacts': [{'path': 'prediction.bin',
       'sha256': '0' * 64 if mode == 'hash' else hashlib.sha256(b'prediction').hexdigest()}]})
if mode == 'nonzero':
    sys.exit(2)
