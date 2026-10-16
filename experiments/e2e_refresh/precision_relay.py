"""Relay fixed official ESMFold2 assets without reading the benchmark host."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from urllib.parse import quote

manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest['destination_root'] == '/media/990Pro/fireprot'
prefix = 'runtime/weights/esmfold2/snapshot/'
base = 'https://huggingface.co/biohub/ESMFold2/resolve/69869f737beffec5294845ede23db5fc0b4f509e/'
log = Path(sys.argv[2])
for entry in manifest['files']:
    relative = entry['relative']
    if not relative.startswith(prefix):
        continue
    name = relative[len(prefix):]
    assert '..' not in Path(name).parts
    origin = base + quote(name, safe='/')
    print('Relaying', relative, entry['bytes'], flush=True)
    started = time.monotonic()
    receiver = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', 'Precision', shlex.join(['python3', '/tmp/fireprot-precision-receive.py', relative, origin])], stdin=subprocess.PIPE)
    try:
        # No in-stream retry: restarting a partial response on stdout would
        # concatenate bytes. Failure leaves no installed destination file.
        download = subprocess.run(['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error', '--connect-timeout', '20', '--max-time', '3600', origin], stdout=receiver.stdin)
    finally:
        receiver.stdin.close()
    receive_status = receiver.wait()
    record = dict(relative=relative, origin=origin, curl_exit=download.returncode, receiver_exit=receive_status, elapsed_seconds=time.monotonic()-started)
    with log.open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    if download.returncode or receive_status:
        raise SystemExit('Relay stopped; original DiamondHill transfer remains the fallback')
print('Official ESMFold2 relay complete', flush=True)
