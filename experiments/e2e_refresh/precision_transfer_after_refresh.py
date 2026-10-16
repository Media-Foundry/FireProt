"""Resume verified asset migration after an explicit end-of-benchmark release."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path(sys.argv[1])
manifest_path = Path('/tmp/fireprot-precision-transfer-manifest.json')
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == sys.argv[2]
manifest = json.loads(manifest_path.read_text())
assert manifest['destination_root'] == '/media/990Pro/fireprot'
release = root / 'release-asset-transfer.json'
log = root / 'asset-transfer-v2.jsonl'
status = root / 'asset-transfer-v2-status.json'
assert not status.exists() and not log.exists()
print('Waiting for explicit release after all GPU trials and evidence checks', flush=True)
while not release.exists():
    time.sleep(2)
decision = json.loads(release.read_text())
assert decision['all_gpu_work_complete'] is True
target = decision.get('ssh_target', 'husrcf@Precision')
assert re.fullmatch(r'husrcf@(Precision|[0-9.]+)', target)
try:
    for entry in manifest['files']:
        if 'reuse_source' in entry:
            continue
        relative = entry['relative']
        assert relative.startswith('runtime/') and '..' not in Path(relative).parts
        destination = manifest['destination_root'] + '/' + relative
        # Existing final files were fully hashed before rescheduling. Public
        # relay files are installed atomically only after the same full hash.
        # Interrupted rsync data remains in a partial directory, not at the
        # final pathname, so --ignore-existing cannot accept an incomplete file.
        command = ['rsync', '-az', '--partial-dir=.fireprot-partial', '--ignore-existing', '--protect-args', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=20', entry['source'], target + ':' + destination]
        for attempt in range(3):
            started = time.monotonic()
            result = subprocess.run(command)
            record = dict(relative=relative, bytes=entry['bytes'], attempt=attempt, exit_code=result.returncode, elapsed_seconds=time.monotonic()-started, ssh_target=target)
            with log.open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if result.returncode == 0:
                print('Ensured', relative, flush=True)
                break
            if attempt == 2:
                raise RuntimeError(f'rsync failed for {relative}: {result.returncode}')
            print('Transfer failed; retrying after 120 seconds', flush=True)
            time.sleep(60)
            time.sleep(60)
    status.write_text(json.dumps(dict(status='transfer_finished_destination_hash_verification_still_required', exit_code=0))+'\n')
    print('Transfer finished; final 49-file verifier is authoritative', flush=True)
except BaseException as error:
    status.write_text(json.dumps(dict(status='failed', error=str(error)))+'\n')
    raise
