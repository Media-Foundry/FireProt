"""Install a streamed public asset only after size and complete SHA-256 checks."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

root = Path('/media/990Pro/fireprot').resolve()
receipt_root = root / 'runtime/migration-20261016'
manifest = json.loads((receipt_root / 'transfer-manifest.json').read_text())
relative, origin = sys.argv[1:]
assert relative.startswith('runtime/weights/esmfold2/snapshot/')
entry = next(e for e in manifest['files'] if e['relative'] == relative)
destination = root / relative
assert destination.resolve().is_relative_to(root)
destination.parent.mkdir(parents=True, exist_ok=True)

def verify_existing():
    assert destination.is_file() and not destination.is_symlink()
    assert destination.stat().st_size == entry['bytes']
    h = hashlib.sha256()
    with destination.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    assert h.hexdigest() == entry['sha256']

incoming = receipt_root / 'incoming'
incoming.mkdir(exist_ok=True)
with tempfile.NamedTemporaryFile(dir=incoming, prefix='public-', suffix='.partial') as temporary:
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: sys.stdin.buffer.read(8 * 1024 * 1024), b''):
        count += len(block)
        assert count <= entry['bytes'], 'Received more bytes than the pinned artifact'
        digest.update(block)
        temporary.write(block)
    assert count == entry['bytes'] and digest.hexdigest() == entry['sha256'], 'Incomplete or mismatched public artifact'
    temporary.flush()
    os.fsync(temporary.fileno())
    os.chmod(temporary.name, 0o644)
    try:
        os.link(temporary.name, destination)
        action = 'installed_verified_stream'
    except FileExistsError:
        # A concurrent rsync may have completed the same pinned file. Never
        # overwrite a conflicting destination or accept an incomplete stream.
        verify_existing()
        action = 'existing_destination_verified'
record = dict(relative=relative, origin=origin, bytes=count, sha256=digest.hexdigest(), action=action, completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
with (receipt_root / 'public-download-progress.jsonl').open('a') as log:
    log.write(json.dumps(record) + '\n')
print(json.dumps(record), flush=True)
