"""Rehash existing ESMFold2 resources without modifying the shared checkpoints."""
from pathlib import Path
import hashlib
import json
import sys
import time


def verify(root, destination):
    root, destination = Path(root), Path(destination)
    revision = json.loads((root / 'revision.json').read_text())
    if revision != dict(repo_id='biohub/ESMFold2', revision='69869f737beffec5294845ede23db5fc0b4f509e'):
        raise ValueError('Unexpected ESMFold2 checkpoint revision')
    records = []
    for path in sorted((root / 'snapshot').iterdir()):
        if path.is_file() and (path.suffix in ('.safetensors', '.json') or path.name == 'ccd.pkl'):
            start = time.monotonic()
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            record = dict(path=str(path), bytes=path.stat().st_size,
                          sha256=digest.hexdigest(), seconds=time.monotonic() - start)
            records.append(record)
            print(path.name, record['sha256'], flush=True)
    with destination.open('x') as handle:
        json.dump(dict(revision=revision, files=records), handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    verify(*sys.argv[1:])
