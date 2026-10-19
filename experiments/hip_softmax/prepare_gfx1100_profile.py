"""Freeze a diagnostic copy of the qualified gfx1100 reuse composition."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


repo, root = (Path(v).resolve() for v in sys.argv[1:])
source = repo / 'outputs/experiments/gfx11-of3-reuse-cycles-20261018-v1'
preparation = json.loads((source/'preparation.json').read_text())
for item in preparation['files']:
    assert pin(Path(item['path'])) == item
assert json.loads((source/'audit-final.json').read_text())['all_passed']
root.mkdir(parents=True, exist_ok=False)
for name in ('python', 'fireprot', 'kernels', 'licenses'):
    shutil.copytree(source/name, root/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ('LICENSE', 'THIRD_PARTY.md'):
    shutil.copy2(source/name, root/name)
worker = root/'python/fireprot_workers/openfold3.py'
code = worker.read_text()
old = '        with reuse_scope(model, output):'
assert code.count(old) == 1
worker.write_text(code.replace(old, '        from .gfx1100_profile import profile_scope\n        with reuse_scope(model, output), profile_scope(model, output):'))
shutil.copy2(repo/'experiments/hip_softmax/gfx1100_profile.py', worker.parent/'gfx1100_profile.py')
transport = worker.parent/'transport.py'
code = transport.read_text()
old = "'reuse-runtime.json')"
assert code.count(old) == 1
transport.write_text(code.replace(old, "'reuse-runtime.json', 'stage-profile.json')"))
profile = json.loads((source/'profile-3.json').read_text())
profile['workers'] = str(root/'python')
request = json.loads((source/'request-3.json').read_text())
request.update(run_id=root.name, output_dir=str(root/'prediction'))
for name, value in [('profile.json', profile), ('request.json', request)]:
    (root/name).write_text(json.dumps(value, indent=2)+'\n')
spec = dict(schema_version=1, timeout_seconds=1800,
            request=pin(root/'request.json'), profile=pin(root/'profile.json'))
(root/'measure.json').write_text(json.dumps(spec, indent=2)+'\n')
(root/'preparation.json').write_text(json.dumps(dict(
    base=pin(source/'preparation.json'), base_audit=pin(source/'audit-final.json'),
    preparer=pin(Path(__file__).resolve()),
    files=[pin(p) for p in sorted(root.rglob('*')) if p.is_file()]), indent=2)+'\n')
print(root)
