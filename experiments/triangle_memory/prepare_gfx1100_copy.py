"""Freeze an N1000 copy candidate and paired controls after the operator gate."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def pin(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


repo, root = (Path(v).resolve() for v in sys.argv[1:])
source = repo/'outputs/experiments/gfx11-of3-reuse-cycles-20261018-v1'
for item in json.loads((source/'preparation.json').read_text())['files']:
    assert pin(Path(item['path'])) == item
assert json.loads((source/'audit-final.json').read_text())['all_passed']
assert json.loads((root/'screen.json').read_text())['correctness_passed']
build = json.loads((root/'build.json').read_text())
assert pin(Path(build['library']))['sha256'] == build['library_sha256']
for name in ('python', 'fireprot', 'kernels', 'licenses'):
    shutil.copytree(source/name, root/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ('LICENSE', 'THIRD_PARTY.md'):
    shutil.copy2(source/name, root/name)
workers = root/'python/fireprot_workers'
for name in ('gfx1100_copy.py', 'copy_runtime.py'):
    shutil.copy2(repo/'experiments/triangle_memory'/name, workers/name)
worker = workers/'openfold3.py'
code = worker.read_text()
old = '        with reuse_scope(model, output):'
assert code.count(old) == 1
new = '''        from .gfx1100_copy import triangle_copy_scope
        with reuse_scope(model, output), triangle_copy_scope(model, output,
                os.environ['FIREPROT_GFX1100_COPY_LIBRARY'],
                os.environ['FIREPROT_GFX1100_COPY_SHA256'],
                enabled=os.environ['FIREPROT_GFX1100_COPY'] == '1'):'''
worker.write_text(code.replace(old, new))
transport = workers/'transport.py'
code = transport.read_text()
assert code.count("'reuse-runtime.json')") == 1
transport.write_text(code.replace("'reuse-runtime.json')", "'reuse-runtime.json', 'triangle-copy.json')"))
profile = json.loads((source/'profile-3.json').read_text())
profile['workers'] = str(root/'python')
profile['environment'].update(FIREPROT_GFX1100_COPY_LIBRARY=build['library'],
                              FIREPROT_GFX1100_COPY_SHA256=build['library_sha256'])
profiles = {}
for arm, flag in [('baseline', '0'), ('candidate', '1')]:
    profile['environment']['FIREPROT_GFX1100_COPY'] = flag
    write(root/f'profile-{arm}.json', profile)
    profiles[arm] = pin(root/f'profile-{arm}.json')
template = json.loads((source/'request-3.json').read_text())
cycle = dict(template, run_id=root.name+'-cycle', output_dir=str(root/'prediction-cycle'))
write(root/'request-cycle.json', cycle)
write(root/'measure-cycle.json', dict(schema_version=1, timeout_seconds=1800,
      request=pin(root/'request-cycle.json'), profile=profiles['candidate']))
pairs = []
for index in range(3):
    pair = dict(first='candidate' if index % 2 else 'baseline')
    for arm in profiles:
        request = dict(template, run_id=f'{root.name}-{arm}-{index}',
                       output_dir=str(root/f'prediction-{arm}-{index}'))
        path = root/f'request-{arm}-{index}.json'
        write(path, request)
        pair[arm] = dict(request=pin(path), profile=profiles[arm])
    pairs.append(pair)
write(root/'pairs.json', dict(schema_version=1,
    execution=dict(schema_version=1, timeout_seconds=10800, invocation_timeout_seconds=1800, pairs=pairs),
    comparison=dict(scope='end_to_end',
        cache_state='Fresh workers; unflushed filesystem/GPU caches; prebuilt kernels; no asset transfers; clocks unlocked',
        treatment_description='Qualified gfx1100 softmax/reuse versus plus HIP triangle copies and explicit native operand packing; FP32 N1000 R3/D200/S1; incremental comparison',
        expected_changes={
            'launch.environment.FIREPROT_GFX1100_COPY': dict(baseline='0', candidate='1'),
            'invocation.profile_sha256': {arm:p['sha256'] for arm,p in profiles.items()}})))
write(root/'preparation.json', dict(base=pin(source/'preparation.json'),
    base_audit=pin(source/'audit-final.json'), build=pin(root/'build.json'),
    library=pin(Path(build['library'])), cli=pin(repo/'target/release/fireprot'),
    bridge=pin(Path(profile['environment']['FIREPROT_BRIDGE_LIBRARY'])),
    preparer=pin(Path(__file__).resolve()),
    files=[pin(p) for p in sorted(root.rglob('*')) if p.is_file()]))
print(root)
