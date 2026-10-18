"""Prepare three isolated gfx1100 reuse cycles; Rust owns execution and gates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    source = repo / 'outputs/experiments/gfx11-of3-pairs-20261017-v2'
    original = json.loads((source / 'preparation.json').read_text())
    for item in original['files'] + [original['library'], original['bridge'], original['cli']]:
        assert pin(Path(item['path']))['sha256'] == item['sha256'], item['path']
    root.mkdir(parents=True, exist_ok=False)
    for name in ('python', 'fireprot', 'kernels', 'licenses'):
        shutil.copytree(source / name, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('LICENSE', 'THIRD_PARTY.md'):
        shutil.copy2(source / name, root / name)
    workers = root / 'python/fireprot_workers'
    for name in ('pair_cache', 'verified_reuse', 'atom_cache', 'verified_atom_reuse'):
        shutil.copy2(repo / f'experiments/conditioning_reuse/{name}.py', workers / f'{name}.py')
    template = (repo / 'experiments/triangle_memory/template_reuse.py').read_text()
    assert template.count('1536') == 5
    # Keep the original N1536 experiment untouched; this copy admits only N1000.
    (workers / 'template_reuse.py').write_text(template.replace('1536', '1000'))
    guard = (repo / 'experiments/triangle_memory/template_reuse_screen.py').read_text()
    (root / 'template_guard_screen.py').write_text(guard.replace('1536', '1000'))
    worker = workers / 'openfold3.py'
    code = worker.read_text()
    old = "        result = forward_with_backend(model, original_forward, args, kwargs,\n                                      backend=request['backend'], output=output)"
    new = """        from .gfx1100_reuse import reuse_scope
        with reuse_scope(model, output):
            result = forward_with_backend(model, original_forward, args, kwargs,
                                          backend=request['backend'], output=output)"""
    assert code.count(old) == 1
    worker.write_text(code.replace(old, new))
    shutil.copy2(repo / 'experiments/hip_softmax/gfx1100_reuse.py', workers / 'gfx1100_reuse.py')
    transport = workers / 'transport.py'
    code = transport.read_text()
    old = "('operator-runtime.json', 'atlasfold-model.json')"
    new = "('operator-runtime.json', 'atlasfold-model.json', 'conditioning-cache.json', 'atom-cache.json', 'template-reuse.json', 'reuse-runtime.json')"
    assert code.count(old) == 1
    transport.write_text(code.replace(old, new))
    profile = json.loads((source / 'profile.json').read_text())
    profile['workers'] = str(root / 'python')
    request = json.loads((source / 'request-candidate-0.json').read_text())
    for mode in range(4):
        profile['environment']['FIREPROT_GFX1100_REUSE_MODE'] = str(mode)
        write(root / f'profile-{mode}.json', profile)
        run = dict(request, run_id=f'{root.name}-cycle-{mode}', output_dir=str(root / f'prediction-cycle-{mode}'))
        write(root / f'request-{mode}.json', run)
        write(root / f'measure-{mode}.json', dict(schema_version=1, timeout_seconds=1800,
            request=pin(root / f'request-{mode}.json'), profile=pin(root / f'profile-{mode}.json')))
    write(root / 'preparation.json', dict(schema_version=1, base=pin(source / 'preparation.json'),
        reuse_modules='Existing source-pinned pair/atom caches; copied N1000 template guard specialization',
        modes={'0':'previous HIP softmax only', '1':'plus pair cache', '2':'plus atom cache', '3':'plus exact duplicate template reuse'},
        files=[pin(p) for p in sorted(root.rglob('*')) if p.is_file()]))
    print(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repo')
    parser.add_argument('root')
    args = parser.parse_args()
    prepare(args.repo, args.root)
