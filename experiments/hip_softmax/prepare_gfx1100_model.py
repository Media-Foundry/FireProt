"""Freeze an experimental wave32 worker and Rust-owned full-model comparison.

Run on Precision after the maintained native entry passes. This does not enable
gfx1100 in the maintained gfx90a profile or change any numerical acceptance gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ('python', 'fireprot', 'kernels', 'licenses'):
        shutil.copytree(repo / name, root / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('LICENSE', 'THIRD_PARTY.md'):
        shutil.copy2(repo / name, root / name)
    workers = root / 'python/fireprot_workers'
    scope = repo / 'experiments/hip_softmax/gfx1100_scope.py'
    shutil.copy2(scope, workers / 'gfx1100_scope.py')
    path = workers / 'openfold3.py'
    before = path.read_text()
    old = ('    from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32\n'
           '    from fireprot.adapters.openfold3.attention import AttentionDispatcher, attention_scope\n'
           '    dispatcher = AttentionDispatcher(SoftmaxFP32())')
    new = ('    from .gfx1100_scope import Gfx1100Dispatcher, Gfx1100Softmax\n'
           '    from fireprot.adapters.openfold3.attention import attention_scope\n'
           '    dispatcher = Gfx1100Dispatcher(Gfx1100Softmax(\n'
           '        os.environ["FIREPROT_GFX1100_SOFTMAX_LIBRARY"],\n'
           '        os.environ["FIREPROT_GFX1100_SOFTMAX_SHA256"]))')
    assert before.count(old) == 1
    after = before.replace(old, new)
    old_call = ('        with attention_scope(dispatcher):\n'
                '            return forward(model, *args, **kwargs)')
    new_call = ('        with attention_scope(dispatcher):\n'
                '            result = forward(model, *args, **kwargs)\n'
                '        if dispatcher.kernel.launches_submitted == 0:\n'
                '            raise RuntimeError("Experimental model executed no HIP softmax calls")\n'
                '        return result')
    assert after.count(old_call) == 1
    after = after.replace(old_call, new_call)
    compile(after, str(path), 'exec')
    path.write_text(after)
    library = repo / 'runtime/kernels/openfold3-gfx1100-softmax-screen-20261017/wave32-v1.so'
    library_pin = pin(library)
    assert library_pin['sha256'] == 'df2c109098b48b019dc95bd120503af301d6d4cec5762f0bbc1fedab05303ad5'
    bridge = repo / 'target/release/lib_fireprot_bridge.so'
    profile = json.loads((repo / 'configs/precision/openfold3-native.json').read_text())
    profile['workers'] = str(root / 'python')
    profile['environment'].update(
        FIREPROT_BRIDGE_LIBRARY=str(bridge),
        FIREPROT_GFX1100_SOFTMAX_LIBRARY=str(library),
        FIREPROT_GFX1100_SOFTMAX_SHA256=library_pin['sha256'])
    write(root / 'profile.json', profile)
    template = json.loads((repo / 'configs/requests/precision-openfold3-n1000-native.json').read_text())
    pairs = []
    for index in range(3):
        pair = dict(first='candidate' if index % 2 else 'baseline')
        for arm, backend in (('baseline', 'native'), ('candidate', 'hip_fused')):
            request = dict(template, backend=backend,
                           run_id=f'gfx11-of3-n1000-{root.name}-{arm}-{index}',
                           output_dir=str(root / f'prediction-{arm}-{index}'))
            request_path = root / f'request-{arm}-{index}.json'
            write(request_path, request)
            pair[arm] = dict(request=pin(request_path), profile=pin(root / 'profile.json'))
        pairs.append(pair)
    bridge_identity = dict(name='rust-bridge', canonical_path=str(bridge), sha256=pin(bridge)['sha256'])
    write(root / 'pairs.json', dict(schema_version=1,
        execution=dict(schema_version=1, timeout_seconds=10800,
                       invocation_timeout_seconds=1800, pairs=pairs),
        comparison=dict(scope='end_to_end',
            cache_state='Fresh processes; filesystem/GPU caches unflushed; prebuilt hash-pinned experimental wave32 library; no asset transfers',
            treatment_description='Native OpenFold3 versus experimental exact gfx1100 mask/bias/softmax; same frozen workers and strict FP32 R3/D200/S1',
            expected_changes={
                'backend': dict(baseline='native', candidate='hip_fused'),
                'launch.inputs.rust-bridge': dict(baseline=None, candidate=bridge_identity)})))
    write(root / 'preparation.json', dict(schema_version=1, scope='Experimental model qualification, not production promotion',
        worker_before_sha256=hashlib.sha256(before.encode()).hexdigest(), worker_after=pin(path),
        dispatcher=pin(workers / 'gfx1100_scope.py'), library=library_pin, bridge=pin(bridge),
        cli=pin(repo / 'target/release/fireprot'),
        environment_inventory=None,
        limitations=['The venv inherits BIO packages; complete environment content inventory is not yet qualified.',
                     'Only N1000 is tested here; no N1536 full-model claim.',
                     'No locked-clock or cold-cache claim.'],
        files=[pin(p) for p in sorted(root.rglob('*')) if p.is_file()]))
    print(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('repo')
    parser.add_argument('root')
    args = parser.parse_args()
    prepare(args.repo, args.root)
