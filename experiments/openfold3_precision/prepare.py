"""Freeze a declared PairFormer precision diagnostic around the qualified route."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def prepare(root, scope):
    root = Path(root).absolute()
    source = root.parent / 'openfold3-hip-bias-pack-model-20261010'
    prep, qualified = read(source / 'preparation.json'), read(source / 'collected.json')
    assert qualified['all11_exact'] and qualified['full_cif_exact']
    for item in prep['files'] + [prep[k] for k in ('cli', 'reference_npz', 'reference_cif')]:
        assert pin(item['path'])['sha256'] == item['sha256'], item['path']
    root.mkdir(exist_ok=False)
    for name in ('python', 'fireprot', 'kernels', 'licenses', 'LICENSE', 'THIRD_PARTY.md',
                 'norm.so', 'gate.so', 'qkv.so', 'pack.so', 'pack-build.json'):
        path = source / name
        if path.is_dir():
            shutil.copytree(path, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        else:
            shutil.copy2(path, root / name)
    workers = root / 'python/fireprot_workers'
    shutil.copy2(scope, workers / 'precision_scope.py')
    path = workers / 'transport.py'
    code = path.read_text()
    needle = "'global-softmax.json',"
    assert code.count(needle) == 1
    code = code.replace(needle, "'pairformer-precision.json', 'precision-policy.json', 'runner.json', 'triangle-key-copy.json', 'isolated-data.json', " + needle)
    path.write_text(code)
    path = workers / 'openfold3.py'
    code = path.read_text()
    needle = '        with scope, atom_scope,'
    assert code.count(needle) == 1
    code = code.replace(needle, '''        from .precision_scope import pairformer_precision_scope
        import hashlib
        policy_path = Path(os.environ['FIREPROT_PRECISION_POLICY'])
        policy_bytes = policy_path.read_bytes()
        if hashlib.sha256(policy_bytes).hexdigest() != os.environ['FIREPROT_PRECISION_POLICY_SHA256']:
            raise RuntimeError('Precision policy digest mismatch')
        policy = json.loads(policy_bytes)
        if policy['base_request_precision'] != workload['precision'] or not policy['diagnostic_only']:
            raise RuntimeError('Expected an explicit diagnostic precision override')
        if policy['pairformer_autocast'] not in ('float32', 'bfloat16', 'float16'):
            raise RuntimeError('Unsupported PairFormer precision')
        (output / 'precision-policy.json').write_bytes(policy_bytes)
        mixed_scope = pairformer_precision_scope(model, output, policy['pairformer_autocast'], diagnostic=True)
        with mixed_scope, scope, atom_scope,''')
    compile(code, str(path), 'exec')
    path.write_text(code)
    for dtype in ('float32', 'bfloat16', 'float16'):
        request = read(source / 'request.json')
        request.update(run_id=f'of3-pairformer-{dtype}-diagnostic-20261010',
                       output_dir=f'/media/IntelSSD/fireprot/outputs/of3-pairformer-{dtype}-diagnostic-20261010')
        assert request['hip_device'] == 7 and not Path(request['output_dir']).exists()
        write(root / f'{dtype}-request.json', request)
        write(root / f'{dtype}-policy.json', dict(schema_version=1, diagnostic_only=True,
            base_request_precision='float32', pairformer_autocast=dtype,
            other_stages='float32', master_parameters='float32',
            recycle_count_unchanged=True, diffusion_steps_unchanged=True,
            acceptance='Report deviations first; no mixed-precision quality threshold or promotion authorized.',
            timing='Activation scans are instrumented; no E2E speedup claim.'))
        policy_pin = pin(root / f'{dtype}-policy.json')
        profile = read(source / 'profile.json')
        profile['workers'] = str(root / 'python')
        profile['environment'].update(FIREPROT_PRECISION_POLICY=policy_pin['path'],
                                      FIREPROT_PRECISION_POLICY_SHA256=policy_pin['sha256'])
        profile['environment_inventory'] = dict(
            path=str(root.parent / 'openfold3-environment-audit-20261010/current.json'),
            sha256='e9a14970d1d8c805cd808bd9d28e6d7333cb2e8a17b87a7b13e0a08c7d08ab45')
        write(root / f'{dtype}-profile.json', profile)
        write(root / f'{dtype}-spec.json', dict(schema_version=1, timeout_seconds=3600,
            request=pin(root / f'{dtype}-request.json'), profile=pin(root / f'{dtype}-profile.json')))
        subprocess.run([prep['cli']['path'], 'inspect', str(root / f'{dtype}-request.json')],
                       check=True, stdout=subprocess.DEVNULL)
    write(root / 'preparation.json', dict(cli=prep['cli'],
        scope='Explicit PairFormer-only precision override; base request remains the FP32 anchor, and the sealed policy/report declare effective arithmetic. Diagnostic data only.',
        source_qualification=pin(source / 'collected.json'),
        reference_npz=qualified['npz'], reference_cif=qualified['cif'],
        files=[pin(path) for path in sorted(root.rglob('*')) if path.is_file()]))
    return root


if __name__ == '__main__':
    print(prepare(*sys.argv[1:]))
