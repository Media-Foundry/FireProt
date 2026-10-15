"""Freeze the qualified FP32 worker with boundary-only timing instrumentation."""
from pathlib import Path
import hashlib
import json
import shutil
import sys


def pin(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare(root, cli):
    root, cli = Path(root).absolute(), Path(cli).absolute()
    source = root.parent / 'openfold3-hip-bias-pack-model-20261010'
    prep = json.loads((source / 'preparation.json').read_text())
    qualified = json.loads((source / 'collected.json').read_text())
    assert qualified['all11_exact'] and qualified['full_cif_exact']
    for item in prep['files'] + [qualified[k] for k in ('npz', 'cif')]:
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
    (workers / 'phase_boundary.py').write_text('''import os
import sys
import time

def phase(label):
    print('FIREPROT_PHASE', os.getpid(), time.time_ns(), label, file=sys.stderr, flush=True)
''')
    edits = {
        '__main__.py': [
            ('        import torch', '        phase("torch_import_begin")\n        import torch\n        phase("torch_import_end")'),
            ('        channel.ready(torch)', '        channel.ready(torch)\n        phase("worker_authorized")'),
            ('        bundle = predict(request, channel)', '        phase("predict_begin")\n        bundle = predict(request, channel)\n        phase("predict_end")'),
            ('        announce_outputs(channel, bundle)', '        announce_outputs(channel, bundle)\n        phase("outputs_announced")'),
        ],
        'openfold3.py': [
            ('    workload = request[\'workload\']', '    phase("model_imports_end")\n    workload = request[\'workload\']'),
            ('        check_determinism(torch, workload)', '        phase("forward_begin")\n        check_determinism(torch, workload)'),
            ('        model_seconds += time.perf_counter() - started', '        model_seconds += time.perf_counter() - started\n        phase("forward_end")'),
            ('        return result\n\n    OpenFold3.forward', '        phase("prediction_arrays_saved")\n        return result\n\n    OpenFold3.forward'),
            ('    started = time.perf_counter()\n    try:', '    started = time.perf_counter()\n    phase("cli_begin")\n    try:'),
            ('    if calls != 1 or len(predictions)', '    phase("cli_end")\n    if calls != 1 or len(predictions)'),
        ],
    }
    changes = []
    for filename, replacements in edits.items():
        path = workers / filename
        original = path.read_bytes()
        code = original.decode()
        for before, after in replacements:
            assert code.count(before) == 1, (filename, before)
            code = code.replace(before, after)
        # Keep __future__ imports in their legal location.
        anchor = 'import json'
        assert code.count(anchor) == 1
        code = code.replace(anchor, 'from .phase_boundary import phase\n' + anchor)
        compile(code, str(path), 'exec')
        path.write_text(code)
        changes.append(dict(path=str(path), before=hashlib.sha256(original).hexdigest(), after=pin(path)['sha256']))
    request = json.loads((source / 'request.json').read_text())
    request.update(run_id='of3-overhead-phase-20261015',
                   output_dir='/media/IntelSSD/fireprot/outputs/of3-overhead-phase-20261015')
    assert request['hip_device'] == 7 and not Path(request['output_dir']).exists()
    profile = json.loads((source / 'profile.json').read_text())
    profile['workers'] = str(root / 'python')
    profile['environment_inventory'] = dict(
        path=str(root.parent / 'openfold3-environment-audit-20261015/current.json'),
        sha256='14c60e2a88af9c7149b739e0af142154c8c1faf01096e014a7d04f1b26d3214f')
    for name, value in [('request.json', request), ('profile.json', profile)]:
        (root / name).write_text(json.dumps(value, indent=2) + '\n')
    (root / 'spec.json').write_text(json.dumps(dict(schema_version=1, timeout_seconds=3600,
        request=pin(root / 'request.json'), profile=pin(root / 'profile.json')), indent=2) + '\n')
    (root / 'preparation.json').write_text(json.dumps(dict(
        schema_version=1, cli=pin(cli), source_qualification=pin(source / 'collected.json'),
        reference_npz=qualified['npz'], reference_cif=qualified['cif'], changes=changes,
        scope='Boundary instrumentation only; all checks and scientific settings retained.',
        files=[pin(p) for p in sorted(root.rglob('*')) if p.is_file()]), indent=2) + '\n')


if __name__ == '__main__':
    prepare(*sys.argv[1:])
