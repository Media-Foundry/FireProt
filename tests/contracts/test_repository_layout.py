"""Both supported import roots resolve one maintained source tree."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_historical_path_map_has_one_canonical_target_per_file():
    mapping = json.loads((ROOT / 'provenance/layout_path_map.json').read_text())
    assert not (ROOT / 'fireprot').is_symlink()
    assert (ROOT / 'fireprot/__init__.py').is_file()
    assert not (ROOT / 'python/fireprot/backends/amd/csrc').exists()
    assert len({item['new'] for item in mapping['entries']}) == len(mapping['entries'])
    for item in mapping['entries']:
        old, new = ROOT / item['old'], ROOT / item['new']
        assert new.is_file() and not old.exists()
        assert new.is_relative_to(ROOT / 'python') or new.is_relative_to(ROOT / 'kernels/amd')


@pytest.mark.parametrize('import_root', [ROOT, ROOT / 'python'])
def test_paths_and_bridge_work_outside_the_repository(import_root, tmp_path):
    code = '''
import json
from fireprot._paths import REPOSITORY_ROOT, KERNEL_ROOT
from fireprot.backends.amd.extension import _source_paths
from fireprot.ops._bridge import identity
print(json.dumps(dict(root=str(REPOSITORY_ROOT), kernels=str(KERNEL_ROOT),
    sources=[str(p) for p in _source_paths()], bridge=identity['library'])))
'''
    environment = dict(os.environ, PYTHONPATH=str(import_root))
    environment.pop('FIREPROT_BRIDGE_LIBRARY', None)
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, check=True)
    paths = json.loads(result.stdout)
    assert paths['root'] == str(ROOT)
    assert paths['kernels'] == str(ROOT / 'kernels/amd')
    assert paths['bridge'] == str(ROOT / 'target/release/lib_fireprot_bridge.so')
    assert all(Path(p).parent == ROOT / 'kernels/amd' for p in paths['sources'])
