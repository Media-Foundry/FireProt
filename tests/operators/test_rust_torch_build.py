"""Exercise real Rust process orchestration through the in-process bridge."""
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from fireprot.ops._bridge import bridge


def test_process_bridge_concurrency_and_diagnostics(tmp_path):
    source = tmp_path / 'kernel.cu'
    source.write_text('input')
    helper = tmp_path / 'compile.py'
    helper.write_text('''import json, pathlib, sys, time
r=json.load(open(sys.argv[1]))
time.sleep(0.1)
pathlib.Path(r['build_directory'], r['name']+'.so').write_bytes(b'fixture')
''')
    request = dict(build=dict(schema_version=1, package='bridge_process', target='gfx90a',
        files=[dict(name=n, path=str(p)) for n, p in [('python', sys.executable),
               ('helper', helper), ('source', source)]],
        arguments=dict(cxx=[], hip=[]), environment={}, versions=dict(recipe='fixture')),
        python='python', helper='helper', sources=['source'], timeout_ms=5000)
    def run(_):
        return json.loads(bridge.build_torch(json.dumps(request), tmp_path / 'cache'))
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(run, range(3)))
    assert sum(r['cache_hit'] for r in results) == 2
    helper.write_text("raise RuntimeError('fixture compiler failed')\n")
    with pytest.raises(OSError, match='fixture compiler failed'):
        run(0)
    request['sources'] = ['undeclared']
    with pytest.raises(OSError, match='undeclared Torch build input'):
        run(0)
