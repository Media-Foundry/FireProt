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
               ('helper', helper), ('source', source),
               ('guard', Path('target/release/fireprot-compiler-guard').resolve())]],
        arguments=dict(cxx=[], hip=[]), environment={}, versions=dict(recipe='fixture')),
        python='python', helper='helper', guard='guard', lock_timeout_ms=5000,
        sources=['source'], timeout_ms=5000)
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


def cancellable_request(tmp_path):
    source = tmp_path / 'cancel.cu'
    source.write_text('fixture')
    helper = tmp_path / 'cancel.py'
    helper.write_text('import pathlib, time\n'
                      f'pathlib.Path({str(tmp_path / "ready")!r}).write_text("ready")\n'
                      'time.sleep(30)\n')
    return dict(build=dict(schema_version=1, package='cancel_process', target='gfx90a',
        files=[dict(name=n, path=str(p)) for n, p in [('python', sys.executable),
               ('helper', helper), ('source', source),
               ('guard', Path('target/release/fireprot-compiler-guard').resolve())]],
        arguments=dict(cxx=[], hip=[]), environment={}, versions=dict(recipe='fixture')),
        python='python', helper='helper', guard='guard', lock_timeout_ms=5000,
        sources=['source'], timeout_ms=30000)


@pytest.mark.parametrize('entry', ['bridge', 'cli'])
def test_build_entry_cancellation(tmp_path, entry):
    import os
    import signal
    import subprocess
    import time
    request = cancellable_request(tmp_path)
    spec = tmp_path / 'request.json'
    spec.write_text(json.dumps(request))
    cache = tmp_path / 'cache'
    if entry == 'bridge':
        program = '''import pathlib,sys
from fireprot.ops._bridge import bridge
try:
    bridge.build_torch(pathlib.Path(sys.argv[1]).read_text(),pathlib.Path(sys.argv[2]))
except KeyboardInterrupt:
    print('original KeyboardInterrupt',flush=True)
else:
    raise AssertionError('cancelled compilation returned success')
'''
        command = [sys.executable, '-c', program, str(spec), str(cache)]
        sig = signal.SIGINT
    else:
        command = [str(Path('target/release/fireprot').resolve()), 'build-torch', str(spec), str(cache)]
        sig = signal.SIGTERM
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / 'ready').exists():
            assert child.poll() is None, child.communicate()
            assert time.monotonic() < deadline, 'compiler never started'
            time.sleep(.01)
        os.kill(child.pid, sig)
        out, err = child.communicate(timeout=5)
        if entry == 'bridge':
            assert child.returncode == 0, (out, err)
            assert 'original KeyboardInterrupt' in out
        else:
            assert child.returncode != 0
            assert 'build cancelled' in err
        assert not list(cache.glob('*/fireprot-build.json'))
        assert not list(cache.glob('.building-*'))
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_callback_cannot_reenter_process_builder(tmp_path):
    request = cancellable_request(tmp_path)
    def reenter(*args):
        bridge.build_torch(json.dumps(request), tmp_path / 'cache')
        raise AssertionError('reentry was accepted')
    with pytest.raises(RuntimeError, match='must not reenter build_torch'):
        bridge.cached_build(json.dumps(request['build']), tmp_path / 'cache', reenter)
