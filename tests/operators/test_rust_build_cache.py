"""Rust cache callbacks preserve Python errors and release the GIL while waiting."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from fireprot.ops._bridge import bridge


def spec(tmp_path):
    source = tmp_path / 'kernel.cu'
    source.write_text('source')
    return json.dumps(dict(schema_version=1, package='test_kernel', target='gfx90a',
        files=[dict(name='source', path=str(source))], arguments={}, environment={},
        versions=dict(recipe='test')))


def compile_fake(directory, name):
    Path(directory, name + '.so').write_bytes(b'library')
    return [name + '.so']


def test_cache_hit_and_corruption_are_rust_decisions(tmp_path):
    request = spec(tmp_path)
    result = json.loads(bridge.cached_build(request, tmp_path / 'cache', compile_fake))
    assert not result['cache_hit']
    def forbidden(*args):
        raise AssertionError('must reuse verified cache')
    assert json.loads(bridge.cached_build(request, tmp_path / 'cache', forbidden))['cache_hit']
    path = Path(result['directory']) / next(iter(result['receipt']['artifacts']))
    path.write_bytes(b'corrupt')
    with pytest.raises(OSError, match='SHA-256 mismatch'):
        bridge.cached_build(request, tmp_path / 'cache', forbidden)


@pytest.mark.parametrize('error', [ValueError('compile failed'), KeyboardInterrupt(), SystemExit(3)])
def test_callback_exception_identity_and_cleanup(tmp_path, error):
    request = spec(tmp_path)
    def fail(directory, name):
        compile_fake(directory, name)
        raise error
    with pytest.raises(type(error)) as raised:
        bridge.cached_build(request, tmp_path / 'cache', fail)
    assert raised.value is error
    result = json.loads(bridge.cached_build(request, tmp_path / 'cache', compile_fake))
    assert not result['cache_hit']


def test_reentrant_callback_fails_without_deadlocking_or_poisoning(tmp_path):
    request = spec(tmp_path)
    def reenter(*args):
        return bridge.cached_build(request, tmp_path / 'cache', compile_fake)
    with pytest.raises(RuntimeError, match='must not reenter'):
        bridge.cached_build(request, tmp_path / 'cache', reenter)
    assert not json.loads(bridge.cached_build(request, tmp_path / 'cache', compile_fake))['cache_hit']


def test_waiting_for_cache_lock_does_not_hold_gil(tmp_path):
    # Subprocess timeout makes a GIL/lock deadlock a bounded test failure.
    request = spec(tmp_path)
    program = '''
import json, pathlib, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from fireprot.ops._bridge import bridge
request, root = sys.argv[1:]
barrier = threading.Barrier(4)
calls = []
def compile(directory, name):
    calls.append(name)
    time.sleep(0.1)  # Other threads wait for Rust lock while this thread releases GIL.
    pathlib.Path(directory, name + '.so').write_bytes(b'library')
    return [name + '.so']
def run(_):
    barrier.wait()
    return json.loads(bridge.cached_build(request, pathlib.Path(root), compile))['cache_hit']
with ThreadPoolExecutor(4) as pool:
    results = list(pool.map(run, range(4)))
assert sum(results) == 3, results
assert len(calls) == 1, calls
'''
    subprocess.run([sys.executable, '-c', program, request, str(tmp_path / 'cache')],
                   check=True, capture_output=True, text=True, timeout=15)


def test_directory_changes_during_callback_cannot_publish(tmp_path):
    request = json.loads(spec(tmp_path))
    headers = tmp_path / 'headers'
    headers.mkdir()
    request['trees'] = [dict(name='headers', path=str(headers), suffixes=[])]
    def mutate(directory, name):
        (headers / 'new.h').write_text('added while compiling')
        return compile_fake(directory, name)
    with pytest.raises(OSError, match='build inputs or receipt changed'):
        bridge.cached_build(json.dumps(request), tmp_path / 'cache', mutate)
    result = json.loads(bridge.cached_build(json.dumps(request), tmp_path / 'cache', compile_fake))
    assert not result['cache_hit']
    assert len(result['receipt']['identity']['tree_files']) == 1
    (headers / 'another.h').write_text('new dependency')
    changed = json.loads(bridge.cached_build(json.dumps(request), tmp_path / 'cache', compile_fake))
    assert not changed['cache_hit']
    assert changed['directory'] != result['directory']


def test_cache_wait_preserves_keyboard_interrupt(tmp_path):
    # The child owns its flock, then requests the same identity on its main thread.
    # A timer supplies SIGINT, so a blocking Rust lock would hit subprocess timeout.
    program = '''
import fcntl,json,os,pathlib,signal,sys,threading
from fireprot.ops._bridge import bridge
request,root=sys.argv[1:]
def build(directory,name):
    pathlib.Path(directory,name+'.so').write_bytes(b'fixture')
    return [name+'.so']
first=json.loads(bridge.cached_build(request,pathlib.Path(root),build))
lock=open(pathlib.Path(first['directory']).with_suffix('.lock'),'a')
fcntl.flock(lock,fcntl.LOCK_EX)
threading.Timer(.2,lambda:os.kill(os.getpid(),signal.SIGINT)).start()
try:
    bridge.cached_build(request,pathlib.Path(root),build)
except KeyboardInterrupt:
    print('interrupted lock wait')
else:
    raise AssertionError('lock wait ignored interruption')
'''
    result = subprocess.run([sys.executable, '-c', program, spec(tmp_path), str(tmp_path / 'cache')],
                            text=True, capture_output=True, timeout=10, check=True)
    assert 'interrupted lock wait' in result.stdout
