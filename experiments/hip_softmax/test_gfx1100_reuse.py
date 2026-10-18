"""CPU checks for experimental composition gates and scope restoration."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def harness(monkeypatch):
    state = SimpleNamespace(closed=[], counts=dict(calls=200, hits=199, misses=1, fallbacks=0),
                            arch='gfx1100', reused_slots=3)
    package = '_fireprot_reuse_test'
    parent = ModuleType(package)
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, package, parent)

    def cache(name):
        @contextmanager
        def scope(model, output):
            item = dict(scopes=[dict(state.counts, completed=False)])
            try:
                yield item
                item['scopes'][0]['completed'] = True
            finally:
                state.closed.append(name)
        return scope

    @contextmanager
    def template(model, output, source_hash):
        assert len(source_hash) == 64
        try:
            yield
        finally:
            state.closed.append('template')
            (output / 'template-reuse.json').write_text(json.dumps(
                dict(counts=dict(reused_slots=state.reused_slots))))

    for name, attr, value in [('pair_cache', 'pair_cache', cache('pair')),
                              ('atom_cache', 'atom_cache', cache('atom')),
                              ('template_reuse', 'template_reuse_scope', template)]:
        module = ModuleType(f'{package}.{name}')
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    cuda = SimpleNamespace(current_device=lambda: 0,
        get_device_properties=lambda index: SimpleNamespace(gcnArchName=state.arch),
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 100, max_memory_reserved=lambda: 200)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    spec = importlib.util.spec_from_file_location(f'{package}.gfx1100_reuse',
        Path(__file__).with_name('gfx1100_reuse.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reuse_scope, SimpleNamespace(sample_diffusion=object()), state


@pytest.mark.parametrize('mode', range(4))
def test_modes_and_reverse_restoration(harness, monkeypatch, tmp_path, mode):
    scope, model, state = harness
    monkeypatch.setenv('FIREPROT_GFX1100_REUSE_MODE', str(mode))
    with scope(model, tmp_path):
        pass
    report = json.loads((tmp_path / 'reuse-runtime.json').read_text())
    assert report['completed'] and report['mode'] == mode
    expected = ['pair', 'atom', 'template'][:mode]
    assert state.closed == expected[::-1]
    assert set(report['caches']) == set(expected) - {'template'}
    assert report['peak_allocated_bytes'] == 100


@pytest.mark.parametrize('bad_count', ['calls', 'hits', 'misses', 'fallbacks'])
def test_wrong_coverage_fails(harness, monkeypatch, tmp_path, bad_count):
    scope, model, state = harness
    state.counts[bad_count] += 1
    monkeypatch.setenv('FIREPROT_GFX1100_REUSE_MODE', '2')
    with pytest.raises(RuntimeError, match='200-call pattern'):
        with scope(model, tmp_path):
            pass
    assert state.closed == ['atom', 'pair']
    assert not json.loads((tmp_path / 'reuse-runtime.json').read_text())['completed']


def test_zero_template_reuse_fails(harness, monkeypatch, tmp_path):
    scope, model, state = harness
    state.reused_slots = 0
    monkeypatch.setenv('FIREPROT_GFX1100_REUSE_MODE', '3')
    with pytest.raises(RuntimeError, match='Template reuse did not execute'):
        with scope(model, tmp_path):
            pass
    assert state.closed == ['template', 'atom', 'pair']


def test_forward_failure_restores_all(harness, monkeypatch, tmp_path):
    scope, model, state = harness
    monkeypatch.setenv('FIREPROT_GFX1100_REUSE_MODE', '3')
    with pytest.raises(ValueError, match='forward failed'):
        with scope(model, tmp_path):
            raise ValueError('forward failed')
    assert state.closed == ['template', 'atom', 'pair']
    assert not json.loads((tmp_path / 'reuse-runtime.json').read_text())['completed']


@pytest.mark.parametrize('mode,arch,error', [('4', 'gfx1100', ValueError), ('0', 'gfx90a', RuntimeError)])
def test_unsupported_configuration(harness, monkeypatch, tmp_path, mode, arch, error):
    scope, model, state = harness
    state.arch = arch
    monkeypatch.setenv('FIREPROT_GFX1100_REUSE_MODE', mode)
    with pytest.raises(error):
        with scope(model, tmp_path):
            pass
    assert state.closed == []
