"""CPU checks for the maintained gfx1100 composition and fallback lifetime."""
from contextlib import contextmanager
import importlib
import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def harness(monkeypatch):
    route = importlib.import_module('fireprot.adapters.openfold3.gfx1100')
    state = SimpleNamespace(events=[])
    monkeypatch.setattr(route, 'Gfx1100Softmax', lambda: object())
    monkeypatch.setattr(route, 'Gfx1100Dispatcher', lambda kernel: SimpleNamespace(
        manifest=lambda: dict(launches_submitted=0, test=True)))

    @contextmanager
    def attention(dispatcher):
        state.events.append('attention-enter')
        try:
            yield
        finally:
            state.events.append('attention-exit')

    def cache(name):
        @contextmanager
        def scope(model, output):
            state.events.append(name + '-enter')
            try:
                yield dict(scopes=[dict(calls=200, hits=199, misses=1, fallbacks=0, completed=True)])
            finally:
                state.events.append(name + '-exit')
        return scope

    @contextmanager
    def template(model, output, source):
        state.events.append('template-enter')
        try:
            yield
        finally:
            state.events.append('template-exit')
            (output / 'template-reuse.json').write_text(json.dumps(dict(counts=dict(reused_slots=27))))

    monkeypatch.setattr(route, 'attention_scope', attention)
    monkeypatch.setattr(route, 'pair_cache', cache('pair'))
    monkeypatch.setattr(route, 'atom_cache', cache('atom'))
    monkeypatch.setattr(route, 'template_reuse_scope', template)
    monkeypatch.setattr(route.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(route.torch.cuda, 'max_memory_allocated', lambda: 100)
    monkeypatch.setattr(route.torch.cuda, 'max_memory_reserved', lambda: 200)
    return route, state


@pytest.mark.parametrize('length,steps,recycles,samples,reuse', [
    (1000, 200, 3, 1, True), (1536, 200, 3, 1, True),
    (124, 200, 3, 1, True), (256, 200, 3, 1, True),
    (800, 200, 3, 1, True), (1001, 200, 3, 1, True),
    (1537, 200, 3, 1, True), (0, 200, 3, 1, False),
    (1000, 20, 3, 1, False), (1000, 200, 2, 1, False),
    (1000, 200, 3, 2, False),
])
@pytest.mark.parametrize('fail', [False, True])
def test_scope_fallback_and_exception(harness, tmp_path, length, steps, recycles, samples, reuse, fail):
    route, state = harness
    shared = SimpleNamespace(num_recycles=recycles, diffusion=SimpleNamespace(
        no_full_rollout_steps=steps, no_full_rollout_samples=samples))
    model = SimpleNamespace(config=SimpleNamespace(architecture=SimpleNamespace(shared=shared)),
                            sample_diffusion=object())
    batch = dict(token_mask=SimpleNamespace(shape=(1, length)))
    def original(obj, data):
        assert obj is model and data is batch
        if fail:
            raise ValueError('upstream failed')
        return 'native-return'
    if fail:
        with pytest.raises(ValueError, match='upstream failed'):
            route.forward(model, original, (batch,), {}, tmp_path)
    else:
        assert route.forward(model, original, (batch,), {}, tmp_path) == 'native-return'
    expected = ['attention-enter']
    if reuse:
        expected += ['pair-enter', 'atom-enter', 'template-enter', 'template-exit', 'atom-exit', 'pair-exit']
    expected += ['attention-exit']
    assert state.events == expected
    report = json.loads((tmp_path / 'gfx1100-composition.json').read_text())
    assert report['completed'] is not fail
    assert report['reuse_enabled'] is reuse
    assert report['token_count'] == length
    assert set(report['caches']) == ({'pair', 'atom'} if reuse else set())
    assert (tmp_path / 'operator-runtime.json').exists()


def test_explicit_native_ignores_default_composition(monkeypatch, tmp_path):
    from fireprot_workers.openfold3 import forward_with_backend
    route = importlib.import_module('fireprot.adapters.openfold3.gfx1100')
    monkeypatch.setenv('FIREPROT_OPENFOLD3_GFX1100', '1')
    def forbidden(*args, **kwargs):
        raise AssertionError('Explicit native must not enter the composition')
    monkeypatch.setattr(route, 'forward', forbidden)
    assert forward_with_backend(None, lambda model: 'native', (), {},
                                backend='native', output=tmp_path) == 'native'
