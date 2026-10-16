"""Explicit upstream deterministic execution must be enforced before forward."""
from types import SimpleNamespace
from contextlib import contextmanager
import json

import pytest

from python.fireprot_workers.openfold3 import check_determinism, forward_with_backend


@pytest.mark.parametrize('requested,enabled,warn,passes', [
    (None, False, False, True),
    (None, True, True, True),
    (True, True, False, True),
    (False, False, False, True),
    (True, False, False, False),
    (False, True, False, False),
    (True, True, True, False),
])
def test_upstream_policy_is_enforced(requested, enabled, warn, passes):
    torch = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: enabled,
        is_deterministic_algorithms_warn_only_enabled=lambda: warn,
    )
    workload = {} if requested is None else {'deterministic': requested}
    if passes:
        check_determinism(torch, workload)
    else:
        with pytest.raises(RuntimeError):
            check_determinism(torch, workload)


def test_native_forward_does_not_load_hip_assets(monkeypatch, tmp_path):
    from fireprot.backends.amd import softmax_fp32
    def forbidden():
        raise AssertionError('Native requests must not load HIP artifacts')
    monkeypatch.setattr(softmax_fp32, 'SoftmaxFP32', forbidden)
    model = object()
    result = forward_with_backend(model, lambda m, x, *, value: (m, x, value),
                                  (2,), {'value': 3}, backend='native', output=tmp_path)
    assert result == (model, 2, 3)
    assert not (tmp_path / 'operator-runtime.json').exists()


@pytest.mark.parametrize('fail', [False, True])
def test_hip_forward_restores_scope_and_records_manifest(monkeypatch, tmp_path, fail):
    from fireprot.backends.amd import softmax_fp32
    from fireprot.adapters.openfold3 import attention
    states = []
    kernel = object()
    monkeypatch.setattr(softmax_fp32, 'SoftmaxFP32', lambda: kernel)
    class Dispatcher:
        def __init__(self, selected):
            assert selected is kernel
        def manifest(self):
            return {'launches_submitted': 1}
    @contextmanager
    def scope(dispatcher):
        states.append('entered')
        try:
            yield
        finally:
            states.append('exited')
    monkeypatch.setattr(attention, 'AttentionDispatcher', Dispatcher)
    monkeypatch.setattr(attention, 'attention_scope', scope)
    def forward(model, *, token):
        assert states == ['entered'] and token == 7
        if fail:
            raise RuntimeError('model error')
        return model
    model = object()
    if fail:
        with pytest.raises(RuntimeError, match='model error'):
            forward_with_backend(model, forward, (), {'token': 7}, backend='hip_fused', output=tmp_path)
    else:
        assert forward_with_backend(model, forward, (), {'token': 7}, backend='hip_fused', output=tmp_path) is model
    assert states == ['entered', 'exited']
    assert json.loads((tmp_path / 'operator-runtime.json').read_text()) == {'launches_submitted': 1}


@pytest.mark.parametrize('backend,accepted', [('native', True), ('hip_fused', True), ('fireprot', False)])
def test_worker_backend_admission(monkeypatch, tmp_path, backend, accepted):
    from python.fireprot_workers import __main__ as entry, openfold3
    path = tmp_path / 'request.json'
    path.write_text(json.dumps({'backend': backend, 'workload': {'model': 'openfold3'}}))
    monkeypatch.setenv('FIREPROT_REQUEST', str(path))
    events, outputs, closed = [], [], []
    def ready(torch, prepare_cpu=None):
        assert prepare_cpu is None
    channel = SimpleNamespace(ready=ready, emit=events.append,
                              socket=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(entry, 'Channel', lambda request: channel)
    monkeypatch.setattr(openfold3, 'predict', lambda request, channel: {'model_called': True})
    monkeypatch.setattr(entry, 'announce_outputs', lambda channel, bundle: outputs.append(bundle))
    if accepted:
        entry.main()
        assert outputs[0]['model_called']
    else:
        with pytest.raises(ValueError, match='not integrated'):
            entry.main()
        assert not outputs and events[-1]['event'] == 'failed'
    assert closed == [True]
