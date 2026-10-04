"""Integration checks at the CPython/Rust policy boundary (no GPU required)."""
from __future__ import annotations

import gc
import json
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from fireprot.ops._bridge import bridge
from fireprot.ops.registry import OperationRegistry, OperationSignature, signature_from_tensor


def signature(**overrides):
    values = dict(operation="test", device_type="cuda", architecture="gfx90a",
                  dtype="float32", shape=(4, 4), strides=(4, 1))
    values.update(overrides)
    return OperationSignature.create(**values)


def register(registry, name="one", supports=lambda _: True, **kw):
    return registry.register(operation="test", name=name, function=lambda: name,
                             supports=supports, **kw)


def test_cache_identity_includes_strides_and_attributes():
    calls = []
    registry = OperationRegistry()
    register(registry, supports=lambda s: calls.append(s) or s.strides == (4, 1))
    assert not registry.resolve(signature()).cache_hit
    assert registry.resolve(signature()).cache_hit
    other = signature(strides=(1, 4))
    assert registry.resolve(other).implementation is None
    assert not registry.resolve(signature(attributes={"policy": "new"})).cache_hit
    assert len(calls) == 3
    assert calls[0] == signature()
    assert registry.manifest()["policy_engine"]["name"] == "fireprot-core"


def test_ties_removal_history_and_reset():
    registry = OperationRegistry()
    register(registry, "z", priority=10)
    register(registry, "a", priority=10)
    assert registry.dispatch(signature(), lambda: "native") == "a"
    with pytest.raises(ValueError):
        register(registry, "a", priority=100)
    assert registry.resolve(signature()).cache_hit
    registry.unregister("test", "a")
    assert registry.manifest()["decisions"][0]["selected"] == "a"
    selected = registry.resolve(signature())
    assert selected.implementation.name == "z" and not selected.cache_hit
    with pytest.raises(KeyError):
        registry.unregister("test", "missing")
    registry.clear_runtime_state()
    assert registry.manifest()["selection_counts"] == []
    assert registry.manifest()["decisions"] == []
    assert registry.dispatch(signature(), lambda: "native") == "z"


def test_concurrent_resolution_when_predicate_releases_gil():
    registry = OperationRegistry()
    calls = []

    def supports(s):
        calls.append(s)
        time.sleep(0.01)
        return True

    register(registry, supports=supports)
    with ThreadPoolExecutor(max_workers=8) as pool:
        selections = list(pool.map(lambda _: registry.resolve(signature()), range(40)))
    assert sum(not selection.cache_hit for selection in selections) == 1
    assert len(calls) == 1
    assert registry.manifest()["selection_counts"] == [
        dict(operation="test", implementation="one", count=40)]


def test_reentrant_mutation_is_rejected_without_corrupting_policy():
    registry = OperationRegistry()
    register(registry, supports=lambda _: registry.clear_runtime_state() or True)
    result = registry.resolve(signature())
    assert result.implementation is None
    assert "Already borrowed" in result.rejected[0][1]
    assert registry.resolve(signature()).cache_hit
    registry.unregister("test", "one")
    register(registry, "valid")
    assert registry.dispatch(signature(), lambda: "native") == "valid"


@pytest.mark.parametrize("raw_bridge", [False, True])
def test_callbacks_participate_in_python_cycle_collection(raw_bridge):
    class Owner:
        def __init__(self):
            self.registry = bridge.OperatorRegistry() if raw_bridge else OperationRegistry()

        def supports(self, _):
            return self.registry is not None

    owner = Owner()
    reference = weakref.ref(owner)
    if raw_bridge:
        owner.registry.register(json.dumps(dict(operation="test", name="cycle", priority=0,
                                                version="1")), owner.supports)
    else:
        register(owner.registry, supports=owner.supports)
    del owner
    gc.collect()
    assert reference() is None


def test_tensor_signature_records_actual_strides():
    import torch
    tensor = torch.empty(4, 4).t()
    observed = signature_from_tensor("test", tensor, architecture="cpu", mode="inference",
                                     mask="none", inplace=False)
    assert observed.strides == (1, 4)
    assert observed.as_dict()["strides"] == [1, 4]
    assert observed.layout == "strided"


def test_invalid_metadata_cannot_select_a_candidate():
    registry = OperationRegistry()
    with pytest.raises(ValueError):
        register(registry, "")
    with pytest.raises(ValueError):
        register(registry, priority=2**40)
    assert registry.dispatch(signature(), lambda: "native") == "native"


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate_without_cached_fallback(error_type):
    registry = OperationRegistry()
    registry.resolve(signature(operation='history'))
    before = registry.manifest()
    calls = []

    def interrupted(_):
        calls.append('interrupted')
        raise error_type('stop now')

    register(registry, 'interrupt', supports=interrupted, priority=100)
    register(registry, 'unchecked', supports=lambda _: calls.append('unchecked') or True)
    with pytest.raises(error_type, match='stop now'):
        registry.resolve(signature())
    assert calls == ['interrupted']
    assert registry.manifest()['selection_counts'] == before['selection_counts']
    assert registry.manifest()['decisions'] == before['decisions']
    registry.unregister('test', 'interrupt')
    assert registry.dispatch(signature(), lambda: 'native') == 'unchecked'
