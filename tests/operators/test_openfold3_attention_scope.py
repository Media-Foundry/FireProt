"""CPU checks for scope isolation and Rust-owned fallback selection."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest
import torch

from fireprot.adapters.openfold3.attention import AttentionDispatcher, _scoped_hook


def test_scope_isolates_other_threads_and_restores_original():
    native = lambda x: ("native", x)
    target = SimpleNamespace(_attention=native)
    selected = lambda original, x: ("selected", x)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with _scoped_hook(target, selected):
            assert target._attention(1) == ("selected", 1)
            assert pool.submit(target._attention, 2).result() == ("native", 2)
            leaked = target._attention
            inherited = copy_context()
        assert target._attention is native
        assert leaked(3) == ("native", 3)
        assert inherited.run(leaked, 4) == ("native", 4)


def test_nested_and_concurrent_installs_rejected_without_displacing_outer():
    target = SimpleNamespace(_attention=lambda: "native")
    def attempt():
        with _scoped_hook(target, lambda original: "inner"):
            pytest.fail("Concurrent installation was accepted")
    with _scoped_hook(target, lambda original: "outer"):
        with pytest.raises(RuntimeError, match="already installed"):
            attempt()
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="already installed"):
                pool.submit(attempt).result()
        assert target._attention() == "outer"
    assert target._attention() == "native"


def test_exception_cleanup_and_external_replacement_detection():
    native = lambda: "native"
    target = SimpleNamespace(_attention=native)
    with pytest.raises(ValueError, match="model failure"):
        with _scoped_hook(target, lambda original: "selected"):
            raise ValueError("model failure")
    assert target._attention is native
    replacement = lambda: "foreign"
    with pytest.raises(RuntimeError, match="changed during"):
        with _scoped_hook(target, lambda original: "selected"):
            target._attention = replacement
    assert target._attention is replacement
    # A failed cleanup releases the installation guard without overwriting the
    # foreign owner. A later explicit scope is independently usable.
    with _scoped_hook(target, lambda original: original()):
        assert target._attention() == "foreign"


def test_native_fallback_uses_rust_registry_and_preserves_call_arguments():
    runtime = AttentionDispatcher(None)
    query = torch.zeros(1, 4, 3, 32)
    biases = [torch.zeros(1)]
    calls = []
    def native(q, k, v, b, use_high_precision=False):
        calls.append((q, k, v, b, use_high_precision))
        return q
    with torch.no_grad():
        for _ in range(2):
            assert runtime(native, query, query, query, biases, True) is query
    assert len(calls) == 2
    assert all(call[3] is biases and call[4] is True for call in calls)
    report = runtime.manifest()
    assert report["kernel"] is None
    assert report["registry"]["policy_engine"]["name"] == "fireprot-core"
    assert all(d["selected"] == "native" for d in report["registry"]["decisions"])
