"""Checkpoint coverage and random-state contracts for deferred construction."""
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch
from scipy.stats import truncnorm

from fireprot.adapters.protenix import construction as c


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("FIREPROT_ENABLE_FAST_MODEL_INIT", "1")
    monkeypatch.setattr(c, "_counts", dict(attempted=0, selected=0, recovered=0, native=0, errors=0))
    monkeypatch.setattr(c, "_records", [])
    monkeypatch.setattr(c, "_reasons", {})

    def initialize(weights, scale=1., fan="fan_in"):
        values = truncnorm.rvs(-2, 2, size=weights.numel())
        with torch.no_grad():
            weights.copy_(torch.tensor(values).reshape_as(weights))

    layers = SimpleNamespace(trunc_normal_init_=initialize)
    primitives = SimpleNamespace(trunc_normal_init_=initialize)

    class Runner:
        def __init__(self, missing=False, strict=False, fail=False):
            self.configs = SimpleNamespace(model_name="protenix_mini_default_v0.5.0", dtype="fp32")
            self.device = "cpu"
            self.missing, self.strict, self.fail = missing, strict, fail
            self.loads = 0

        def init_model(self):
            self.model = torch.nn.Linear(4, 4)
            layers.trunc_normal_init_(self.model.weight)
            random.random()
            if self.fail:
                raise ValueError("constructor failure")

        def load_checkpoint(self):
            self.loads += 1
            state = {"bias": torch.ones(4)}
            if not self.missing:
                state["weight"] = torch.ones(4, 4)
            self.model.load_state_dict(state, strict=self.strict)
            self.model.eval()

    return Runner, layers, primitives, initialize


def same_rng(a, b):
    assert a["python"] == b["python"]
    assert a["numpy"][0] == b["numpy"][0]
    np.testing.assert_array_equal(a["numpy"][1], b["numpy"][1])
    assert a["numpy"][2:] == b["numpy"][2:]
    assert torch.equal(a["cpu"], b["cpu"])


@pytest.mark.parametrize("missing", [False, True])
def test_checkpoint_and_rng_match_native(setup, missing):
    Runner, layers, primitives, original = setup
    before = c._capture_rng("cpu")
    reference = Runner(missing=missing)
    reference.init_model()
    reference.load_checkpoint()
    after = c._capture_rng("cpu")
    c._restore_rng(before)
    c._patch_runner(Runner, layers, primitives, truncnorm)
    candidate = Runner(missing=missing)
    candidate.init_model()
    candidate.load_checkpoint()
    same_rng(after, c._capture_rng("cpu"))
    for key, value in reference.model.state_dict().items():
        assert torch.equal(value, candidate.model.state_dict()[key])
    assert candidate.loads == 1
    assert not candidate.model.training
    assert not hasattr(candidate, c._PENDING)
    assert "load_state_dict" not in candidate.model.__dict__
    assert layers.trunc_normal_init_ is primitives.trunc_normal_init_ is original
    assert c._counts["recovered" if missing else "selected"] == 1
    assert c._records[0]["uniform_elements"] == 16


@pytest.mark.parametrize("failure", ["constructor", "checkpoint"])
def test_exception_restores_bindings(setup, failure):
    Runner, layers, primitives, original = setup
    c._patch_runner(Runner, layers, primitives, truncnorm)
    runner = Runner(missing=True, strict=True, fail=failure == "constructor")
    with pytest.raises((ValueError, RuntimeError)):
        runner.init_model()
        runner.load_checkpoint()
    assert layers.trunc_normal_init_ is primitives.trunc_normal_init_ is original
    assert not hasattr(runner, c._PENDING)
    assert "load_state_dict" not in runner.model.__dict__
    assert c._counts["errors"] == 1


def test_disabled_is_native(setup, monkeypatch):
    Runner, layers, primitives, _ = setup
    c._patch_runner(Runner, layers, primitives, truncnorm)
    monkeypatch.setenv("FIREPROT_ENABLE_FAST_MODEL_INIT", "0")
    runner = Runner()
    runner.init_model()
    runner.load_checkpoint()
    assert c._counts["native"] == 1
    assert c._counts["attempted"] == 0


def test_replay_restores_rng_on_error():
    before = c._capture_rng("cpu")
    random.random()
    np.random.uniform()
    torch.rand(1)
    current = c._capture_rng("cpu")
    with pytest.raises(ValueError), c._replay_rng(before):
        same_rng(before, c._capture_rng("cpu"))
        raise ValueError("replay error")
    same_rng(current, c._capture_rng("cpu"))


@pytest.mark.parametrize("field,value,reason", [
    ("model_name", "unknown", "unsupported_model"),
    ("dtype", "bf16", "unsupported_dtype"),
])
def test_unsupported_configuration_is_native(setup, field, value, reason):
    Runner, layers, primitives, _ = setup
    c._patch_runner(Runner, layers, primitives, truncnorm)
    runner = Runner()
    setattr(runner.configs, field, value)
    runner.init_model()
    assert c._reasons == {reason: 1}
    assert not hasattr(runner, c._PENDING)


def test_other_python_thread_disables_skip(setup):
    import threading
    Runner, layers, primitives, _ = setup
    c._patch_runner(Runner, layers, primitives, truncnorm)
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait)
    thread.start()
    try:
        runner = Runner()
        runner.init_model()
        assert c._reasons == {"requires_single_python_thread": 1}
    finally:
        stop.set()
        thread.join()


def test_nonparameter_initializer_uses_native(setup):
    _, layers, primitives, original = setup
    before = c._capture_rng("cpu")
    expected = torch.empty(4, 4)
    original(expected)
    after = c._capture_rng("cpu")
    c._restore_rng(before)
    stats = dict(uniform_calls=0, uniform_elements=0, native_initializer_calls=0)
    with c._defer_initializers(layers, primitives, truncnorm, stats):
        actual = torch.empty(4, 4)
        layers.trunc_normal_init_(actual)
    assert torch.equal(expected, actual)
    same_rng(after, c._capture_rng("cpu"))
    assert stats["native_initializer_calls"] == 1
    assert stats["uniform_calls"] == 0
