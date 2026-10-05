"""Cache lifecycle tests; these do not replace real-model numerical gates."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
try:
    import pair_cache as cache
finally:
    sys.path.pop(0)

spec = importlib.util.spec_from_file_location("reuse_standins", Path(__file__).with_name("test_probe.py"))
standins = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standins)


class Sample(standins.Sample):
    def _sample_rollout(self, inputs, replace_at=None):
        result = []
        for index in range(3):
            if index == replace_at:
                inputs = dict(inputs, zij_trunk=inputs["zij_trunk"] + 4)
            values = dict(inputs, t=torch.tensor(float(index)))
            result.append(self.diffusion_module(**values))
        return result


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.setattr(cache, "validate_sources", lambda *args: None)
    monkeypatch.setattr(cache, "eligible", lambda inputs: True)
    monkeypatch.setattr(cache, "single_conditioning",
                        lambda module, inputs, chunk: inputs["si_trunk"] + inputs["t"] + 1)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield dict(batch={"token_mask": torch.ones(2)}, si_input=torch.zeros(2),
                   si_trunk=torch.ones(2), zij_trunk=torch.zeros(2),
                   use_conditioning=True, chunk_size=8)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_hits_and_rollout_invalidation(tmp_path, experiment):
    sample = Sample().eval()
    with torch.inference_mode():
        expected = sample._sample_rollout(experiment)
        with cache.pair_cache(sample, tmp_path) as report:
            for _ in range(2):
                actual = sample._sample_rollout(experiment)
                assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert [(s["hits"], s["misses"]) for s in report["scopes"]] == [(2, 1), (2, 1)]
    assert "_sample_rollout" not in sample.__dict__


def test_input_replacement_invalidates_pair(tmp_path, experiment):
    sample = Sample().eval()
    with torch.inference_mode():
        expected = sample._sample_rollout(experiment, replace_at=1)
        with cache.pair_cache(sample, tmp_path) as report:
            actual = sample._sample_rollout(experiment, replace_at=1)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert report["scopes"][0]["misses"] == 2
    assert report["scopes"][0]["hits"] == 1


def test_unsupported_input_uses_native(tmp_path, experiment, monkeypatch):
    monkeypatch.setattr(cache, "eligible", lambda inputs: False)
    sample = Sample().eval()
    with torch.inference_mode(), cache.pair_cache(sample, tmp_path) as report:
        sample._sample_rollout(experiment)
    assert report["scopes"][0]["fallbacks"] == 3
    assert report["scopes"][0]["hits"] == 0


def test_consumer_failure_restores_methods(tmp_path, experiment):
    sample = Sample(fail=True).eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="consumer failed"):
        with cache.pair_cache(sample, tmp_path) as report:
            sample._sample_rollout(experiment)
    assert not report["scopes"][0]["completed"]
    assert "forward" not in sample.diffusion_module.diffusion_conditioning.__dict__
    assert "_sample_rollout" not in sample.__dict__


def test_source_guard_rejects_unpinned_module(tmp_path):
    sample = Sample().eval()
    with pytest.raises(RuntimeError, match="source mismatch"):
        with cache.pair_cache(sample, tmp_path):
            pass
    assert "_sample_rollout" not in sample.__dict__


def test_warn_only_determinism_rejected(tmp_path, experiment):
    sample = Sample().eval()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="strict deterministic"):
        with cache.pair_cache(sample, tmp_path):
            sample._sample_rollout(experiment)
