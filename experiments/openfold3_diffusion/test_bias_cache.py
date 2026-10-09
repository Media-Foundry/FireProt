"""CPU lifecycle regression checks; real GPU/model gates remain mandatory."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("global_bias_experiment", Path(__file__).with_name("bias_cache.py"))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class Diffusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        transformer = torch.nn.Module()
        transformer.layer_norm_z = torch.nn.LayerNorm(4)
        transformer.blocks = torch.nn.ModuleList()
        for _ in range(24):
            block = torch.nn.Module()
            block.attention_pair_bias = torch.nn.Module()
            block.attention_pair_bias.linear_z = torch.nn.Linear(4, 2, bias=False)
            transformer.blocks.append(block)
        self.diffusion_transformer = transformer

    def forward(self, z, step, mutate=False, fail=False):
        if fail:
            raise RuntimeError("consumer failed")
        t = self.diffusion_transformer
        normal = t.layer_norm_z(z)
        outputs = [b.attention_pair_bias.linear_z(normal) for b in t.blocks]
        result = sum(outputs) + step
        if mutate:
            outputs[-1].add_(1)
        return result


class Sample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_module = Diffusion()

    def _sample_rollout(self, z, replace=False, mutate=False, fail=False):
        result = []
        for i in range(3):
            if replace and i == 1:
                z = z + torch.tensor([0., 1., 2., 3.])
            result.append(self.diffusion_module(z, i, mutate, fail))
        return result


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(cache, "validate_sources", lambda s: s.diffusion_module.diffusion_transformer)
    monkeypatch.setattr(cache, "eligible", lambda value: True)
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True)
    try:
        yield Sample().eval(), torch.randn(1, 1, 3, 3, 4)
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


@pytest.mark.parametrize("verify", [False, True])
def test_repeated_rollouts_release_cache(tmp_path, setup, verify):
    sample, z = setup
    with torch.inference_mode():
        expected = sample._sample_rollout(z)
        with cache.global_bias_cache(sample, tmp_path, verify) as report:
            for _ in range(2):
                actual = sample._sample_rollout(z)
                assert all(cache.exact(a, b) for a, b in zip(actual, expected))
    for record in report["scopes"]:
        assert record["completed"] and record["misses"] == 25 and record["hits"] == 50
        assert record["native_checks"] == (50 if verify else 0)
        assert record["consumer_checks"] == (75 if verify else 0)
    assert "_sample_rollout" not in sample.__dict__
    assert "forward" not in sample.diffusion_module.__dict__
    for block in sample.diffusion_module.diffusion_transformer.blocks:
        assert "forward" not in block.attention_pair_bias.linear_z.__dict__


def test_replaced_pair_invalidates_all_dependent_entries(tmp_path, setup):
    sample, z = setup
    with torch.inference_mode():
        expected = sample._sample_rollout(z, replace=True)
        with cache.global_bias_cache(sample, tmp_path) as report:
            actual = sample._sample_rollout(z, replace=True)
    assert all(cache.exact(a, b) for a, b in zip(actual, expected))
    assert report["scopes"][0]["invalidations"] == 25
    assert report["scopes"][0]["misses"] == 50


def test_verification_detects_consumer_mutation(tmp_path, setup):
    sample, z = setup
    with torch.inference_mode(), pytest.raises(RuntimeError, match="modified by a consumer"):
        with cache.global_bias_cache(sample, tmp_path, verify=True) as report:
            sample._sample_rollout(z, mutate=True)
    assert not report["scopes"][0]["completed"]
    assert "forward" not in sample.diffusion_module.diffusion_transformer.layer_norm_z.__dict__


def test_exception_restores_existing_instance_override(tmp_path, setup):
    sample, z = setup
    previous = sample.diffusion_module.forward
    sample.diffusion_module.forward = previous
    with torch.inference_mode(), pytest.raises(RuntimeError, match="consumer failed"):
        with cache.global_bias_cache(sample, tmp_path) as report:
            sample._sample_rollout(z, fail=True)
    assert sample.diffusion_module.forward is previous
    assert not report["scopes"][0]["completed"]


def test_source_guard_rejects_standin(tmp_path):
    with pytest.raises(RuntimeError, match="source mismatch"):
        with cache.global_bias_cache(Sample().eval(), tmp_path):
            pass


@pytest.mark.parametrize("verify", [False, True])
def test_packed_cache_preserves_values_and_changes_bias_stride(tmp_path, setup, verify):
    sample, z = setup
    with torch.inference_mode():
        expected = sample._sample_rollout(z)
        with cache.global_bias_cache(sample, tmp_path, verify=verify, pack_bias=True) as report:
            actual = sample._sample_rollout(z)
    assert all(cache.exact(a, b) for a, b in zip(actual, expected))
    record = report["scopes"][0]
    assert record["misses"] == 25 and record["hits"] == 50
    assert record["entries"]["projection.0"]["strides"][-3:] == [3, 1, 9]
    assert report["pack_bias"]


def test_packing_preserves_float_bits():
    raw = torch.tensor([0, -2147483648, 2139095040, -8388608, 2143289345, 1065353216],
                       dtype=torch.int32).reshape(1, 1, 1, 2, 3)
    value = raw.view(torch.float32)
    packed = cache.pack_projection(value)
    assert cache.exact(value, packed)
    assert packed.permute(0, 1, 4, 2, 3).is_contiguous()
