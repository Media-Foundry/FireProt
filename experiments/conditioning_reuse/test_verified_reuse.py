"""Adversarial lifecycle checks; upstream arithmetic requires GPU validation."""
import importlib.util
from pathlib import Path

import pytest
import torch


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reuse = load("verified_reuse", "verified_reuse.py")
standins = load("probe_standins", "test_probe.py")


class Sample(standins.Sample):
    def _sample_rollout(self, pair):
        return super().forward(pair)

    def forward(self, pair):
        return self._sample_rollout(pair)


@pytest.fixture(autouse=True)
def substitute_single_branch(monkeypatch):
    # Exercise lifecycle independently of the production arithmetic helper.
    monkeypatch.setattr(reuse, "single_conditioning",
                        lambda module, inputs, chunk: inputs["si_trunk"] + inputs["t"] + 1)


def test_every_call_compared_and_scope_reset(tmp_path):
    model = Sample().eval()
    with torch.inference_mode():
        with reuse.verified_reuse(model, tmp_path) as report:
            for pair in (torch.tensor([float("nan"), -0.0]), torch.ones(2)):
                result = model(pair)
                assert len(result) == 3
        assert len(report["scopes"]) == 2
        for scope in report["scopes"]:
            assert scope["completed"]
            assert [c["reused"] for c in scope["calls"]] == [False, True, True]
            assert all(c["pair_unchanged_after_consumers"] for c in scope["calls"])
    assert "_sample_rollout" not in model.__dict__
    assert "forward" not in model.diffusion_module.__dict__
    assert "_forward" not in model.diffusion_module.diffusion_conditioning.__dict__


@pytest.mark.parametrize("option,error", [
    ("time_dependent", "differs from native"),
    ("random", "differs from native"),
    ("mutate", "mutated retained pair"),
    ("fail", "consumer failed"),
])
def test_rejects_wrong_reuse_and_restores_on_exception(tmp_path, option, error):
    model = Sample(**{option: True}).eval()
    conditioning = model.diffusion_module.diffusion_conditioning
    original = conditioning.forward
    conditioning.forward = original
    with torch.inference_mode(), pytest.raises(RuntimeError, match=error):
        with reuse.verified_reuse(model, tmp_path) as report:
            model(torch.zeros(2))
    assert conditioning.forward is original
    assert not report["scopes"][0]["completed"]
    assert "_sample_rollout" not in model.__dict__


def test_rejects_wrong_single_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(reuse, "single_conditioning",
                        lambda module, inputs, chunk: inputs["si_trunk"])
    model = Sample().eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="differs from native"):
        with reuse.verified_reuse(model, tmp_path):
            model(torch.zeros(2))


def test_grad_mode_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="requires inference"):
        model = Sample().eval()
        with reuse.verified_reuse(model, tmp_path):
            model(torch.zeros(2))
