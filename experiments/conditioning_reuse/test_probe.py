"""CPU checks of diagnostic sensitivity and method restoration, not model parity."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("conditioning_probe", Path(__file__).with_name("probe.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Conditioning(torch.nn.Module):
    def __init__(self, time_dependent=False, random=False):
        super().__init__()
        self.time_dependent = time_dependent
        self.random = random

    def _forward(self, si, zij, token_mask, chunk_size=None):
        return si + 1, zij.clone()

    def forward(self, batch, t, si_input, si_trunk, zij_trunk,
                use_conditioning, chunk_size=None):
        if self.random:
            torch.rand(1)
        pair = zij_trunk.clone()
        if self.time_dependent:
            # Numeric equality would miss this deliberate bit change.
            pair[0] = -0.0 if t.item() else 0.0
        return self._forward(si_trunk + t, pair, batch["token_mask"], chunk_size)


class Diffusion(torch.nn.Module):
    def __init__(self, mutate=False, fail=False, **kwargs):
        super().__init__()
        self.diffusion_conditioning = Conditioning(**kwargs)
        self.mutate = mutate
        self.fail = fail

    def forward(self, **kwargs):
        single, pair = self.diffusion_conditioning(**kwargs)
        if self.mutate:
            pair.add_(1)
        if self.fail:
            raise RuntimeError("consumer failed")
        return single + pair


class Sample(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.diffusion_module = Diffusion(**kwargs)

    def forward(self, pair):
        return [self.diffusion_module(
            batch={"token_mask": torch.ones(2)}, t=torch.tensor(float(i)),
            si_input=torch.zeros(2), si_trunk=torch.ones(2), zij_trunk=pair,
            use_conditioning=True, chunk_size=8) for i in range(3)]


def test_scopes_rng_chunks_and_original_results(tmp_path):
    model = Sample()
    pair = torch.tensor([float("nan"), -0.0])
    with torch.inference_mode():
        reference = model(pair)
        with module.conditioning_probe(model, tmp_path, (0, 1, 2)) as report:
            for shift in (0, 2):
                result = model(pair + shift)
                if shift == 0:
                    for a, b in zip(reference, result):
                        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert len(report["scopes"]) == 2
    for scope in report["scopes"]:
        assert scope["completed"]
        for call in scope["calls"]:
            assert call["chunks"] == [8]
            assert call["cpu_rng_unchanged"]
            assert call["pair_unchanged_after_consumers"]
            assert call.get("equal_to_first_sampled_pair", True)
    assert "forward" not in model.__dict__
    assert "_forward" not in model.diffusion_module.diffusion_conditioning.__dict__
    assert json.loads((tmp_path / "conditioning-audit.json").read_text()) == report


@pytest.mark.parametrize("option,field", [
    ("time_dependent", "equal_to_first_sampled_pair"),
    ("mutate", "pair_unchanged_after_consumers"),
    ("random", "cpu_rng_unchanged"),
])
def test_detects_invalid_reuse_evidence(tmp_path, option, field):
    model = Sample(**{option: True})
    with module.conditioning_probe(model, tmp_path, (0, 1)) as report:
        model(torch.zeros(2))
    assert report["scopes"][0]["calls"][1][field] is False


def test_exception_restores_preexisting_override(tmp_path):
    model = Sample(fail=True)
    conditioning = model.diffusion_module.diffusion_conditioning
    original = conditioning.forward
    conditioning.forward = original
    with pytest.raises(RuntimeError, match="consumer failed"):
        with module.conditioning_probe(model, tmp_path) as report:
            model(torch.zeros(2))
    assert conditioning.forward is original
    assert "forward" not in model.__dict__
    assert "forward" not in model.diffusion_module.__dict__
    assert not report["scopes"][0]["completed"]
    assert "pair_unchanged_after_consumers" not in report["scopes"][0]["calls"][0]
