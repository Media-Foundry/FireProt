"""Explicit opt-in and nontrivial numerical gates for N1536 pair epilogues."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fireprot.backends.amd import recycle
from fireprot.ops.registry import OperationSignature


def test_long1536_epilogues_require_opt_in_and_exact_shape(monkeypatch):
    monkeypatch.setattr(recycle, "load_amd_extension", lambda: object())
    monkeypatch.setattr(recycle, "_long_layer_norm_enabled", True)
    monkeypatch.setattr(recycle, "_long_transition_enabled", True)
    common = dict(device_type="cuda", architecture="gfx90a", dtype="float32",
                  shape=(1536, 1536, 128), layout="contiguous", mode="inference",
                  mask="none", inplace=False)
    for operation, predicate, attributes in (
        (recycle.LAYER_NORM_OPERATION, recycle._supports_long_layer_norm,
         dict(scale=True, offset=True, epsilon=1e-5)),
        (recycle.TRANSITION_OPERATION, recycle._supports_long_transition, dict(expansion=4)),
    ):
        values = dict(common, operation=operation, attributes=attributes)
        monkeypatch.setattr(recycle, "_long1536_epilogues_enabled", False)
        assert predicate(OperationSignature.create(**values)) is not True
        monkeypatch.setattr(recycle, "_long1536_epilogues_enabled", True)
        assert predicate(OperationSignature.create(**values)) is True
        for change in (dict(shape=(1536, 1535, 128)), dict(dtype="bfloat16"),
                       dict(layout="strided"), dict(mode="training"), dict(architecture="gfx942")):
            assert predicate(OperationSignature.create(**(values | change))) is not True
    monkeypatch.setattr(recycle, "load_amd_extension", lambda: None)
    assert recycle._supports_long_transition(OperationSignature.create(
        **dict(common, operation=recycle.TRANSITION_OPERATION, attributes=dict(expansion=4)))) is not True


hip_only = pytest.mark.skipif(
    os.environ.get("FIREPROT_RUN_HIP_TESTS") != "1" or not torch.version.hip,
    reason="explicit HIP test opt-in required",
)


@hip_only
def test_long1536_layernorm_dispatch_matches_torch(monkeypatch):
    monkeypatch.setattr(recycle, "_long1536_epilogues_enabled", True)
    assert recycle.load_amd_extension() is not None
    torch.manual_seed(631)
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        source = torch.randn(1536, 1536, 128, device="cuda")
        original = source.clone()
        module = SimpleNamespace(training=False, eps=1e-5,
            weight=torch.randn(128, device="cuda"), bias=torch.randn(128, device="cuda"))
        expected = F.layer_norm(source, (128,), module.weight, module.bias, module.eps)
        actual = recycle.dispatch_layer_norm(module=module, x=source,
                                             native=lambda: pytest.fail("native selected"))
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)
        torch.testing.assert_close(source, original, atol=0, rtol=0)
    torch.cuda.synchronize()


@hip_only
def test_long1536_transition_dispatch_matches_torch(monkeypatch):
    monkeypatch.setenv("LAYERNORM_TYPE", "torch")
    from protenix.model.modules.primitives import Transition

    monkeypatch.setattr(recycle, "_long1536_epilogues_enabled", True)
    assert recycle.load_amd_extension() is not None
    torch.manual_seed(633)
    module = Transition(c_in=128, n=4).cuda().eval()
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        for parameter in module.parameters():
            parameter.normal_(0, 0.05)
        source = torch.randn(1536, 1536, 128, device="cuda")
        original = source.clone()
        norm = module.layernorm1
        normalized = F.layer_norm(source, (128,), norm.weight, norm.bias, norm.eps)
        expected = F.linear(
            F.silu(F.linear(normalized, module.linear_no_bias_a.weight))
            * F.linear(normalized, module.linear_no_bias_b.weight), module.linear_no_bias.weight)
        del normalized
        actual = recycle.dispatch_transition(module=module, x=source,
                                             native=lambda: pytest.fail("native selected"))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-5)
        torch.testing.assert_close(source, original, atol=0, rtol=0)
    torch.cuda.synchronize()
