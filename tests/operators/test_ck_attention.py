from __future__ import annotations

import dataclasses
import os

import pytest
import torch

from fireprot.backends.amd import ck_attention as ck
from fireprot.backends.amd import triangle_attention as ta
from fireprot.ops.registry import OperationSignature


def test_ck_attention_capability_and_build_fallback(monkeypatch):
    signature = OperationSignature.create(
        operation=ta.OPERATION, device_type="cuda", architecture="gfx90a",
        dtype="float32", shape=ta.LONG_QKV_SHAPE, layout="strided",
        mode="inference", mask="triangle-bias", inplace=False,
        attributes={"bias_shapes": repr(ta.LONG_UNMASKED_BIAS_SHAPES),
                    "qkv_strides": repr((ck.STRIDES,) * 3), "ck_tile_compatible": True},
    )
    monkeypatch.setattr(ck, "load_extension", lambda: object())
    monkeypatch.setattr(ck, "_enabled", False)
    assert ck._supports(signature)[0] is False
    monkeypatch.setattr(ck, "_enabled", True)
    monkeypatch.setattr(ta, "_long_sequence_enabled", True)
    monkeypatch.setattr(ta, "_long_direct_enabled", True)
    assert ck._supports(signature) is True
    monkeypatch.setattr(ta, "_long_direct_enabled", False)
    assert ck._supports(signature)[0] is False
    monkeypatch.setattr(ta, "_long_direct_enabled", True)
    for changes in (dict(dtype="bfloat16"), dict(architecture="gfx942"),
                    dict(mode="training"), dict(shape=ta.SUPPORTED_QKV_SHAPE)):
        assert ck._supports(dataclasses.replace(signature, **changes))[0] is False
    for key, value in (("ck_tile_compatible", "False"), ("qkv_strides", "other"),
                       ("bias_shapes", repr(ta.LONG_BIAS_SHAPES))):
        attributes = dict(signature.attributes)
        attributes[key] = value
        assert ck._supports(dataclasses.replace(signature, attributes=tuple(sorted(attributes.items()))))[0] is False
    monkeypatch.setattr(ck, "load_extension", lambda: None)
    assert ck._supports(signature)[0] is False


def test_ck_attention_disable_prevents_build(monkeypatch):
    monkeypatch.setattr(ck, "_entry", None)
    monkeypatch.setattr(ck, "_error", None)
    monkeypatch.setenv("FIREPROT_DISABLE_CK_EXTENSION", "1")
    from fireprot.backends.amd import _native_hip
    monkeypatch.setattr(_native_hip, "load", lambda *a, **k: pytest.fail("unexpected build"))
    assert ck.load_extension() is None
    assert ck.extension_status()["error"] == "disabled by FIREPROT_DISABLE_CK_EXTENSION"


@pytest.mark.skipif(os.environ.get("FIREPROT_RUN_HIP_TESTS") != "1" or not torch.version.hip,
                    reason="explicit HIP test opt-in required")
def test_ck_attention_gpu_dispatch_stream_and_extreme_bias(monkeypatch):
    from types import SimpleNamespace
    from fireprot.ops.registry import registry

    monkeypatch.setattr(ck, "_enabled", True)
    registry.clear_runtime_state()
    assert ck.load_extension() is not None, ck.extension_status()
    torch.manual_seed(521)
    try:
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            q, k, v = [torch.randn(1000, 1000, 4, 32, device="cuda").transpose(1, 2) for _ in range(3)]
            q /= 32**0.5
            saved = [x.clone() for x in (q, k, v)]
            bias = torch.randn(1000, 1000, 4, device="cuda").permute(2, 0, 1).unsqueeze(0)
            for exceptional in (False, True):
                if exceptional:
                    bias[:, :, 0, :] = float("-inf")
                    bias[:, :, 1, 0] = 80
                arguments = dict(module=SimpleNamespace(training=False), q=q, k=k, v=v,
                                 biases=[bias], native=lambda: pytest.fail("native selected"))
                expected = ta._direct_sdpa(**arguments)
                actual = ta.dispatch_triangle_attention(**arguments)
                assert actual.is_contiguous()
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            for x, before in zip((q, k, v), saved):
                torch.testing.assert_close(x, before, atol=0, rtol=0)
            assert any(row["implementation"] == ck.IMPLEMENTATION and row["count"] == 2
                       for row in registry.manifest()["selection_counts"])
    finally:
        registry.clear_runtime_state()
