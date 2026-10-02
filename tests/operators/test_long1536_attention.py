from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import pytest
import torch

from fireprot.backends.amd import ck_attention as ck
from fireprot.backends.amd import triangle_attention as ta
from fireprot.ops.registry import OperationSignature, registry


def test_long1536_attention_requires_exact_upstream_chunk(monkeypatch):
    signature = OperationSignature.create(
        operation=ta.OPERATION, device_type="cuda", architecture="gfx90a",
        dtype="float32", shape=(512, 4, 1536, 32), layout="strided",
        mode="inference", mask="triangle-bias", inplace=False,
        attributes={"bias_shapes": repr(((1, 4, 1536, 1536),)),
                    "qkv_strides": repr(((1536 * 128, 32, 128, 1),) * 3),
                    "ck_tile_compatible": True},
    )
    monkeypatch.setattr(ck, "_enabled", True)
    monkeypatch.setattr(ck, "_long1536_enabled", True)
    monkeypatch.setattr(ta, "_long_direct_enabled", True)
    monkeypatch.setattr(ta, "_long_sequence_enabled", True)
    monkeypatch.setattr(ck, "load_extension", lambda: object())
    monkeypatch.setattr(ta, "_long1536_enabled", False)
    assert ck._supports_long1536(signature)[0] is False
    monkeypatch.setattr(ta, "_long1536_enabled", True)
    monkeypatch.setattr(ck, "_long1536_enabled", False)
    assert ck._supports_long1536(signature)[0] is False
    assert ta._supports_long1536_direct(signature) is True
    monkeypatch.setattr(ck, "_long1536_enabled", True)
    assert ck._supports_long1536(signature) is True
    assert ta._supports_long1536_direct(signature) is True
    assert ck._supports(signature)[0] is False
    for change in (dict(shape=(1536, 4, 1536, 32)), dict(shape=(256, 4, 1536, 32)),
                   dict(dtype="bfloat16"), dict(mode="training"), dict(architecture="gfx942")):
        assert ck._supports_long1536(dataclasses.replace(signature, **change))[0] is False
    monkeypatch.setattr(ta, "_long_direct_enabled", False)
    assert ck._supports_long1536(signature)[0] is False


@pytest.mark.skipif(os.environ.get("FIREPROT_RUN_HIP_TESTS") != "1" or not torch.version.hip,
                    reason="explicit HIP test opt-in required")
def test_long1536_ck_matches_explicit_torch_attention(monkeypatch):
    monkeypatch.setenv("LAYERNORM_TYPE", "torch")
    from protenix.model.triangular.layers import _attention

    monkeypatch.setattr(ck, "_enabled", True)
    monkeypatch.setattr(ck, "_long1536_enabled", True)
    monkeypatch.setattr(ta, "_long1536_enabled", True)
    registry.clear_runtime_state()
    assert ck.load_extension() is not None, ck.extension_status()
    torch.manual_seed(613)
    try:
        with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
            q, k, v = [torch.randn(512, 1536, 4, 32, device="cuda").transpose(1, 2) for _ in range(3)]
            q /= 32**0.5
            originals = [x.clone() for x in (q, k, v)]
            bias = torch.randn(1536, 1536, 4, device="cuda").permute(2, 0, 1).unsqueeze(0)
            expected = torch.empty_like(q).transpose(1, 2)
            for start in range(0, 512, 16):
                expected[start:start+16] = _attention(q[start:start+16], k[start:start+16],
                                                      v[start:start+16], [bias]).transpose(1, 2)
            actual = ta.dispatch_triangle_attention(module=SimpleNamespace(training=False),
                        q=q, k=k, v=v, biases=[bias], native=lambda: pytest.fail("native selected"))
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            for current, before in zip((q, k, v), originals):
                torch.testing.assert_close(current, before, atol=0, rtol=0)
            assert registry.manifest()["selection_counts"] == [dict(
                operation=ta.OPERATION, implementation=ck.LONG1536_IMPLEMENTATION, count=1)]
    finally:
        registry.clear_runtime_state()
