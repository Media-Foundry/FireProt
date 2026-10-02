from __future__ import annotations

import dataclasses
import os

import pytest
import torch

from fireprot.backends.amd import triangle_attention as ta
from fireprot.ops.registry import OperationSignature


def test_direct_attention_capability_can_be_disabled_and_requires_implicit_long_mask(monkeypatch):
    signature = OperationSignature.create(
        operation=ta.OPERATION, device_type="cuda", architecture="gfx90a",
        dtype="float32", shape=ta.LONG_QKV_SHAPE, layout="strided",
        mode="inference", mask="triangle-bias", inplace=False,
        attributes={"bias_shapes": repr(ta.LONG_UNMASKED_BIAS_SHAPES)},
    )
    monkeypatch.setattr(ta, "_long_direct_enabled", False)
    assert ta._supports_long_direct(signature)[0] is False
    monkeypatch.setattr(ta, "_long_direct_enabled", True)
    monkeypatch.setattr(ta, "_long_sequence_enabled", True)
    assert ta._supports_long_direct(signature) is True
    for change in (
        dict(shape=ta.SUPPORTED_QKV_SHAPE), dict(dtype="bfloat16"),
        dict(architecture="gfx942"), dict(mode="training"), dict(layout="contiguous"),
        dict(attributes=(("bias_shapes", repr(ta.LONG_BIAS_SHAPES)),)),
    ):
        assert ta._supports_long_direct(dataclasses.replace(signature, **change))[0] is False
    monkeypatch.setattr(ta, "_long_sequence_enabled", False)
    assert ta._supports_long_direct(signature)[0] is False


@pytest.mark.parametrize("masked", [False, True])
def test_direct_attention_matches_explicit_softmax_and_mask_fallback(masked):
    torch.manual_seed(431)
    q, k, v = [torch.randn(5, 19, 4, 8).transpose(1, 2) for _ in range(3)]
    q /= 8**0.5
    bias = torch.randn(19, 19, 4).permute(2, 0, 1).unsqueeze(0)
    biases = [bias]
    if masked:
        biases.insert(0, -1e4 * torch.randint(0, 2, (5, 1, 1, 19)))
    logits = q @ k.transpose(-1, -2)
    for value in biases:
        logits = logits + value
    expected = (logits.softmax(-1) @ v).transpose(-2, -3)
    actual = ta._direct_sdpa(module=None, q=q, k=k, v=v, biases=biases, native=lambda: None)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(
    os.environ.get("FIREPROT_RUN_HIP_TESTS") != "1" or not torch.version.hip,
    reason="explicit HIP test opt-in required",
)
def test_long_direct_attention_gpu_matches_chunked_and_preserves_inputs():
    torch.manual_seed(433)
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        q, k, v = [torch.randn(1000, 1000, 4, 32, device="cuda").transpose(1, 2) for _ in range(3)]
        q /= 32**0.5
        before = [x.clone() for x in (q, k, v)]
        bias = torch.randn(1000, 1000, 4, device="cuda").permute(2, 0, 1).unsqueeze(0)
        arguments = dict(module=None, q=q, k=k, v=v, biases=[bias], native=lambda: None)
        expected = ta._efficient_sdpa(**arguments)
        actual = ta._direct_sdpa(**arguments)
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        for value, saved in zip((q, k, v), before):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
