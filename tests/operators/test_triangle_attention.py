from __future__ import annotations

import os

import pytest
import torch

os.environ["LAYERNORM_TYPE"] = "torch"

from fireprot.backends.amd.triangle_attention import (
    LONG_BIAS_SHAPES,
    LONG_QKV_SHAPE,
    LONG_UNMASKED_BIAS_SHAPES,
    SUPPORTED_BIAS_SHAPES,
    SUPPORTED_QKV_SHAPE,
    SUPPORTED_UNMASKED_BIAS_SHAPES,
    _efficient_sdpa,
    _supports_efficient_sdpa,
    dispatch_triangle_attention,
)
from fireprot.ops.registry import OperationSignature
from protenix.model.triangular.layers import Attention, _attention
from protenix.model.triangular.triangular import TriangleAttention


@pytest.mark.parametrize("tokens", [1, 16, 31, 32, 62, 124, 127, 128])
def test_efficient_sdpa_matches_native_fp32(tokens: int) -> None:
    torch.manual_seed(101 + tokens)
    heads, hidden = 2, 8
    q = torch.randn(tokens, heads, tokens, hidden)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask = torch.randint(0, 2, (tokens, 1, 1, tokens), dtype=q.dtype)
    mask_bias = 1e4 * (mask - 1)
    triangle_bias = torch.randn(tokens, tokens, heads).permute(2, 0, 1).unsqueeze(0)
    if tokens > 1:
        assert not triangle_bias.is_contiguous()
    biases = [mask_bias, triangle_bias]

    expected = _attention(q, k, v, biases).transpose(-2, -3)
    actual = _efficient_sdpa(
        module=None,
        q=q,
        k=k,
        v=v,
        biases=biases,
        native=lambda **_: None,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_efficient_sdpa_handles_all_masked_rows_and_extreme_biases() -> None:
    torch.manual_seed(202)
    q = torch.randn(7, 2, 7, 8)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    mask_bias = torch.full((7, 1, 1, 7), -1e9)
    triangle_bias = torch.linspace(-80, 80, 7).view(1, 1, 1, 7).expand(1, 2, 7, 7)
    biases = [mask_bias, triangle_bias]

    expected = _attention(q, k, v, biases).transpose(-2, -3)
    actual = _efficient_sdpa(
        module=None,
        q=q,
        k=k,
        v=v,
        biases=biases,
        native=lambda **_: None,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_efficient_sdpa_matches_native_without_explicit_mask() -> None:
    torch.manual_seed(211)
    tokens, heads, hidden = 127, 2, 8
    q = torch.randn(tokens, heads, tokens, hidden)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    triangle_bias = torch.randn(tokens, tokens, heads).permute(2, 0, 1).unsqueeze(0)
    biases = [triangle_bias]

    expected = _attention(q, k, v, biases).transpose(-2, -3)
    actual = _efficient_sdpa(
        module=None,
        q=q,
        k=k,
        v=v,
        biases=biases,
        native=lambda **_: None,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_capability_accepts_only_validated_gfx90a_signature() -> None:
    base = {
        "operation": "triangle_attention",
        "device_type": "cuda",
        "architecture": "gfx90a",
        "dtype": "float32",
        "shape": SUPPORTED_QKV_SHAPE,
        "layout": "strided",
        "mode": "inference",
        "mask": "additive-mask-and-triangle-bias",
        "inplace": False,
        "attributes": {"bias_shapes": repr(SUPPORTED_BIAS_SHAPES)},
    }
    assert _supports_efficient_sdpa(OperationSignature.create(**base)) is True

    unsupported = (
        {"device_type": "cpu"},
        {"architecture": "gfx942"},
        {"dtype": "bfloat16"},
        {"shape": (127, 4, 127, 32)},
        {"layout": "contiguous"},
        {"mode": "training"},
        {"attributes": {"bias_shapes": "other"}},
    )
    for change in unsupported:
        signature = OperationSignature.create(**{**base, **change})
        assert _supports_efficient_sdpa(signature)[0] is False


def test_long_capability_can_be_disabled(monkeypatch) -> None:
    base = {
        "operation": "triangle_attention",
        "device_type": "cuda",
        "architecture": "gfx90a",
        "dtype": "float32",
        "shape": LONG_QKV_SHAPE,
        "layout": "strided",
        "mode": "inference",
        "mask": "additive-mask-and-triangle-bias",
        "inplace": False,
        "attributes": {"bias_shapes": repr(LONG_BIAS_SHAPES)},
    }
    signature = OperationSignature.create(**base)
    monkeypatch.setattr(
        "fireprot.backends.amd.triangle_attention._long_sequence_enabled", False
    )
    assert _supports_efficient_sdpa(signature)[0] is False
    monkeypatch.setattr(
        "fireprot.backends.amd.triangle_attention._long_sequence_enabled", True
    )
    assert _supports_efficient_sdpa(signature) is True
    unmasked = OperationSignature.create(
        **{
            **base,
            "mask": "triangle-bias",
            "attributes": {"bias_shapes": repr(LONG_UNMASKED_BIAS_SHAPES)},
        }
    )
    assert _supports_efficient_sdpa(unmasked) is True


def test_short_capability_accepts_implicit_mask_signature() -> None:
    signature = OperationSignature.create(
        operation="triangle_attention",
        device_type="cuda",
        architecture="gfx90a",
        dtype="float32",
        shape=SUPPORTED_QKV_SHAPE,
        layout="strided",
        mode="inference",
        mask="triangle-bias",
        inplace=False,
        attributes={"bias_shapes": repr(SUPPORTED_UNMASKED_BIAS_SHAPES)},
    )
    assert _supports_efficient_sdpa(signature) is True


def test_dispatch_falls_back_for_cpu() -> None:
    module = TriangleAttention(c_in=16, c_hidden=8, no_heads=2).eval()
    q = torch.randn(5, 2, 5, 8)
    biases = [torch.zeros(5, 1, 1, 5), torch.zeros(1, 2, 5, 5)]
    with torch.inference_mode():
        result = dispatch_triangle_attention(
            module=module,
            q=q,
            k=q,
            v=q,
            biases=biases,
            native=lambda **_: "native",
        )
    assert result == "native"


def test_protenix_fireprot_option_preserves_native_cpu_fallback() -> None:
    torch.manual_seed(303)
    module = Attention(16, 16, 16, c_hidden=8, no_heads=2).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)
        q_x = torch.randn(3, 5, 16)
        biases = [torch.zeros(3, 1, 1, 5), torch.randn(1, 2, 5, 5)]
        expected = module(q_x, q_x, biases, triangle_attention="torch")
        actual = module(q_x, q_x, biases, triangle_attention="fireprot")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_triangle_attention_implicit_mask_preserves_cpu_fallback() -> None:
    torch.manual_seed(307)
    module = TriangleAttention(c_in=16, c_hidden=8, no_heads=2).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.1)
        source = torch.randn(5, 5, 16)
        expected = module(source, mask=None, triangle_attention="torch")
        actual = module(source, mask=None, triangle_attention="fireprot")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
