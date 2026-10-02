from __future__ import annotations

import os
from unittest import mock

import torch

os.environ["LAYERNORM_TYPE"] = "torch"

from fireprot.backends.amd import ck_gemm
from fireprot.ops.registry import OperationSignature
from protenix.model.triangular.layers import OpenfoldLinear


def _signature(**changes: object) -> OperationSignature:
    values = {
        "operation": ck_gemm.OPERATION,
        "device_type": "cuda",
        "architecture": "gfx90a",
        "dtype": "float32",
        "shape": (1, 124, 64),
        "layout": "contiguous",
        "mode": "inference",
        "mask": "none",
        "inplace": False,
        "attributes": {
            "out_features": 32,
            "bias": False,
            "weight_contiguous": True,
        },
    }
    values.update(changes)
    return OperationSignature.create(**values)


def test_ck_linear_capability_is_exact_and_requires_extension() -> None:
    with mock.patch.object(ck_gemm, "load_ck_extension", return_value=object()):
        assert ck_gemm._supports_linear(_signature()) is True
        for change in (
            {"architecture": "gfx942"},
            {"dtype": "bfloat16"},
            {"shape": (1, 125, 64)},
            {"shape": (1, 124, 128)},
            {"layout": "strided"},
            {"mode": "training"},
            {"attributes": {"out_features": 64, "bias": False}},
        ):
            assert ck_gemm._supports_linear(_signature(**change)) is not True

    with mock.patch.object(ck_gemm, "load_ck_extension", return_value=None):
        assert ck_gemm._supports_linear(_signature()) is not True


def test_ck_linear_preserves_bias_free_linear_expression_and_shape() -> None:
    class _FakeExtension:
        @staticmethod
        def mm_f32(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return torch.mm(left, right)

    torch.manual_seed(101)
    module = OpenfoldLinear(64, 32, bias=False).eval()
    tensor = torch.randn(1, 124, 64)
    expected = module(tensor)

    with mock.patch.object(
        ck_gemm, "load_ck_extension", return_value=_FakeExtension()
    ):
        actual = ck_gemm._ck_linear(module=module, x=tensor, native=lambda: None)

    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 124, 32)


def test_packed_weight_cache_invalidates_after_parameter_update() -> None:
    ck_gemm._reset_for_testing()
    module = OpenfoldLinear(64, 32, bias=False).eval()

    first = ck_gemm._packed_weight(module)
    assert ck_gemm._packed_weight(module) is first
    with torch.no_grad():
        module.weight.add_(1.0)
    second = ck_gemm._packed_weight(module)

    assert second is not first
    torch.testing.assert_close(second, module.weight.transpose(0, 1))


def test_ck_outer_product_mean_preserves_projection_and_normalization() -> None:
    class _FakeExtension:
        @staticmethod
        def mm_f32(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return torch.mm(left, right)

    class _FakeOuterProductMean:
        def __init__(self) -> None:
            self.linear_1 = OpenfoldLinear(64, 32, bias=False).eval()
            self.linear_2 = OpenfoldLinear(64, 32, bias=False).eval()
            self.eps = 1e-3

        @staticmethod
        def layer_norm(value: torch.Tensor) -> torch.Tensor:
            return value

        @staticmethod
        def _opm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            outer = torch.einsum("...bac,...dae->...bdce", left, right)
            return outer.sum(dim=(-1, -2)).unsqueeze(-1)

    torch.manual_seed(103)
    module = _FakeOuterProductMean()
    tensor = torch.randn(1, 1, 124, 64)
    mask = torch.ones(1, 1, 124)
    expanded_mask = mask.unsqueeze(-1)
    normalized = module.layer_norm(tensor)
    left = module.linear_1(normalized) * expanded_mask
    right = module.linear_2(normalized) * expanded_mask
    expected = module._opm(
        left.transpose(-2, -3), right.transpose(-2, -3)
    )
    expected = expected / (
        torch.einsum("...abc,...adc->...bdc", expanded_mask, expanded_mask)
        + module.eps
    )

    actual = ck_gemm._ck_outer_product_mean(
        module=module,
        m=tensor,
        mask=mask,
        chunk_size=None,
        inplace_safe=False,
        native=lambda: None,
        extension=_FakeExtension(),
    )

    # F.linear and the mm-based CK stand-in may choose different FP32
    # accumulation orders before the OPM reduction.
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-4)


def test_ck_outer_product_mean_dispatch_falls_back_on_cpu() -> None:
    class _FakeOuterProductMean:
        training = False
        c_m = 64
        c_hidden = 32
        linear_1 = OpenfoldLinear(64, 32, bias=False).eval()
        linear_2 = OpenfoldLinear(64, 32, bias=False).eval()

    tensor = torch.randn(1, 1, 124, 64)

    with torch.inference_mode():
        result = ck_gemm.dispatch_ck_outer_product_mean(
            module=_FakeOuterProductMean(),
            m=tensor,
            mask=None,
            chunk_size=None,
            inplace_safe=False,
            native=lambda: "native",
        )

    assert result == "native"
