from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as functional

os.environ["LAYERNORM_TYPE"] = "torch"

from fireprot.backends.amd import recycle
from fireprot.ops.registry import OperationSignature


def _signature(operation: str, **changes: object) -> OperationSignature:
    values = {
        "operation": operation,
        "device_type": "cuda",
        "architecture": "gfx90a",
        "dtype": "float32",
        "shape": (124, 124, 128),
        "layout": "contiguous",
        "mode": "inference",
        "mask": "none",
        "inplace": False,
        "attributes": (
            {"scale": True, "offset": True, "epsilon": 1e-5}
            if operation == recycle.LAYER_NORM_OPERATION
            else {"expansion": 4, "channels": 128}
        ),
    }
    values.update(changes)
    return OperationSignature.create(**values)


@pytest.mark.parametrize(
    "change",
    [
        {"device_type": "cpu"},
        {"architecture": "gfx942"},
        {"dtype": "bfloat16"},
        {"shape": (127, 127, 128)},
        {"shape": (124, 124, 64)},
        {"layout": "strided"},
        {"mode": "training"},
    ],
)
def test_layer_norm_capability_rejects_unvalidated_signatures(
    change: dict[str, object],
) -> None:
    result = recycle._supports_layer_norm(
        _signature(recycle.LAYER_NORM_OPERATION, **change)
    )
    assert result is not True


def test_valid_capabilities_require_the_extension() -> None:
    with mock.patch.object(recycle, "load_amd_extension", return_value=object()):
        assert (
            recycle._supports_layer_norm(
                _signature(recycle.LAYER_NORM_OPERATION)
            )
            is True
        )
        assert (
            recycle._supports_channel_first_layer_norm(
                _signature(
                    recycle.LAYER_NORM_OPERATION,
                    layout="strided",
                    attributes={
                        "scale": True,
                        "offset": True,
                        "epsilon": 1e-5,
                        "stride": (124, 1, 15376),
                    },
                )
            )
            is True
        )
        assert (
            recycle._supports_transition(
                _signature(recycle.TRANSITION_OPERATION)
            )
            is True
        )


@pytest.mark.parametrize(
    "shape",
    [
        (1000, 244, 128),
        (1000, 256, 128),
        (256, 1000, 128),
        (1000, 1000, 128),
    ],
)
def test_long_layer_norm_accepts_profiled_row_buckets(
    shape: tuple[int, ...],
) -> None:
    with mock.patch.object(recycle, "load_amd_extension", return_value=object()):
        assert (
            recycle._supports_long_layer_norm(
                _signature(recycle.LAYER_NORM_OPERATION, shape=shape)
            )
            is True
        )


def test_long_layer_norm_rejects_unprofiled_row_bucket() -> None:
    with mock.patch.object(recycle, "load_amd_extension", return_value=object()):
        result = recycle._supports_long_layer_norm(
            _signature(
                recycle.LAYER_NORM_OPERATION,
                shape=(1000, 255, 128),
            )
        )
    assert result is not True


@pytest.mark.parametrize(
    "change",
    [
        {"architecture": "gfx942"},
        {"dtype": "bfloat16"},
        {"shape": (124, 125, 128)},
        {"layout": "contiguous"},
        {
            "attributes": {
                "scale": True,
                "offset": True,
                "epsilon": 1e-5,
                "stride": (15872, 128, 1),
            }
        },
    ],
)
def test_channel_first_layer_norm_rejects_unvalidated_signatures(
    change: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "layout": "strided",
        "attributes": {
            "scale": True,
            "offset": True,
            "epsilon": 1e-5,
            "stride": (124, 1, 15376),
        },
    }
    values.update(change)
    result = recycle._supports_channel_first_layer_norm(
        _signature(recycle.LAYER_NORM_OPERATION, **values)
    )
    assert result is not True


def test_dispatches_fall_back_on_cpu() -> None:
    module = SimpleNamespace(
        training=False,
        weight=torch.ones(128),
        bias=torch.zeros(128),
        eps=1e-5,
        n=4,
        c_in=128,
    )
    tensor = torch.randn(124, 124, 128)
    with torch.inference_mode():
        assert (
            recycle.dispatch_layer_norm(
                module=module, x=tensor, native=lambda **_: "native"
            )
            == "native"
        )
        assert (
            recycle.dispatch_transition(
                module=module, x=tensor, native=lambda **_: "native"
            )
            == "native"
        )


def test_channel_first_layer_norm_preserves_reference_expression() -> None:
    class _FakeExtension:
        @staticmethod
        def layer_norm_f32_128_channel_first(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            epsilon: float,
        ) -> torch.Tensor:
            assert x.stride() == (124, 1, 15376)
            return functional.layer_norm(x, (128,), weight, bias, epsilon)

    torch.manual_seed(107)
    storage = torch.randn(128, 124, 124)
    tensor = storage.permute(1, 2, 0)
    module = SimpleNamespace(
        weight=torch.randn(128),
        bias=torch.randn(128),
        eps=1e-5,
    )
    expected = functional.layer_norm(
        tensor, (128,), module.weight, module.bias, module.eps
    )
    with mock.patch.object(
        recycle, "load_amd_extension", return_value=_FakeExtension()
    ):
        actual = recycle._hip_channel_first_layer_norm(
            module=module,
            x=tensor,
            native=lambda: None,
        )

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    os.environ.get("FIREPROT_RUN_HIP_TESTS") != "1"
    or not torch.cuda.is_available()
    or torch.version.hip is None,
    reason="requires FIREPROT_RUN_HIP_TESTS=1 with PyTorch HIP",
)
def test_channel_first_layer_norm_hip_extension_matches_torch() -> None:
    torch.manual_seed(109)
    storage = torch.randn(128, 124, 124, device="cuda", dtype=torch.float32)
    tensor = storage.permute(1, 2, 0)
    weight = torch.randn(128, device="cuda", dtype=torch.float32)
    bias = torch.randn(128, device="cuda", dtype=torch.float32)
    expected = functional.layer_norm(tensor, (128,), weight, bias, 1e-5)
    extension = recycle.load_amd_extension(torch_module=torch)

    assert extension is not None
    actual = extension.layer_norm_f32_128_channel_first(
        tensor, weight, bias, 1e-5
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)


def test_fused_transition_preserves_reference_expression() -> None:
    from protenix.model.modules.primitives import Transition

    class _FakeExtension:
        @staticmethod
        def silu_mul_f32(activation: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            return functional.silu(activation) * gate

    torch.manual_seed(101)
    module = Transition(c_in=128, n=4).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.05)
        tensor = torch.randn(3, 5, 128)
        expected = module(tensor)
        with mock.patch.object(
            recycle, "load_amd_extension", return_value=_FakeExtension()
        ):
            actual = recycle._hip_transition(
                module=module,
                x=tensor,
                native=lambda: None,
            )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
