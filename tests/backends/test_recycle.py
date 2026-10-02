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
            recycle._supports_transition(
                _signature(recycle.TRANSITION_OPERATION)
            )
            is True
        )


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
