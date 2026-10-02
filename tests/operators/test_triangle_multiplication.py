from __future__ import annotations

import os

import pytest
import torch

os.environ["LAYERNORM_TYPE"] = "torch"

from fireprot.backends.amd.triangle_multiplication import (
    _short_sequence_torch,
    _supports_short_sequence,
    dispatch_triangle_multiplication,
)
from fireprot.ops.registry import OperationSignature
from protenix.model.triangular.triangular import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


@pytest.mark.parametrize(
    "module_class", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming]
)
@pytest.mark.parametrize("add_with_inplace", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
def test_short_sequence_path_matches_native_inplace_semantics(
    module_class: type, add_with_inplace: bool, use_mask: bool
) -> None:
    torch.manual_seed(101)
    module = module_class(c_z=8, c_hidden=8).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.2)
        source = torch.randn(7, 7, 8)
        mask = (
            torch.randint(0, 2, (7, 7), dtype=source.dtype) if use_mask else None
        )
        native_input = source.clone()
        expected = module(
            native_input,
            mask=mask,
            inplace_safe=True,
            _add_with_inplace=add_with_inplace,
            triangle_multiplicative="torch",
        ).clone()
        candidate_input = source.clone()
        pointer = candidate_input.data_ptr()
        actual = _short_sequence_torch(
            module=module,
            z=candidate_input,
            mask=mask,
            inplace_safe=True,
            add_with_inplace=add_with_inplace,
            inplace_chunk_size=256,
            native=lambda: None,
        )

    assert actual.data_ptr() == pointer
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_dispatch_falls_back_for_cpu() -> None:
    module = TriangleMultiplicationOutgoing(c_z=8, c_hidden=8).eval()
    with torch.no_grad():
        result = dispatch_triangle_multiplication(
            module=module,
            z=torch.randn(4, 4, 8),
            mask=None,
            inplace_safe=True,
            add_with_inplace=True,
            inplace_chunk_size=256,
            native=lambda **_: "native",
        )
    assert result == "native"


def test_capability_rejects_long_or_training_signature() -> None:
    base = {
        "operation": "triangle_multiplication",
        "device_type": "cuda",
        "architecture": "gfx90a",
        "dtype": "float32",
        "shape": (124, 124, 128),
        "layout": "contiguous",
        "mode": "inference",
        "mask": "dense",
        "inplace": True,
    }
    assert _supports_short_sequence(OperationSignature.create(**base)) is True
    long = OperationSignature.create(**{**base, "shape": (257, 257, 128)})
    training = OperationSignature.create(**{**base, "mode": "training"})
    assert _supports_short_sequence(long)[0] is False
    assert _supports_short_sequence(training)[0] is False
