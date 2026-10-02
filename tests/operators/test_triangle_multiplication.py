from __future__ import annotations

import os

import pytest
import torch

os.environ["LAYERNORM_TYPE"] = "torch"

from fireprot.backends.amd.triangle_multiplication import (
    _short_sequence_torch,
    _supports_fused_gates,
    _supports_long_fused_gates,
    _supports_short_sequence,
    dispatch_triangle_multiplication,
)
from fireprot.backends.amd.extension import load_amd_extension
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


def test_fused_gate_capability_is_limited_to_validated_bucket(monkeypatch) -> None:
    monkeypatch.setattr(
        "fireprot.backends.amd.triangle_multiplication.load_amd_extension",
        lambda: object(),
    )
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
        "attributes": {"c_z": 128, "c_hidden": 128},
    }
    assert _supports_fused_gates(OperationSignature.create(**base)) is True
    wrong_shape = OperationSignature.create(**{**base, "shape": (123, 123, 128)})
    no_mask = OperationSignature.create(**{**base, "mask": "none"})
    assert _supports_fused_gates(wrong_shape)[0] is False
    assert _supports_fused_gates(no_mask) is True


def test_long_fused_gate_capability_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "fireprot.backends.amd.triangle_multiplication.load_amd_extension",
        lambda: object(),
    )
    base = {
        "operation": "triangle_multiplication",
        "device_type": "cuda",
        "architecture": "gfx90a",
        "dtype": "float32",
        "shape": (1000, 1000, 128),
        "layout": "contiguous",
        "mode": "inference",
        "mask": "none",
        "inplace": True,
        "attributes": {"c_z": 128, "c_hidden": 128},
    }
    signature = OperationSignature.create(**base)
    monkeypatch.setattr(
        "fireprot.backends.amd.triangle_multiplication._long_fused_gate_enabled",
        False,
    )
    assert _supports_long_fused_gates(signature)[0] is False
    monkeypatch.setattr(
        "fireprot.backends.amd.triangle_multiplication._long_fused_gate_enabled",
        True,
    )
    assert _supports_long_fused_gates(signature) is True


@pytest.mark.parametrize(
    "module_class", [TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming]
)
@pytest.mark.parametrize("add_with_inplace", [False, True])
def test_inference_fusion_callbacks_preserve_no_mask_semantics(
    module_class: type, add_with_inplace: bool
) -> None:
    torch.manual_seed(107)
    module = module_class(c_z=8, c_hidden=8).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.2)
        source = torch.randn(7, 7, 8)
        expected = module(
            source.clone(),
            mask=None,
            inplace_safe=True,
            _add_with_inplace=add_with_inplace,
            triangle_multiplicative="torch",
        ).clone()

        def projection_fusion(gate, projection, mask):
            assert mask is None
            return torch.sigmoid(gate) * projection

        def gate_fusion(gate, output):
            return output * torch.sigmoid(gate)

        actual_input = source.clone()
        actual = module._inference_forward(
            actual_input,
            mask=None,
            inplace_chunk_size=256,
            with_add=add_with_inplace,
            projection_fusion=projection_fusion,
            gate_fusion=gate_fusion,
        )

    assert actual.data_ptr() == actual_input.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires PyTorch HIP",
)
def test_hip_sigmoid_multiply_fusions_match_torch() -> None:
    extension = load_amd_extension()
    if extension is None:
        pytest.skip("FireProt AMD extension is unavailable")

    torch.manual_seed(103)
    device = torch.device("cuda")
    activation = torch.randn(124, 124, 128, device=device)
    projection = torch.randn_like(activation)
    mask = torch.randint(0, 2, (124, 124), device=device).float()

    expected_masked = (mask.unsqueeze(-1) * torch.sigmoid(activation)) * projection
    masked_output = projection.clone()
    masked_pointer = masked_output.data_ptr()
    actual_masked = extension.sigmoid_mul_mask_f32(
        activation, masked_output, mask
    )
    assert actual_masked.data_ptr() == masked_pointer
    torch.testing.assert_close(actual_masked, expected_masked, rtol=2e-6, atol=2e-6)

    expected = projection * torch.sigmoid(activation)
    output = projection.clone()
    pointer = output.data_ptr()
    actual = extension.sigmoid_mul_f32(activation, output)
    assert actual.data_ptr() == pointer
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
