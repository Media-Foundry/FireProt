"""Numerics and fallback contracts for the packed HIP triangle candidate."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

os.environ["LAYERNORM_TYPE"] = "torch"

from fireprot.backends.amd import triangle_multiplication as tm
from fireprot.backends.amd.extension import extension_status, load_amd_extension
from fireprot.ops.registry import OperationSignature
from protenix.model.triangular.triangular import TriangleMultiplicativeUpdate


def test_packed_capability_and_fallback(monkeypatch):
    monkeypatch.setattr(tm, "load_amd_extension", lambda: object())
    monkeypatch.setattr(tm, "_long_fused_gate_enabled", True)
    monkeypatch.setattr(tm, "_long_packed_enabled", True)
    base = dict(operation=tm.OPERATION, device_type="cuda", architecture="gfx90a",
                dtype="float32", shape=(1000, 1000, 128), layout="contiguous",
                mode="inference", mask="none", inplace=True,
                attributes={"c_z": 128, "c_hidden": 128})
    assert tm._supports_long_packed(OperationSignature.create(**base)) is True
    for change in (
        {"shape": (1001, 1001, 128)}, {"shape": (1, 1000, 1000, 128)},
        {"mask": "dense"}, {"dtype": "bfloat16"}, {"architecture": "gfx942"},
        {"layout": "strided"}, {"mode": "training"}, {"inplace": False},
    ):
        assert tm._supports_long_packed(OperationSignature.create(**(base | change))) is not True
    monkeypatch.setattr(tm, "_long_packed_enabled", False)
    assert tm._supports_long_packed(OperationSignature.create(**base)) is not True
    monkeypatch.setattr(tm, "_long_packed_enabled", True)
    monkeypatch.setattr(tm, "load_amd_extension", lambda: None)
    assert tm._supports_long_packed(OperationSignature.create(**base)) is not True


hip_only = pytest.mark.skipif(
    os.environ.get("FIREPROT_RUN_HIP_TESTS") != "1"
    or not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires FIREPROT_RUN_HIP_TESTS=1 with PyTorch HIP",
)


@hip_only
@pytest.mark.parametrize("n", [7, 33, 1000])
def test_pack_and_unpack_match_reference_on_nondefault_stream(n):
    extension = load_amd_extension()
    assert extension is not None, extension_status()
    torch.manual_seed(311)
    with torch.cuda.stream(torch.cuda.Stream()), torch.inference_mode():
        gate = torch.randn(n, n, 128, device="cuda")
        projection = torch.randn_like(gate)
        original = projection.clone()
        packed = extension.sigmoid_pack_f32_128(gate, projection)
        expected = (torch.sigmoid(gate) * projection).permute(2, 0, 1).contiguous()
        assert packed.is_contiguous()
        torch.testing.assert_close(packed, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(projection, original, atol=0, rtol=0)
        weight, bias = [torch.randn(128, device="cuda") for _ in range(2)]
        actual = extension.packed_layer_norm_f32_128(packed, weight, bias, 1e-5)
        expected = F.layer_norm(packed.permute(1, 2, 0), (128,), weight, bias, 1e-5)
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)
    torch.cuda.synchronize()


@hip_only
def test_packed_norm_handles_large_means_and_constant_rows():
    extension = load_amd_extension()
    assert extension is not None, extension_status()
    torch.manual_seed(313)
    with torch.inference_mode():
        packed = torch.randn(128, 7, 7, device="cuda") + 1000
        packed[:, 0, 0] = 1000
        weight = torch.ones(128, device="cuda")
        bias = torch.zeros_like(weight)
        actual = extension.packed_layer_norm_f32_128(packed, weight, bias, 1e-5)
        expected = F.layer_norm(packed.permute(1, 2, 0).double(), (128,), eps=1e-5).float()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        assert actual[0, 0].count_nonzero() == 0


@hip_only
@pytest.mark.parametrize("n", [1000, 1536])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("with_add", [True, False])
def test_packed_triangle_matches_native_chunked_and_preserves_alias(n, outgoing, with_add):
    assert load_amd_extension() is not None, extension_status()
    torch.manual_seed(317)
    module = TriangleMultiplicativeUpdate(128, 128, outgoing).cuda().eval()
    with torch.inference_mode():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
        source = torch.randn(n, n, 128, device="cuda")
        expected = module(source.clone(), inplace_safe=True, _add_with_inplace=with_add)
        candidate = source.clone()
        actual = tm._long_sequence_packed(
            module=module, z=candidate, mask=None, inplace_safe=True,
            add_with_inplace=with_add, inplace_chunk_size=256, native=lambda: None,
        )
        assert actual.data_ptr() == candidate.data_ptr()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-5)


def test_long1536_packed_capability_is_explicit_and_exact(monkeypatch):
    monkeypatch.setattr(tm, "load_amd_extension", lambda: object())
    monkeypatch.setattr(tm, "_long_fused_gate_enabled", True)
    monkeypatch.setattr(tm, "_long_packed_enabled", True)
    base = dict(operation=tm.OPERATION, device_type="cuda", architecture="gfx90a",
                dtype="float32", shape=(1536, 1536, 128), layout="contiguous",
                mode="inference", mask="none", inplace=True,
                attributes={"c_z": 128, "c_hidden": 128})
    monkeypatch.setattr(tm, "_long1536_enabled", False)
    assert tm._supports_long1536_packed(OperationSignature.create(**base)) is not True
    monkeypatch.setattr(tm, "_long1536_enabled", True)
    assert tm._supports_long1536_packed(OperationSignature.create(**base)) is True
    assert tm._supports_long_packed(OperationSignature.create(**base)) is not True
    for change in ({"shape": (1537, 1537, 128)}, {"shape": (1, 1536, 1536, 128)},
                   {"dtype": "bfloat16"}, {"mask": "dense"}, {"mode": "training"},
                   {"layout": "strided"}, {"inplace": False}, {"architecture": "gfx942"},
                   {"attributes": {"c_z": 128, "c_hidden": 64}}):
        assert tm._supports_long1536_packed(OperationSignature.create(**(base | change))) is not True
    monkeypatch.setattr(tm, "_long_packed_enabled", False)
    assert tm._supports_long1536_packed(OperationSignature.create(**base)) is not True
    monkeypatch.setattr(tm, "_long_packed_enabled", True)
    monkeypatch.setattr(tm, "_long_fused_gate_enabled", False)
    assert tm._supports_long1536_packed(OperationSignature.create(**base)) is not True
    monkeypatch.setattr(tm, "_long_fused_gate_enabled", True)
    monkeypatch.setattr(tm, "load_amd_extension", lambda: None)
    assert tm._supports_long1536_packed(OperationSignature.create(**base)) is not True
