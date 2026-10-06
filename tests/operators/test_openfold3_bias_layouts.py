"""Metadata gates for explicit bias normalization before the packed HIP ABI."""
import pytest
import torch

from experiments.mfma_qk.workspace_attention import bias_layout_supported


@pytest.mark.parametrize('transpose', [False, True])
def test_model_head_last_requires_normalization(transpose):
    bias = torch.empty(1536, 1536, 4, device='meta').permute(2, 0, 1).unsqueeze(0)
    assert bias.stride() == (4, 1, 6144, 4)
    if transpose:
        bias = bias.transpose(-1, -2)
    assert bias_layout_supported(bias, normalize=True)
    assert not bias_layout_supported(bias, normalize=False)


@pytest.mark.parametrize('transpose', [False, True])
def test_existing_head_first_layouts_remain_supported(transpose):
    bias = torch.empty(1, 4, 1536, 1536, device='meta')
    if transpose:
        bias = bias.transpose(-1, -2)
    assert bias_layout_supported(bias, normalize=False)
    assert bias_layout_supported(bias, normalize=True)


def test_overlapping_and_gapped_layouts_rejected():
    overlap = torch.empty(1, 4, 1, 1, device='meta').expand(1, 4, 1536, 1536)
    gapped = torch.empty(3072, 3072, 4, device='meta')[::2, ::2].permute(2, 0, 1).unsqueeze(0)
    assert not bias_layout_supported(overlap, normalize=True)
    assert not bias_layout_supported(gapped, normalize=True)


def test_negative_view_and_wrong_dtype_rejected():
    bias = torch.empty(1, 4, 1536, 1536, device='meta')
    assert not bias_layout_supported(torch._neg_view(bias), normalize=True)
    assert not bias_layout_supported(bias.to(torch.float64), normalize=True)
