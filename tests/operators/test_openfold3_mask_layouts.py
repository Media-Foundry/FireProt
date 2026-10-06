"""Exact metadata scope and one-copy behavior for transposed pair masks."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.mfma_qk.model_workspace import ModelWorkspaceFusion
from test_openfold3_model_workspace import FakePool, TensorInput


def test_normalized_mask_is_contiguous_before_slicing_and_input_unchanged():
    with patch('experiments.mfma_qk.model_workspace.WorkspaceFusion', FakePool):
        p = ModelWorkspaceFusion({}, microbatch=128, normalize_transposed_mask=True)
    q = TensorInput(device=SimpleNamespace(index=7), dtype=torch.float32)
    source = torch.arange(1536 * 1536, dtype=torch.float32).reshape(1536, 1536)
    mask = source.T[17:17 + 256, None, None, :]
    original = mask.clone()
    bias = torch.empty(1, 4, 1536, 1536, device='meta')
    seen = []
    def submit(_q, _k, _v, part, _bias):
        seen.append(part)
        assert part.is_contiguous()
        return torch.ones(1, 1)
    allocation = torch.zeros(1)
    with patch.object(p, '_check', return_value=(256, 123)), patch(
            'experiments.mfma_qk.model_workspace.torch.empty', return_value=allocation), patch.object(
            FakePool, '__call__', side_effect=submit):
        p(q, q, q, mask, bias)
    assert len(seen) == 2 and p.manifest()['mask_copies'] == 1
    assert torch.equal(torch.cat(seen), original) and torch.equal(mask, original)
    assert seen[0].untyped_storage().data_ptr() == seen[1].untyped_storage().data_ptr()
    assert seen[0].untyped_storage().data_ptr() != mask.untyped_storage().data_ptr()
    p.close()


@pytest.mark.parametrize('batch', [1, 128, 256, 512, 1024])
@pytest.mark.parametrize('offset', [0, 17])
def test_only_exact_transposed_pair_storage_layout_is_supported(batch, offset):
    from experiments.mfma_qk.model_workspace import transposed_pair_mask_supported
    mask = torch.empty(1536, 1536, device='meta').T[offset:offset + batch, None, None, :]
    assert transposed_pair_mask_supported(mask)
    assert not transposed_pair_mask_supported(mask.to(torch.float64))
    assert not transposed_pair_mask_supported(torch._neg_view(mask))
    assert not transposed_pair_mask_supported(mask[..., ::2])


def test_overlapping_and_unmeasured_pitch_rejected():
    from experiments.mfma_qk.model_workspace import transposed_pair_mask_supported
    assert not transposed_pair_mask_supported(torch.empty(1, 1, 1, 1536, device='meta').expand(128, 1, 1, 1536))
    assert not transposed_pair_mask_supported(torch.empty(1536, 1600, device='meta').T[:128, None, None, :])
    assert not transposed_pair_mask_supported(torch.empty(1536, 1536, device='meta').T[:64, None, None, :])


@pytest.mark.parametrize('value', [1, 'true', None])
def test_normalization_option_requires_plain_bool(value):
    with patch('experiments.mfma_qk.model_workspace.WorkspaceFusion') as loader:
        with pytest.raises(ValueError, match='normalization'):
            ModelWorkspaceFusion({}, normalize_transposed_mask=value)
        loader.assert_not_called()


def test_mask_copy_interruption_poisoned_before_pool_reservation():
    with patch('experiments.mfma_qk.model_workspace.WorkspaceFusion', FakePool):
        p = ModelWorkspaceFusion({}, microbatch=128, normalize_transposed_mask=True)
    q = TensorInput(device=SimpleNamespace(index=7), dtype=torch.float32)
    mask = torch.empty(1536, 1536, device='meta').T[:128, None, None, :]
    bias = torch.empty(1, 4, 1536, 1536, device='meta')
    error = KeyboardInterrupt()
    with patch.object(p, '_check', return_value=(128, 123)), patch.object(
            torch.Tensor, 'contiguous', side_effect=[bias, error]):
        with pytest.raises(KeyboardInterrupt) as caught:
            p(q, q, q, mask, bias)
    assert caught.value is error
    r = p.manifest()
    assert r['closed'] and r['lease']['state'] == 'failed'
    assert r['mask_copies'] == r['allocations'] == r['successful_kernel_calls'] == 0
    assert r['lease']['charged_bytes'] == 0
    p.close()
    assert p.manifest()['lease']['state'] == 'released'
