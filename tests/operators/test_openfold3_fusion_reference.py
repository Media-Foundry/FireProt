"""CPU checks for experimental reference partitioning, not GPU acceptance."""
import pytest
import torch

from experiments.mfma_qk.probe_fusion import bitwise_mismatch_count, staged_mismatches


@pytest.mark.parametrize('reference_batch', [1, 2, 3, 8])
def test_complete_rows_and_tail_match_full_cpu_reference(reference_batch):
    generator = torch.Generator().manual_seed(101)
    scores = torch.randn(5, 4, 17, 17, generator=generator)
    mask = torch.zeros(5, 1, 1, 17)
    mask[1, :, :, 3:] = -float('inf')
    mask[4].fill_(-float('inf'))
    bias = torch.randn(1, 4, 17, 17, generator=generator).transpose(-1, -2)
    expected = torch.softmax(scores + mask + bias, -1)
    saved_bits = scores.view(torch.int32).clone()
    assert staged_mismatches(scores, expected, reference_batch, mask, bias) == 0
    assert torch.equal(scores.view(torch.int32), saved_bits)


def test_bit_differences_at_first_and_last_chunk_boundaries():
    scores = torch.zeros(5, 4, 17, 17)
    actual = scores.clone()
    actual.view(torch.int32)[0, 0, 0, 0] ^= 1
    actual.view(torch.int32)[4, 3, 16, 16] ^= 1
    assert staged_mismatches(scores, actual, 2) == 2


def test_signed_zero_and_nan_payloads_are_not_collapsed():
    scores = torch.tensor([-0., float('nan')]).reshape(2, 1, 1, 1)
    actual = scores.clone()
    actual.view(torch.int32)[0, 0, 0, 0] = 0
    actual.view(torch.int32)[1, 0, 0, 0] ^= 1
    assert staged_mismatches(scores, actual, 1) == 2


def test_invalid_partition_rejected():
    scores = torch.zeros(1, 1, 1, 1)
    with pytest.raises(ValueError, match='Invalid staged comparison'):
        staged_mismatches(scores, scores, 0)
    with pytest.raises(ValueError, match='Invalid staged comparison'):
        staged_mismatches(scores, scores.expand(2, 1, 1, 1), 1)


def test_count_bounds_reduction_buffer_and_includes_tail(monkeypatch):
    scores = torch.zeros(5, 4, 17, 17)
    actual = scores.clone()
    actual.view(torch.int32)[0, 0, 0, 0] ^= 1
    actual.view(torch.int32)[4, 3, 16, 16] ^= 1
    original = torch.count_nonzero
    observed = []

    def bounded_count(value):
        assert value.shape[0] <= 2
        observed.append(value.shape[0])
        return original(value)

    monkeypatch.setattr(torch, 'count_nonzero', bounded_count)
    assert bitwise_mismatch_count(scores, actual, outer_batch=2) == 2
    assert observed == [2, 2, 1]
