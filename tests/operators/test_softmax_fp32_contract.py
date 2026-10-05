"""CPU/meta contract checks; numerical HIP acceptance is a separate GPU gate."""
import pytest
import torch

from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32, compatible_tensors


def tensors(n=1000, batch=8, ending=False):
    scores = torch.empty(batch, 4, n, n, device="meta")
    mask = torch.empty(1, 1, 1, n, device="meta").expand(batch, 1, 1, n)
    bias = torch.empty(n, n, 4, device="meta").permute(
        (2, 1, 0) if ending else (2, 0, 1)).unsqueeze(0)
    return scores, [mask, bias]


@pytest.mark.parametrize("n", (1000, 1536))
@pytest.mark.parametrize("ending", (False, True))
def test_real_bias_strides_and_broadcast_mask(n, ending):
    scores, biases = tensors(n, ending=ending)
    assert biases[0].stride(0) == 0
    assert not biases[1].is_contiguous()
    assert compatible_tensors(scores, biases)


@pytest.mark.parametrize("change", ("score_view", "mask_heads", "bias_length", "dtype", "grad", "negative"))
def test_unsupported_storage_and_semantics_rejected(change):
    scores, biases = tensors()
    if change == "score_view":
        scores = scores.transpose(-2, -1)
    elif change == "mask_heads":
        biases[0] = biases[0].expand(8, 4, 1, 1000)
    elif change == "bias_length":
        biases[1] = biases[1][..., :-1]
    elif change == "dtype":
        biases[1] = biases[1].to(torch.bfloat16)
    elif change == "grad":
        scores.requires_grad_(True)
    elif change == "negative":
        biases[1] = torch._neg_view(biases[1])
    assert not compatible_tensors(scores, biases)


def test_unmatched_lengths_and_non_tensor_inputs():
    for n in (999, 1024, 1537):
        assert not compatible_tensors(*tensors(n))
    scores, biases = tensors()
    assert not compatible_tensors(scores, biases[:1])
    assert not compatible_tensors(scores, [object(), biases[1]])
    assert not compatible_tensors(object(), biases)


def test_non_gpu_call_never_reaches_c_abi():
    kernel = SoftmaxFP32.__new__(SoftmaxFP32)
    kernel.device = 0
    def unexpected(*args):
        raise AssertionError("C ABI must not run for meta/CPU tensors")
    kernel.entry = unexpected
    with torch.no_grad(), pytest.raises(ValueError, match="Unsupported"):
        kernel(*tensors())
