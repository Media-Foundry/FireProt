"""Experimental overwrite of private QK scores in the source-pinned OF3 adapter.

The caller must create scores for this attention invocation and give them no
other consumers. Storage checks cannot establish that ownership by themselves.
The existing C ABI and reduction order are unchanged.
"""
import torch
import torch.utils.deterministic

from fireprot.backends.amd.softmax_fp32 import SoftmaxFP32


def disjoint_full_storage(scores, biases):
    storage = scores.untyped_storage()
    if (scores.storage_offset()
            or scores.numel() * scores.element_size() != storage.nbytes()):
        return False
    low, high = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
    for bias in biases:
        other = bias.untyped_storage()
        other_low = other.data_ptr()
        if low < other_low + other.nbytes() and other_low < high:
            return False
    return True


class OwnedScoreSoftmax(SoftmaxFP32):
    def __init__(self):
        super().__init__()
        self.inplace_calls = {}
        self.storage_fallbacks = 0

    def __call__(self, scores, biases):
        if (not torch.is_inference_mode_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.utils.deterministic.fill_uninitialized_memory):
            raise RuntimeError("Owned scores require strict inference with memory filling")
        if not self.supports(scores, biases):
            raise ValueError("Unsupported FP32 softmax tensors or device")
        if not disjoint_full_storage(scores, biases):
            self.storage_fallbacks += 1
            return super().__call__(scores, biases)
        mask, bias = biases
        with torch.cuda.device(self.device):
            bias = bias.contiguous()
            status = self.entry(
                scores.data_ptr(), mask.data_ptr(), bias.data_ptr(), scores.data_ptr(),
                scores.shape[0], scores.shape[-1],
                mask.stride(0) if mask.shape[0] > 1 else 0, mask.stride(-1),
                *bias.stride()[-3:], 2, 1, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f"In-place FP32 softmax HIP launch failed: {status}")
        self.launches_submitted += 1
        key = f"B{scores.shape[0]}_N{scores.shape[-1]}"
        self.inplace_calls[key] = self.inplace_calls.get(key, 0) + 1
        return scores

    def experiment_manifest(self):
        return dict(mode="owned_scores_strict", submitted_shapes=self.inplace_calls,
                    storage_fallbacks=self.storage_fallbacks,
                    limits=["Shape counts include tuning; not steady-state frequency.",
                            "Private-score ownership requires the audited adapter."])
