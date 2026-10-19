"""Rust-registry attention selection with gfx1100 tensor eligibility."""
import torch
from fireprot.adapters.openfold3.attention import AttentionDispatcher
from fireprot.backends.amd.softmax_gfx1100 import CASES
from fireprot.ops.registry import signature_from_tensor

class Gfx1100Dispatcher(AttentionDispatcher):
    def _eligible(self, query, key, value, biases):
        return (super()._eligible(query, key, value, biases)
                and (query.shape[0], query.shape[2]) in CASES)

    def __call__(self, native, query, key, value, biases, use_high_precision=False):
        if type(query) is not torch.Tensor:
            return native(query, key, value, biases, use_high_precision=use_high_precision)
        signature = signature_from_tensor("attention", query, architecture="gfx1100",
            mode="training" if torch.is_grad_enabled() else "inference", mask="pair_and_token",
            inplace=False, attributes={"eligible": self._eligible(query, key, value, biases),
                                       "high_precision": use_high_precision})
        return self.registry.dispatch(signature, native, query, key, value, biases,
                                      use_high_precision=use_high_precision)
