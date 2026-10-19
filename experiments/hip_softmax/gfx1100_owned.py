"""Private QK-score reuse for the pinned experimental N1000 OF3 dispatcher.

Storage checks do not prove ownership. Only the audited dispatcher's fresh
einsum result may be passed here with enabled=True. Native QK and PV stay intact.
"""
import ctypes
import hashlib
from pathlib import Path

import torch
import torch.utils.deterministic

try:
    from .gfx1100_scope import Gfx1100Softmax
except ImportError:
    from gfx1100_scope import Gfx1100Softmax


def disjoint_full_storage(scores, biases):
    storage = scores.untyped_storage()
    if scores.storage_offset() or scores.numel()*scores.element_size() != storage.nbytes():
        return False
    low, high = storage.data_ptr(), storage.data_ptr()+storage.nbytes()
    for bias in biases:
        other = bias.untyped_storage()
        if low < other.data_ptr()+other.nbytes() and other.data_ptr() < high:
            return False
    return True


class Gfx1100OwnedSoftmax(Gfx1100Softmax):
    def __init__(self, path, sha256, owned_path, owned_sha256, enabled):
        super().__init__(path, sha256)
        owned_path = Path(owned_path).resolve()
        if hashlib.sha256(owned_path.read_bytes()).hexdigest() != owned_sha256:
            raise ValueError('Owned softmax library SHA-256 mismatch')
        self.owned_library = ctypes.CDLL(str(owned_path))
        self.owned_entry = self.owned_library.fireprot_softmax_gfx1100_owned
        self.owned_entry.argtypes = self.entry.argtypes
        self.owned_entry.restype = ctypes.c_int
        self.enabled = enabled
        self.receipt['owned_scores'] = dict(enabled=enabled, calls=0, fallbacks=0,
            library=str(owned_path), sha256=owned_sha256,
            ownership='Private fresh QK result in source-pinned AttentionDispatcher._fused')

    def __call__(self, scores, biases):
        receipt = self.receipt['owned_scores']
        if not self.enabled:
            return super().__call__(scores, biases)
        if (not torch.is_inference_mode_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.utils.deterministic.fill_uninitialized_memory):
            raise RuntimeError('Owned scores require strict inference with memory filling')
        if not self.supports(scores, biases):
            raise ValueError('Unsupported experimental owned-score input')
        if scores.shape[-1] != 1000 or not disjoint_full_storage(scores, biases):
            receipt['fallbacks'] += 1
            return super().__call__(scores, biases)
        mask, bias = biases
        batch, n = scores.shape[0], scores.shape[-1]
        waves, reuse = (8, 1) if bias.stride(-1) > bias.stride(-2) else (1, 0)
        with torch.cuda.device(self.device):
            status = self.owned_entry(scores.data_ptr(), mask.data_ptr(), bias.data_ptr(),
                scores.data_ptr(), batch, n,
                mask.stride(0) if mask.shape[0] > 1 else 0, mask.stride(-1),
                *bias.stride()[-3:], waves, reuse, torch.cuda.current_stream().cuda_stream)
        if status:
            raise RuntimeError(f'Owned softmax launch failed: {status}')
        self.launches_submitted += 1
        receipt['calls'] += 1
        key = f'n{n}-b{batch}-w{waves}-r{reuse}-packed0'
        self.receipt['variant_calls'][key] = self.receipt['variant_calls'].get(key, 0)+1
        return scores
