"""Experimental B512/B1024 attention using one initialized B256 workspace."""
import threading

import torch

from experiments.mfma_qk.workspace_attention import WorkspaceFusion, bias_layout_supported
from experiments.mfma_qk.probe_fusion import bitwise_mismatch_count


class MicrobatchFusion:
    def __init__(self, control, batch):
        if batch not in (512, 1024):
            raise ValueError('Only experimental outer batches512/1024 are supported')
        self.batch = batch
        self._owner = threading.get_ident()
        self._lock = threading.Lock()
        self._closed = False
        self._pool = WorkspaceFusion(control, 256, max_bytes=9 * 2**30)
        self.calls = 0

    def _check(self, query, key, value, mask, bias):
        if self._closed or threading.get_ident() != self._owner:
            raise RuntimeError('Microbatch workspace is closed or owned by another thread')
        tensors = (query, key, value, mask, bias)
        if any(type(t) is not torch.Tensor for t in tensors):
            raise ValueError('Only plain tensors are supported')
        if any(tuple(t.shape) != (self.batch, 4, 1536, 32) for t in tensors[:3]):
            raise ValueError('Unsupported full-batch Q/K/V shape')
        if (tuple(mask.shape) not in ((1, 1, 1, 1536), (self.batch, 1, 1, 1536))
                or tuple(bias.shape) != (1, 4, 1536, 1536)):
            raise ValueError('Unsupported full-batch mask/bias shape')
        if any(t.is_neg() or t.is_conj() for t in tensors):
            raise ValueError('Unresolved negative/conjugate tensor views are forbidden')
        if any(not t.is_cuda or t.device != query.device or t.dtype != torch.float32
               or t.requires_grad for t in tensors):
            raise ValueError('Expected inference FP32 tensors on one HIP device')
        if not mask.is_contiguous() or not bias_layout_supported(bias, normalize=True):
            raise ValueError('Unsupported mask/bias layout')
        if (not torch.is_inference_mode_enabled() or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.utils.deterministic.fill_uninitialized_memory):
            raise RuntimeError('Strict inference and memory filling are required')
        stream = torch.cuda.current_stream(query.device).cuda_stream
        if self._pool._device is not None and (query.device != self._pool._device
                or stream != self._pool._stream):
            raise RuntimeError('Microbatch workspace requires its original device/stream')
        if torch.cuda.current_device() != query.device.index or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Current device must match; graph capture is unsupported')

    def __call__(self, query, key, value, mask, bias, *, reference_probabilities=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent microbatch use is forbidden')
        try:
            self._check(query, key, value, mask, bias)
            if reference_probabilities is not None and (
                    tuple(reference_probabilities.shape) != (self.batch, 4, 1536, 1536)
                    or reference_probabilities.device != query.device
                    or reference_probabilities.dtype != torch.float32):
                raise ValueError('Invalid full-batch probability reference')
            # One conversion per complete call; its cost stays inside timing.
            bias = bias.contiguous()
            results = []
            for start in range(0, self.batch, 256):
                end = start + 256
                if reference_probabilities is not None and self._pool._scores is not None:
                    self._pool._scores.fill_(float('nan'))
                part = self._pool(query[start:end], key[start:end], value[start:end],
                    mask if mask.shape[0] == 1 else mask[start:end], bias)
                if reference_probabilities is not None:
                    if bitwise_mismatch_count(reference_probabilities[start:end], self._pool._scores):
                        raise RuntimeError('Microbatch probability mismatch')
                results.append(part)
            result = torch.cat(results, dim=0)
            self.calls += 1
            return result
        finally:
            self._lock.release()

    def close(self):
        if threading.get_ident() != self._owner or not self._lock.acquire(blocking=False):
            raise RuntimeError('Microbatch workspace requires its idle owner for close')
        try:
            self._pool.close()
            self._closed = True
        finally:
            self._lock.release()
