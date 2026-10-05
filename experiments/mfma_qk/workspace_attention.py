"""Bounded, stream-owned score workspace experiment; no model integration."""
import threading

import torch
import torch.utils.deterministic

from experiments.mfma_qk.benchmark_attention import PackedFusion
from experiments.mfma_qk.probe_fusion import validate_inputs


class WorkspaceFusion(PackedFusion):
    """Reuse initialized private scores; only the independent PV result escapes."""

    def __init__(self, control, batch, *, max_bytes, contiguous_bias=False):
        needed = batch * 4 * 1536 * 1536 * 4
        if batch not in (32, 256) or needed > max_bytes or max_bytes > 9 * 2**30:
            raise ValueError('Experimental workspace requires tested batch and explicit <=9GiB budget')
        super().__init__(control, batch)
        self.batch = batch
        self.retained_bytes = needed
        self.contiguous_bias = contiguous_bias
        self._owner = threading.get_ident()
        self._lock = threading.Lock()
        self._device = self._stream = self._scores = None
        self._closed = False
        self.allocations = self.calls = 0

    def _check_inputs(self, query, key, value, mask, bias):
        if self._closed or threading.get_ident() != self._owner:
            raise RuntimeError('Workspace is closed or called from a different thread')
        if (not torch.is_inference_mode_enabled()
                or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.utils.deterministic.fill_uninitialized_memory):
            raise RuntimeError('Strict inference and uninitialized-memory filling must remain enabled')
        if any(tuple(t.shape) != (self.batch, 4, 1536, 32) for t in (query, key, value)):
            raise ValueError('Workspace Q/K/V shape is fixed for its entire lifetime')
        if (tuple(mask.shape) not in ((1, 1, 1, 1536), (self.batch, 1, 1, 1536))
                or tuple(bias.shape) != (1, 4, 1536, 1536)):
            raise ValueError('Unsupported mask/bias shape')
        if any(not t.is_cuda or t.device != query.device or t.dtype != torch.float32
               or t.requires_grad for t in (query, key, value, mask, bias)):
            raise ValueError('Workspace accepts inference FP32 tensors on one HIP device')
        if not mask.is_contiguous() or not (bias.is_contiguous()
                or bias.transpose(-1, -2).is_contiguous()):
            raise ValueError('Unsupported mask/bias layout')

    def __call__(self, query, key, value, mask, bias):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent workspace use is forbidden')
        submitted = False
        try:
            self._check_inputs(query, key, value, mask, bias)
            stream = torch.cuda.current_stream(query.device).cuda_stream
            if self._device is not None and (query.device != self._device or stream != self._stream):
                raise RuntimeError('Workspace is bound to one device and stream')
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Workspace experiment does not support graph capture')
            packed_q = query.reshape(self.batch * 4, 1536, 32)
            packed_k = key.transpose(-1, -2).reshape(self.batch * 4, 32, 1536)
            if self.contiguous_bias:
                bias = bias.contiguous()
            if validate_inputs(packed_q, packed_k, mask, bias) != self.batch:
                raise ValueError('Packed operand batch differs from workspace')
            if self._scores is None:
                # First allocation uses the ordinary strict Torch NaN fill.
                self._scores = torch.empty(self.batch, 4, 1536, 1536,
                                           device=query.device, dtype=query.dtype)
                self._device, self._stream = query.device, stream
                self.allocations += 1
            submitted = True
            status = self.entry(packed_q.data_ptr(), packed_k.data_ptr(), mask.data_ptr(),
                bias.data_ptr(), self._scores.data_ptr(), self.batch,
                mask.stride(0) if mask.shape[0] != 1 else 0, mask.stride(-1),
                bias.stride(1), bias.stride(2), bias.stride(3), 1, stream)
            if status:
                raise RuntimeError(f'Workspace fusion launch failed: {status}')
            result = torch.einsum('...qk, ...kc->...qc', self._scores, value)
            if result.untyped_storage().data_ptr() == self._scores.untyped_storage().data_ptr():
                raise RuntimeError('Private score storage escaped through PV')
            # Same-stream ordering puts every PV read before the next overwrite.
            self.calls += 1
            return result
        except Exception:
            if submitted:
                self._closed = True
            raise
        finally:
            self._lock.release()

    def close(self):
        if threading.get_ident() != self._owner or not self._lock.acquire(blocking=False):
            raise RuntimeError('Workspace must be closed by its idle owner')
        try:
            if self._device is not None:
                torch.cuda.synchronize(self._device)
            self._scores = None
            self._closed = True
        finally:
            self._lock.release()
