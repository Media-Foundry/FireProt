"""Opt-in model tensor owner; one Rust-governed pool across measured batches.

No model dispatch is installed by importing this module. GPU allocation remains
ordinary strict Torch allocation; Rust owns budget and lifecycle accounting.
"""
import json
import threading

import torch

from fireprot.ops._bridge import bridge, identity
from experiments.mfma_qk.probe_fusion import bitwise_mismatch_count
from experiments.mfma_qk.workspace_attention import WorkspaceFusion, bias_layout_supported

BYTES = 9 * 2**30
SCORE_SHAPE = (256, 4, 1536, 1536)


class ModelWorkspaceFusion:
    def __init__(self, control, *, microbatch=256, allocation_policy='ordinary'):
        if type(microbatch) is not int or microbatch not in (32, 128, 256):
            raise ValueError('Only explicitly qualified capacities32/128/256 are supported')
        if type(allocation_policy) is not str or allocation_policy not in (
                'ordinary', 'release_cached_once'):
            raise ValueError('Unknown experimental allocation policy')
        self.allocation_policy = allocation_policy
        self.cache_release_attempts = self.cache_releases_completed = 0
        self.microbatch = microbatch
        self.score_shape = (microbatch, 4, 1536, 1536)
        self.retained_bytes = microbatch * 4 * 1536 * 1536 * 4
        self._owner = threading.get_ident()
        self._lock = threading.Lock()
        self._closed = self._released = False
        self._lease = bridge.WorkspaceLease(self.score_shape, 4, self.retained_bytes)
        self._pool = WorkspaceFusion(control, microbatch, max_bytes=self.retained_bytes)
        self.calls_by_batch = {}

    def eligibility(self, query, key, value, mask, bias):
        """Immutable tensor metadata only; runtime state is checked on each call."""
        tensors = (query, key, value, mask, bias)
        if any(type(t) is not torch.Tensor for t in tensors):
            return False, 'plain_tensors_required'
        if (query.ndim != 4 or query.shape[0] not in (128, 256, 512, 1024)
                or query.shape[0] % self.microbatch):
            return False, 'unmeasured_outer_batch'
        batch = query.shape[0]
        if any(tuple(t.shape) != (batch, 4, 1536, 32) for t in tensors[:3]):
            return False, 'unmeasured_qkv_shape'
        if (tuple(mask.shape) not in ((1, 1, 1, 1536), (batch, 1, 1, 1536))
                or tuple(bias.shape) != (1, 4, 1536, 1536)):
            return False, 'unmeasured_broadcast'
        if any(t.is_neg() or t.is_conj() for t in tensors):
            return False, 'unresolved_tensor_view'
        if any(not t.is_cuda or t.device != query.device or t.dtype != torch.float32
               or t.requires_grad for t in tensors):
            return False, 'inference_fp32_hip_required'
        if not mask.is_contiguous() or not bias_layout_supported(bias, normalize=True):
            return False, 'unmeasured_bias_or_mask_layout'
        return True, ''

    def _check(self, query, key, value, mask, bias):
        if self._closed or threading.get_ident() != self._owner:
            raise RuntimeError('Model workspace is closed or owned by another thread')
        supported, reason = self.eligibility(query, key, value, mask, bias)
        if not supported:
            raise ValueError(reason)
        if (not torch.is_inference_mode_enabled() or not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
                or not torch.utils.deterministic.fill_uninitialized_memory):
            raise RuntimeError('Strict inference and memory filling are required')
        if torch.cuda.current_device() != query.device.index or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Current device must match; graph capture is unsupported')
        stream = torch.cuda.current_stream(query.device).cuda_stream
        if self._pool._device is not None and (query.device != self._pool._device
                or stream != self._pool._stream):
            raise RuntimeError('Model workspace requires its original device/stream')
        return query.shape[0], stream

    def _allocate(self, query, stream):
        self._lease.reserve(query.device.index, stream)
        # Bind before allocation so interrupted initialization is also drained.
        self._pool._device, self._pool._stream = query.device, stream
        if self.allocation_policy == 'release_cached_once':
            # A retained pool can otherwise pin part of a much larger cached
            # segment needed by subsequent upstream autotuning allocations.
            # Preserve live tensors, the ordinary allocator and strict fill.
            self.cache_release_attempts += 1
            torch.cuda.empty_cache()
            self.cache_releases_completed += 1
        self._pool._scores = torch.empty(self.score_shape, device=query.device, dtype=query.dtype)
        self._pool.allocations += 1
        self._lease.allocation_committed()

    def __call__(self, query, key, value, mask, bias, *, reference_probabilities=None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Concurrent model workspace use is forbidden')
        started = False
        try:
            batch, stream = self._check(query, key, value, mask, bias)
            if reference_probabilities is not None and (
                    type(reference_probabilities) is not torch.Tensor
                    or tuple(reference_probabilities.shape) != (batch, 4, 1536, 1536)
                    or reference_probabilities.device != query.device
                    or reference_probabilities.dtype != torch.float32):
                raise ValueError('Invalid probability reference')
            bias = bias.contiguous()
            started = True
            if self._pool._scores is None:
                self._allocate(query, stream)
            self._lease.begin_use(query.device.index, stream)
            results = []
            for start in range(0, batch, self.microbatch):
                end = start + self.microbatch
                if reference_probabilities is not None:
                    self._pool._scores.fill_(float('nan'))
                part = self._pool(query[start:end], key[start:end], value[start:end],
                    mask if mask.shape[0] == 1 else mask[start:end], bias)
                if reference_probabilities is not None and bitwise_mismatch_count(
                        reference_probabilities[start:end], self._pool._scores):
                    raise RuntimeError('Model workspace probability mismatch')
                results.append(part)
            result = results[0] if batch == self.microbatch else torch.cat(results, dim=0)
            self._lease.finish_use()
            self.calls_by_batch[str(batch)] = self.calls_by_batch.get(str(batch), 0) + 1
            return result
        except BaseException as error:
            if started:
                self._closed = True
                self._lease.fail(type(error).__name__ + ': ' + str(error))
            raise
        finally:
            self._lock.release()

    def close(self):
        if threading.get_ident() != self._owner or not self._lock.acquire(blocking=False):
            raise RuntimeError('Model workspace must be closed by its idle owner')
        try:
            if self._released:
                return
            self._closed = True
            self._lease.begin_close()
            try:
                self._pool.close()
            except BaseException as error:
                self._lease.fail('cleanup: ' + type(error).__name__ + ': ' + str(error))
                raise
            self._lease.release_after_drain()
            self._released = True
        finally:
            self._lock.release()

    def manifest(self):
        return dict(policy_engine=identity, lease=json.loads(self._lease.manifest()),
                    microbatch=self.microbatch, retained_bytes=self.retained_bytes,
                    allocation_policy=self.allocation_policy,
                    cache_release_attempts=self.cache_release_attempts,
                    cache_releases_completed=self.cache_releases_completed,
                    kernel=self._pool.receipt, allocations=self._pool.allocations,
                    successful_kernel_calls=self._pool.calls,
                    calls_by_batch=dict(self.calls_by_batch), closed=self._closed,
                    released=self._released)

    def __enter__(self):
        if self._closed or threading.get_ident() != self._owner:
            raise RuntimeError('Model workspace requires its open owner')
        return self

    def __exit__(self, _kind, error, _traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            if hasattr(error, 'add_note'):
                error.add_note('Workspace cleanup also failed: ' + repr(cleanup))
        return False
