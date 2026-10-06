"""CPU lifecycle fault injection with the real Rust lease; not GPU parity tests."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.mfma_qk.model_workspace import BYTES, ModelWorkspaceFusion


class FakePool:
    """Small independent outputs let tests detect reuse of returned storage."""
    def __init__(self, *args, **kwargs):
        self._scores = self._device = self._stream = None
        self.allocations = self.calls = 0
        self.receipt = {'test_double': True}
        self.failure = self.cleanup_failure = None

    def __call__(self, *args):
        if self.failure is not None:
            raise self.failure
        self.calls += 1
        return torch.tensor([[float(self.calls)]])

    def close(self):
        if self.cleanup_failure is not None:
            raise self.cleanup_failure
        self._scores = None


def instance():
    with patch('experiments.mfma_qk.model_workspace.WorkspaceFusion', FakePool):
        return ModelWorkspaceFusion({})


def fake_call(p, batch=256, allocation_failure=None):
    # These tests replace device validation and allocation only. Real validation
    # is tested separately below; the Rust transition code is never mocked.
    q = TensorInput(device=SimpleNamespace(index=7), dtype=torch.float32)
    q.batch = batch
    mask = torch.empty(1, 1, 1, 1536, device='meta')
    bias = torch.empty(1, 4, 1536, 1536, device='meta')
    allocation = (dict(return_value=torch.zeros(1)) if allocation_failure is None
                  else dict(side_effect=allocation_failure))
    with patch.object(p, '_check', return_value=(batch, 123)), patch(
            'experiments.mfma_qk.model_workspace.torch.empty', **allocation):
        return p(q, q, q, mask, bias)


class TensorInput(SimpleNamespace):
    def __getitem__(self, _slice):
        return self


def test_mixed_batches_one_pool_and_previous_outputs_survive():
    p = instance()
    first = fake_call(p)
    original = first.clone()
    fake_call(p, 512)
    fake_call(p, 1024)
    assert torch.equal(first, original)
    r = p.manifest()
    assert r['allocations'] == r['lease']['allocation_count'] == 1
    assert r['successful_kernel_calls'] == 7
    assert r['calls_by_batch'] == {'256': 1, '512': 1, '1024': 1}
    assert r['lease']['charged_bytes'] == BYTES
    p.close()
    p.close()
    assert p.manifest()['lease']['release_count'] == 1
    assert p.manifest()['lease']['charged_bytes'] == 0


@pytest.mark.parametrize('failure', [RuntimeError('HIP launch failed'), KeyboardInterrupt()])
def test_submission_exception_poisoned_and_propagated(failure):
    p = instance()
    p._pool.failure = failure
    with pytest.raises(type(failure)) as caught:
        fake_call(p)
    assert caught.value is failure
    r = p.manifest()
    assert r['closed'] and not r['released'] and r['lease']['state'] == 'failed'
    assert r['lease']['charged_bytes'] == BYTES and r['lease']['completed_calls'] == 0
    with pytest.raises(RuntimeError, match='closed'):
        p(None, None, None, None, None)
    p.close()
    assert p.manifest()['lease']['charged_bytes'] == 0


def test_cleanup_failure_never_reports_release_and_can_be_retried():
    p = instance()
    fake_call(p)
    p._pool.cleanup_failure = RuntimeError('sync failed')
    with pytest.raises(RuntimeError, match='sync failed'):
        p.close()
    r = p.manifest()
    assert not r['released'] and r['lease']['release_count'] == 0
    assert r['lease']['charged_bytes'] == r['lease']['committed_bytes'] == BYTES
    p._pool.cleanup_failure = None
    p.close()
    assert p.manifest()['lease']['charged_bytes'] == 0


def test_context_cleanup_preserves_original_interruption():
    p = instance()
    fake_call(p)
    p._pool.cleanup_failure = RuntimeError('sync failed')
    original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with p:
            raise original
    assert caught.value is original
    assert p.manifest()['lease']['charged_bytes'] == BYTES
    assert 'sync failed' in str(original.__notes__)


def test_unallocated_close_and_concurrent_access():
    p = instance()
    p._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='Concurrent'):
            p(None, None, None, None, None)
        with pytest.raises(RuntimeError, match='idle owner'):
            p.close()
    finally:
        p._lock.release()
    with ThreadPoolExecutor(max_workers=1) as executor:
        f = executor.submit(p.close)
        with pytest.raises(RuntimeError, match='idle owner'):
            f.result()
    p.close()
    assert p.manifest()['allocations'] == 0


@pytest.mark.parametrize('shape,reason', [((64, 4, 1536, 32), 'outer_batch'),
    ((256, 4, 1000, 32), 'qkv_shape'), ((256, 4, 1536, 16), 'qkv_shape')])
def test_unmeasured_signatures_rejected_without_allocation(shape, reason):
    p = instance()
    q = torch.empty(shape, device='meta')
    mask = torch.empty(1, 1, 1, shape[-2], device='meta')
    bias = torch.empty(1, 4, shape[-2], shape[-2], device='meta')
    supported, why = p.eligibility(q, q, q, mask, bias)
    assert not supported and reason in why
    with pytest.raises(ValueError, match=reason):
        p(q, q, q, mask, bias)
    assert p.manifest()['lease']['charged_bytes'] == 0


@pytest.mark.parametrize('failure', [torch.OutOfMemoryError('allocation failed'), KeyboardInterrupt()])
def test_allocation_failure_retains_conservative_charge_until_cleanup(failure):
    p = instance()
    with pytest.raises(type(failure)) as caught:
        fake_call(p, allocation_failure=failure)
    assert caught.value is failure
    r = p.manifest()
    assert r['allocations'] == r['successful_kernel_calls'] == 0
    assert r['lease']['committed_bytes'] == 0
    assert r['lease']['charged_bytes'] == BYTES and r['lease']['state'] == 'failed'
    p.close()
    assert p.manifest()['lease']['charged_bytes'] == 0


def test_partial_microbatch_failure_cannot_return_or_resume():
    p = instance()
    error = RuntimeError('second microchunk PV failed')
    with patch.object(FakePool, '__call__', side_effect=[torch.ones(1, 1), error]) as submit:
        with pytest.raises(RuntimeError, match='second microchunk'):
            fake_call(p, 1024)
        assert submit.call_count == 2
    assert p.manifest()['lease']['completed_calls'] == 0
    assert p.manifest()['calls_by_batch'] == {}
    with pytest.raises(RuntimeError, match='closed'):
        p(None, None, None, None, None)
    p.close()


def test_dynamic_context_checked_on_every_call_even_with_cached_metadata():
    p = instance()
    q = TensorInput(device=SimpleNamespace(index=7), shape=(256, 4, 1536, 32))
    with patch.object(p, 'eligibility', return_value=(True, '')), patch(
            'torch.is_inference_mode_enabled', return_value=True), patch(
            'torch.are_deterministic_algorithms_enabled', return_value=True), patch(
            'torch.is_deterministic_algorithms_warn_only_enabled', return_value=False), patch(
            'torch.utils.deterministic', SimpleNamespace(fill_uninitialized_memory=True)), patch(
            'torch.cuda.current_device', return_value=7), patch(
            'torch.cuda.is_current_stream_capturing', return_value=False), patch(
            'torch.cuda.current_stream', return_value=SimpleNamespace(cuda_stream=123)):
        assert p._check(q, q, q, q, q) == (256, 123)
        p._pool._device, p._pool._stream = q.device, 124
        with pytest.raises(RuntimeError, match='original device/stream'):
            p._check(q, q, q, q, q)
        p._pool._stream = 123
        with patch('torch.cuda.is_current_stream_capturing', return_value=True):
            with pytest.raises(RuntimeError, match='graph capture'):
                p._check(q, q, q, q, q)
        with patch('torch.cuda.current_device', return_value=6):
            with pytest.raises(RuntimeError, match='Current device'):
                p._check(q, q, q, q, q)
        with patch.object(torch.utils.deterministic, 'fill_uninitialized_memory', False):
            with pytest.raises(RuntimeError, match='Strict inference'):
                p._check(q, q, q, q, q)


@pytest.mark.parametrize('capacity', [32, 128, 256])
def test_fixed_capacities_derive_one_pool_and_correct_slice_counts(capacity):
    with patch('experiments.mfma_qk.model_workspace.WorkspaceFusion', FakePool):
        p = ModelWorkspaceFusion({}, microbatch=capacity)
    batches = [b for b in (128, 256, 512, 1024) if b % capacity == 0]
    for batch in batches:
        fake_call(p, batch)
    r = p.manifest()
    assert p.score_shape == (capacity, 4, 1536, 1536)
    assert r['microbatch'] == capacity and r['retained_bytes'] == capacity * BYTES // 256
    assert r['lease']['charged_bytes'] == r['lease']['budget_bytes'] == r['retained_bytes']
    assert r['lease']['completed_calls'] == len(batches)
    assert r['allocations'] == 1 and r['successful_kernel_calls'] == sum(b // capacity for b in batches)
    p.close()
    assert p.manifest()['lease']['charged_bytes'] == 0


@pytest.mark.parametrize('capacity', [True, 64, 0, 512, '128'])
def test_unqualified_capacity_rejected_before_loading(capacity):
    with patch('experiments.mfma_qk.model_workspace.WorkspaceFusion') as loader:
        with pytest.raises(ValueError, match='qualified capacities'):
            ModelWorkspaceFusion({}, microbatch=capacity)
        loader.assert_not_called()


def test_smaller_outer_batch_rejected_by_default_pool():
    p = instance()
    q = torch.empty(128, 4, 1536, 32, device='meta')
    mask = torch.empty(1, 1, 1, 1536, device='meta')
    bias = torch.empty(1, 4, 1536, 1536, device='meta')
    assert p.eligibility(q, q, q, mask, bias) == (False, 'unmeasured_outer_batch')
