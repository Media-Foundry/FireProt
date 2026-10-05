"""Reject unsafe microbatch metadata before any GPU operation."""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from experiments.mfma_qk.microbatch_attention import MicrobatchFusion


def instance():
    p = object.__new__(MicrobatchFusion)
    p.batch = 512
    p._owner = threading.get_ident()
    p._lock = threading.Lock()
    p._closed = False
    p._pool = SimpleNamespace(close=lambda: None)
    p.calls = 0
    return p


@pytest.mark.parametrize('batch', [256, 768, 1536])
def test_unmeasured_batch_rejected_before_loader(batch):
    with patch('experiments.mfma_qk.microbatch_attention.WorkspaceFusion') as loader:
        with pytest.raises(ValueError):
            MicrobatchFusion({}, batch)
        loader.assert_not_called()


@pytest.mark.parametrize('kind', ['negative', 'conjugate'])
def test_lazy_view_rejected_before_device_access(kind):
    p = instance()
    q = torch.empty(512, 4, 1536, 32, device='meta')
    mask = torch.empty(1, 1, 1, 1536, device='meta')
    bias = torch.empty(1, 4, 1536, 1536, device='meta')
    bad = torch._neg_view(q) if kind == 'negative' else q.to(torch.complex64).conj()
    with pytest.raises(ValueError, match='Unresolved'):
        p(bad, q, q, mask, bias)
    assert p.calls == 0


def test_owner_and_closed_state_rejected():
    p = instance()
    errors = []
    def call():
        try:
            p(None, None, None, None, None)
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=call)
    thread.start()
    thread.join()
    assert len(errors) == 1 and 'another thread' in errors[0]
    p.close()
    with pytest.raises(RuntimeError, match='closed'):
        p(None, None, None, None, None)


def test_concurrent_call_and_close_rejected():
    p = instance()
    p._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='Concurrent'):
            p(None, None, None, None, None)
        with pytest.raises(RuntimeError, match='idle owner'):
            p.close()
    finally:
        p._lock.release()
