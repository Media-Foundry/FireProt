"""CPU rejection gates for the experimental stream-owned score workspace."""
import threading
from unittest.mock import patch

import pytest
import torch
import torch.utils.deterministic

from experiments.mfma_qk.benchmark_attention import PackedFusion
from experiments.mfma_qk.workspace_attention import WorkspaceFusion


def pool():
    with patch.object(PackedFusion, '__init__', return_value=None):
        return WorkspaceFusion({}, 32, max_bytes=9 * 2**30)


@pytest.mark.parametrize('batch,budget', [(256, 1), (512, 9 * 2**30), (32, 10 * 2**30)])
def test_budget_rejected_before_loading(batch, budget):
    with patch.object(PackedFusion, '__init__', side_effect=AssertionError('loader called')):
        with pytest.raises(ValueError):
            WorkspaceFusion({}, batch, max_bytes=budget)


def test_owner_rejected_before_tensor_access():
    p = pool()
    errors = []
    def attempt():
        try:
            p(None, None, None, None, None)
        except RuntimeError as error:
            errors.append(str(error))
    t = threading.Thread(target=attempt)
    t.start()
    t.join()
    assert len(errors) == 1 and 'different thread' in errors[0]
    assert p.calls == p.allocations == 0


def test_closed_and_concurrent_access_rejected():
    p = pool()
    p._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match='Concurrent'):
            p(None, None, None, None, None)
        with pytest.raises(RuntimeError, match='idle owner'):
            p.close()
    finally:
        p._lock.release()
    p.close()
    with pytest.raises(RuntimeError, match='closed'):
        p(None, None, None, None, None)


def test_strict_settings_and_cpu_input_rejection():
    p = pool()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        with torch.inference_mode():
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.utils.deterministic.fill_uninitialized_memory = False
            with pytest.raises(RuntimeError, match='filling'):
                p(None, None, None, None, None)
            torch.utils.deterministic.fill_uninitialized_memory = True
            q = torch.empty(32, 4, 1536, 32, device='meta')
            mask = torch.empty(32, 1, 1, 1536, device='meta')
            bias = torch.empty(1, 4, 1536, 1536, device='meta')
            with pytest.raises(ValueError, match='fixed'):
                p(q[:1], q, q, mask, bias)
            with pytest.raises(ValueError, match='one HIP device'):
                p(q, q, q, mask, bias)
        assert p.allocations == p.calls == 0
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        torch.utils.deterministic.fill_uninitialized_memory = fill
