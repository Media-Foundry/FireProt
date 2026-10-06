"""Use the actual Rust registry to check candidate/fallback routing."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from experiments.mfma_qk.workspace_dispatch import WorkspaceDispatcher, load_workspace


def setup(eligible=True):
    accepted = Mock()
    accepted._eligible.return_value = True
    accepted.manifest.return_value = {'accepted': True}
    accepted.return_value = 'accepted fallback'
    pool = Mock()
    pool.eligibility.return_value = (eligible, '' if eligible else 'unmeasured_shape')
    pool.return_value = 'workspace'
    pool.manifest.return_value = {'calls': 0}
    return WorkspaceDispatcher(accepted, pool), accepted, pool


def invoke(dispatcher, **kwargs):
    q = torch.zeros(1)
    with patch('torch.amp.autocast', return_value=nullcontext()):
        return dispatcher('native sentinel', q, q, q, [q, q], **kwargs)


def test_cached_selection_still_invokes_workspace_every_time():
    d, accepted, pool = setup()
    assert invoke(d) == invoke(d) == 'workspace'
    assert pool.call_count == 2 and accepted.call_count == 0
    pool.side_effect = RuntimeError('dynamic lease guard failed')
    with pytest.raises(RuntimeError, match='dynamic lease'):
        invoke(d)
    assert accepted.call_count == 0


def test_eligibility_changes_bypass_previous_cached_choice():
    d, accepted, pool = setup()
    assert invoke(d) == 'workspace'
    pool.eligibility.return_value = (False, 'unmeasured_shape')
    assert invoke(d, use_high_precision=True) == 'accepted fallback'
    assert accepted.call_args.kwargs['use_high_precision'] is True
    assert d.fallback_reasons == {'unmeasured_shape': 1}
    assert pool.call_count == 1
    d.accepted._eligible.return_value = False
    assert invoke(d) == 'accepted fallback'
    assert d.fallback_reasons['accepted_adapter_ineligible'] == 1


def test_disabled_workspace_calls_original_dispatcher_directly():
    d, accepted, _ = setup()
    d.workspace = None
    assert invoke(d) == 'accepted fallback'
    assert d.manifest()['experimental_workspace']['enabled'] is False
    assert d.registry.manifest()['selection_counts'] == []


def test_control_hash_is_checked_before_loader(tmp_path):
    p = tmp_path / 'control.json'
    p.write_text('{}')
    with patch('experiments.mfma_qk.workspace_dispatch.ModelWorkspaceFusion') as loader:
        with pytest.raises(ValueError, match='hash mismatch'):
            load_workspace(p, '0' * 64)
        loader.assert_not_called()
