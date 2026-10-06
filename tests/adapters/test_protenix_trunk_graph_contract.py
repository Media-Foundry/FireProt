"""CPU coverage for experiment bookkeeping; these do not qualify GPU capture."""
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[2] / 'experiments/protenix_base/trunk_graph.py'
spec = importlib.util.spec_from_file_location('trunk_graph_contract', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ReplayContract = module.ReplayContract


@pytest.mark.parametrize('group', [4, 12, 48])
def test_graph_tracks_capture_separately_from_scientific_replay(group):
    contract = ReplayContract(True, group)
    checks = []
    contract.guards.append(lambda: checks.append('checked'))
    for _ in range(48 // group):
        with contract.capture():
            assert contract.capturing
        for _ in range(9):
            contract.before_replay()
            contract.replays += 1
    contract.finish(10)
    contract.validate_completed()
    assert len(checks) == 9 * (48 // group)
    assert contract.python_contractions == 280
    assert contract.logical_contractions == 1048
    with pytest.raises(RuntimeError, match='outside captured'):
        contract.before_replay()


def test_capture_exception_clears_state_without_claiming_completion():
    contract = ReplayContract(True, 12)
    with pytest.raises(ValueError, match='capture failed'):
        with contract.capture():
            raise ValueError('capture failed')
    assert not contract.capturing and contract.captures == 0
    with pytest.raises(RuntimeError, match='outside captured'):
        contract.before_replay()
    with pytest.raises(RuntimeError, match='did not complete'):
        contract.validate_completed()


def test_missing_replays_or_cycles_are_rejected():
    contract = ReplayContract(True, 48)
    with contract.capture():
        pass
    contract.replays = 8
    with pytest.raises(RuntimeError, match='Incomplete'):
        contract.finish(10)
    contract.replays = 9
    with pytest.raises(RuntimeError, match='Incomplete'):
        contract.finish(9)


def test_workspace_guard_failure_prevents_replay():
    contract = ReplayContract(True, 48)
    with contract.capture():
        pass
    def changed_stream():
        raise RuntimeError('workspace stream changed')
    contract.guards.append(changed_stream)
    with pytest.raises(RuntimeError, match='workspace stream changed'):
        contract.before_replay()
    assert contract.replays == 0


def test_eager_control_keeps_complete_python_coverage():
    contract = ReplayContract(False, 12)
    assert contract.python_contractions == contract.logical_contractions == 1048
    with pytest.raises(RuntimeError, match='Unexpected'):
        with contract.capture():
            pass
    contract.finish(10)
    contract.validate_completed()


@pytest.mark.parametrize('enabled,group', [(1, 12), (True, 0), (True, 8)])
def test_unqualified_partitions_are_rejected(enabled, group):
    with pytest.raises(ValueError):
        ReplayContract(enabled, group)
