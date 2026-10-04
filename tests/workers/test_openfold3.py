"""Explicit upstream deterministic execution must be enforced before forward."""
from types import SimpleNamespace

import pytest

from python.fireprot_workers.openfold3 import check_determinism


@pytest.mark.parametrize('requested,enabled,warn,passes', [
    (None, False, False, True),
    (None, True, True, True),
    (True, True, False, True),
    (False, False, False, True),
    (True, False, False, False),
    (False, True, False, False),
    (True, True, True, False),
])
def test_upstream_policy_is_enforced(requested, enabled, warn, passes):
    torch = SimpleNamespace(
        are_deterministic_algorithms_enabled=lambda: enabled,
        is_deterministic_algorithms_warn_only_enabled=lambda: warn,
    )
    workload = {} if requested is None else {'deterministic': requested}
    if passes:
        check_determinism(torch, workload)
    else:
        with pytest.raises(RuntimeError):
            check_determinism(torch, workload)
