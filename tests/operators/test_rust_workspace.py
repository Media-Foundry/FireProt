"""Real CPython/Rust lease boundary, without allocating GPU storage."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from fireprot.ops._bridge import bridge

BYTES = 9 * 2**30


def lease():
    return bridge.WorkspaceLease([256, 4, 1536, 1536], 4, BYTES)


def receipt(p):
    return json.loads(p.manifest())


def test_checked_shape_budget_and_overflow():
    for shape, width, budget in [([], 4, BYTES), ([1], 0, BYTES),
                                  ([256, 4, 1536, 1536], 4, BYTES - 1),
                                  ([2**63, 4], 4, 2**64 - 1)]:
        with pytest.raises(ValueError):
            bridge.WorkspaceLease(shape, width, budget)
    with pytest.raises(OverflowError):
        bridge.WorkspaceLease([-1], 4, BYTES)


def test_shared_lease_receipt_and_additive_abi():
    assert bridge.ABI_VERSION == 1
    assert callable(bridge.OperatorRegistry)
    p = lease()
    p.reserve(7, 123)
    p.allocation_committed()
    for _batch in (256, 512, 1024):
        p.begin_use(7, 123)
        with pytest.raises(RuntimeError, match='Busy'):
            p.begin_use(7, 123)
        p.finish_use()
    r = receipt(p)
    assert r['allocation_count'] == 1 and r['completed_calls'] == 3
    assert r['charged_bytes'] == r['committed_bytes'] == BYTES
    assert r['binding'] == dict(device=7, stream=123)
    p.begin_close()
    p.release_after_drain()
    r = receipt(p)
    assert r['state'] == 'released' and r['release_count'] == 1
    assert r['charged_bytes'] == r['committed_bytes'] == 0
    with pytest.raises(RuntimeError, match='Released'):
        p.begin_use(7, 123)


def test_thread_and_stream_rejection_leave_state_unchanged():
    p = lease()
    p.reserve(7, 123)
    p.allocation_committed()
    before = receipt(p)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(p.begin_use, 7, 123)
        with pytest.raises(RuntimeError, match='another thread'):
            future.result()
    for binding in [(6, 123), (7, 124)]:
        with pytest.raises(RuntimeError, match='device or stream'):
            p.begin_use(*binding)
    assert receipt(p) == before


def test_cleanup_failure_retains_charge_until_acknowledged():
    p = lease()
    p.reserve(7, 123)
    p.allocation_committed()
    p.begin_use(7, 123)
    p.fail('PV failed')
    p.begin_close()
    p.fail('drain failed')
    r = receipt(p)
    assert r['state'] == 'failed'
    assert r['committed_bytes'] == r['charged_bytes'] == BYTES
    assert r['completed_calls'] == r['release_count'] == 0
    with pytest.raises(RuntimeError):
        p.release_after_drain()
    p.begin_close()
    p.release_after_drain()
    assert receipt(p)['last_failure'] == 'drain failed'


def test_preallocation_rollback_is_distinct_from_committed_failure():
    p = lease()
    p.reserve(7, 123)
    p.rollback_reservation()
    assert receipt(p)['charged_bytes'] == 0
    p.reserve(7, 123)
    p.allocation_committed()
    with pytest.raises(RuntimeError):
        p.rollback_reservation()
    assert receipt(p)['charged_bytes'] == BYTES
