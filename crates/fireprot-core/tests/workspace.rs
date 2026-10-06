use fireprot_core::workspace::{Binding, State, WorkspaceError, WorkspaceLease};

const BYTES: u64 = 9 * (1 << 30);
const BINDING: Binding = Binding {
    device: 7,
    stream: 42,
};

fn lease() -> WorkspaceLease {
    WorkspaceLease::new(&[256, 4, 1536, 1536], 4, BYTES).unwrap()
}

#[test]
fn invalid_budget_and_overflow_rejected() {
    for shape in [&[][..], &[0][..], &[1, 0][..]] {
        assert_eq!(
            WorkspaceLease::new(shape, 4, BYTES).unwrap_err(),
            WorkspaceError::Budget
        );
    }
    assert_eq!(
        WorkspaceLease::new(&[u64::MAX, 2], 4, u64::MAX).unwrap_err(),
        WorkspaceError::Overflow
    );
    assert_eq!(
        WorkspaceLease::new(&[256, 4, 1536, 1536], 4, BYTES - 1).unwrap_err(),
        WorkspaceError::Budget
    );
    assert_eq!(
        WorkspaceLease::new(&[1], 0, BYTES).unwrap_err(),
        WorkspaceError::Budget
    );
}

#[test]
fn mixed_outer_calls_share_exactly_one_allocation() {
    let mut p = lease();
    p.reserve(BINDING).unwrap();
    assert_eq!(p.receipt().charged_bytes, BYTES);
    assert_eq!(p.receipt().committed_bytes, 0);
    assert!(p.reserve(BINDING).is_err());
    p.allocation_committed().unwrap();
    assert!(p.rollback_reservation().is_err());
    for _outer_batch in [256, 512, 1024, 256] {
        p.begin_use(BINDING).unwrap();
        assert!(p.begin_use(BINDING).is_err());
        assert!(p.begin_close().is_err());
        p.finish_use().unwrap();
    }
    assert_eq!(p.receipt().allocation_count, 1);
    assert_eq!(p.receipt().completed_calls, 4);
    assert_eq!(p.receipt().committed_bytes, BYTES);
    assert!(p.release_after_drain().is_err());
    p.begin_close().unwrap();
    p.release_after_drain().unwrap();
    assert_eq!(p.receipt().state, State::Released);
    assert_eq!(p.receipt().charged_bytes, 0);
    assert_eq!(p.receipt().committed_bytes, 0);
    assert_eq!(p.receipt().release_count, 1);
    assert!(p.begin_close().is_err());
    assert!(p.reserve(BINDING).is_err());
    assert!(p.begin_use(BINDING).is_err());
    assert!(p.fail("late".into()).is_err());
}

#[test]
fn failed_allocation_can_roll_back_only_before_commit() {
    let mut p = lease();
    p.reserve(BINDING).unwrap();
    p.rollback_reservation().unwrap();
    assert_eq!(p.receipt().state, State::New);
    assert_eq!(p.receipt().charged_bytes, 0);
    assert!(
        p.reserve(Binding {
            device: 6,
            ..BINDING
        })
        .is_err()
    );
    p.reserve(BINDING).unwrap();
    p.allocation_committed().unwrap();
    assert!(p.rollback_reservation().is_err());
    assert_eq!(p.receipt().committed_bytes, BYTES);
}

#[test]
fn mismatched_binding_rejected_without_mutation() {
    let mut p = lease();
    p.reserve(BINDING).unwrap();
    p.allocation_committed().unwrap();
    for other in [
        Binding {
            device: 6,
            ..BINDING
        },
        Binding {
            stream: 43,
            ..BINDING
        },
    ] {
        let before = serde_json::to_string(p.receipt()).unwrap();
        assert_eq!(p.begin_use(other), Err(WorkspaceError::Binding));
        assert_eq!(serde_json::to_string(p.receipt()).unwrap(), before);
    }
}

#[test]
fn wrong_thread_cannot_mutate_or_release_the_lease() {
    let mut p = lease();
    p.reserve(BINDING).unwrap();
    p.allocation_committed().unwrap();
    let p = std::thread::spawn(move || {
        assert_eq!(p.begin_use(BINDING), Err(WorkspaceError::Owner));
        assert_eq!(p.begin_close(), Err(WorkspaceError::Owner));
        assert_eq!(p.fail("wrong owner".into()), Err(WorkspaceError::Owner));
        p
    })
    .join()
    .unwrap();
    assert_eq!(p.receipt().state, State::Ready);
    assert_eq!(p.receipt().committed_bytes, BYTES);
}

#[test]
fn submission_and_cleanup_failures_keep_accounting_until_confirmed_release() {
    let mut p = lease();
    p.reserve(BINDING).unwrap();
    p.allocation_committed().unwrap();
    p.begin_use(BINDING).unwrap();
    p.fail("launch or PV failed".into()).unwrap();
    assert!(p.begin_use(BINDING).is_err());
    assert_eq!(p.receipt().completed_calls, 0);
    p.begin_close().unwrap();
    p.fail("synchronization failed".into()).unwrap();
    assert_eq!(p.receipt().state, State::Failed);
    assert_eq!(p.receipt().charged_bytes, BYTES);
    assert_eq!(p.receipt().committed_bytes, BYTES);
    assert_eq!(p.receipt().release_count, 0);
    assert!(p.release_after_drain().is_err());
    p.begin_close().unwrap();
    p.release_after_drain().unwrap();
    assert_eq!(p.receipt().charged_bytes, 0);
    assert!(p.receipt().last_failure.is_some());
}

#[test]
fn uncertain_allocation_stays_charged_and_unused_lease_can_close() {
    let mut p = lease();
    p.reserve(BINDING).unwrap();
    p.fail("interrupted allocation acknowledgment".into())
        .unwrap();
    assert_eq!(p.receipt().charged_bytes, BYTES);
    assert_eq!(p.receipt().committed_bytes, 0);
    assert!(p.rollback_reservation().is_err());
    p.begin_close().unwrap();
    p.release_after_drain().unwrap();
    let mut unused = lease();
    unused.begin_close().unwrap();
    unused.release_after_drain().unwrap();
    assert_eq!(unused.receipt().allocation_count, 0);
    assert_eq!(unused.receipt().binding, None);
}
