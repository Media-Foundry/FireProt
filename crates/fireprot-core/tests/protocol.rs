use fireprot_core::protocol::*;
use fireprot_core::request::ArtifactRef;
use fireprot_core::{Request, ValidatedRequest};
use serde_json::json;

fn session() -> Session {
    let artifact = json!({"path":"/assets/file","sha256":"a".repeat(64)});
    let r:Request=serde_json::from_value(json!({"schema_version":1,"run_id":"run-1","backend":"native","seed":101,"hip_device":7,
        "workload":{"model":"openfold3","model_name":"openbind-0","recycles":3,"diffusion_steps":200,"samples":1,"precision":"float32"},
        "source":{"root":"/source","commit":"b".repeat(40)},"checkpoints":{"model":artifact},"input":artifact,"output_dir":"/output"})).unwrap();
    Session::new(&ValidatedRequest::try_from(r).unwrap(), 42).unwrap()
}
fn event(sequence: u64, payload: WorkerEvent) -> Envelope {
    Envelope {
        schema_version: 1,
        run_id: "run-1".into(),
        sequence,
        payload,
    }
}
fn ready() -> WorkerEvent {
    WorkerEvent::Ready {
        device: DeviceIdentity {
            worker_pid: 42,
            logical_device: 0,
            pci_address: "0000:93:00.0".into(),
            architecture: "gfx90a".into(),
        },
    }
}
fn outputs() -> WorkerEvent {
    WorkerEvent::Outputs {
        artifacts: vec![ArtifactRef {
            path: "prediction.npz".into(),
            sha256: "c".repeat(64),
        }],
    }
}
fn checks() -> CompletionChecks {
    CompletionChecks {
        exit_code: Some(0),
        verified_artifacts: 1,
        isolation_samples: 2,
        external_pids: vec![],
        monitor_errors: vec![],
    }
}
fn pending() -> Session {
    let mut s = session();
    s.receive(event(0, ready())).unwrap();
    s.authorize_start().unwrap();
    s.receive(event(1, outputs())).unwrap();
    s
}

#[test]
fn worker_cannot_self_declare_success() {
    let mut s = pending();
    assert_eq!(s.phase(), Phase::AwaitingVerification);
    s.finish(checks()).unwrap();
    assert_eq!(s.phase(), Phase::Succeeded);
    assert_eq!(s.receive(event(2, outputs())), Err(ProtocolError::Terminal));
    assert_eq!(s.phase(), Phase::Succeeded);
}
#[test]
fn clean_exit_without_predictions_fails() {
    let mut s = session();
    s.receive(event(0, ready())).unwrap();
    s.authorize_start().unwrap();
    let mut c = checks();
    c.verified_artifacts = 0;
    assert_eq!(s.finish(c), Err(ProtocolError::Verification));
    assert_eq!(s.phase(), Phase::Failed);
}
#[test]
fn ready_requires_supervisor_authorization() {
    let mut s = session();
    s.receive(event(0, ready())).unwrap();
    assert_eq!(s.phase(), Phase::AwaitingStart);
    assert_eq!(s.receive(event(1, outputs())), Err(ProtocolError::Order));
    assert_eq!(s.authorize_start(), Err(ProtocolError::Order));
}
#[test]
fn identity_version_and_sequence_errors_are_permanent() {
    for (kind, expected) in [
        (0, ProtocolError::Schema),
        (1, ProtocolError::RunId),
        (2, ProtocolError::Sequence),
        (3, ProtocolError::Identity),
    ] {
        let mut s = session();
        let mut e = event(0, ready());
        match kind {
            0 => e.schema_version = 9,
            1 => e.run_id = "other".into(),
            2 => e.sequence = 1,
            _ => {
                if let WorkerEvent::Ready { device } = &mut e.payload {
                    device.worker_pid = 43;
                }
            }
        }
        assert_eq!(s.receive(e), Err(expected));
        assert_eq!(s.phase(), Phase::Failed);
        assert_eq!(s.receive(event(0, ready())), Err(ProtocolError::Terminal));
    }
}
#[test]
fn duplicate_and_out_of_order_events_fail() {
    let mut s = session();
    s.receive(event(0, ready())).unwrap();
    s.authorize_start().unwrap();
    assert_eq!(
        s.receive(event(
            0,
            WorkerEvent::Progress {
                stage: "trunk".into()
            }
        )),
        Err(ProtocolError::Sequence)
    );
    let mut s = pending();
    assert_eq!(s.receive(event(2, outputs())), Err(ProtocolError::Order));
}
#[test]
fn all_completion_checks_are_required() {
    for mode in 0..6 {
        let mut s = pending();
        let mut c = checks();
        match mode {
            0 => c.exit_code = None,
            1 => c.exit_code = Some(1),
            2 => c.verified_artifacts = 0,
            3 => c.isolation_samples = 0,
            4 => c.external_pids.push(99),
            _ => c.monitor_errors.push("SMI unavailable".into()),
        }
        assert_eq!(s.finish(c), Err(ProtocolError::Verification));
        assert_eq!(s.phase(), Phase::Failed);
    }
}
#[test]
fn unsafe_or_duplicate_artifacts_fail() {
    for path in ["/tmp/elsewhere", "../escape", "", "."] {
        let mut s = session();
        s.receive(event(0, ready())).unwrap();
        s.authorize_start().unwrap();
        let e = WorkerEvent::Outputs {
            artifacts: vec![ArtifactRef {
                path: path.into(),
                sha256: "a".repeat(64),
            }],
        };
        assert_eq!(s.receive(event(1, e)), Err(ProtocolError::Artifacts));
    }
    let mut s = session();
    s.receive(event(0, ready())).unwrap();
    s.authorize_start().unwrap();
    let WorkerEvent::Outputs { mut artifacts } = outputs() else {
        unreachable!()
    };
    artifacts.push(artifacts[0].clone());
    assert_eq!(
        s.receive(event(1, WorkerEvent::Outputs { artifacts })),
        Err(ProtocolError::Artifacts)
    );
}
#[test]
fn cancellation_and_worker_failures_cannot_be_completed() {
    let mut s = pending();
    s.abort("cancelled").unwrap();
    assert_eq!(s.failure(), Some("cancelled"));
    assert_eq!(s.finish(checks()), Err(ProtocolError::Terminal));
    let mut s = session();
    s.receive(event(
        0,
        WorkerEvent::Failed {
            message: "checkpoint mismatch".into(),
        },
    ))
    .unwrap();
    assert_eq!(s.phase(), Phase::Failed);
    assert_eq!(s.failure(), Some("checkpoint mismatch"));
}
#[test]
fn protocol_round_trip_rejects_unknown_worker_success_fields() {
    let mut value = serde_json::to_value(event(0, ready())).unwrap();
    let decoded: Envelope = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(decoded, event(0, ready()));
    value["payload"]["isolation_passed"] = true.into();
    assert!(serde_json::from_value::<Envelope>(value).is_err());
}
