use fireprot_core::{
    Request, ValidatedRequest,
    protocol::{DeviceIdentity, Phase},
};
use fireprot_runtime::supervisor::*;
use serde_json::json;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

struct Monitor {
    calls: usize,
    fail_at: usize,
    external: bool,
    wrong_device: bool,
}
impl DeviceMonitor for Monitor {
    fn verify_device(&mut self, hip: u32, device: &DeviceIdentity) -> Result<(), String> {
        assert_eq!(hip, 7);
        assert_eq!(device.pci_address, "0000:93:00.0");
        if self.wrong_device {
            Err("PCI mismatch".into())
        } else {
            Ok(())
        }
    }
    fn sample(&mut self, _: u32) -> Result<Vec<u32>, String> {
        self.calls += 1;
        if self.calls == self.fail_at {
            return Err("SMI unavailable".into());
        }
        Ok(if self.external { vec![999] } else { vec![] })
    }
}
fn monitor() -> Monitor {
    Monitor {
        calls: 0,
        fail_at: usize::MAX,
        external: false,
        wrong_device: false,
    }
}
fn request(output: PathBuf) -> fireprot_runtime::preflight::PreparedRequest {
    let root = output.parent().unwrap().join("source");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("model.py"), "model source").unwrap();
    let git = |args: &[&str]| {
        let o = std::process::Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(args)
            .output()
            .unwrap();
        assert!(o.status.success(), "{:?}", o);
        String::from_utf8(o.stdout).unwrap()
    };
    if !root.join(".git").exists() {
        git(&["init", "-q"]);
        git(&["add", "model.py"]);
        git(&[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ]);
    }
    let commit = git(&["rev-parse", "HEAD"]);
    let path = root.join("model.py");
    let hash = fireprot_runtime::artifacts::sha256(&path).unwrap();
    let a = json!({"path":path,"sha256":hash});
    let r: Request = serde_json::from_value(json!({"schema_version":1,"run_id":"test","backend":"native","seed":101,"hip_device":7,
        "workload":{"model":"openfold3","model_name":"openbind-0","recycles":3,"diffusion_steps":200,"samples":1,"precision":"float32"},
        "source":{"root":root,"commit":commit.trim()},"checkpoints":{"model":a},"input":a,"output_dir":output})).unwrap();
    fireprot_runtime::preflight::prepare(
        ValidatedRequest::try_from(r).unwrap(),
        fireprot_runtime::preflight::SourceOrigin::GitCheckout,
    )
    .unwrap()
}
fn worker(mode: &str) -> WorkerCommand {
    WorkerCommand {
        identity_files: vec![fireprot_runtime::build::BuildFile {
            name: "worker-source".into(),
            path: concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py").into(),
        }],
        identity_trees: vec![],
        environment_inventory: None,
        program: "python3".into(),
        args: vec![
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/worker.py").into(),
            mode.into(),
        ],
        working_directory: std::env::current_dir().unwrap(),
        environment: vec![
            ("ROCR_VISIBLE_DEVICES".into(), "wrong".into()),
            ("HIP_VISIBLE_DEVICES".into(), "wrong".into()),
        ],
    }
}
fn limits() -> Limits {
    Limits {
        timeout: Duration::from_secs(3),
        poll_interval: Duration::from_millis(2),
        monitor_interval: Duration::from_millis(5),
        max_frame_bytes: 4096,
    }
}
#[test]
fn success_requires_outputs_exit_monitor_and_content() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("run");
    let report = run(
        &request(output.clone()),
        &worker("success"),
        &mut monitor(),
        &limits(),
        &AtomicBool::new(false),
        |paths| {
            assert_eq!(std::fs::read(&paths[0]).unwrap(), b"prediction");
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.phase, Phase::Succeeded);
    assert!(report.isolation_samples >= 2);
    assert!(report.device_verified);
    assert_eq!(
        report.announced_device.as_ref().unwrap().pci_address,
        "0000:93:00.0"
    );
    let launch: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("execution.json")).unwrap()).unwrap();
    assert_eq!(launch["sha256"], report.execution_sha256);
    assert_eq!(launch["environment"]["HIP_VISIBLE_DEVICES"], "7");
    assert!(launch["environment"].get("ROCR_VISIBLE_DEVICES").is_none());
    assert_eq!(launch["inputs"][0]["name"], "worker-source");
    assert!(
        std::fs::read_to_string(output.join("stdout.log"))
            .unwrap()
            .contains("model logs")
    );
    assert!(output.join("supervisor.json").exists());
    assert!(
        run(
            &request(output),
            &worker("success"),
            &mut monitor(),
            &limits(),
            &AtomicBool::new(false),
            |_| Ok(())
        )
        .is_err()
    );
}
#[test]
fn protocol_and_process_failures_never_succeed() {
    for mode in [
        "exit",
        "malformed",
        "oversize",
        "partial",
        "failure",
        "sequence",
        "hash",
        "escape",
        "nonzero",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("run");
        let report = run(
            &request(output.clone()),
            &worker(mode),
            &mut monitor(),
            &limits(),
            &AtomicBool::new(false),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(report.phase, Phase::Failed, "{mode}");
        assert!(report.failure.is_some(), "{mode}");
        assert!(output.join("supervisor.json").exists());
    }
}
#[test]
fn content_validation_is_mandatory() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(
        &request(dir.path().join("run")),
        &worker("success"),
        &mut monitor(),
        &limits(),
        &AtomicBool::new(false),
        |_| Err("nonfinite coordinates".into()),
    )
    .unwrap();
    assert_eq!(r.phase, Phase::Failed);
    assert_eq!(r.failure.as_deref(), Some("nonfinite coordinates"));
}
#[test]
fn device_and_monitor_failures_reject_run() {
    for failure in 0..4 {
        let dir = tempfile::tempdir().unwrap();
        let mut m = monitor();
        match failure {
            0 => m.wrong_device = true,
            1 => m.external = true,
            2 => m.fail_at = 1,
            _ => m.fail_at = 2,
        }
        let r = run(
            &request(dir.path().join("run")),
            &worker("wait"),
            &mut m,
            &limits(),
            &AtomicBool::new(false),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(r.phase, Phase::Failed);
        assert!(r.elapsed_seconds < 2.0);
        assert_eq!(r.device_verified, failure != 0);
    }
}
#[test]
fn timeout_kills_entire_process_group() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("run");
    let mut l = limits();
    l.timeout = Duration::from_millis(350);
    let r = run(
        &request(output.clone()),
        &worker("descendant"),
        &mut monitor(),
        &l,
        &AtomicBool::new(false),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(r.failure.as_deref(), Some("worker timeout"));
    std::thread::sleep(Duration::from_millis(1100));
    assert!(!output.join("escaped").exists());
}
#[test]
fn cancellation_stops_live_worker() {
    let dir = tempfile::tempdir().unwrap();
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(150));
            cancelled.store(true, Ordering::Relaxed);
        });
        let r = run(
            &request(dir.path().join("run")),
            &worker("wait"),
            &mut monitor(),
            &limits(),
            &cancelled,
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(r.failure.as_deref(), Some("run cancelled"));
    });
}

#[test]
fn source_changes_before_start_or_after_predictions_fail_acceptance() {
    for mode in ["mutate-before-ready", "mutate-after-ready"] {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("declared.py");
        std::fs::write(&input, "initial").unwrap();
        let mut worker = worker(mode);
        worker
            .identity_files
            .push(fireprot_runtime::build::BuildFile {
                name: "mutable-input".into(),
                path: input.clone(),
            });
        worker
            .environment
            .push(("MUTATE_INPUT".into(), input.into()));
        let r = run(
            &request(dir.path().join("run")),
            &worker,
            &mut monitor(),
            &limits(),
            &AtomicBool::new(false),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(r.phase, Phase::Failed);
        assert_eq!(
            r.failure.as_deref(),
            Some("worker launch inputs changed during execution")
        );
        assert_eq!(r.device_verified, mode == "mutate-after-ready");
    }
}

#[test]
fn accepted_archive_binds_metadata_and_outputs_and_rejects_later_changes() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("run");
    let report = run(
        &request(root.clone()),
        &worker("success"),
        &mut monitor(),
        &limits(),
        &AtomicBool::new(false),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(report.phase, Phase::Succeeded);
    let path = root.join("accepted.json");
    let pin = fireprot_core::request::ArtifactRef {
        sha256: fireprot_runtime::artifacts::sha256(&path).unwrap(),
        path,
    };
    let archive = fireprot_runtime::run_archive::verify(&pin).unwrap();
    assert_eq!(archive.receipt.metadata.len(), 4);
    assert_eq!(archive.outputs.len(), 1);
    for name in [
        "prediction.bin",
        "request.json",
        "provenance.json",
        "execution.json",
        "supervisor.json",
        "accepted.json",
    ] {
        let file = root.join(name);
        let original = std::fs::read(&file).unwrap();
        std::fs::write(&file, b"changed").unwrap();
        assert!(
            fireprot_runtime::run_archive::verify(&pin).is_err(),
            "{name}"
        );
        std::fs::write(&file, original).unwrap();
    }
}

#[test]
fn changed_metadata_or_receipt_collision_cannot_leave_success_report() {
    for name in [
        "request.json",
        "provenance.json",
        "execution.json",
        "accepted.json",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("run");
        let report = run(
            &request(root.clone()),
            &worker("success"),
            &mut monitor(),
            &limits(),
            &AtomicBool::new(false),
            |_| {
                std::fs::write(root.join(name), b"changed").unwrap();
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(report.phase, Phase::Failed, "{name}");
        assert!(
            report
                .failure
                .unwrap()
                .starts_with("acceptance archive failed:")
        );
        let stored: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("supervisor.json")).unwrap()).unwrap();
        assert_eq!(stored["phase"], "failed");
    }
}

#[test]
fn pinned_environment_changes_veto_start_and_final_acceptance() {
    use fireprot_runtime::{
        artifacts::sha256, build::BuildFile, environment_inventory::InventorySpec,
    };
    for mode in ["mutate-before-ready", "mutate-after-ready"] {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("package.py");
        std::fs::write(&input, "initial").unwrap();
        let spec = InventorySpec {
            schema_version: 1,
            files: vec![BuildFile {
                name: "package".into(),
                path: input.clone(),
            }],
            trees: vec![],
        };
        let manifest = dir.path().join("environment.json");
        std::fs::write(
            &manifest,
            serde_json::to_vec(&spec.capture().unwrap()).unwrap(),
        )
        .unwrap();
        let mut worker = worker(mode);
        worker.environment_inventory = Some(fireprot_core::request::ArtifactRef {
            sha256: sha256(&manifest).unwrap(),
            path: manifest,
        });
        worker
            .environment
            .push(("MUTATE_INPUT".into(), input.into()));
        let r = run(
            &request(dir.path().join("run")),
            &worker,
            &mut monitor(),
            &limits(),
            &AtomicBool::new(false),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(r.phase, Phase::Failed);
        assert_eq!(
            r.failure.as_deref(),
            Some("environment files or topology differ from pinned inventory")
        );
        assert!(!dir.path().join("run/accepted.json").exists());
    }
}

#[test]
fn cancellation_during_output_validation_cannot_accept_predictions() {
    let dir = tempfile::tempdir().unwrap();
    let cancelled = AtomicBool::new(false);
    let r = run(
        &request(dir.path().join("run")),
        &worker("valid"),
        &mut monitor(),
        &limits(),
        &cancelled,
        |_| {
            cancelled.store(true, Ordering::Relaxed);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(r.phase, Phase::Failed);
    assert_eq!(r.failure.as_deref(), Some("run cancelled"));
    assert!(!dir.path().join("run/accepted.json").exists());
}
