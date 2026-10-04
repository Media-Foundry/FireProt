#![cfg(unix)]
use fireprot_core::protocol::Phase;
use fireprot_runtime::{
    artifacts::sha256,
    batch::{self, BatchPhase, BatchPlan, JobPhase},
    supervisor::RunReport,
};
use serde_json::json;
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

fn spec(root: &Path) -> serde_json::Value {
    let mut jobs = Vec::new();
    for i in 0..3 {
        let mut request: serde_json::Value = serde_json::from_str(include_str!(
            "../../../configs/requests/protenix-n1000-native.json"
        ))
        .unwrap();
        request["run_id"] = format!("job-{i}").into();
        request["output_dir"] = json!(root.join(format!("output-{i}")));
        let r = root.join(format!("request-{i}.json"));
        let p = root.join(format!("profile-{i}.json"));
        fs::write(&r, serde_json::to_vec(&request).unwrap()).unwrap();
        fs::write(&p, format!("{{\"fixture\":{i}}}")).unwrap();
        jobs.push(json!({"request":{"path":r,"sha256":sha256(&r).unwrap()},"profile":{"path":p,"sha256":sha256(&p).unwrap()}}));
    }
    json!({"schema_version":1,"timeout_seconds":600,"jobs":jobs})
}
fn plan(v: serde_json::Value) -> std::io::Result<BatchPlan> {
    BatchPlan::load(serde_json::from_value(v).unwrap())
}
fn success(job: &batch::LoadedJob) -> RunReport {
    RunReport {
        run_id: job.request().as_request().run_id.clone(),
        phase: Phase::Succeeded,
        worker_pid: 1,
        elapsed_seconds: 1.,
        isolation_samples: 1,
        failure: None,
        announced_device: None,
        device_verified: true,
        physical_device: None,
        execution_sha256: "a".repeat(64),
    }
}
#[test]
fn control_bytes_are_pinned_once_and_not_reread_during_execution() {
    let d = tempfile::tempdir().unwrap();
    let value = spec(d.path());
    let loaded = plan(value).unwrap();
    fs::write(d.path().join("request-0.json"), "invalid replacement").unwrap();
    fs::write(d.path().join("profile-0.json"), "invalid replacement").unwrap();
    let report = batch::execute(&loaded, &AtomicBool::new(false), |i, j| {
        assert_eq!(j.request().as_request().run_id, format!("job-{i}"));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(j.profile()).unwrap(),
            json!({"fixture":i})
        );
        Ok(success(j))
    });
    assert_eq!(report.phase, BatchPhase::Succeeded);
    assert_eq!(report.jobs.len(), 3);
    assert!(report.jobs.iter().all(|j| j.phase == JobPhase::Succeeded));
}
#[test]
fn missing_changed_duplicate_and_existing_destinations_fail_before_execution() {
    for mode in [
        "digest",
        "missing",
        "duplicate_id",
        "duplicate_output",
        "existing",
        "empty",
        "schema",
    ] {
        let d = tempfile::tempdir().unwrap();
        let mut v = spec(d.path());
        match mode {
            "digest" => v["jobs"][2]["profile"]["sha256"] = json!("0".repeat(64)),
            "missing" => {
                fs::remove_file(d.path().join("profile-2.json")).unwrap();
            }
            "existing" => {
                fs::create_dir(d.path().join("output-2")).unwrap();
            }
            "empty" => v["jobs"] = json!([]),
            "schema" => v["schema_version"] = json!(2),
            _ => {
                let path = d.path().join("request-2.json");
                let mut r: serde_json::Value =
                    serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
                if mode == "duplicate_id" {
                    r["run_id"] = json!("job-0");
                } else {
                    r["output_dir"] = json!(d.path().join("./output-0"));
                }
                fs::write(&path, serde_json::to_vec(&r).unwrap()).unwrap();
                v["jobs"][2]["request"]["sha256"] = json!(sha256(&path).unwrap());
            }
        }
        assert!(plan(v).is_err(), "{mode}");
        assert!(!d.path().join("output-0").exists());
    }
}
#[test]
fn failed_jobs_stop_the_batch_and_keep_later_jobs_explicitly_skipped() {
    let d = tempfile::tempdir().unwrap();
    let loaded = plan(spec(d.path())).unwrap();
    for mode in ["error", "failed_report", "wrong_id"] {
        let mut calls = 0;
        let report = batch::execute(&loaded, &AtomicBool::new(false), |i, j| {
            calls += 1;
            if i == 0 {
                return Ok(success(j));
            }
            if mode == "error" {
                return Err("asset failure".into());
            }
            let mut r = success(j);
            if mode == "failed_report" {
                r.phase = Phase::Failed;
            } else {
                r.run_id = "wrong".into();
            }
            Ok(r)
        });
        assert_eq!(calls, 2);
        assert_eq!(report.phase, BatchPhase::Failed);
        assert_eq!(
            report.jobs.iter().map(|j| j.phase).collect::<Vec<_>>(),
            vec![JobPhase::Succeeded, JobPhase::Failed, JobPhase::Skipped]
        );
    }
}
#[test]
fn cancellation_before_or_between_jobs_never_starts_the_next_model() {
    let d = tempfile::tempdir().unwrap();
    let loaded = plan(spec(d.path())).unwrap();
    for already_cancelled in [false, true] {
        let cancelled = AtomicBool::new(already_cancelled);
        let mut calls = 0;
        let report = batch::execute(&loaded, &cancelled, |_, j| {
            calls += 1;
            cancelled.store(true, Ordering::Relaxed);
            Ok(success(j))
        });
        assert_eq!(calls, usize::from(!already_cancelled));
        assert_eq!(report.phase, BatchPhase::Cancelled);
        assert_eq!(report.jobs[1].phase, JobPhase::Skipped);
        assert_eq!(report.jobs[2].phase, JobPhase::Skipped);
    }
}

#[test]
fn deadline_cancels_the_active_job_and_leaves_remaining_jobs_unstarted() {
    let d = tempfile::tempdir().unwrap();
    let mut v = spec(d.path());
    v["timeout_seconds"] = json!(1);
    let loaded = plan(v).unwrap();
    let cancelled = AtomicBool::new(false);
    let mut calls = 0;
    let report = batch::execute(&loaded, &cancelled, |_, _| {
        calls += 1;
        let started = std::time::Instant::now();
        while !cancelled.load(Ordering::Relaxed) {
            assert!(started.elapsed() < std::time::Duration::from_secs(5));
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Err("active supervisor cancelled".into())
    });
    assert_eq!(calls, 1);
    assert_eq!(report.phase, BatchPhase::TimedOut);
    assert_eq!(report.timeout_seconds, 1);
    assert_eq!(report.jobs[1].phase, JobPhase::Skipped);
    assert_eq!(report.jobs[2].phase, JobPhase::Skipped);
}

#[test]
fn zero_or_excessive_batch_deadline_is_rejected() {
    for seconds in [0, 604801] {
        let d = tempfile::tempdir().unwrap();
        let mut v = spec(d.path());
        v["timeout_seconds"] = json!(seconds);
        assert!(plan(v).is_err());
    }
}
