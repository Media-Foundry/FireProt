#![cfg(unix)]
use fireprot_runtime::invocation::{StopReason, measure};
use std::{
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[test]
#[cfg(target_os = "linux")]
fn cancellation_allows_supervisor_to_reap_a_separate_worker_group() {
    let dir = tempfile::tempdir().unwrap();
    let ready = dir.path().join("ready");
    let worker_pid = dir.path().join("worker-pid");
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let start = Instant::now();
            while !ready.exists() {
                assert!(start.elapsed() < Duration::from_secs(5));
                std::thread::sleep(Duration::from_millis(1));
            }
            cancelled.store(true, Ordering::Relaxed);
        });
        let mut command = Command::new("sh");
        command
            .args([
                "-c",
                r#"trap '/bin/kill -TERM -- -"$worker"; wait "$worker"; exit 0' TERM
setsid sh -c 'echo $$ > "$1"; touch "$2"; exec sleep 30' fixture "$1" "$2" &
worker=$!
wait "$worker"
"#,
                "fixture",
            ])
            .arg(&worker_pid)
            .arg(&ready);
        let result = measure(&mut command, Duration::from_secs(5), &cancelled).unwrap();
        assert_eq!(result.stop_reason, Some(StopReason::Cancelled));
        assert_eq!(result.status.code(), Some(0));
        let pid: i32 = std::fs::read_to_string(&worker_pid)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert_eq!(
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None),
            Err(nix::errno::Errno::ESRCH)
        );
    });
}

#[test]
fn measures_until_exit_and_preserves_failure_status() {
    for code in [0, 23] {
        let mut command = Command::new("sh");
        command.args(["-c", &format!("sleep 0.05; exit {code}")]);
        let outer = Instant::now();
        let result = measure(
            &mut command,
            Duration::from_secs(5),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(result.status.code(), Some(code));
        assert_eq!(result.succeeded(), code == 0);
        assert_eq!(result.stop_reason, None);
        assert!(result.elapsed >= Duration::from_millis(50));
        assert!(result.elapsed <= outer.elapsed());
    }
}

#[test]
fn prelaunch_errors_do_not_execute_command() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("spawned");
    let mut command = Command::new("touch");
    command.arg(&marker);
    assert!(measure(&mut command, Duration::ZERO, &AtomicBool::new(false)).is_err());
    assert!(measure(&mut command, Duration::from_secs(1), &AtomicBool::new(true)).is_err());
    assert!(!marker.exists());
    assert!(
        measure(
            &mut Command::new(dir.path().join("missing")),
            Duration::from_secs(1),
            &AtomicBool::new(false)
        )
        .is_err()
    );
}

#[test]
fn cancellation_waits_for_cleanup_and_rejects_zero_exit() {
    let dir = tempfile::tempdir().unwrap();
    let ready = dir.path().join("ready");
    let cleaned = dir.path().join("cleaned");
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let start = Instant::now();
            while !ready.exists() {
                assert!(start.elapsed() < Duration::from_secs(5));
                std::thread::sleep(Duration::from_millis(1));
            }
            cancelled.store(true, Ordering::Relaxed);
        });
        let mut command = Command::new("sh");
        command.args(["-c", "trap 'sleep 0.1; touch \"$2\"; exit 0' TERM; touch \"$1\"; while :; do sleep 0.01; done", "fixture"])
            .arg(&ready).arg(&cleaned);
        let result = measure(&mut command, Duration::from_secs(5), &cancelled).unwrap();
        assert!(cleaned.exists());
        assert!(result.elapsed >= Duration::from_millis(100));
        assert_eq!(result.status.code(), Some(0));
        assert_eq!(result.stop_reason, Some(StopReason::Cancelled));
        assert!(!result.succeeded());
    });
}

#[test]
fn deadline_waits_for_cooperative_cleanup() {
    let dir = tempfile::tempdir().unwrap();
    let cleaned = dir.path().join("cleaned");
    let mut command = Command::new("sh");
    command
        .args([
            "-c",
            "trap 'sleep 0.1; touch \"$1\"; exit 0' TERM; while :; do sleep 0.01; done",
            "fixture",
        ])
        .arg(&cleaned);
    let result = measure(
        &mut command,
        Duration::from_millis(200),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(cleaned.exists());
    assert!(result.elapsed >= Duration::from_millis(300));
    assert_eq!(result.stop_reason, Some(StopReason::Deadline));
    assert!(!result.succeeded());
}
