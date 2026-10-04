use fireprot_runtime::command::output;
use std::{
    process::Command,
    time::{Duration, Instant},
};
#[test]
fn captures_stdout_separately_and_checks_exit_status() {
    let mut c = Command::new("python3");
    c.args([
        "-c",
        "import sys; print('result'); print('diagnostic',file=sys.stderr)",
    ]);
    assert_eq!(
        output(&mut c, Duration::from_secs(2), 1024).unwrap(),
        b"result\n"
    );
    let mut c = Command::new("python3");
    c.args([
        "-c",
        "import sys; print('failure',file=sys.stderr); sys.exit(3)",
    ]);
    assert!(
        output(&mut c, Duration::from_secs(2), 1024)
            .unwrap_err()
            .to_string()
            .contains("failure")
    );
}
#[test]
fn slow_and_oversized_probes_fail() {
    let start = Instant::now();
    let mut c = Command::new("python3");
    c.args(["-c", "import time; time.sleep(30)"]);
    assert_eq!(
        output(&mut c, Duration::from_millis(100), 1024)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::TimedOut
    );
    assert!(start.elapsed() < Duration::from_secs(2));
    let mut c = Command::new("python3");
    c.args(["-c", "print('x'*5000)"]);
    assert!(
        output(&mut c, Duration::from_secs(2), 1024)
            .unwrap_err()
            .to_string()
            .contains("exceeds limit")
    );
}
