use std::process::Command;

fn invoke(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn help_and_version_are_successful_without_model_environments() {
    for args in [vec!["--help"], vec!["-h"], vec!["help"]] {
        let result = invoke(&args);
        assert!(result.status.success());
        assert!(result.stderr.is_empty());
        assert!(String::from_utf8_lossy(&result.stdout).contains("collect-timings"));
    }
    for command in [
        "inspect",
        "infer",
        "summarize-timings",
        "collect-timings",
        "export-source",
        "compare-npz",
        "build-torch",
        "build-hip",
        "publish-kernel",
        "verify-kernel",
    ] {
        let a = invoke(&[command, "--help"]);
        let b = invoke(&["help", command]);
        assert!(a.status.success(), "{command}");
        assert_eq!(a.stdout, b.stdout);
        assert!(
            String::from_utf8_lossy(&a.stdout).contains(&format!("Usage: fireprot {command} "))
        );
    }
    let version = invoke(&["--version"]);
    assert!(version.status.success());
    assert_eq!(
        String::from_utf8(version.stdout).unwrap(),
        format!("fireprot {}\n", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn invalid_arguments_and_invalid_json_are_nonzero_with_actionable_stderr() {
    for args in [
        vec![],
        vec!["unknown"],
        vec!["help", "unknown"],
        vec!["infer", "request-only"],
        vec!["inspect", "a", "extra"],
        vec!["--version", "extra"],
    ] {
        let result = invoke(&args);
        assert_eq!(result.status.code(), Some(1));
        assert!(result.stdout.is_empty());
        assert!(!result.stderr.is_empty());
    }
    let file = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let result = invoke(&["inspect", file]);
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("invalid JSON"));
    assert!(error.contains(file));
}

#[test]
fn inspect_every_checked_in_request_and_compare_real_npz_without_weights() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for file in std::fs::read_dir(root.join("configs/requests")).unwrap() {
        let path = file.unwrap().path();
        let result = invoke(&["inspect", path.to_str().unwrap()]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let original: fireprot_core::Request =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let printed: fireprot_core::Request = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(original, printed);
    }
    let npz = root.join("crates/fireprot-validation/tests/fixtures/openfold_sample.npz");
    let path = npz.to_str().unwrap();
    let result = invoke(&["compare-npz", path, path]);
    assert!(result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert!(
        report
            .as_object()
            .unwrap()
            .values()
            .all(|v| v["passed"] == true)
    );
}

#[cfg(unix)]
#[test]
fn non_utf8_argument_is_reported_without_panicking() {
    use std::os::unix::ffi::OsStringExt;
    let result = Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .arg(std::ffi::OsString::from_vec(vec![0xff]))
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(result.stderr).unwrap(),
        "fireprot: arguments must be UTF-8\n"
    );
}
