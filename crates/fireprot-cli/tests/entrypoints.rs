#![cfg(unix)]
use std::{fs, os::unix::fs::PermissionsExt, path::PathBuf, process::Command};
fn script(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts")
        .join(name)
}
#[test]
fn shell_entry_points_use_rust_help_and_reject_legacy_argument_shape() {
    for (script_name, command) in [("infer.sh", "infer"), ("smoke.sh", "infer-batch")] {
        let help = Command::new("bash")
            .arg(script(script_name))
            .arg("--help")
            .env("FIREPROT_BIN", env!("CARGO_BIN_EXE_fireprot"))
            .output()
            .unwrap();
        assert!(help.status.success());
        assert!(String::from_utf8_lossy(&help.stdout).contains(&format!("fireprot {command}")));
        let bad = Command::new("bash")
            .arg(script(script_name))
            .args(["--input", "old.json", "--out_dir", "old-output"])
            .env("FIREPROT_BIN", env!("CARGO_BIN_EXE_fireprot"))
            .output()
            .unwrap();
        assert!(!bad.status.success());
        assert!(String::from_utf8_lossy(&bad.stderr).contains("invalid arguments"));
    }
}
#[test]
fn wrapper_preserves_literal_arguments_working_directory_and_exit_status() {
    let d = tempfile::Builder::new()
        .prefix("fireprot shell ")
        .tempdir()
        .unwrap();
    let binary = d.path().join("fake rust binary");
    fs::write(&binary,"#!/usr/bin/env bash\nprintf '%s\\0' \"$@\" > \"$ARG_CAPTURE\"\nprintf '%s' \"$PWD\" > \"$CWD_CAPTURE\"\nexit 23\n").unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
    let capture = d.path().join("argv");
    let cwd = d.path().join("cwd");
    let output = Command::new("bash")
        .arg(script("infer.sh"))
        .args(["request with spaces.json", "profile;literal.json"])
        .current_dir(d.path())
        .env("FIREPROT_BIN", binary)
        .env("ARG_CAPTURE", &capture)
        .env("CWD_CAPTURE", &cwd)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(23));
    assert_eq!(
        fs::read(capture).unwrap(),
        b"infer\0request with spaces.json\0profile;literal.json\0"
    );
    assert_eq!(fs::read_to_string(cwd).unwrap(), d.path().to_str().unwrap());
}
#[test]
fn missing_rust_binary_has_an_actionable_error() {
    let d = tempfile::tempdir().unwrap();
    let output = Command::new("bash")
        .arg(script("infer.sh"))
        .env("FIREPROT_BIN", d.path().join("missing"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(127));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Build fireprot-cli"));
}
