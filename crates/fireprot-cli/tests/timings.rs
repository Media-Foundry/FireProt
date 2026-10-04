use std::process::Command;

#[test]
fn timing_command_preserves_observations_and_emits_statistics() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/timings/illustrative.json"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .args(["summarize-timings", path])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(result["series"], original);
    assert_eq!(result["summary"]["count"], 4);
    assert_eq!(result["summary"]["median"], 3.0);
    assert_eq!(result["summary"]["mean"], 4.0);
    let invalid = Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .args([
            "summarize-timings",
            concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"),
        ])
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    assert!(invalid.stdout.is_empty());
}
