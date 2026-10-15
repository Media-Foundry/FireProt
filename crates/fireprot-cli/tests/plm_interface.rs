use std::process::Command;
#[test]
fn dedicated_plm_commands_reject_wrong_families_before_environment_access() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (command, request, message) in [
        (
            "esm",
            "atlasfold-n1000-native.json",
            "requires a plm workload",
        ),
        ("esm", "plm-atlaslm-3b-n1000-native.json", "model family"),
        ("atlaslm", "plm-esm2-3b-n1000-native.json", "model family"),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_fireprot"))
            .arg(command)
            .arg(root.join("configs/requests").join(request))
            .arg("/nonexistent-profile")
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains(message));
    }
    for command in ["esm", "atlaslm"] {
        let result = Command::new(env!("CARGO_BIN_EXE_fireprot"))
            .args([command, "--help"])
            .output()
            .unwrap();
        assert!(result.status.success());
    }
}
