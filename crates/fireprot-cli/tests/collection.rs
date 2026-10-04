use fireprot_core::request::ArtifactRef;
use fireprot_runtime::{artifacts::sha256, run_archive::AcceptanceReceipt};
use fireprot_validation::numpy::read_npz;
use serde_json::{Value, json};
use std::{fs, path::Path, process::Command};

fn write(root: &Path, name: &str, value: &Value) {
    fs::write(root.join(name), serde_json::to_vec(value).unwrap()).unwrap();
}
fn repin(root: &Path) -> ArtifactRef {
    let refs = |names: &[&str]| {
        names
            .iter()
            .map(|name| ArtifactRef {
                path: (*name).into(),
                sha256: sha256(&root.join(name)).unwrap(),
            })
            .collect()
    };
    let receipt = AcceptanceReceipt {
        schema_version: 1,
        metadata: refs(&[
            "request.json",
            "provenance.json",
            "execution.json",
            "supervisor.json",
        ]),
        outputs: refs(&["predictions.json", "sample-0.npz"]),
    };
    fs::write(
        root.join("accepted.json"),
        serde_json::to_vec(&receipt).unwrap(),
    )
    .unwrap();
    ArtifactRef {
        path: root.join("accepted.json"),
        sha256: sha256(&root.join("accepted.json")).unwrap(),
    }
}
fn fixture(root: &Path, run: &str, seconds: f64) -> ArtifactRef {
    fs::create_dir(root).unwrap();
    let mut request: Value = serde_json::from_str(include_str!(
        "../../../configs/requests/openfold3-n1000-native.json"
    ))
    .unwrap();
    request["run_id"] = run.into();
    request["hip_device"] = 7.into();
    request["output_dir"] = "/original/now-unavailable/location".into();
    write(root, "request.json", &request);
    fs::write(
        root.join("sample-0.npz"),
        include_bytes!("../../fireprot-validation/tests/fixtures/openfold_sample.npz"),
    )
    .unwrap();
    let arrays = read_npz(&root.join("sample-0.npz"), 100000).unwrap();
    let descriptors: serde_json::Map<String, Value> = arrays
        .iter()
        .map(|(k, a)| {
            (
                k.clone(),
                json!({"shape":a.shape(),"storage_dtype":a.dtype(),"original_dtype":a.dtype()}),
            )
        })
        .collect();
    write(
        root,
        "predictions.json",
        &json!({"schema_version":1,"run_id":run,"seed":request["seed"],"workload":request["workload"],
        "predictions":[{"sample":0,"archive":"sample-0.npz","arrays":descriptors,"confidence":null}],
        "model_seconds":seconds,"request_seconds":seconds+1.0,"environment":{"torch":"fixture","hip":"fixture","python":"fixture"}}),
    );
    write(
        root,
        "supervisor.json",
        &json!({"run_id":run,"phase":"succeeded","worker_pid":123,"elapsed_seconds":seconds+2.0,
        "isolation_samples":5,"failure":null,"device_verified":true,"execution_sha256":"fixture",
        "announced_device":{"worker_pid":123,"logical_device":0,"pci_address":"0000:93:00.0","architecture":"gfx90a"},
        "physical_device":{"hip_device":7,"smi_device":2,"pci_address":"0000:93:00.0","architecture":"gfx90a"}}),
    );
    write(
        root,
        "execution.json",
        &json!({"schema_version":1,"environment_policy":"explicit-v1","sha256":"fixture",
        "environment":{"HIP_VISIBLE_DEVICES":"7","FIREPROT_OUTPUT_DIR":root,"FIREPROT_REQUEST":root.join("request.json"),"FIREPROT_WORKER_SOCKET":run}}),
    );
    write(
        root,
        "provenance.json",
        &json!({"fixture":"synthetic archive, not hardware evidence"}),
    );
    repin(root)
}
fn collect(root: &Path, pins: Vec<ArtifactRef>, scope: &str) -> std::process::Output {
    let config = root.join("collection.json");
    fs::write(&config, serde_json::to_vec(&json!({"schema_version":1,"scope":scope,"cache_state":"synthetic fixtures","runs":pins})).unwrap()).unwrap();
    Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .arg("collect-timings")
        .arg(config)
        .output()
        .unwrap()
}
#[test]
fn copied_archives_collect_real_fields_and_reject_duplicate_runs_and_e2e() {
    let dir = tempfile::tempdir().unwrap();
    let a = fixture(&dir.path().join("a"), "one", 2.0);
    let b = fixture(&dir.path().join("b"), "two", 4.0);
    for (scope, expected) in [("model", 3.0), ("worker_request", 4.0), ("supervisor", 5.0)] {
        let result = collect(dir.path(), vec![a.clone(), b.clone()], scope);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let value: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(value["summary"]["median"], expected);
        assert_eq!(value["receipts"][0]["sha256"], a.sha256);
        assert_eq!(
            value["series"]["observations"][0]["request"]["output_dir"],
            "/original/now-unavailable/location"
        );
    }
    assert!(
        !collect(dir.path(), vec![a.clone(), a.clone()], "model")
            .status
            .success()
    );
    assert!(!collect(dir.path(), vec![a], "end_to_end").status.success());
    fs::write(dir.path().join("b/sample-0.npz"), b"changed").unwrap();
    assert!(!collect(dir.path(), vec![b], "model").status.success());
}
#[test]
fn semantic_failures_remain_failures_even_with_recomputed_receipt_hashes() {
    for (name, key, value) in [
        ("supervisor.json", "phase", json!("failed")),
        ("supervisor.json", "failure", json!("monitor failed")),
        ("supervisor.json", "isolation_samples", json!(0)),
        ("supervisor.json", "device_verified", json!(false)),
        ("supervisor.json", "physical_device", json!(null)),
        ("supervisor.json", "execution_sha256", json!("different")),
        ("predictions.json", "seed", json!(99)),
        ("predictions.json", "predictions", json!([])),
        ("predictions.json", "model_seconds", json!(-1)),
        (
            "execution.json",
            "environment",
            json!({"HIP_VISIBLE_DEVICES":"0"}),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("run");
        fixture(&root, "one", 2.0);
        let mut data: Value = serde_json::from_slice(&fs::read(root.join(name)).unwrap()).unwrap();
        data[key] = value;
        write(&root, name, &data);
        let pin = repin(&root);
        let output = collect(dir.path(), vec![pin], "model");
        assert!(!output.status.success(), "{name}/{key}");
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn run_local_working_directories_compare_without_hiding_fixed_directory_changes() {
    let dir = tempfile::tempdir().unwrap();
    let roots = [dir.path().join("a"), dir.path().join("b")];
    for (index, root) in roots.iter().enumerate() {
        fixture(root, &format!("run-{index}"), 2.0 + index as f64);
        let original = format!("/original/unavailable/run-{index}");
        for (file, key) in [
            ("request.json", "output_dir"),
            ("execution.json", "working_directory"),
        ] {
            let mut data: Value =
                serde_json::from_slice(&fs::read(root.join(file)).unwrap()).unwrap();
            data[key] = original.clone().into();
            write(root, file, &data);
        }
    }
    let pins = || roots.iter().map(|root| repin(root)).collect();
    let output = collect(dir.path(), pins(), "model");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["summary"]["median"], 2.5);
    assert_eq!(
        result["series"]["observations"][0]["request"]["output_dir"],
        "/original/unavailable/run-0"
    );

    // A fixed directory is still a condition, including a marker-like string.
    for fixed in ["/different/fixed/directory", "<run-output-directory>"] {
        let mut launch: Value =
            serde_json::from_slice(&fs::read(roots[1].join("execution.json")).unwrap()).unwrap();
        launch["working_directory"] = fixed.into();
        write(&roots[1], "execution.json", &launch);
        assert!(!collect(dir.path(), pins(), "model").status.success());
    }
    // Historical runs sharing a fixed cwd still form a valid series.
    for root in &roots {
        let mut launch: Value =
            serde_json::from_slice(&fs::read(root.join("execution.json")).unwrap()).unwrap();
        launch["working_directory"] = "/shared/upstream".into();
        write(root, "execution.json", &launch);
    }
    assert!(collect(dir.path(), pins(), "model").status.success());
}
