#![cfg(unix)]
use fireprot_runtime::artifacts::sha256;
use fireprot_validation::numpy::read_npz;
use serde_json::{Value, json};
use std::{fs, os::unix::fs::PermissionsExt, path::Path, process::Command};

fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn write(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}
fn fixture(root: &Path) -> (Value, Value) {
    for name in [
        "source",
        "python/fireprot_workers",
        "python/fireprot",
        "fireprot",
        "kernels/amd",
        "fixture",
    ] {
        fs::create_dir_all(root.join(name)).unwrap();
    }
    fs::write(root.join("source/model.py"), "# synthetic source\n").unwrap();
    let git = |args: &[&str]| {
        let r = Command::new("git")
            .arg("-C")
            .arg(root.join("source"))
            .args(args)
            .output()
            .unwrap();
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        String::from_utf8(r.stdout).unwrap().trim().to_owned()
    };
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
    let commit = git(&["rev-parse", "HEAD"]);
    fs::write(
        root.join("python/fireprot_workers/__main__.py"),
        include_str!("fixtures/model_worker.py"),
    )
    .unwrap();
    fs::write(root.join("fireprot/__init__.py"), "# fixture\n").unwrap();
    executable(
        &root.join("hip-probe"),
        "#!/usr/bin/python3\nimport os,json\nassert os.environ['HIP_VISIBLE_DEVICES']=='7'\nprint(json.dumps(dict(pci_address='0000:93:00.0',architecture='gfx90a')))\n",
    );
    executable(
        &root.join("amd-smi"),
        "#!/usr/bin/python3\nimport sys,json,os\nassert 'HIP_VISIBLE_DEVICES' not in os.environ\nif sys.argv[1]=='list': print(json.dumps([dict(gpu=2,bdf='0000:93:00.0')]))\nelse: print(json.dumps([dict(gpu=2,process_list=[dict(process_info='No running processes detected')])]))\n",
    );
    let archive = root.join("fixture/sample-0.npz");
    fs::write(
        root.join("fixture/sample.cif"),
        include_bytes!("../../fireprot-validation/tests/fixtures/prediction-zero.cif"),
    )
    .unwrap();
    fs::write(
        &archive,
        include_bytes!("../../fireprot-validation/tests/fixtures/openfold_sample.npz"),
    )
    .unwrap();
    let descriptors: serde_json::Map<String, Value> = read_npz(&archive, 100000)
        .unwrap()
        .iter()
        .map(|(k, a)| {
            (
                k.clone(),
                json!({"shape":a.shape(),"storage_dtype":a.dtype(),"original_dtype":a.dtype()}),
            )
        })
        .collect();
    write(
        &root.join("fixture/predictions.json"),
        &json!({"schema_version":2,"run_id":"filled-by-worker","seed":101,"workload":{},
        "predictions":[{"sample":0,"archive":"sample-0.npz","structure":"sample.cif","arrays":descriptors,"confidence":null}],
        "model_seconds":0.001,"request_seconds":0.002,"environment":{"torch":"fixture","hip":"fixture","python":"fixture"}}),
    );
    let artifact = json!({"path":root.join("source/model.py"),"sha256":sha256(&root.join("source/model.py")).unwrap()});
    let request = json!({"schema_version":1,"run_id":"cli-fixture","backend":"native","seed":101,"hip_device":7,
        "workload":{"model":"openfold3","model_name":"openbind-0","recycles":3,"diffusion_steps":200,"samples":1,"precision":"float32"},
        "source":{"root":root.join("source"),"commit":commit},"checkpoints":{"model":artifact},"input":artifact,"output_dir":root.join("run")});
    let profile = json!({"python":"/usr/bin/python3","workers":root.join("python"),"hip_probe":root.join("hip-probe"),"amd_smi":root.join("amd-smi"),
        "environment":{"FIREPROT_TEST_FIXTURE":root.join("fixture"),"HIP_VISIBLE_DEVICES":"wrong","ROCR_VISIBLE_DEVICES":"wrong","CUDA_VISIBLE_DEVICES":"wrong","GPU_DEVICE_ORDINAL":"wrong"}});
    (request, profile)
}
fn infer(root: &Path, request: &Value, profile: &Value) -> std::process::Output {
    write(&root.join("request.json"), request);
    write(&root.join("profile.json"), profile);
    Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .current_dir(root)
        .args(["infer", "request.json", "profile.json"])
        .output()
        .unwrap()
}

#[test]
fn full_cli_inference_to_archive_collection_without_model_dependencies() {
    let dir = tempfile::Builder::new()
        .prefix("fireprot cli ")
        .tempdir()
        .unwrap();
    let root = dir.path();
    let (request, profile) = fixture(root);
    let result = infer(root, &request, &profile);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["phase"], "succeeded");
    assert_eq!(report["physical_device"]["smi_device"], 2);
    let receipt = root.join("run/accepted.json");
    write(
        &root.join("collection.json"),
        &json!({"schema_version":1,"scope":"model","cache_state":"synthetic CLI fixture",
        "runs":[{"path":receipt,"sha256":sha256(&receipt).unwrap()}]}),
    );
    let collected = Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .current_dir(root)
        .args(["collect-timings", "collection.json"])
        .output()
        .unwrap();
    assert!(
        collected.status.success(),
        "{}",
        String::from_utf8_lossy(&collected.stderr)
    );
    let value: Value = serde_json::from_slice(&collected.stdout).unwrap();
    assert_eq!(value["summary"]["median"], 0.001);
    let again = infer(root, &request, &profile);
    assert!(!again.status.success());
    assert!(again.stdout.is_empty());
}

#[test]
fn worker_content_failure_reaches_cli_status_and_never_creates_accepted_receipt() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (request, mut profile) = fixture(root);
    profile["environment"]["FIREPROT_TEST_INVALID"] = "1".into();
    let result = infer(root, &request, &profile);
    assert_eq!(result.status.code(), Some(1));
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["phase"], "failed");
    assert!(
        report["failure"]
            .as_str()
            .unwrap()
            .contains("identity/workload mismatch")
    );
    assert!(!root.join("run/accepted.json").exists());
}

#[test]
fn upstream_relative_files_are_scoped_to_the_new_run_directory() {
    let dir = tempfile::tempdir().unwrap();
    let (request, profile) = fixture(dir.path());
    let result = infer(dir.path(), &request, &profile);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let execution: Value =
        serde_json::from_slice(&fs::read(dir.path().join("run/execution.json")).unwrap()).unwrap();
    assert_eq!(
        execution["working_directory"],
        dir.path()
            .join("run")
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
    );
}

fn batch_fixture(
    root: &Path,
    bad_second_profile: bool,
    fail_first_worker: bool,
) -> std::process::Output {
    let mut jobs = Vec::new();
    for i in 0..2 {
        let job_root = root.join(format!("job {i}"));
        let (mut request, mut profile) = fixture(&job_root);
        request["run_id"] = json!(format!("batch-job-{i}"));
        if i == 0 && fail_first_worker {
            profile["environment"]["FIREPROT_TEST_INVALID"] = "1".into();
        }
        if i == 1 && bad_second_profile {
            profile["unknown_option"] = true.into();
        }
        let r = job_root.join("request.json");
        let p = job_root.join("profile.json");
        write(&r, &request);
        write(&p, &profile);
        jobs.push(json!({"request":{"path":r,"sha256":sha256(&r).unwrap()},"profile":{"path":p,"sha256":sha256(&p).unwrap()}}));
    }
    let manifest = root.join("batch.json");
    write(
        &manifest,
        &json!({"schema_version":1,"timeout_seconds":600,"jobs":jobs}),
    );
    Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .arg("infer-batch")
        .arg(manifest)
        .output()
        .unwrap()
}

#[test]
fn batch_cli_runs_each_pinned_job_and_seals_individual_archives() {
    let d = tempfile::tempdir().unwrap();
    let output = batch_fixture(d.path(), false, false);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["phase"], "succeeded");
    assert_eq!(report["jobs"].as_array().unwrap().len(), 2);
    for i in 0..2 {
        assert!(
            d.path()
                .join(format!("job {i}/run/accepted.json"))
                .is_file()
        );
    }
}

#[test]
fn batch_cli_validates_all_profiles_before_start_and_stops_after_worker_failure() {
    let d = tempfile::tempdir().unwrap();
    let output = batch_fixture(d.path(), true, false);
    assert!(!output.status.success());
    assert!(!d.path().join("job 0/run").exists());
    assert!(!d.path().join("job 1/run").exists());
    let d = tempfile::tempdir().unwrap();
    let output = batch_fixture(d.path(), false, true);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["phase"], "failed");
    assert_eq!(report["jobs"][0]["phase"], "failed");
    assert_eq!(report["jobs"][1]["phase"], "skipped");
    assert!(!d.path().join("job 0/run/accepted.json").exists());
    assert!(!d.path().join("job 1/run").exists());
}

#[test]
fn maintained_shell_entrypoint_executes_the_validated_model_pipeline() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    let (request, profile) = fixture(root);
    write(&root.join("request.json"), &request);
    write(&root.join("profile.json"), &profile);
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/infer.sh");
    let result = Command::new("bash")
        .arg(script)
        .args(["request.json", "profile.json"])
        .current_dir(root)
        .env("FIREPROT_BIN", env!("CARGO_BIN_EXE_fireprot"))
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&result.stdout).unwrap()["phase"],
        "succeeded"
    );
    assert!(root.join("run/accepted.json").is_file());
}
