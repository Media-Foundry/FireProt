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

fn measure_fixture(
    root: &Path,
    request: &Value,
    profile: &Value,
    name: &str,
    timeout: u64,
) -> std::process::Output {
    let r = root.join("measurement-request.json");
    let p = root.join("measurement-profile.json");
    write(&r, request);
    write(&p, profile);
    write(
        &root.join("measurement-spec.json"),
        &json!({
            "schema_version":1,"timeout_seconds":timeout,
            "request":{"path":r,"sha256":sha256(&r).unwrap()},
            "profile":{"path":p,"sha256":sha256(&p).unwrap()},
        }),
    );
    Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .current_dir(root)
        .args(["measure-inference", "measurement-spec.json", name])
        .output()
        .unwrap()
}

fn collect_invocations(root: &Path, runs: &Value, invocations: &Value) -> std::process::Output {
    write(
        &root.join("e2e-collection.json"),
        &json!({
            "schema_version":1,"scope":"end_to_end","cache_state":"synthetic CLI fixture",
            "runs":runs,"invocations":invocations,
        }),
    );
    Command::new(env!("CARGO_BIN_EXE_fireprot"))
        .current_dir(root)
        .args(["collect-timings", "e2e-collection.json"])
        .output()
        .unwrap()
}

#[test]
fn complete_invocations_collect_repeated_and_relocated_archives() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (mut request, profile) = fixture(root);
    let mut invocations = Vec::new();
    let mut runs = Vec::new();
    for index in 0..2 {
        request["run_id"] = json!(format!("measured-{index}"));
        request["output_dir"] = json!(root.join(format!("run-{index}")));
        let output = measure_fixture(
            root,
            &request,
            &profile,
            &format!("measurement-{index}"),
            30,
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let pin: Value = serde_json::from_slice(&output.stdout).unwrap();
        let record: Value =
            serde_json::from_slice(&fs::read(pin["path"].as_str().unwrap()).unwrap()).unwrap();
        let supervisor: Value = serde_json::from_slice(
            &fs::read(root.join(format!("run-{index}/supervisor.json"))).unwrap(),
        )
        .unwrap();
        assert!(
            record["elapsed_seconds"].as_f64().unwrap()
                >= supervisor["elapsed_seconds"].as_f64().unwrap()
        );
        assert_eq!(record["launch"]["arguments"][0], "infer");
        assert_eq!(record["exit_code"], 0);
        invocations.push(pin);
        runs.push(record["accepted"].clone());
    }
    let output = collect_invocations(root, &json!(runs), &json!(invocations));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["summary"]["count"], 2);
    assert_eq!(result["invocations"], json!(invocations));
    let swapped = json!([invocations[1], invocations[0]]);
    assert!(
        !collect_invocations(root, &json!(runs), &swapped)
            .status
            .success()
    );
    // Replaying requires the two archives, not original configs/model assets.
    fs::remove_file(root.join("measurement-request.json")).unwrap();
    fs::remove_file(root.join("measurement-profile.json")).unwrap();
    fs::remove_dir_all(root.join("source")).unwrap();
    for index in 0..2 {
        for (old, new, name, pins) in [
            (
                format!("run-{index}"),
                format!("relocated-run-{index}"),
                "accepted.json",
                &mut runs,
            ),
            (
                format!("measurement-{index}"),
                format!("relocated-measurement-{index}"),
                "invocation.json",
                &mut invocations,
            ),
        ] {
            fs::rename(root.join(old), root.join(&new)).unwrap();
            pins[index]["path"] = json!(root.join(new).join(name));
        }
    }
    let output = collect_invocations(root, &json!(runs), &json!(invocations));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Control snapshots remain bound after relocation.
    fs::write(root.join("relocated-measurement-0/profile.json"), b"{}").unwrap();
    assert!(
        !collect_invocations(root, &json!(runs), &json!(invocations))
            .status
            .success()
    );
}

#[test]
fn invocation_record_semantics_are_checked_after_receipt_repinning() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (request, profile) = fixture(root);
    let output = measure_fixture(root, &request, &profile, "measurement", 30);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut pin: Value = serde_json::from_slice(&output.stdout).unwrap();
    let path = root.join("measurement/invocation.json");
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let runs = json!([original["accepted"]]);
    for (field, value) in [
        ("exit_code", json!(1)),
        ("elapsed_seconds", json!(0.0)),
        ("elapsed_seconds", json!(0.0000001)),
        ("elapsed_seconds", json!(31.0)),
        ("boundary", json!("model-time-only")),
        ("schema_version", json!(99)),
    ] {
        let mut changed = original.clone();
        changed[field] = value;
        write(&path, &changed);
        pin["sha256"] = sha256(&path).unwrap().into();
        assert!(
            !collect_invocations(root, &runs, &json!([pin]))
                .status
                .success(),
            "{field}"
        );
    }
    let mut changed = original.clone();
    changed["launch"]["program"]["sha256"] = "0".repeat(64).into();
    write(&path, &changed);
    pin["sha256"] = sha256(&path).unwrap().into();
    assert!(
        !collect_invocations(root, &runs, &json!([pin]))
            .status
            .success()
    );
}

#[test]
fn failed_or_timed_out_invocations_never_publish_timing_records() {
    for timeout in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let (request, mut profile) = fixture(root);
        if timeout {
            let worker = root.join("python/fireprot_workers/__main__.py");
            let source = fs::read_to_string(&worker).unwrap();
            fs::write(
                &worker,
                source.replace(
                    "fixture = Path",
                    "import time; time.sleep(30)\nfixture = Path",
                ),
            )
            .unwrap();
        } else {
            profile["environment"]["FIREPROT_TEST_INVALID"] = "1".into();
        }
        let output = measure_fixture(
            root,
            &request,
            &profile,
            "measurement",
            if timeout { 1 } else { 30 },
        );
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(!root.join("measurement/invocation.json").exists());
        assert!(!root.join("run/accepted.json").exists());
        let failed: Value =
            serde_json::from_slice(&fs::read(root.join("measurement/failure.json")).unwrap())
                .unwrap();
        if timeout {
            assert_eq!(failed["stop_reason"], "Deadline");
        }
        assert!(failed["elapsed_seconds"].as_f64().unwrap() > 0.0);
    }
}

#[test]
fn paired_cli_withholds_performance_when_resealed_finite_outputs_disagree() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let (mut request, profile) = fixture(root);
    let mut pins = Vec::new();
    for arm in ["baseline", "candidate"] {
        request["run_id"] = json!(arm);
        request["output_dir"] = json!(root.join(arm));
        let output = infer(root, &request, &profile);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let receipt = root.join(arm).join("accepted.json");
        pins.push(json!({"path":receipt,"sha256":sha256(&receipt).unwrap()}));
    }
    let collection = |pin: &Value| json!({"schema_version":1,"scope":"model","cache_state":"synthetic paired fixture","runs":[pin]});
    let mut specification = json!({"schema_version":1,"treatment_description":"same-code synthetic repeat",
        "expected_changes":{},"baseline":collection(&pins[0]),"candidate":collection(&pins[1])});
    let run = |spec: &Value| {
        write(&root.join("comparison.json"), spec);
        Command::new(env!("CARGO_BIN_EXE_fireprot"))
            .current_dir(root)
            .args(["compare-runs", "comparison.json"])
            .output()
            .unwrap()
    };
    let output = run(&specification);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["correctness_passed"], true);
    assert_eq!(report["performance"]["median_speedup"], 1.0);
    assert_eq!(
        report["performance"]["observed_trend"],
        "insufficient_repeats"
    );
    assert_eq!(report["pairs"][0]["cif_atom_identities_checked"], true);
    // Keep the candidate finite, structurally valid and independently re-pinned.
    // Only its model agreement is now wrong; archive/content checks still pass.
    let archive = root.join("candidate/sample-0.npz");
    let changed=Command::new("/usr/bin/python3").arg("-c").arg(
        "import sys,zipfile,struct\np=sys.argv[1]\nwith zipfile.ZipFile(p) as z: d={k:z.read(k) for k in z.namelist()}\nk='confidence.ptm.npy'\nd[k]=d[k][:-4]+struct.pack('<f',0.75)\nwith zipfile.ZipFile(p,'w') as z:\n for k,v in d.items(): z.writestr(k,v)\n"
    ).arg(&archive).output().unwrap();
    assert!(
        changed.status.success(),
        "{}",
        String::from_utf8_lossy(&changed.stderr)
    );
    let receipt = root.join("candidate/accepted.json");
    let mut content: Value = serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
    for artifact in content["outputs"].as_array_mut().unwrap() {
        if artifact["path"] == "sample-0.npz" {
            artifact["sha256"] = sha256(&archive).unwrap().into();
        }
    }
    write(&receipt, &content);
    specification["candidate"]["runs"][0]["sha256"] = sha256(&receipt).unwrap().into();
    let output = run(&specification);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["correctness_passed"], false);
    assert!(report["performance"].is_null());
    assert_eq!(
        report["pairs"][0]["samples"][0]["exact_arrays"]["confidence.ptm"],
        false
    );
}

fn benchmark_spec(root: &Path, mode: &str) -> Value {
    let (request, profile) = fixture(root);
    let mut other_profile = profile.clone();
    if mode == "monitor" {
        let source = fs::read_to_string(root.join("amd-smi")).unwrap();
        executable(
            &root.join("other-monitor"),
            &(source + "\n# independently identified monitor fixture\n"),
        );
        other_profile["amd_smi"] = json!(root.join("other-monitor"));
    }
    if mode == "numeric_failure" {
        let bad = root.join("bad-fixture");
        fs::create_dir(&bad).unwrap();
        for file in ["sample.cif", "sample-0.npz", "predictions.json"] {
            fs::copy(root.join("fixture").join(file), bad.join(file)).unwrap();
        }
        let changed=Command::new("/usr/bin/python3").arg("-c").arg(
            "import sys,zipfile,struct\np=sys.argv[1]\nwith zipfile.ZipFile(p) as z: d={k:z.read(k) for k in z.namelist()}\nk='confidence.ptm.npy'\nd[k]=d[k][:-4]+struct.pack('<f',0.75)\nwith zipfile.ZipFile(p,'w') as z:\n for k,v in d.items(): z.writestr(k,v)\n"
        ).arg(bad.join("sample-0.npz")).output().unwrap();
        assert!(changed.status.success());
        other_profile["environment"]["FIREPROT_TEST_FIXTURE"] = json!(bad);
    }
    if mode == "bad_profile" {
        other_profile["unknown_option"] = json!(true);
    }
    if mode == "cancel" || mode == "deadline" {
        let path = root.join("python/fireprot_workers/__main__.py");
        let source = fs::read_to_string(&path).unwrap();
        fs::write(&path,source.replace("fixture = Path","Path('worker.pid').write_text(str(os.getpid()))\nimport time; time.sleep(30)\nfixture = Path")).unwrap();
    }
    let mut pairs = Vec::new();
    for pair in 0..2 {
        let mut arms = serde_json::Map::new();
        for arm in ["baseline", "candidate"] {
            let id = format!("bench-{arm}-{pair}");
            let mut r = request.clone();
            r["run_id"] = json!(id);
            r["output_dir"] = json!(root.join(&id));
            let rp = root.join(format!("{id}-request.json"));
            let pp = root.join(format!("{id}-profile.json"));
            write(&rp, &r);
            write(
                &pp,
                if arm == "baseline" {
                    &profile
                } else {
                    &other_profile
                },
            );
            arms.insert(arm.into(),json!({"request":{"path":rp,"sha256":sha256(&rp).unwrap()},"profile":{"path":pp,"sha256":sha256(&pp).unwrap()}}));
        }
        arms.insert(
            "first".into(),
            json!(if pair == 0 { "baseline" } else { "candidate" }),
        );
        pairs.push(Value::Object(arms));
    }
    let changes = if mode == "monitor" {
        let identity = |name: &str| {
            json!({"name":"amd-smi","canonical_path":root.join(name).canonicalize().unwrap(),
            "sha256":sha256(&root.join(name)).unwrap()})
        };
        json!({"launch.inputs.amd-smi":{"baseline":identity("amd-smi"),"candidate":identity("other-monitor")},
            "invocation.profile_sha256":{"baseline":pairs[0]["baseline"]["profile"]["sha256"],
                                          "candidate":pairs[0]["candidate"]["profile"]["sha256"]}})
    } else if mode == "numeric_failure" {
        json!({"launch.environment.FIREPROT_TEST_FIXTURE":{
        "baseline":profile["environment"]["FIREPROT_TEST_FIXTURE"],"candidate":other_profile["environment"]["FIREPROT_TEST_FIXTURE"]}})
    } else {
        json!({})
    };
    let mut specification = json!({"schema_version":1,"execution":{"schema_version":1,"timeout_seconds":if mode=="deadline" {1} else {60},
        "invocation_timeout_seconds":30,"pairs":pairs},
        "comparison":{"scope":if mode=="numeric_failure" {"model"} else {"end_to_end"},
            "cache_state":"synthetic scheduler fixture","treatment_description":"synthetic repeat comparison","expected_changes":changes}});
    if mode == "monitor" {
        specification["comparison"]["monitor_treatment"] = json!(true);
    }
    specification
}

fn benchmark_command(root: &Path, spec: &Value) -> Command {
    write(&root.join("benchmark-spec.json"), spec);
    let mut command = Command::new(env!("CARGO_BIN_EXE_fireprot"));
    command
        .current_dir(root)
        .args(["benchmark-pairs", "benchmark-spec.json", "benchmark"]);
    command
}

#[test]
fn benchmark_monitor_treatment_is_explicit_and_preserves_all_archive_checks() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    let mut spec = benchmark_spec(root, "monitor");
    spec["execution"]["timeout_seconds"] = json!(180);
    let output = benchmark_command(root, &spec).output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        fs::read_to_string(root.join("benchmark/benchmark.json")).unwrap_or_default()
    );
    let report: Value =
        serde_json::from_slice(&fs::read(root.join("benchmark/benchmark.json")).unwrap()).unwrap();
    assert_eq!(report["phase"], "succeeded");
    assert_eq!(report["checks"].as_array().unwrap().len(), 5);
    let comparison: Value =
        serde_json::from_slice(&fs::read(report["comparison"]["path"].as_str().unwrap()).unwrap())
            .unwrap();
    assert_eq!(comparison["monitor_treatment"], true);
    assert_eq!(comparison["correctness_passed"], true);
    assert_eq!(
        comparison["performance"]["changes"],
        spec["comparison"]["expected_changes"]
    );
}

#[test]
fn monitor_treatment_in_model_scope_is_rejected_before_any_launch() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    let mut spec = benchmark_spec(root, "success");
    spec["comparison"]["scope"] = json!("model");
    spec["comparison"]["monitor_treatment"] = json!(true);
    let output = benchmark_command(root, &spec).output().unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("monitor treatment requires end-to-end")
    );
    assert!(!root.join("benchmark").exists());
    assert!(!root.join("bench-baseline-0").exists());
}

#[test]
fn benchmark_records_actual_order_and_all_reference_pair_and_aggregate_checks() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path();
    let mut spec = benchmark_spec(root, "success");
    // Four separately pinned invocations plus five archive comparisons can exceed
    // 60 seconds on shared CI runners. This test checks ordering and correctness;
    // the separate deadline/cancellation fixture keeps its explicit short limit.
    spec["execution"]["timeout_seconds"] = json!(180);
    let output = benchmark_command(root, &spec).output().unwrap();
    assert!(
        output.status.success(),
        "stderr: {}\nstdout: {}\nreport: {}\norder: {}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
        fs::read_to_string(root.join("benchmark/benchmark.json")).unwrap_or_default(),
        fs::read_to_string(root.join("benchmark/order.jsonl")).unwrap_or_default()
    );
    let outcome: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(outcome["phase"], "succeeded");
    assert_eq!(
        outcome["report"]["sha256"],
        sha256(Path::new(outcome["report"]["path"].as_str().unwrap())).unwrap()
    );
    let report: Value =
        serde_json::from_slice(&fs::read(root.join("benchmark/benchmark.json")).unwrap()).unwrap();
    assert_eq!(report["measurements"].as_array().unwrap().len(), 4);
    assert_eq!(report["checks"].as_array().unwrap().len(), 5);
    let lines = fs::read_to_string(root.join("benchmark/order.jsonl")).unwrap();
    let events: Vec<Value> = lines
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(
        events
            .iter()
            .filter(|e| e["event"] == "started")
            .map(|e| e["run_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec![
            "bench-baseline-0",
            "bench-candidate-0",
            "bench-candidate-1",
            "bench-baseline-1"
        ]
    );
    assert_eq!(events.len(), 12);
    assert!(
        events
            .windows(2)
            .all(|w| w[0]["offset_seconds"].as_f64().unwrap()
                <= w[1]["offset_seconds"].as_f64().unwrap())
    );
    let comparison: Value =
        serde_json::from_slice(&fs::read(root.join("benchmark/comparison.json")).unwrap()).unwrap();
    assert_eq!(comparison["correctness_passed"], true);
    assert_eq!(comparison["performance"]["baseline"]["count"], 2);
}

#[test]
fn benchmark_rejects_bad_profiles_before_launch_and_stops_on_numeric_failure() {
    for mode in ["bad_profile", "numeric_failure"] {
        let d = tempfile::tempdir().unwrap();
        let root = d.path();
        let spec = benchmark_spec(root, mode);
        let output = benchmark_command(root, &spec).output().unwrap();
        assert!(!output.status.success());
        if mode == "bad_profile" {
            assert!(!root.join("benchmark").exists());
            assert!(!root.join("bench-baseline-0").exists());
        } else {
            let result: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(result["phase"], "failed");
            assert!(result["comparison"].is_null());
            let report: Value =
                serde_json::from_slice(&fs::read(root.join("benchmark/benchmark.json")).unwrap())
                    .unwrap();
            assert_eq!(report["batch"]["jobs"][2]["phase"], "skipped");
            assert_eq!(report["batch"]["jobs"][3]["phase"], "skipped");
            assert!(!root.join("bench-candidate-1").exists());
            assert!(!root.join("benchmark/comparison.json").exists());
            let check: Value = serde_json::from_slice(
                &fs::read(root.join("benchmark/reference-check-001.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(check["correctness_passed"], false);
            assert!(check["performance"].is_null());
        }
    }
}

#[test]
fn benchmark_cancellation_and_global_deadline_stop_later_jobs_and_reap_workers() {
    for mode in ["cancel", "deadline"] {
        let d = tempfile::tempdir().unwrap();
        let root = d.path();
        let spec = benchmark_spec(root, mode);
        let mut command = benchmark_command(root, &spec);
        command
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let mut child = command.spawn().unwrap();
        let pid_file = root.join("bench-baseline-0/worker.pid");
        if mode == "cancel" {
            let start = std::time::Instant::now();
            while !pid_file.exists() {
                assert!(
                    child.try_wait().unwrap().is_none(),
                    "benchmark exited before worker ready"
                );
                assert!(start.elapsed() < std::time::Duration::from_secs(15));
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            assert!(
                Command::new("kill")
                    .args(["-TERM", &child.id().to_string()])
                    .status()
                    .unwrap()
                    .success()
            );
        }
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        let outcome: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            outcome["phase"],
            if mode == "cancel" {
                "cancelled"
            } else {
                "timed_out"
            }
        );
        assert!(outcome["comparison"].is_null());
        assert!(!root.join("bench-candidate-0").exists());
        if pid_file.exists() {
            let pid = fs::read_to_string(pid_file).unwrap();
            assert!(
                !Command::new("kill")
                    .args(["-0", pid.trim()])
                    .output()
                    .unwrap()
                    .status
                    .success()
            );
        }
    }
}
