use fireprot_core::{Request, ValidatedRequest};
use fireprot_validation::{bundle::validate, numpy::read_npz};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};
fn setup() -> (tempfile::TempDir, ValidatedRequest, Value, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let archive = root.join("sample-0.npz");
    fs::write(&archive, include_bytes!("fixtures/openfold_sample.npz")).unwrap();
    let artifact = json!({"path":"unused","sha256":"a".repeat(64)});
    let r:Request=serde_json::from_value(json!({"schema_version":1,"run_id":"sample","backend":"native","seed":101,"hip_device":7,
        "workload":{"model":"openfold3","model_name":"openbind-0","recycles":3,"diffusion_steps":200,"samples":1,"precision":"float32"},
        "source":{"root":"unused","commit":"b".repeat(40)},"checkpoints":{"model":artifact},"input":artifact,"output_dir":root})).unwrap();
    let arrays = read_npz(&archive, 100000).unwrap();
    let descriptors: serde_json::Map<String, Value> = arrays
        .iter()
        .map(|(k, a)| {
            (
                k.clone(),
                json!({"shape":a.shape(),"storage_dtype":a.dtype(),"original_dtype":a.dtype()}),
            )
        })
        .collect();
    let bundle = json!({"schema_version":1,"run_id":"sample","seed":101,"workload":r.workload,"predictions":[{"sample":0,"archive":"sample-0.npz","arrays":descriptors,"confidence":null}],"model_seconds":1.,"request_seconds":2.,"environment":{"torch":"fixture","hip":"fixture","python":"fixture"}});
    let verified = vec![root.join("predictions.json"), archive];
    (
        dir,
        ValidatedRequest::try_from(r).unwrap(),
        bundle,
        verified,
    )
}
#[test]
fn complete_model_bundle_passes_content_gate() {
    let (_d, r, b, v) = setup();
    fs::write(&v[0], serde_json::to_vec(&b).unwrap()).unwrap();
    assert!(validate(&r, &v).is_ok());
}
#[test]
fn stale_identity_workload_missing_samples_and_descriptors_fail() {
    for change in 0..9 {
        let (_d, r, mut b, v) = setup();
        match change {
            0 => b["run_id"] = json!("stale"),
            1 => b["seed"] = json!(102),
            2 => b["workload"]["recycles"] = json!(2),
            3 => b["predictions"] = json!([]),
            4 => b["predictions"][0]["sample"] = json!(1),
            5 => b["predictions"][0]["arrays"]["coordinates"]["shape"] = json!([12]),
            6 => b["predictions"][0]["arrays"]["coordinates"]["original_dtype"] = json!("bfloat16"),
            7 => b["request_seconds"] = json!(0.5),
            _ => {
                b["predictions"][0]["arrays"]
                    .as_object_mut()
                    .unwrap()
                    .remove("confidence.ptm");
            }
        }
        fs::write(&v[0], serde_json::to_vec(&b).unwrap()).unwrap();
        assert!(validate(&r, &v).is_err(), "change {change}");
    }
}
#[test]
fn only_supervisor_verified_files_are_accepted() {
    let (_d, r, b, v) = setup();
    fs::write(&v[0], serde_json::to_vec(&b).unwrap()).unwrap();
    assert!(validate(&r, &v[..1]).is_err());
    assert!(validate(&r, &v[1..]).is_err());
}
