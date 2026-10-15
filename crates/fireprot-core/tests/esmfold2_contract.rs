use fireprot_core::{Request, ValidatedRequest};
use serde_json::{Value, json};

fn request() -> Value {
    let pin = json!({"path":"/asset", "sha256":"a".repeat(64)});
    json!({"schema_version":1,"run_id":"esm2","backend":"native","seed":101,"hip_device":7,
        "workload":{"model":"esmfold2","model_name":"biohub/ESMFold2","loops":3,
                    "sampling_steps":50,"samples":1,"precision":"native","chunk_size":64,"deterministic":true},
        "source":{"root":"/source","commit":"b".repeat(40)},"checkpoints":{"model":pin},
        "input":pin,"output_dir":"/output"})
}

#[test]
fn loops_are_additional_passes_and_sampling_is_a_distinct_field() {
    for loops in [0, 3, 20] {
        let mut r = request();
        r["workload"]["loops"] = json!(loops);
        let parsed: Request = serde_json::from_value(r.clone()).unwrap();
        let checked = ValidatedRequest::try_from(parsed).unwrap();
        assert_eq!(
            checked.as_request().workload.trunk_passes().unwrap(),
            loops + 1
        );
        assert_eq!(
            serde_json::to_value(checked.as_request()).unwrap()["workload"],
            r["workload"]
        );
    }
    let mut r = request();
    r["workload"]["diffusion_steps"] = json!(50);
    assert!(serde_json::from_value::<Request>(r).is_err());
}

#[test]
fn unsupported_variants_precision_and_incomplete_workloads_are_rejected() {
    for (field, value) in [
        ("loops", json!(u32::MAX)),
        ("sampling_steps", json!(0)),
        ("samples", json!(0)),
        ("chunk_size", json!(0)),
        ("precision", json!("float32")),
        ("model_name", json!("biohub/ESMFold2-Fast")),
    ] {
        let mut r = request();
        r["workload"][field] = value;
        assert!(
            ValidatedRequest::try_from(serde_json::from_value::<Request>(r).unwrap()).is_err(),
            "{field}"
        );
    }
    let mut r = request();
    r["backend"] = json!("fireprot");
    assert!(
        ValidatedRequest::try_from(serde_json::from_value::<Request>(r.clone()).unwrap()).is_ok()
    );
    r["workload"]["deterministic"] = json!(false);
    assert!(ValidatedRequest::try_from(serde_json::from_value::<Request>(r).unwrap()).is_err());
}
