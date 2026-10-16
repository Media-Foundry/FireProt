use fireprot_core::{Request, ValidatedRequest};
use serde_json::json;
#[test]
fn standalone_models_reject_folding_settings_and_unqualified_precision() {
    let pin = json!({"path":"/asset","sha256":"a".repeat(64)});
    let mut r = json!({"schema_version":1,"run_id":"plm","backend":"native","seed":101,"hip_device":6,
        "workload":{"model":"plm","model_name":"esmc-fold2-6b","precision":"native"},
        "source":{"root":"/source","commit":"b".repeat(40)},"checkpoints":{"model":pin},"input":pin,"output_dir":"/output"});
    for name in ["esmc-fold2-6b", "esm2-3b", "esm3-open", "atlaslm-3b"] {
        r["workload"]["model_name"] = json!(name);
        assert!(
            ValidatedRequest::try_from(serde_json::from_value::<Request>(r.clone()).unwrap())
                .is_ok()
        );
    }
    for name in ["esmc-fold2-6b", "esm2-3b", "atlaslm-3b", "esm3-open"] {
        let mut candidate = r.clone();
        candidate["workload"]["model_name"] = json!(name);
        candidate["backend"] = json!("fireprot");
        assert!(
            ValidatedRequest::try_from(
                serde_json::from_value::<Request>(candidate.clone()).unwrap()
            )
            .is_ok()
        );
        candidate["resources"] = json!({"hip_esm":pin});
        assert!(
            ValidatedRequest::try_from(serde_json::from_value::<Request>(candidate).unwrap())
                .is_ok()
        );
    }
    for (key, value) in [("model_name", "unknown"), ("precision", "bfloat16")] {
        let mut invalid = r.clone();
        invalid["workload"][key] = json!(value);
        assert!(
            ValidatedRequest::try_from(serde_json::from_value::<Request>(invalid).unwrap())
                .is_err()
        );
    }
    let mut invalid = r.clone();
    invalid["backend"] = json!("hip_fused");
    assert!(
        ValidatedRequest::try_from(serde_json::from_value::<Request>(invalid).unwrap()).is_err()
    );
    r["workload"]["recycles"] = json!(1);
    assert!(serde_json::from_value::<Request>(r).is_err());
}
