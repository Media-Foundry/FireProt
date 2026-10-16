use fireprot_core::{Request, ValidatedRequest};
use serde_json::{Value, json};

fn old_request() -> Value {
    serde_json::from_str(include_str!(
        "../../../configs/requests/plm-atlaslm-3b-n1000-native.json"
    ))
    .unwrap()
}

#[test]
fn optional_batch_preserves_singleton_wire_identity() {
    let old = old_request();
    let parsed: Request = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(
        serde_json::to_value(&parsed).unwrap()["workload"],
        old["workload"]
    );
    ValidatedRequest::try_from(parsed).unwrap();
}

#[test]
fn validates_batch_bounds_and_model_backend_without_folding_fields() {
    for backend in ["native", "fireprot"] {
        let mut value = old_request();
        value["backend"] = json!(backend);
        value["workload"]["batch"] = json!({"items":4,"max_batch_size":4,"max_padded_tokens":4008,"grouping":"exact_length"});
        ValidatedRequest::try_from(serde_json::from_value::<Request>(value.clone()).unwrap())
            .unwrap();
        for (field, bad) in [
            ("items", 0),
            ("items", 257),
            ("max_batch_size", 0),
            ("max_batch_size", 17),
            ("max_padded_tokens", 16033),
        ] {
            let mut bad_value = value.clone();
            bad_value["workload"]["batch"][field] = json!(bad);
            assert!(
                ValidatedRequest::try_from(serde_json::from_value::<Request>(bad_value).unwrap())
                    .is_err()
            );
        }
        value["workload"]["samples"] = json!(2);
        assert!(serde_json::from_value::<Request>(value).is_err());
    }
    let mut value = old_request();
    value["backend"] = json!("fireprot");
    value["workload"]["model_name"] = json!("esm3-open");
    value["workload"]["batch"] =
        json!({"items":4,"max_batch_size":4,"max_padded_tokens":4008,"grouping":"exact_length"});
    ValidatedRequest::try_from(serde_json::from_value::<Request>(value.clone()).unwrap()).unwrap();
    value["resources"]["hip_esm"] = json!({"path":"/hip.so","sha256":"a".repeat(64)});
    ValidatedRequest::try_from(serde_json::from_value::<Request>(value).unwrap()).unwrap();
}
