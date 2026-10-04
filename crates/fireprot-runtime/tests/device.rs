use fireprot_runtime::device::{normalize_pci, process_ids, smi_device};
use serde_json::json;

#[test]
fn pci_identity_not_ordinal_determines_smi_index() {
    let list = json!([{"gpu":0,"bdf":"0000:11:00.0"}, {"gpu":5,"bdf":"0000:93:00.0"}]);
    assert_eq!(smi_device(&list, "0000:93:00.0").unwrap(), 5);
    assert_eq!(normalize_pci("0000:AE:00.0").unwrap(), "0000:ae:00.0");
    for bad in [
        "",
        "0000:11:00.8",
        "0000:11:20.0",
        "0000:ZZ:00.0",
        "é000:11:00.0",
    ] {
        assert!(normalize_pci(bad).is_err(), "{bad}");
    }
}
#[test]
fn ambiguous_missing_or_malformed_device_records_fail() {
    for list in [
        json!({}),
        json!([]),
        json!([{"gpu":0}]),
        json!([{"gpu":-1,"bdf":"0000:11:00.0"}]),
        json!([{"gpu":0,"bdf":"0000:11:00.0"},{"gpu":0,"bdf":"0000:93:00.0"}]),
        json!([{"gpu":0,"bdf":"0000:11:00.0"},{"gpu":1,"bdf":"0000:11:00.0"}]),
    ] {
        assert!(smi_device(&list, "0000:11:00.0").is_err());
    }
}
#[test]
fn valid_idle_and_active_process_records() {
    for entries in [
        json!([]),
        json!([{"process_info":"No running processes detected"}]),
    ] {
        assert!(
            process_ids(&json!([{"gpu":5,"process_list":entries}]), 5)
                .unwrap()
                .is_empty()
        );
    }
    let active = json!([{"gpu":0,"process_list":[]},{"gpu":5,"process_list":[{"process_info":{"pid":"123"}}, {"process_info":{"pid":456}}]}]);
    assert_eq!(
        process_ids(&active, 5)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        vec![123, 456]
    );
}
#[test]
fn missing_or_bad_process_rows_never_mean_idle() {
    for payload in [
        json!({}),
        json!([]),
        json!([{"gpu":5}]),
        json!([{"gpu":0,"process_list":[]}]),
        json!([{"gpu":5,"process_list":[]},{"gpu":5,"process_list":[]}]),
    ] {
        assert!(process_ids(&payload, 5).is_err());
    }
    for entries in [
        json!([{}]),
        json!([{"process_info":"unavailable"}]),
        json!([{"process_info":{"pid":0}}]),
        json!([{"process_info":{"pid":"oops"}}]),
        json!([{"process_info":{"pid":123}},{"process_info":{"pid":123}}]),
        json!([{"process_info":"No running processes detected"},{"process_info":{"pid":123}}]),
    ] {
        assert!(process_ids(&json!([{"gpu":5,"process_list":entries}]), 5).is_err());
    }
}
