use fireprot_core::protocol::DeviceIdentity;
use fireprot_runtime::{device::AmdSmiMonitor, supervisor::DeviceMonitor};
use std::{os::unix::fs::PermissionsExt, time::Duration};

#[test]
fn independent_probe_mapping_pid_and_reverification_are_enforced() {
    let d = tempfile::tempdir().unwrap();
    let hip = d.path().join("hip");
    let smi = d.path().join("smi");
    std::fs::write(&hip,"#!/usr/bin/env python3\nimport os\nassert os.environ['HIP_VISIBLE_DEVICES']=='7'\nassert 'ROCR_VISIBLE_DEVICES' not in os.environ\nprint('{\"pci_address\":\"0000:93:00.0\",\"architecture\":\"gfx90a\"}')\n").unwrap();
    std::fs::write(&smi,"#!/usr/bin/env python3\nimport os,sys,pathlib\nassert 'HIP_VISIBLE_DEVICES' not in os.environ\nprint(pathlib.Path(__file__).with_name(sys.argv[1]+'.json').read_text())\n").unwrap();
    for path in [&hip, &smi] {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let list = d.path().join("list.json");
    let process = d.path().join("process.json");
    std::fs::write(&list, r#"[{"gpu":5,"bdf":"0000:93:00.0"}]"#).unwrap();
    std::fs::write(
        &process,
        r#"[{"gpu":5,"process_list":[{"process_info":{"pid":42}},{"process_info":{"pid":43}}]}]"#,
    )
    .unwrap();
    let mut m = AmdSmiMonitor::new(hip, smi, Duration::from_secs(2));
    assert!(m.sample(42).is_err());
    let mut identity = DeviceIdentity {
        worker_pid: 42,
        logical_device: 0,
        pci_address: "0000:93:00.0".into(),
        architecture: "gfx90a".into(),
    };
    m.verify_device(7, &identity).unwrap();
    assert_eq!(m.selected().unwrap().smi_device, 5);
    assert_eq!(m.sample(42).unwrap(), vec![43]);
    assert!(m.sample(43).is_err());
    std::fs::write(&list, r#"[{"gpu":7,"bdf":"0000:93:00.0"}]"#).unwrap();
    assert!(m.sample(42).unwrap_err().contains("mapping changed"));
    identity.pci_address = "0000:11:00.0".into();
    assert!(m.verify_device(7, &identity).is_err());
    assert!(m.selected().is_none());
    assert!(m.sample(42).is_err());
}

#[test]
fn combined_snapshot_preserves_mapping_pid_and_failure_checks_without_fallback() {
    use fireprot_runtime::device::SmiProtocol;
    use serde_json::json;
    let d = tempfile::tempdir().unwrap();
    let hip = d.path().join("hip");
    let smi = d.path().join("smi");
    std::fs::write(&hip, "#!/usr/bin/env python3\nprint('{\"pci_address\":\"0000:93:00.0\",\"architecture\":\"gfx90a\"}')\n").unwrap();
    std::fs::write(&smi, r#"#!/usr/bin/env python3
import os,sys,pathlib,time
root=pathlib.Path(__file__).parent
assert all(k not in os.environ for k in ('HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL'))
assert sys.argv[2:] == ['--json']
with (root/'calls').open('a') as f:f.write(sys.argv[1]+'\n')
if (root/'fail').exists():
    print((root/'snapshot-v1.json').read_text())
    sys.exit(1)
if (root/'slow').exists():time.sleep(5)
print((root/(sys.argv[1]+'.json')).read_text())
"#).unwrap();
    for path in [&hip, &smi] {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let list = json!([{"gpu":5,"bdf":"0000:93:00.0"}]);
    std::fs::write(d.path().join("list.json"), list.to_string()).unwrap();
    let valid = json!({"schema_version":1,"devices":[
        {"gpu":5,"bdf":"0000:93:00.0","process_list":[{"process_info":{"pid":42}},{"process_info":{"pid":43}}]},
        {"gpu":6,"bdf":"0000:94:00.0","process_list":[{"process_info":"No running processes detected"}]}
    ]});
    let snapshot = d.path().join("snapshot-v1.json");
    std::fs::write(&snapshot, valid.to_string()).unwrap();
    let mut monitor = AmdSmiMonitor::new(hip, smi, Duration::from_millis(500))
        .with_protocol(SmiProtocol::SnapshotV1);
    let identity = DeviceIdentity {
        worker_pid: 42,
        logical_device: 0,
        pci_address: "0000:93:00.0".into(),
        architecture: "gfx90a".into(),
    };
    assert!(monitor.sample(42).is_err());
    monitor.verify_device(7, &identity).unwrap();
    assert_eq!(monitor.sample(42).unwrap(), vec![43]);
    assert!(monitor.sample(43).is_err());
    assert_eq!(
        std::fs::read_to_string(d.path().join("calls")).unwrap(),
        "list\nsnapshot-v1\n"
    );
    let mut rejected = Vec::new();
    for version in [json!(0), json!(2), json!("1"), json!(null)] {
        let mut bad = valid.clone();
        bad["schema_version"] = version;
        rejected.push(bad);
    }
    let mut extra = valid.clone();
    extra["cached"] = json!(true);
    rejected.push(extra);
    for field in ["gpu", "bdf", "process_list"] {
        let mut bad = valid.clone();
        bad["devices"][0].as_object_mut().unwrap().remove(field);
        rejected.push(bad);
    }
    let mut changed = valid.clone();
    changed["devices"][0]["gpu"] = json!(7);
    rejected.push(changed);
    let mut duplicate = valid.clone();
    duplicate["devices"][1] = duplicate["devices"][0].clone();
    rejected.push(duplicate);
    let mut gone = valid.clone();
    gone["devices"].as_array_mut().unwrap().remove(0);
    rejected.push(gone);
    for entries in [
        json!([{"process_info":{"pid":0}}]),
        json!([{"process_info":"Unknown"}]),
        json!([{"process_info":{"pid":42}},{"process_info":{"pid":42}}]),
    ] {
        for row in 0..2 {
            let mut bad = valid.clone();
            bad["devices"][row]["process_list"] = entries.clone();
            rejected.push(bad);
        }
    }
    for bad in rejected {
        std::fs::write(&snapshot, bad.to_string()).unwrap();
        assert!(monitor.sample(42).is_err(), "accepted {bad}");
    }
    std::fs::write(&snapshot, "not json").unwrap();
    assert!(monitor.sample(42).is_err());
    std::fs::write(&snapshot, valid.to_string()).unwrap();
    std::fs::write(d.path().join("fail"), "").unwrap();
    assert!(monitor.sample(42).is_err());
    std::fs::remove_file(d.path().join("fail")).unwrap();
    std::fs::write(d.path().join("slow"), "").unwrap();
    assert!(monitor.sample(42).is_err());
    std::fs::remove_file(d.path().join("slow")).unwrap();
    let mut idle = valid.clone();
    idle["devices"][0]["process_list"] = json!([{"process_info":{"pid":42}}]);
    std::fs::write(&snapshot, idle.to_string()).unwrap();
    assert!(monitor.sample(42).unwrap().is_empty());
    let calls = std::fs::read_to_string(d.path().join("calls")).unwrap();
    assert_eq!(calls.lines().filter(|s| *s == "list").count(), 1);
    assert!(!calls.lines().any(|s| s == "process"));
    assert!(SmiProtocol::try_from("automatic").is_err());
    assert_eq!(SmiProtocol::try_from("cli").unwrap(), SmiProtocol::Cli);
}
