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
