//! Explicit hardware gate: FIREPROT_HIP_PROBE and FIREPROT_GPU_FIXTURE must be
//! compiled HIP executables. Defaults target DiamondHill HIP 7 / SMI 5.
use fireprot_core::protocol::DeviceIdentity;
use fireprot_runtime::{device::AmdSmiMonitor, supervisor::DeviceMonitor};
use std::{
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
struct Fixture(Child);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
#[ignore = "requires an authorized idle AMD GPU and compiled HIP fixtures"]
fn identifies_real_gpu_process_and_rejects_it_unless_allowed() {
    let probe = std::env::var("FIREPROT_HIP_PROBE").expect("HIP probe path");
    let fixture = std::env::var("FIREPROT_GPU_FIXTURE").expect("GPU fixture path");
    let ordinal = std::env::var("FIREPROT_TEST_HIP_DEVICE").unwrap_or("7".into());
    let pci = std::env::var("FIREPROT_TEST_PCI").unwrap_or("0000:93:00.0".into());
    let mut m = AmdSmiMonitor::new(probe.into(), "amd-smi".into(), Duration::from_secs(10));
    let mut identity = DeviceIdentity {
        worker_pid: std::process::id(),
        logical_device: 0,
        pci_address: pci,
        architecture: "gfx90a".into(),
    };
    m.verify_device(ordinal.parse().unwrap(), &identity)
        .unwrap();
    assert!(
        m.sample(identity.worker_pid).unwrap().is_empty(),
        "test requires an idle selected GPU"
    );
    let dir = tempfile::tempdir().unwrap();
    let ready = dir.path().join("ready");
    let mut child = Fixture(
        Command::new(fixture)
            .arg(&ready)
            .env_remove("ROCR_VISIBLE_DEVICES")
            .env_remove("CUDA_VISIBLE_DEVICES")
            .env_remove("GPU_DEVICE_ORDINAL")
            .env("HIP_VISIBLE_DEVICES", ordinal.clone())
            .stdin(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    while !ready.exists() {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "GPU fixture readiness timeout"
        );
        assert!(child.0.try_wait().unwrap().is_none(), "GPU fixture failed");
        std::thread::sleep(Duration::from_millis(20));
    }
    let external = m.sample(identity.worker_pid).unwrap();
    assert!(external.contains(&child.0.id()));
    identity.worker_pid = child.0.id();
    m.verify_device(ordinal.parse().unwrap(), &identity)
        .unwrap();
    assert!(m.sample(identity.worker_pid).unwrap().is_empty());
    println!(
        "{}",
        serde_json::json!({"device":m.selected(),"detected_external_pid":external,"allowed_pid":identity.worker_pid})
    );
}
