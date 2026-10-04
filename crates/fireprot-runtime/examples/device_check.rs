//! Explicit host diagnostic; no inference or performance claim.
use fireprot_core::protocol::DeviceIdentity;
use fireprot_runtime::{device::AmdSmiMonitor, supervisor::DeviceMonitor};
use std::{path::PathBuf, time::Duration};
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: device_check HIP_PROBE HIP_ORDINAL WORKER_PID PCI ARCHITECTURE".into());
    }
    let mut monitor = AmdSmiMonitor::new(
        PathBuf::from(&args[1]),
        "amd-smi".into(),
        Duration::from_secs(10),
    );
    let pid = args[3].parse().map_err(|_| "invalid PID")?;
    monitor.verify_device(
        args[2].parse().map_err(|_| "invalid HIP ordinal")?,
        &DeviceIdentity {
            worker_pid: pid,
            logical_device: 0,
            pci_address: args[4].clone(),
            architecture: args[5].clone(),
        },
    )?;
    let external = monitor.sample(pid)?;
    println!(
        "{}",
        serde_json::json!({"device":monitor.selected(),"external_pids":external})
    );
    if !external.is_empty() {
        return Err("external GPU processes detected".into());
    }
    Ok(())
}
