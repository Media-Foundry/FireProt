//! Independent HIP-to-SMI identity matching and strict process observations.
use crate::{command::output, supervisor::DeviceMonitor};
use fireprot_core::protocol::DeviceIdentity;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, path::PathBuf, process::Command, time::Duration};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HipIdentity {
    pub pci_address: String,
    pub architecture: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalDevice {
    pub hip_device: u32,
    pub smi_device: u32,
    pub pci_address: String,
    pub architecture: String,
}

/// Normalize the PCI address while rejecting malformed identities.
pub fn normalize_pci(value: &str) -> Result<String, String> {
    let b = value.as_bytes();
    if b.len() != 12
        || b[4] != b':'
        || b[7] != b':'
        || b[10] != b'.'
        || !b
            .iter()
            .enumerate()
            .all(|(i, c)| matches!(i, 4 | 7 | 10) || c.is_ascii_hexdigit())
        || !(b'0'..=b'7').contains(&b[11])
        || u8::from_str_radix(&value[8..10], 16).map_err(|e| e.to_string())? > 31
    {
        return Err("invalid PCI address".into());
    }
    Ok(value.to_ascii_lowercase())
}
fn index(value: &Value) -> Result<u32, String> {
    let n = value
        .as_u64()
        .or_else(|| value.as_str()?.parse().ok())
        .ok_or("invalid SMI integer")?;
    u32::try_from(n).map_err(|_| "SMI integer overflow".into())
}
/// Reject duplicate indices/addresses, including duplicates unrelated to target.
pub fn smi_device(list: &Value, pci: &str) -> Result<u32, String> {
    let target = normalize_pci(pci)?;
    let rows = list.as_array().ok_or("SMI list must be an array")?;
    let mut ids = BTreeSet::new();
    let mut addresses = BTreeSet::new();
    let mut selected = None;
    for row in rows {
        let id = index(&row["gpu"])?;
        let bdf = normalize_pci(row["bdf"].as_str().ok_or("missing SMI PCI address")?)?;
        if !ids.insert(id) || !addresses.insert(bdf.clone()) {
            return Err("ambiguous SMI device records".into());
        }
        if bdf == target {
            selected = Some(id);
        }
    }
    selected.ok_or("selected PCI device missing from SMI list".into())
}
/// The AMD SMI no-process sentinel is accepted only as the sole entry. Unknown
/// strings, missing records/lists and malformed PID rows are errors, not idle.
pub fn process_ids(payload: &Value, selected: u32) -> Result<BTreeSet<u32>, String> {
    let rows = payload
        .as_array()
        .ok_or("SMI process output must be an array")?;
    let mut ids = BTreeSet::new();
    let mut selected_row = None;
    for row in rows {
        let id = index(&row["gpu"])?;
        if !ids.insert(id) {
            return Err("duplicate SMI process device".into());
        }
        if id == selected {
            selected_row = Some(row);
        }
    }
    let processes =
        selected_row.ok_or("selected device missing from process records")?["process_list"]
            .as_array()
            .ok_or("missing SMI process list")?;
    let mut pids = BTreeSet::new();
    for process in processes {
        let info = &process["process_info"];
        if info.as_str() == Some("No running processes detected") && processes.len() == 1 {
            continue;
        }
        if !info.is_object() {
            return Err("invalid SMI process record".into());
        }
        let pid = index(&info["pid"])?;
        if pid == 0 || !pids.insert(pid) {
            return Err("invalid or duplicate SMI PID".into());
        }
    }
    Ok(pids)
}

pub struct AmdSmiMonitor {
    hip_probe: PathBuf,
    amd_smi: PathBuf,
    timeout: Duration,
    selected: Option<PhysicalDevice>,
    worker_pid: Option<u32>,
}
impl AmdSmiMonitor {
    pub fn new(hip_probe: PathBuf, amd_smi: PathBuf, timeout: Duration) -> Self {
        Self {
            hip_probe,
            amd_smi,
            timeout,
            selected: None,
            worker_pid: None,
        }
    }
    pub fn selected(&self) -> Option<&PhysicalDevice> {
        self.selected.as_ref()
    }
    fn smi(&self, operation: &str) -> Result<Value, String> {
        let mut command = Command::new(&self.amd_smi);
        command
            .args([operation, "--json"])
            .env_remove("HIP_VISIBLE_DEVICES")
            .env_remove("ROCR_VISIBLE_DEVICES")
            .env_remove("CUDA_VISIBLE_DEVICES")
            .env_remove("GPU_DEVICE_ORDINAL");
        let bytes = output(&mut command, self.timeout, 1024 * 1024).map_err(|e| e.to_string())?;
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid SMI JSON: {e}"))
    }
}
impl DeviceMonitor for AmdSmiMonitor {
    fn physical_device(&self) -> Option<PhysicalDevice> {
        self.selected.clone()
    }
    fn verify_device(&mut self, hip_device: u32, device: &DeviceIdentity) -> Result<(), String> {
        self.selected = None;
        self.worker_pid = None;
        let mut command = Command::new(&self.hip_probe);
        command
            .env_remove("ROCR_VISIBLE_DEVICES")
            .env_remove("CUDA_VISIBLE_DEVICES")
            .env_remove("GPU_DEVICE_ORDINAL")
            .env("HIP_VISIBLE_DEVICES", hip_device.to_string());
        let bytes = output(&mut command, self.timeout, 4096).map_err(|e| e.to_string())?;
        let identity: HipIdentity =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid HIP identity: {e}"))?;
        let pci = normalize_pci(&identity.pci_address)?;
        if pci != normalize_pci(&device.pci_address)?
            || identity.architecture != device.architecture
            || !identity.architecture.starts_with("gfx")
            || identity.architecture.len() <= 3
            || device.logical_device != 0
            || device.worker_pid == 0
        {
            return Err("worker identity disagrees with independent HIP probe".into());
        }
        let id = smi_device(&self.smi("list")?, &pci)?;
        self.selected = Some(PhysicalDevice {
            hip_device,
            smi_device: id,
            pci_address: pci,
            architecture: identity.architecture,
        });
        self.worker_pid = Some(device.worker_pid);
        Ok(())
    }
    fn sample(&mut self, worker_pid: u32) -> Result<Vec<u32>, String> {
        let device = self.selected.as_ref().ok_or("device not verified")?;
        if self.worker_pid != Some(worker_pid) {
            return Err("monitor worker PID changed".into());
        }
        // Revalidate SMI indexing each sample: device reorder/disappearance fails.
        if smi_device(&self.smi("list")?, &device.pci_address)? != device.smi_device {
            return Err("SMI physical device mapping changed".into());
        }
        Ok(process_ids(&self.smi("process")?, device.smi_device)?
            .into_iter()
            .filter(|&pid| pid != worker_pid)
            .collect())
    }
}
