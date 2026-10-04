//! Shared CLI I/O and cancellation wiring; domain policy stays in library crates.
use fireprot_core::{Request, ValidatedRequest};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)
        .map_err(|e| format!("invalid JSON in {}: {e}", path.display()))?)
}

pub fn read_request(path: &Path) -> Result<ValidatedRequest> {
    let request: Request = read_json(path)?;
    Ok(request.try_into()?)
}

pub fn print_json(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub fn cancellation() -> Result<Arc<AtomicBool>> {
    let cancelled = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&cancelled))?;
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&cancelled))?;
    Ok(cancelled)
}

pub fn check_build_cancelled(cancelled: &AtomicBool) -> std::io::Result<()> {
    if cancelled.load(Ordering::Relaxed) {
        Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "build cancelled",
        ))
    } else {
        Ok(())
    }
}
