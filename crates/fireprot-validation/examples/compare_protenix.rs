//! Apply the established Protenix coordinate/confidence gates to saved arrays.
use fireprot_validation::{metrics, numpy::read_npz};
use std::{fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: compare_protenix REFERENCE_NPZ REFERENCE_CONFIDENCE_JSON CANDIDATE_NPZ CANDIDATE_CONFIDENCE_JSON".into());
    }
    let r = read_npz(Path::new(&args[0]), 512 * 1024 * 1024)?;
    let c = read_npz(Path::new(&args[2]), 512 * 1024 * 1024)?;
    let coords = metrics::coordinates(
        &r.get("coordinates")
            .ok_or("missing reference coordinates")?
            .coordinates()?,
        &c.get("coordinates")
            .ok_or("missing candidate coordinates")?
            .coordinates()?,
    )?;
    let confidence = metrics::confidence(
        &serde_json::from_slice(&fs::read(&args[1])?)?,
        &serde_json::from_slice(&fs::read(&args[3])?)?,
    )?;
    let passed = metrics::protenix_passes(&coords, &confidence, 1e-3, 1e-3)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"coordinates":coords,
        "confidence":confidence,"passed":passed,"coordinate_tolerance_angstrom":1e-3,"confidence_tolerance":1e-3}))?
    );
    if !passed {
        return Err("Protenix prediction comparison failed".into());
    }
    Ok(())
}
