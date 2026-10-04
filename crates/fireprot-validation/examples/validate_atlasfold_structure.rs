//! Diagnostic only: the supplied sequence is not authenticated to a run request.
use fireprot_validation::{atlasfold::validate_atom14, numpy, structure};
use std::{env, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        return Err("usage: validate_atlasfold_structure SEQUENCE ARRAY_NPZ PREDICTION_CIF".into());
    }
    let arrays = numpy::read_npz(Path::new(&args[2]), 512 * 1024 * 1024)?;
    let coordinates = arrays.get("coordinates").ok_or("missing coordinates")?;
    let structure = structure::read(Path::new(&args[3]))?;
    validate_atom14(&args[1], coordinates, &structure)?;
    println!(
        "{}",
        serde_json::json!({"residues":args[1].len(), "atoms":structure.atoms.len(), "passed":true, "sequence_authenticated":false})
    );
    Ok(())
}
