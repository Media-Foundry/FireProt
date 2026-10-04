//! Historical native captures include batch/sample axes [1,1,...]. A migrated
//! single-sample archive removes only those axes; tensor values are untouched.
use fireprot_validation::{
    array::{Array, Policy},
    numpy::{compare_archives, read_npz},
};
use std::collections::BTreeMap;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: compare_openfold3_capture LEGACY_NPZ SAMPLE_NPZ".into());
    }
    let r = read_npz(std::path::Path::new(&args[1]), 512 * 1024 * 1024)?;
    let mut normalized = BTreeMap::new();
    for (name, array) in r {
        if array.shape().len() < 2 || array.shape()[..2] != [1, 1] {
            return Err(format!("unexpected legacy sample axes: {name}").into());
        }
        normalized.insert(
            name,
            Array::new(
                array.dtype(),
                array.shape()[2..].to_vec(),
                array.bytes().to_vec(),
            )?,
        );
    }
    let c = read_npz(std::path::Path::new(&args[2]), 512 * 1024 * 1024)?;
    let result = compare_archives(&normalized, &c, Policy::Exact)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if result.values().any(|r| !r.passed) {
        return Err("migrated output differs from legacy capture".into());
    }
    Ok(())
}
