//! Revalidate saved prediction arrays with exact shape/dtype/value-byte equality.
use fireprot_validation::{
    array::Policy,
    numpy::{compare_archives, read_npz},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: compare_npz REFERENCE_NPZ CANDIDATE_NPZ".into());
    }
    let r = read_npz(std::path::Path::new(&args[1]), 512 * 1024 * 1024)?;
    let c = read_npz(std::path::Path::new(&args[2]), 512 * 1024 * 1024)?;
    let comparison = compare_archives(&r, &c, Policy::Exact)?;
    println!("{}", serde_json::to_string_pretty(&comparison)?);
    if comparison.values().any(|x| !x.passed) {
        return Err("prediction comparison failed".into());
    }
    Ok(())
}
