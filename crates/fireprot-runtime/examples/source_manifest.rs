//! Export pinned source for deployments that do not transfer Git metadata.
use fireprot_runtime::preflight::export_source;
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: source_manifest SOURCE_ROOT COMMIT".into());
    }
    let manifest = export_source(std::path::Path::new(&args[1]), &args[2])?;
    println!(
        "{}",
        serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?
    );
    Ok(())
}
