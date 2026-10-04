//! Inspect a request's actual source, input and checkpoint provenance.
use fireprot_core::{Request, ValidatedRequest, request::ArtifactRef};
use fireprot_runtime::preflight::{SourceOrigin, prepare};
fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if !matches!(args.len(), 2 | 4) {
        return Err("usage: preflight REQUEST_JSON [SOURCE_MANIFEST SHA256]".into());
    }
    let request: Request =
        serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let origin = if args.len() == 4 {
        SourceOrigin::Exported {
            manifest: ArtifactRef {
                path: (&args[2]).into(),
                sha256: args[3].clone(),
            },
        }
    } else {
        SourceOrigin::GitCheckout
    };
    let prepared = prepare(
        ValidatedRequest::try_from(request).map_err(|e| e.to_string())?,
        origin,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(prepared.provenance()).map_err(|e| e.to_string())?
    );
    Ok(())
}
