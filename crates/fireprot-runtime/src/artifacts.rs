use fireprot_core::request::ArtifactRef;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
    path::{Component, Path, PathBuf},
};

/// Stream large checkpoints without retaining them in memory.
pub fn sha256(path: &Path) -> io::Result<String> {
    if !std::fs::metadata(path)?.is_file() {
        return Err(io::Error::other("artifact is not a regular file"));
    }
    let mut file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::other("artifact is not a regular file"));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Verify relative output artifacts after the worker process group has stopped.
/// Content/schema validation is a separate mandatory supervisor callback.
pub fn verify_outputs(root: &Path, artifacts: &[ArtifactRef]) -> io::Result<Vec<PathBuf>> {
    let root = root.canonicalize()?;
    artifacts
        .iter()
        .map(|artifact| {
            if artifact.path.as_os_str().is_empty()
                || !artifact
                    .path
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
            {
                return Err(io::Error::other("invalid relative artifact path"));
            }
            let path = root.join(&artifact.path).canonicalize()?;
            if !path.starts_with(&root) || sha256(&path)? != artifact.sha256 {
                return Err(io::Error::other("artifact containment or SHA-256 mismatch"));
            }
            Ok(path)
        })
        .collect()
}
