//! Source and artifact verification before a worker can be launched.
use crate::{artifacts::sha256, command::output};
use fireprot_core::{
    ValidatedRequest,
    request::{ArtifactRef, CheckpointRole},
};
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Component, Path, PathBuf},
    process::Command,
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceFile {
    pub sha256: String,
    pub executable: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceManifest {
    pub schema_version: u32,
    pub commit: String,
    pub files: BTreeMap<PathBuf, SourceFile>,
}

/// Exported source is authorized by a separately pinned manifest hash. Remote
/// Git metadata can stay untouched; each listed file is checked against bytes
/// exported from the local pinned Git commit.
pub enum SourceOrigin {
    GitCheckout,
    Exported { manifest: ArtifactRef },
}
#[derive(Debug, Clone, Serialize)]
pub struct VerifiedArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct Provenance {
    pub source_root: PathBuf,
    pub source: SourceManifest,
    pub exported_manifest: Option<VerifiedArtifact>,
    pub input: VerifiedArtifact,
    pub checkpoints: BTreeMap<CheckpointRole, VerifiedArtifact>,
}

/// Only `prepare` can construct this token. It is deliberately not deserializable.
pub struct PreparedRequest {
    request: ValidatedRequest,
    provenance: Provenance,
}
impl PreparedRequest {
    pub fn request(&self) -> &ValidatedRequest {
        &self.request
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }
}

fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0");
    output(&mut command, Duration::from_secs(30), 16 * 1024 * 1024).map_err(|e| e.to_string())
}
fn text(bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|e| e.to_string())
}
fn relative(path: &Path) -> bool {
    !path.as_os_str().is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))
}
fn file_bytes(root: &Path, path: &Path) -> Result<(Vec<u8>, bool), String> {
    if !relative(path) {
        return Err("invalid source path".into());
    }
    let full = root.join(path);
    let meta = fs::symlink_metadata(&full).map_err(|e| format!("{}: {e}", full.display()))?;
    if !meta.is_file()
        || !full
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(root)
    {
        return Err(format!(
            "source must be a regular contained file: {}",
            path.display()
        ));
    }
    Ok((
        fs::read(&full).map_err(|e| e.to_string())?,
        meta.permissions().mode() & 0o111 != 0,
    ))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

/// Export every regular tracked file after checking its Git blob hash. This
/// catches modifications hidden by assume-unchanged/skip-worktree flags too.
/// Symlinks and gitlinks are rejected until an explicit source policy supports them.
pub fn export_source(root: &Path, commit: &str) -> Result<SourceManifest, String> {
    let root = root.canonicalize().map_err(|e| e.to_string())?;
    if Path::new(text(git(&root, &["rev-parse", "--show-toplevel"])?)?.trim()) != root {
        return Err("source root is not the Git repository root".into());
    }
    if text(git(&root, &["rev-parse", "HEAD"])?)?.trim() != commit {
        return Err("source HEAD does not match the requested commit".into());
    }
    if !git(&root, &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err("tracked source checkout is dirty".into());
    }
    let tree = git(&root, &["ls-tree", "-r", "-z", "--full-tree", "HEAD"])?;
    let mut files = BTreeMap::new();
    for entry in tree.split(|&b| b == 0).filter(|entry| !entry.is_empty()) {
        let entry = std::str::from_utf8(entry).map_err(|_| "source paths must be UTF-8")?;
        let (header, path) = entry.split_once('\t').ok_or("malformed Git tree")?;
        let fields: Vec<_> = header.split(' ').collect();
        if fields.len() != 3 || fields[1] != "blob" || !matches!(fields[0], "100644" | "100755") {
            return Err(format!("unsupported source entry: {header}"));
        }
        let path = PathBuf::from(path);
        let (bytes, executable) = file_bytes(&root, &path)?;
        let mut hash = Sha1::new();
        hash.update(format!("blob {}\0", bytes.len()));
        hash.update(&bytes);
        if format!("{:x}", hash.finalize()) != fields[2] || executable != (fields[0] == "100755") {
            return Err(format!(
                "source bytes/mode differ from commit: {}",
                path.display()
            ));
        }
        files.insert(
            path,
            SourceFile {
                sha256: digest(&bytes),
                executable,
            },
        );
    }
    if files.is_empty() {
        return Err("empty source tree".into());
    }
    // Recheck the commit/index after reading the tree.
    if text(git(&root, &["rev-parse", "HEAD"])?)?.trim() != commit
        || !git(&root, &["status", "--porcelain", "--untracked-files=no"])?.is_empty()
    {
        return Err("source changed during export".into());
    }
    Ok(SourceManifest {
        schema_version: 1,
        commit: commit.into(),
        files,
    })
}

fn verify_artifact(artifact: &ArtifactRef) -> Result<VerifiedArtifact, String> {
    let path = artifact
        .path
        .canonicalize()
        .map_err(|e| format!("{}: {e}", artifact.path.display()))?;
    let before = fs::metadata(&path).map_err(|e| e.to_string())?;
    let hash = sha256(&path).map_err(|e| e.to_string())?;
    let after = fs::metadata(&path).map_err(|e| e.to_string())?;
    if hash != artifact.sha256
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Err(format!(
            "artifact digest mismatch or changed during verification: {}",
            path.display()
        ));
    }
    Ok(VerifiedArtifact {
        path,
        sha256: hash,
        bytes: after.len(),
    })
}

pub fn prepare(request: ValidatedRequest, origin: SourceOrigin) -> Result<PreparedRequest, String> {
    let mut r = request.into_request();
    let root = r.source.root.canonicalize().map_err(|e| e.to_string())?;
    let (source, exported_manifest) = match origin {
        SourceOrigin::GitCheckout => (export_source(&root, &r.source.commit)?, None),
        SourceOrigin::Exported { manifest } => {
            let verified = verify_artifact(&manifest)?;
            // Verify the exact bytes parsed, not a second unverified file read.
            let bytes = fs::read(&verified.path).map_err(|e| e.to_string())?;
            if digest(&bytes) != verified.sha256 {
                return Err("source manifest changed".into());
            }
            let m: SourceManifest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            if m.schema_version != 1 || m.commit != r.source.commit || m.files.is_empty() {
                return Err("source manifest version, commit or file list is invalid".into());
            }
            for (path, expected) in &m.files {
                let (bytes, executable) = file_bytes(&root, path)?;
                if digest(&bytes) != expected.sha256 || executable != expected.executable {
                    return Err(format!("exported source differs: {}", path.display()));
                }
            }
            (m, Some(verified))
        }
    };
    let input = verify_artifact(&r.input)?;
    let mut checkpoints = BTreeMap::new();
    for (role, artifact) in &r.checkpoints {
        checkpoints.insert(*role, verify_artifact(artifact)?);
    }
    r.source.root = root.clone();
    r.input.path = input.path.clone();
    for (role, artifact) in &mut r.checkpoints {
        artifact.path = checkpoints[role].path.clone();
    }
    // Absolute paths survive changing the worker's current directory.
    if !r.output_dir.is_absolute() {
        r.output_dir = std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(&r.output_dir);
    }
    let request = ValidatedRequest::try_from(r).map_err(|e| e.to_string())?;
    Ok(PreparedRequest {
        request,
        provenance: Provenance {
            source_root: root,
            source,
            exported_manifest,
            input,
            checkpoints,
        },
    })
}
