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
    pub resources: BTreeMap<String, VerifiedArtifact>,
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

fn verify_source(
    root: &Path,
    commit: &str,
    origin: SourceOrigin,
) -> Result<(SourceManifest, Option<VerifiedArtifact>), String> {
    match origin {
        SourceOrigin::GitCheckout => Ok((export_source(root, commit)?, None)),
        SourceOrigin::Exported { manifest } => {
            let verified = verify_artifact(&manifest)?;
            // Verify the exact bytes parsed, not a second unverified file read.
            let bytes = fs::read(&verified.path).map_err(|e| e.to_string())?;
            if digest(&bytes) != verified.sha256 {
                return Err("source manifest changed".into());
            }
            let source: SourceManifest =
                serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            if source.schema_version != 1 || source.commit != commit || source.files.is_empty() {
                return Err("source manifest version, commit or file list is invalid".into());
            }
            for (path, expected) in &source.files {
                let (bytes, executable) = file_bytes(root, path)?;
                if digest(&bytes) != expected.sha256 || executable != expected.executable {
                    return Err(format!("exported source differs: {}", path.display()));
                }
            }
            Ok((source, Some(verified)))
        }
    }
}

pub fn prepare(request: ValidatedRequest, origin: SourceOrigin) -> Result<PreparedRequest, String> {
    prepare_with_hash_threads(request, origin, 1)
}

/// Verify independent assets concurrently without caching or omitting any bytes.
/// All verification threads finish before a prepared request can be returned.
pub fn prepare_with_hash_threads(
    request: ValidatedRequest,
    origin: SourceOrigin,
    hash_threads: usize,
) -> Result<PreparedRequest, String> {
    prepare_with_options(request, origin, hash_threads, false)
}

/// Overlap source verification with independent input, checkpoint and resource hashing.
/// Both results are joined and checked before the request can be prepared.
pub fn prepare_with_concurrent_source(
    request: ValidatedRequest,
    origin: SourceOrigin,
    hash_threads: usize,
) -> Result<PreparedRequest, String> {
    prepare_with_options(request, origin, hash_threads, true)
}

fn prepare_with_options(
    request: ValidatedRequest,
    origin: SourceOrigin,
    hash_threads: usize,
    concurrent_source: bool,
) -> Result<PreparedRequest, String> {
    if !(1..=8).contains(&hash_threads) {
        return Err("artifact verification requires 1..=8 hash threads".into());
    }
    let mut r = request.into_request();
    let root = r.source.root.canonicalize().map_err(|e| e.to_string())?;
    let artifacts: Vec<_> = std::iter::once(&r.input)
        .chain(r.checkpoints.values())
        .chain(r.resources.values())
        .collect();
    let (source_result, artifacts_result) = if concurrent_source {
        std::thread::scope(|scope| {
            let source = scope.spawn(|| verify_source(&root, &r.source.commit, origin));
            let artifacts = verify_artifacts(&artifacts, hash_threads);
            let source = source
                .join()
                .map_err(|_| "source verification thread panicked".to_string())
                .and_then(|result| result);
            (source, artifacts)
        })
    } else {
        (
            verify_source(&root, &r.source.commit, origin),
            verify_artifacts(&artifacts, hash_threads),
        )
    };
    let (source, exported_manifest) = source_result?;
    let mut verified = artifacts_result?.into_iter();
    let input = verified.next().ok_or("missing input verification")?;
    let checkpoints: BTreeMap<_, _> = r
        .checkpoints
        .keys()
        .copied()
        .zip(verified.by_ref().take(r.checkpoints.len()))
        .collect();
    let resources: BTreeMap<_, _> = r.resources.keys().cloned().zip(verified).collect();
    r.source.root = root.clone();
    r.input.path = input.path.clone();
    for (role, artifact) in &mut r.checkpoints {
        artifact.path = checkpoints[role].path.clone();
    }
    for (name, artifact) in &mut r.resources {
        artifact.path = resources[name].path.clone();
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
            resources,
        },
    })
}

fn verify_artifacts(
    artifacts: &[&ArtifactRef],
    threads: usize,
) -> Result<Vec<VerifiedArtifact>, String> {
    if threads == 1 {
        return artifacts
            .iter()
            .map(|artifact| verify_artifact(artifact))
            .collect();
    }
    let workers = threads.min(artifacts.len());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                scope.spawn(move || {
                    artifacts
                        .iter()
                        .enumerate()
                        .skip(worker)
                        .step_by(workers)
                        .map(|(index, artifact)| (index, verify_artifact(artifact)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut results = Vec::with_capacity(artifacts.len());
        let mut panicked = false;
        for handle in handles {
            match handle.join() {
                Ok(items) => results.extend(items),
                Err(_) => panicked = true,
            }
        }
        if panicked {
            return Err("artifact verification thread panicked".into());
        }
        results.sort_unstable_by_key(|(index, _)| *index);
        results.into_iter().map(|(_, result)| result).collect()
    })
}
