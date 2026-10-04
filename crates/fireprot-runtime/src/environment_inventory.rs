//! Pinned content inventories for explicitly declared model environments.
//! Completeness depends on the selected roots; discovery and isolation are separate.
mod scan;
use crate::{
    artifacts::sha256,
    build::{BuildFile, FileIdentity},
    build_tree::BuildTree,
};
use fireprot_core::request::ArtifactRef;
use scan::ContentScan;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs, io,
    path::{Component, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventorySpec {
    pub schema_version: u32,
    pub files: Vec<BuildFile>,
    pub trees: Vec<BuildTree>,
    /// Explicit optional loader inputs that must remain absent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub absent_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvironmentInventory {
    pub specification: InventorySpec,
    pub files: Vec<FileIdentity>,
}

impl InventorySpec {
    pub fn capture(&self) -> io::Result<EnvironmentInventory> {
        if self.schema_version != 1 || (self.files.is_empty() && self.trees.is_empty()) {
            return Err(io::Error::other(
                "environment inventory requires schema 1 and declared inputs",
            ));
        }
        check_absent(&self.absent_paths)?;
        let mut files = Vec::new();
        let mut scan = ContentScan::new(sha256);
        for input in &self.files {
            if !input.path.is_absolute() {
                return Err(io::Error::other("environment input path must be absolute"));
            }
            let canonical_path = input.path.canonicalize()?;
            let sha256 = scan.hash(&canonical_path)?;
            files.push(FileIdentity {
                name: input.name.clone(),
                canonical_path,
                sha256,
            });
        }
        for tree in &self.trees {
            files.extend(tree.identify_with(&mut |path| scan.hash(path))?);
        }
        files.sort_by(|a, b| a.name.cmp(&b.name));
        let mut names = BTreeSet::new();
        if files.is_empty()
            || files
                .iter()
                .any(|f| f.name.is_empty() || !names.insert(&f.name))
        {
            return Err(io::Error::other(
                "environment inventory has no files or duplicate/empty names",
            ));
        }
        scan.finish()?;
        check_absent(&self.absent_paths)?;
        Ok(EnvironmentInventory {
            specification: self.clone(),
            files,
        })
    }
}

/// Read exactly the hashed bytes, then compare the current declared content and
/// topology. Called again by execution verification before Start and acceptance.
pub fn accept(pin: &ArtifactRef) -> io::Result<EnvironmentInventory> {
    if !pin.path.is_absolute() || fs::metadata(&pin.path)?.len() > 256 * 1024 * 1024 {
        return Err(io::Error::other(
            "invalid environment inventory path or size",
        ));
    }
    let bytes = fs::read(&pin.path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != pin.sha256 {
        return Err(io::Error::other("environment inventory digest mismatch"));
    }
    let expected: EnvironmentInventory =
        serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    let current = expected.specification.capture()?;
    if current.files != expected.files {
        return Err(io::Error::other(
            "environment files or topology differ from pinned inventory",
        ));
    }
    Ok(expected)
}

/// Observe absence without following a leaf symlink (including a dangling one).
/// Parents must exist so missing ancestors cannot silently broaden the claim.
fn check_absent(paths: &[PathBuf]) -> io::Result<()> {
    let mut seen = BTreeSet::new();
    for path in paths {
        if !path.is_absolute()
            || !path
                .components()
                .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
            || path.file_name().is_none()
            || !seen.insert(path)
        {
            return Err(io::Error::other(
                "invalid or duplicate absent environment path",
            ));
        }
        // Remove redundant separators before lstat; a trailing slash must not
        // turn a dangling leaf symlink into a false absence observation.
        let normalized: PathBuf = path.components().collect();
        let parent = normalized
            .parent()
            .ok_or_else(|| io::Error::other("absent path has no parent"))?;
        if !fs::metadata(parent)?.is_dir() {
            return Err(io::Error::other(
                "absent environment path parent is not a directory",
            ));
        }
        match fs::symlink_metadata(&normalized) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
            Ok(_) => {
                return Err(io::Error::other(format!(
                    "expected environment path to remain absent: {}",
                    normalized.display()
                )));
            }
        }
    }
    Ok(())
}
