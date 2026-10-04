//! Pinned content inventories for explicitly declared model environments.
//! Completeness depends on the selected roots; discovery and isolation are separate.
use crate::{
    artifacts::sha256,
    build::{BuildFile, FileIdentity},
    build_tree::BuildTree,
};
use fireprot_core::request::ArtifactRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventorySpec {
    pub schema_version: u32,
    pub files: Vec<BuildFile>,
    pub trees: Vec<BuildTree>,
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
        let mut files = Vec::new();
        for input in &self.files {
            if !input.path.is_absolute() {
                return Err(io::Error::other("environment input path must be absolute"));
            }
            files.push(FileIdentity {
                name: input.name.clone(),
                canonical_path: input.path.canonicalize()?,
                sha256: sha256(&input.path)?,
            });
        }
        for tree in &self.trees {
            files.extend(tree.identify()?);
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
