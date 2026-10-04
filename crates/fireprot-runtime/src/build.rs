//! Content identity for native kernel builds. This module never invokes a compiler.
//!
//! Callers must enumerate every effective source/header/tool input and environment
//! value. An identity is a reproducibility record, not proof that an input list is
//! complete or that a compiler is deterministic.
use crate::artifacts::sha256;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildFile {
    /// Stable name including the role, e.g. `source/pair.cu` or `tool/hipcc`.
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildSpec {
    pub schema_version: u32,
    pub package: String,
    pub target: String,
    /// Ordered translation units, headers, compiler executables and build helpers.
    pub files: Vec<BuildFile>,
    /// Exact ordered arguments for each compiler/linker stage.
    pub arguments: BTreeMap<String, Vec<String>>,
    /// Effective values, including explicit absence for relevant unset variables.
    pub environment: BTreeMap<String, Option<String>>,
    /// Torch, HIP, Python ABI and recipe versions; must be explicit at the caller.
    pub versions: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub name: String,
    pub canonical_path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    pub schema_version: u32,
    pub package: String,
    pub target: String,
    pub files: Vec<FileIdentity>,
    pub arguments: BTreeMap<String, Vec<String>>,
    pub environment: BTreeMap<String, Option<String>>,
    pub versions: BTreeMap<String, String>,
    pub sha256: String,
}

fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
}

impl BuildSpec {
    /// Hash file contents in Rust. Absolute paths deliberately participate in the
    /// identity: include search roots and compiler paths can affect generated code.
    pub fn identify(&self) -> io::Result<BuildIdentity> {
        if self.schema_version != 1
            || !identifier(&self.package)
            || !identifier(&self.target)
            || self.files.is_empty()
            || self.versions.is_empty()
        {
            return Err(io::Error::other(
                "invalid or incomplete build specification",
            ));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut files = Vec::new();
        for input in &self.files {
            if input.name.is_empty() || !names.insert(&input.name) {
                return Err(io::Error::other("empty or duplicate build input name"));
            }
            let canonical_path = input.path.canonicalize()?;
            let digest = sha256(&canonical_path)?;
            files.push(FileIdentity {
                name: input.name.clone(),
                canonical_path,
                sha256: digest,
            });
        }
        let mut result = BuildIdentity {
            schema_version: 1,
            package: self.package.clone(),
            target: self.target.clone(),
            files,
            arguments: self.arguments.clone(),
            environment: self.environment.clone(),
            versions: self.versions.clone(),
            sha256: String::new(),
        };
        result.sha256 = result.compute_digest()?;
        Ok(result)
    }
}

impl BuildIdentity {
    fn compute_digest(&self) -> io::Result<String> {
        // Fixed struct field order and BTreeMap key order define schema 1 encoding.
        // Hash framed JSON, never ambiguous concatenated source/version strings.
        let mut value = self.clone();
        value.sha256.clear();
        let bytes = serde_json::to_vec(&value)?;
        let mut digest = Sha256::new();
        digest.update(b"fireprot-native-build-v1\0");
        digest.update(bytes);
        Ok(format!("{:x}", digest.finalize()))
    }

    /// Validate a persisted receipt and rehash all inputs before reusing a build.
    pub fn verify(&self) -> io::Result<()> {
        let spec = BuildSpec {
            schema_version: self.schema_version,
            package: self.package.clone(),
            target: self.target.clone(),
            files: self
                .files
                .iter()
                .map(|f| BuildFile {
                    name: f.name.clone(),
                    path: f.canonical_path.clone(),
                })
                .collect(),
            arguments: self.arguments.clone(),
            environment: self.environment.clone(),
            versions: self.versions.clone(),
        };
        if &spec.identify()? != self {
            return Err(io::Error::other("build inputs or receipt changed"));
        }
        Ok(())
    }

    /// Full digest avoids truncated-name collisions. Verification is mandatory
    /// even when a directory with this name already exists.
    pub fn cache_name(&self) -> io::Result<String> {
        if !identifier(&self.package) || self.compute_digest()? != self.sha256 {
            return Err(io::Error::other("invalid build receipt"));
        }
        Ok(format!("{}_{}", self.package, self.sha256))
    }
}
