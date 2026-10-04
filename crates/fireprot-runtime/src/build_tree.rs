//! Deterministic inventories, including linked directories and topology changes.
use crate::artifacts::sha256;
use crate::build::FileIdentity;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs, io,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BuildTree {
    pub name: String,
    pub path: PathBuf,
    /// Empty means every file. Otherwise match filename suffixes exactly.
    #[serde(default)]
    pub suffixes: Vec<String>,
}

impl BuildTree {
    pub fn identify(&self) -> io::Result<Vec<FileIdentity>> {
        self.identify_with(&mut sha256)
    }

    pub(crate) fn identify_with(
        &self,
        digest: &mut impl FnMut(&Path) -> io::Result<String>,
    ) -> io::Result<Vec<FileIdentity>> {
        if self.name.is_empty()
            || !self.path.is_absolute()
            || !self.path.is_dir()
            || self.suffixes.iter().any(String::is_empty)
        {
            return Err(io::Error::other("invalid build input tree"));
        }
        let mut files = Vec::new();
        walk(
            self,
            &self.path,
            &mut BTreeSet::new(),
            &mut files,
            0,
            digest,
        )?;
        Ok(files)
    }
}

fn walk(
    tree: &BuildTree,
    path: &Path,
    ancestors: &mut BTreeSet<PathBuf>,
    files: &mut Vec<FileIdentity>,
    depth: usize,
    digest: &mut impl FnMut(&Path) -> io::Result<String>,
) -> io::Result<()> {
    if depth > 128 {
        return Err(io::Error::other("build tree exceeds directory depth limit"));
    }
    let canonical = path.canonicalize()?;
    let metadata = fs::metadata(&canonical)?;
    if metadata.is_dir() {
        if !ancestors.insert(canonical.clone()) {
            return Err(io::Error::other("cycle in build input tree"));
        }
        let mut entries: Vec<_> = fs::read_dir(path)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<io::Result<_>>()?;
        entries.sort();
        for entry in entries {
            walk(tree, &entry, ancestors, files, depth + 1, digest)?;
        }
        ancestors.remove(&canonical);
    } else if metadata.is_file() {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| io::Error::other("non-UTF8 build input filename"))?;
        if !tree.suffixes.is_empty() && !tree.suffixes.iter().any(|s| filename.ends_with(s)) {
            return Ok(());
        }
        let relative = path.strip_prefix(&tree.path).map_err(io::Error::other)?;
        let relative = relative
            .to_str()
            .ok_or_else(|| io::Error::other("non-UTF8 build input path"))?;
        files.push(FileIdentity {
            name: format!("{}/{}", tree.name, relative),
            sha256: digest(&canonical)?,
            canonical_path: canonical,
        });
    } else {
        return Err(io::Error::other("non-regular build input"));
    }
    Ok(())
}
