//! Transactional cache for native libraries on a local Unix filesystem.
//! The cache root is owned by the current user, not a hostile shared directory.
use crate::{artifacts::sha256, build::BuildIdentity};
use nix::fcntl::{Flock, FlockArg, OFlag};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Component, Path, PathBuf},
};

const RECEIPT: &str = "fireprot-build.json";

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildReceipt {
    pub schema_version: u32,
    pub identity: BuildIdentity,
    pub artifacts: BTreeMap<String, String>,
}

#[derive(Debug)]
pub struct CachedBuild {
    pub directory: PathBuf,
    pub receipt: BuildReceipt,
    pub cache_hit: bool,
}

fn artifact_path(root: &Path, name: &str) -> io::Result<PathBuf> {
    let path = Path::new(name);
    if name == RECEIPT
        || path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(io::Error::other(
            "artifact must be a single relative filename",
        ));
    }
    let path = root.join(path);
    if !fs::symlink_metadata(&path)?.file_type().is_file() {
        return Err(io::Error::other(
            "build artifact is not a regular non-symlink file",
        ));
    }
    Ok(path)
}

fn read_cached(directory: &Path, identity: &BuildIdentity) -> io::Result<BuildReceipt> {
    if !fs::symlink_metadata(directory)?.file_type().is_dir() {
        return Err(io::Error::other(
            "cache entry is not a non-symlink directory",
        ));
    }
    let path = directory.join(RECEIPT);
    let metadata = fs::symlink_metadata(&path)?;
    if !metadata.file_type().is_file() || metadata.len() > 64 * 1024 * 1024 {
        return Err(io::Error::other("invalid build receipt file"));
    }
    let receipt: BuildReceipt = serde_json::from_reader(File::open(path)?)?;
    if receipt.schema_version != 1 || &receipt.identity != identity || receipt.artifacts.is_empty()
    {
        return Err(io::Error::other("build receipt identity/schema mismatch"));
    }
    identity.verify()?;
    for (name, expected) in &receipt.artifacts {
        if sha256(&artifact_path(directory, name)?)? != *expected {
            return Err(io::Error::other("cached build artifact SHA-256 mismatch"));
        }
    }
    Ok(receipt)
}

/// Serialize builders for an identity and publish only verified successful output.
///
/// The callback must return after all compiler children have exited and closed
/// outputs. Returned names are flat library filenames, not intermediate files.
/// It must not embed its temporary directory as a runtime dependency. Corrupted
/// published entries fail closed; this function never deletes an existing entry.
/// SIGKILL may leave an ignored temporary directory, but releases the OS lock.
/// Inputs are checked before and after compilation; immutable source snapshots
/// are still required to defend against inputs changed and restored mid-build.
pub fn get_or_build<F>(root: &Path, identity: &BuildIdentity, build: F) -> io::Result<CachedBuild>
where
    F: FnOnce(&Path) -> io::Result<Vec<String>>,
{
    identity.verify()?;
    let name = identity.cache_name()?;
    fs::create_dir_all(root)?;
    let root = root.canonicalize()?;
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(OFlag::O_NOFOLLOW.bits())
        .open(root.join(format!("{name}.lock")))?;
    if !lock_file.metadata()?.is_file() {
        return Err(io::Error::other("invalid cache lock file"));
    }
    // Never unlink lock files: waiters must all continue locking the same inode.
    let _lock = Flock::lock(lock_file, FlockArg::LockExclusive)
        .map_err(|(_, error)| io::Error::from_raw_os_error(error as i32))?;
    identity.verify()?;
    let directory = root.join(&name);
    match fs::symlink_metadata(&directory) {
        Ok(_) => {
            return Ok(CachedBuild {
                receipt: read_cached(&directory, identity)?,
                directory,
                cache_hit: true,
            });
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => (),
        Err(error) => return Err(error),
    }
    let staging = tempfile::Builder::new()
        .prefix(".building-")
        .tempdir_in(&root)?;
    let names = build(staging.path())?;
    if names.is_empty() {
        return Err(io::Error::other("build returned no artifacts"));
    }
    let mut artifacts = BTreeMap::new();
    for name in names {
        let path = artifact_path(staging.path(), &name)?;
        File::open(&path)?.sync_all()?;
        if artifacts.insert(name, sha256(&path)?).is_some() {
            return Err(io::Error::other("duplicate build artifact"));
        }
    }
    identity.verify()?;
    let receipt = BuildReceipt {
        schema_version: 1,
        identity: identity.clone(),
        artifacts,
    };
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging.path().join(RECEIPT))?;
    file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    file.sync_all()?;
    File::open(staging.path())?.sync_all()?;
    // Under the identity lock, no cooperating writer can create this destination.
    fs::rename(staging.path(), &directory)?;
    File::open(&root)?.sync_all()?;
    Ok(CachedBuild {
        directory,
        receipt,
        cache_hit: false,
    })
}
