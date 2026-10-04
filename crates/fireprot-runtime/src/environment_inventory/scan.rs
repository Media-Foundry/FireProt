//! Hash canonical files once per capture, never across acceptance checkpoints.
#[cfg(unix)]
use std::{collections::BTreeMap, fs, os::unix::fs::MetadataExt, path::PathBuf};
use std::{io, path::Path};

#[cfg(unix)]
#[derive(Debug, PartialEq, Eq)]
struct Stamp {
    device: u64,
    inode: u64,
    size: u64,
    mode: u32,
    modified: (i64, i64),
    changed: (i64, i64),
}
#[cfg(unix)]
fn stamp(path: &Path) -> io::Result<Stamp> {
    let m = fs::metadata(path)?;
    if !m.is_file() {
        return Err(io::Error::other("environment input is not a regular file"));
    }
    Ok(Stamp {
        device: m.dev(),
        inode: m.ino(),
        size: m.len(),
        mode: m.mode(),
        modified: (m.mtime(), m.mtime_nsec()),
        changed: (m.ctime(), m.ctime_nsec()),
    })
}

pub(super) struct ContentScan<F> {
    digest: F,
    #[cfg(unix)]
    files: BTreeMap<PathBuf, (Stamp, String)>,
}
impl<F: FnMut(&Path) -> io::Result<String>> ContentScan<F> {
    pub(super) fn new(digest: F) -> Self {
        Self {
            digest,
            #[cfg(unix)]
            files: BTreeMap::new(),
        }
    }
    /// The inventory walker supplies canonical paths and retains every alias in
    /// its output. Unix metadata guards reuse; other platforms always rehash.
    pub(super) fn hash(&mut self, canonical: &Path) -> io::Result<String> {
        #[cfg(unix)]
        {
            let before = stamp(canonical)?;
            if let Some((previous, digest)) = self.files.get(canonical) {
                if previous != &before {
                    return Err(changed());
                }
                return Ok(digest.clone());
            }
            let digest = (self.digest)(canonical)?;
            if stamp(canonical)? != before {
                return Err(changed());
            }
            self.files
                .insert(canonical.into(), (before, digest.clone()));
            Ok(digest)
        }
        #[cfg(not(unix))]
        (self.digest)(canonical)
    }
    /// No scan-local digest survives this call. A new verification hashes again.
    pub(super) fn finish(self) -> io::Result<()> {
        #[cfg(unix)]
        for (path, (previous, _)) in self.files {
            if stamp(&path)? != previous {
                return Err(changed());
            }
        }
        Ok(())
    }
}
#[cfg(unix)]
fn changed() -> io::Error {
    io::Error::other("environment file changed during inventory capture")
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::artifacts::sha256;
    use std::{cell::Cell, fs::FileTimes};

    #[test]
    fn aliases_share_one_read_but_new_captures_rehash_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("module");
        fs::write(&path, b"before").unwrap();
        let calls = Cell::new(0);
        let digest = |p: &Path| {
            calls.set(calls.get() + 1);
            sha256(p)
        };
        let mut scan = ContentScan::new(digest);
        let first = scan.hash(&path).unwrap();
        assert_eq!(first, scan.hash(&path).unwrap());
        scan.finish().unwrap();
        assert_eq!(calls.get(), 1);
        fs::write(&path, b"after!").unwrap();
        let mut fresh = ContentScan::new(digest);
        assert_ne!(first, fresh.hash(&path).unwrap());
        fresh.finish().unwrap();
        assert_eq!(calls.get(), 2);
    }
    #[test]
    fn same_size_rewrite_with_restored_mtime_cannot_reuse_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("module");
        fs::write(&path, b"before").unwrap();
        let modified = fs::metadata(&path).unwrap().modified().unwrap();
        let mut scan = ContentScan::new(sha256);
        scan.hash(&path).unwrap();
        fs::write(&path, b"after!").unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(FileTimes::new().set_modified(modified))
            .unwrap();
        assert!(scan.hash(&path).is_err());
        assert!(scan.finish().is_err());
    }
    #[test]
    fn replacement_during_hash_or_after_last_alias_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("module");
        let replacement = dir.path().join("replacement");
        fs::write(&path, b"before").unwrap();
        fs::write(&replacement, b"before").unwrap();
        let mut scan = ContentScan::new(|p: &Path| {
            let digest = sha256(p)?;
            fs::rename(&replacement, p)?;
            Ok(digest)
        });
        assert!(scan.hash(&path).is_err());
        let mut scan = ContentScan::new(sha256);
        scan.hash(&path).unwrap();
        fs::write(&replacement, b"before").unwrap();
        fs::rename(&replacement, &path).unwrap();
        assert!(scan.finish().is_err());
    }
}
