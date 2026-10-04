//! Publish verified build output, then accept pinned artifacts without a compiler.
//! Bundle integrity and declared compatibility do not establish numerical quality.
use crate::{
    artifacts::sha256,
    build_cache::{BuildReceipt, get_or_build},
};
use fireprot_core::request::ArtifactRef;
use nix::fcntl::{Flock, FlockArg, OFlag};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Component, Path, PathBuf},
};

const MANIFEST: &str = "kernel.json";
const RECEIPT: &str = "build-receipt.json";

/// Untagged only to preserve schema-1 Python deployment JSON. The required ABI
/// field and denied unknown fields make the two variants mutually exclusive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged, deny_unknown_fields)]
pub enum KernelCompatibility {
    Python {
        package: String,
        target: String,
        torch: String,
        hip: String,
        python_soabi: String,
    },
    C {
        package: String,
        target: String,
        torch: String,
        hip: String,
        c_abi: String,
    },
}
impl KernelCompatibility {
    fn schema_version(&self) -> u32 {
        match self {
            Self::Python { .. } => 1,
            Self::C { .. } => 2,
        }
    }
    fn module_name(&self, name: &str) -> Option<String> {
        match self {
            Self::Python { .. } => Some(name.into()),
            Self::C { .. } => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelManifest {
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_name: Option<String>,
    pub compatibility: KernelCompatibility,
    pub build_identity_sha256: String,
    pub library: ArtifactRef,
    pub build_receipt: ArtifactRef,
}

#[derive(Debug, Serialize)]
pub struct PublishedKernel {
    pub directory: PathBuf,
    pub manifest_sha256: String,
    pub compatibility: KernelCompatibility,
}

/// Constructed only after hash/schema/compatibility checks. No build-input reads.
#[derive(Debug, Serialize)]
pub struct AcceptedKernel {
    directory: PathBuf,
    library: PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    module_name: Option<String>,
    manifest_sha256: String,
    build_identity_sha256: String,
    compatibility: KernelCompatibility,
}

fn flat_file(root: &Path, name: &Path) -> io::Result<PathBuf> {
    if name.components().count() != 1
        || !matches!(name.components().next(), Some(Component::Normal(_)))
    {
        return Err(io::Error::other(
            "bundle artifact must be a flat relative filename",
        ));
    }
    let path = root.join(name);
    if !fs::symlink_metadata(&path)?.file_type().is_file() {
        return Err(io::Error::other(
            "bundle artifact must be a regular non-symlink file",
        ));
    }
    Ok(path)
}

fn read_json<T: DeserializeOwned>(path: &Path, limit: u64) -> io::Result<T> {
    if fs::metadata(path)?.len() > limit {
        return Err(io::Error::other("bundle JSON exceeds size limit"));
    }
    serde_json::from_reader(File::open(path)?).map_err(io::Error::other)
}

fn compatibility(receipt: &BuildReceipt) -> io::Result<KernelCompatibility> {
    let version = |key| {
        receipt
            .identity
            .versions
            .get(key)
            .filter(|s| !s.is_empty())
            .cloned()
            .ok_or_else(|| io::Error::other(format!("missing build compatibility field: {key}")))
    };
    let package = receipt.identity.package.clone();
    let target = receipt.identity.target.clone();
    let torch = version("torch")?;
    let hip = version("hip")?;
    match receipt.identity.versions.get("abi").map(String::as_str) {
        Some("c") => {
            if receipt.identity.versions.contains_key("soabi") {
                return Err(io::Error::other(
                    "C ABI build cannot declare a Python SOABI",
                ));
            }
            Ok(KernelCompatibility::C {
                package,
                target,
                torch,
                hip,
                c_abi: version("c_abi")?,
            })
        }
        None | Some("python") => {
            if receipt.identity.versions.contains_key("c_abi") {
                return Err(io::Error::other(
                    "Python build cannot declare a C ABI revision",
                ));
            }
            Ok(KernelCompatibility::Python {
                package,
                target,
                torch,
                hip,
                python_soabi: version("soabi")?,
            })
        }
        Some(_) => Err(io::Error::other("unsupported kernel ABI")),
    }
}

fn module_name(receipt: &BuildReceipt) -> io::Result<String> {
    if receipt.schema_version != 1 || receipt.identity.schema_version != 1 {
        return Err(io::Error::other("unsupported kernel build schema"));
    }
    let name = receipt.identity.cache_name()?;
    if !name.as_bytes()[0].is_ascii_alphabetic() && !name.starts_with('_')
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(io::Error::other("invalid Python extension module name"));
    }
    Ok(name)
}

fn write_json(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()
}

/// Revalidate the complete build on its build host and copy one library to a fresh
/// destination. Published directories are immutable to cooperating writers.
pub fn publish(cache_directory: &Path, destination: &Path) -> io::Result<PublishedKernel> {
    let cache_directory = cache_directory.canonicalize()?;
    let input: BuildReceipt = read_json(
        &flat_file(&cache_directory, Path::new("fireprot-build.json"))?,
        64 * 1024 * 1024,
    )?;
    let name = module_name(&input)?;
    if cache_directory.file_name() != Some(std::ffi::OsStr::new(&name)) {
        return Err(io::Error::other(
            "cache directory does not match build identity",
        ));
    }
    let cached = get_or_build(
        cache_directory
            .parent()
            .ok_or_else(|| io::Error::other("missing cache parent"))?,
        &input.identity,
        |_| {
            Err(io::Error::other(
                "publishing requires an existing successful build",
            ))
        },
    )?;
    let receipt = cached.receipt;
    let compatibility = compatibility(&receipt)?;
    let library_name = format!("{name}.so");
    if receipt.artifacts.len() != 1 || !receipt.artifacts.contains_key(&library_name) {
        return Err(io::Error::other(
            "publishing requires one library named by its build identity",
        ));
    }
    let destination = std::path::absolute(destination)?;
    let parent = destination
        .parent()
        .ok_or_else(|| io::Error::other("missing destination parent"))?;
    fs::create_dir_all(parent)?;
    let parent = parent.canonicalize()?;
    let leaf = destination
        .file_name()
        .ok_or_else(|| io::Error::other("invalid bundle destination"))?;
    let destination = parent.join(leaf);
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(OFlag::O_NOFOLLOW.bits())
        .open(parent.join(format!(".{}.publish.lock", leaf.to_string_lossy())))?;
    let _lock = Flock::lock(lock, FlockArg::LockExclusive)
        .map_err(|(_, e)| io::Error::from_raw_os_error(e as i32))?;
    match fs::symlink_metadata(&destination) {
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "bundle destination already exists",
            ));
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => (),
        Err(e) => return Err(e),
    }
    let staging = tempfile::Builder::new()
        .prefix(".publishing-")
        .tempdir_in(&parent)?;
    let library = staging.path().join(&library_name);
    fs::copy(
        flat_file(&cache_directory, Path::new(&library_name))?,
        &library,
    )?;
    File::open(&library)?.sync_all()?;
    let library_hash = sha256(&library)?;
    if receipt.artifacts[&library_name] != library_hash {
        return Err(io::Error::other("library changed during publication"));
    }
    write_json(&staging.path().join(RECEIPT), &receipt)?;
    let manifest = KernelManifest {
        schema_version: compatibility.schema_version(),
        module_name: compatibility.module_name(&name),
        compatibility: compatibility.clone(),
        build_identity_sha256: receipt.identity.sha256.clone(),
        library: ArtifactRef {
            path: library_name.into(),
            sha256: library_hash,
        },
        build_receipt: ArtifactRef {
            path: RECEIPT.into(),
            sha256: sha256(&staging.path().join(RECEIPT))?,
        },
    };
    write_json(&staging.path().join(MANIFEST), &manifest)?;
    let manifest_sha256 = sha256(&staging.path().join(MANIFEST))?;
    File::open(staging.path())?.sync_all()?;
    fs::rename(staging.path(), &destination)?;
    File::open(&parent)?.sync_all()?;
    Ok(PublishedKernel {
        directory: destination,
        manifest_sha256,
        compatibility,
    })
}

/// `manifest_sha256` comes from a trusted request/deployment configuration, not
/// from this bundle. Expected compatibility must describe the loading process.
/// This never executes the recorded inspector or opens recorded build inputs.
pub fn accept(
    directory: &Path,
    manifest_sha256: &str,
    expected: &KernelCompatibility,
) -> io::Result<AcceptedKernel> {
    let directory = directory.canonicalize()?;
    let manifest_path = flat_file(&directory, Path::new(MANIFEST))?;
    if sha256(&manifest_path)? != manifest_sha256 {
        return Err(io::Error::other("kernel manifest SHA-256 mismatch"));
    }
    let manifest: KernelManifest = read_json(&manifest_path, 64 * 1024)?;
    if manifest.schema_version != expected.schema_version() || &manifest.compatibility != expected {
        return Err(io::Error::other("kernel schema or compatibility mismatch"));
    }
    if manifest.build_receipt.path != Path::new(RECEIPT) {
        return Err(io::Error::other("invalid kernel build receipt path"));
    }
    let receipt_path = flat_file(&directory, &manifest.build_receipt.path)?;
    if sha256(&receipt_path)? != manifest.build_receipt.sha256 {
        return Err(io::Error::other("kernel build receipt SHA-256 mismatch"));
    }
    let receipt: BuildReceipt = read_json(&receipt_path, 64 * 1024 * 1024)?;
    let name = module_name(&receipt)?;
    let filename = format!("{name}.so");
    if manifest.module_name != expected.module_name(&name)
        || compatibility(&receipt)? != manifest.compatibility
        || manifest.build_identity_sha256 != receipt.identity.sha256
        || manifest.library.path != Path::new(&filename)
        || receipt.artifacts.len() != 1
        || receipt.artifacts.get(&filename) != Some(&manifest.library.sha256)
    {
        return Err(io::Error::other(
            "kernel manifest and build receipt disagree",
        ));
    }
    let library = flat_file(&directory, &manifest.library.path)?;
    if sha256(&library)? != manifest.library.sha256 {
        return Err(io::Error::other("kernel library SHA-256 mismatch"));
    }
    Ok(AcceptedKernel {
        directory,
        library,
        module_name: expected.module_name(&name),
        manifest_sha256: manifest_sha256.into(),
        build_identity_sha256: receipt.identity.sha256,
        compatibility: manifest.compatibility,
    })
}
