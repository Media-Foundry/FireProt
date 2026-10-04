//! Install the maintained framework independently of its build/source directories.
//! External model environments, checkpoints and kernel bundles remain explicit.
use crate::artifacts::sha256;
use fireprot_core::request::ArtifactRef;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs, io,
    os::unix::fs::PermissionsExt,
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstallSpec {
    pub schema_version: u32,
    pub source_root: PathBuf,
    pub cli: ArtifactRef,
    pub bridge: ArtifactRef,
    pub compiler_guard: ArtifactRef,
    pub hip_probe: ArtifactRef,
    pub cache_root: PathBuf,
    /// Existing execution profiles. Only framework-owned locations are rewritten.
    pub profiles: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InstalledFile {
    pub sha256: String,
    pub executable: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstallManifest {
    pub schema_version: u32,
    pub version: String,
    pub platform: String,
    pub files: BTreeMap<String, InstalledFile>,
}

#[derive(Debug, Serialize)]
pub struct Installation {
    pub directory: PathBuf,
    pub manifest_sha256: String,
    pub file_count: usize,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}
fn regular(path: &Path) -> io::Result<()> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(invalid(format!(
            "expected regular non-symlink file: {}",
            path.display()
        )));
    }
    Ok(())
}
fn relative(path: &Path) -> bool {
    !path.as_os_str().is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))
}
fn inventory(
    root: &Path,
    path: &Path,
    files: &mut BTreeMap<String, InstalledFile>,
    depth: usize,
) -> io::Result<()> {
    if depth > 128 {
        return Err(invalid("installation directory depth exceeds 128"));
    }
    let meta = fs::symlink_metadata(path)?;
    if meta.file_type().is_dir() {
        for entry in fs::read_dir(path)? {
            inventory(root, &entry?.path(), files, depth + 1)?;
        }
    } else {
        regular(path)?;
        let name = path
            .strip_prefix(root)
            .map_err(io::Error::other)?
            .to_str()
            .ok_or_else(|| invalid("non-UTF8 installation path"))?
            .to_owned();
        if name != "installation.json" {
            files.insert(
                name,
                InstalledFile {
                    sha256: sha256(path)?,
                    executable: meta.permissions().mode() & 0o111 != 0,
                },
            );
        }
    }
    Ok(())
}
fn copy_file(source: &Path, target: &Path, executable: bool) -> io::Result<()> {
    regular(source)?;
    let before = sha256(source)?;
    fs::create_dir_all(
        target
            .parent()
            .ok_or_else(|| invalid("missing target parent"))?,
    )?;
    fs::copy(source, target)?;
    fs::set_permissions(
        target,
        fs::Permissions::from_mode(if executable { 0o755 } else { 0o644 }),
    )?;
    if sha256(target)? != before || sha256(source)? != before {
        return Err(invalid("installation input changed while copying"));
    }
    Ok(())
}
fn copy_tree(source: &Path, target: &Path, extensions: &[&str], depth: usize) -> io::Result<()> {
    if depth > 128 {
        return Err(invalid("source directory depth exceeds 128"));
    }
    if !fs::symlink_metadata(source)?.file_type().is_dir() {
        return Err(invalid("source tree must be a non-symlink directory"));
    }
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_name() == "__pycache__" {
            continue;
        }
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_tree(
                &path,
                &target.join(entry.file_name()),
                extensions,
                depth + 1,
            )?;
        } else if kind.is_symlink() {
            return Err(invalid("source tree contains a symlink"));
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| extensions.contains(&e))
        {
            copy_file(&path, &target.join(entry.file_name()), false)?;
        }
    }
    Ok(())
}

/// Build in a sibling temporary directory, then atomically publish without replacement.
pub fn install(spec: &InstallSpec, destination: &Path) -> io::Result<Installation> {
    if spec.schema_version != 1
        || !spec.source_root.is_absolute()
        || !spec.cache_root.is_absolute()
        || spec.profiles.is_empty()
    {
        return Err(invalid(
            "installation requires schema 1, absolute source/cache paths and profiles",
        ));
    }
    let destination = std::path::absolute(destination)?;
    let parent = destination
        .parent()
        .ok_or_else(|| invalid("missing installation parent"))?;
    fs::create_dir_all(parent)?;
    let destination = parent.canonicalize()?.join(
        destination
            .file_name()
            .ok_or_else(|| invalid("missing installation name"))?,
    );
    if destination.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "installation destination exists",
        ));
    }
    if spec.cache_root.starts_with(&destination) {
        return Err(invalid("cache must be outside the installation"));
    }
    if spec
        .cache_root
        .components()
        .any(|c| matches!(c, Component::ParentDir))
        || destination.starts_with(spec.source_root.canonicalize()?)
    {
        return Err(invalid(
            "use a normalized cache path and a destination outside the source tree",
        ));
    }
    let staging = tempfile::Builder::new()
        .prefix(".fireprot-install-")
        .tempdir_in(destination.parent().unwrap())?;
    let root = staging.path();
    let framework = root.join("lib/fireprot");
    for (artifact, name, executable) in [
        (&spec.cli, "bin/fireprot", true),
        (&spec.compiler_guard, "bin/fireprot-compiler-guard", true),
        (&spec.hip_probe, "bin/fireprot-hip-probe", true),
        (&spec.bridge, "lib/fireprot/lib_fireprot_bridge.so", false),
    ] {
        if !artifact.path.is_absolute() || sha256(&artifact.path)? != artifact.sha256 {
            return Err(invalid("binary artifact digest/path mismatch"));
        }
        copy_file(&artifact.path, &root.join(name), executable)?;
        if sha256(&root.join(name))? != artifact.sha256 {
            return Err(invalid("copied binary digest mismatch"));
        }
    }
    for package in ["fireprot", "fireprot_workers"] {
        copy_tree(
            &spec.source_root.join("python").join(package),
            &framework.join("python").join(package),
            &["py"],
            0,
        )?;
    }
    copy_tree(
        &spec.source_root.join("kernels/amd"),
        &framework.join("kernels/amd"),
        &["cpp", "cu", "hip", "h", "hpp"],
        0,
    )?;
    copy_file(
        &spec.source_root.join("fireprot/__init__.py"),
        &framework.join("fireprot/__init__.py"),
        false,
    )?;
    for name in ["README.md", "Cargo.lock", "Cargo.toml"] {
        copy_file(
            &spec.source_root.join(name),
            &root.join("share/fireprot").join(name),
            false,
        )?;
    }
    let installed = destination.join("lib/fireprot");
    for (name, template) in &spec.profiles {
        if name.is_empty()
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            return Err(invalid(
                "profile name must contain only ASCII letters, digits, underscores or hyphens",
            ));
        }
        let mut value = template.clone();
        let object = value
            .as_object_mut()
            .ok_or_else(|| invalid("execution profile must be an object"))?;
        object.insert(
            "workers".into(),
            serde_json::to_value(installed.join("python"))?,
        );
        object.insert(
            "hip_probe".into(),
            serde_json::to_value(destination.join("bin/fireprot-hip-probe"))?,
        );
        let env = object
            .entry("environment")
            .or_insert_with(|| serde_json::json!({}))
            .as_object_mut()
            .ok_or_else(|| invalid("profile environment must be an object"))?;
        for (key, path) in [
            (
                "FIREPROT_BRIDGE_LIBRARY",
                installed.join("lib_fireprot_bridge.so"),
            ),
            (
                "FIREPROT_BUILD_GUARD",
                destination.join("bin/fireprot-compiler-guard"),
            ),
            ("FIREPROT_EXTENSION_CACHE", spec.cache_root.clone()),
        ] {
            env.insert(key.into(), serde_json::to_value(path)?);
        }
        env.insert("PYTHONDONTWRITEBYTECODE".into(), "1".into());
        let directory = root.join("profiles");
        fs::create_dir_all(&directory)?;
        fs::write(
            directory.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&value)?,
        )?;
    }
    let mut files = BTreeMap::new();
    inventory(root, root, &mut files, 0)?;
    let manifest = InstallManifest {
        schema_version: 1,
        version: env!("CARGO_PKG_VERSION").into(),
        platform: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        files,
    };
    fs::write(
        root.join("installation.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let digest = sha256(&root.join("installation.json"))?;
    verify(root, &digest)?;
    nix::fcntl::renameat2(
        nix::fcntl::AT_FDCWD,
        root,
        nix::fcntl::AT_FDCWD,
        &destination,
        nix::fcntl::RenameFlags::RENAME_NOREPLACE,
    )
    .map_err(io::Error::other)?;
    Ok(Installation {
        directory: destination,
        manifest_sha256: digest,
        file_count: manifest.files.len(),
    })
}

/// Verify the pinned manifest and exact file inventory, including execute bits.
pub fn verify(directory: &Path, digest: &str) -> io::Result<Installation> {
    let directory = directory.canonicalize()?;
    let path = directory.join("installation.json");
    regular(&path)?;
    if fs::metadata(&path)?.len() > 16 * 1024 * 1024 || sha256(&path)? != digest {
        return Err(invalid("installation manifest digest/size mismatch"));
    }
    let manifest: InstallManifest =
        serde_json::from_reader(fs::File::open(path)?).map_err(io::Error::other)?;
    if manifest.schema_version != 1
        || manifest.files.is_empty()
        || manifest.files.keys().any(|p| !relative(Path::new(p)))
    {
        return Err(invalid("invalid installation manifest"));
    }
    let mut actual = BTreeMap::new();
    inventory(&directory, &directory, &mut actual, 0)?;
    if actual != manifest.files {
        return Err(invalid("installation file inventory mismatch"));
    }
    Ok(Installation {
        directory,
        manifest_sha256: digest.into(),
        file_count: actual.len(),
    })
}
