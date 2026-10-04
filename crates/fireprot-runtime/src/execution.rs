//! Observable worker launch identity, with an explicit inherited environment.
//! This is not a complete Python package or dynamically loaded library lock.
use crate::{
    artifacts::sha256,
    build::{BuildFile, FileIdentity},
    build_tree::BuildTree,
};
use fireprot_core::request::ArtifactRef;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs, io,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

/// Ambient process values required for ordinary host operation. Model/backend
/// settings must be explicit profile overrides; no other parent keys are copied.
const INHERITED: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PYTHONHASHSEED",
];

pub fn environment(
    ambient: impl IntoIterator<Item = (OsString, OsString)>,
    overrides: &[(OsString, OsString)],
) -> io::Result<BTreeMap<String, String>> {
    let mut result = BTreeMap::new();
    for (key, value) in ambient {
        if key.to_str().is_some_and(|s| INHERITED.contains(&s)) {
            insert(&mut result, &key, &value)?;
        }
    }
    for (key, value) in overrides {
        insert(&mut result, key, value)?;
    }
    Ok(result)
}
fn insert(
    environment: &mut BTreeMap<String, String>,
    key: &std::ffi::OsStr,
    value: &std::ffi::OsStr,
) -> io::Result<()> {
    let key = key
        .to_str()
        .ok_or_else(|| io::Error::other("environment key must be UTF-8"))?;
    let value = value
        .to_str()
        .ok_or_else(|| io::Error::other("environment value must be UTF-8"))?;
    if key.is_empty() || key.contains(['=', '\0']) || value.contains('\0') {
        return Err(io::Error::other("invalid worker environment"));
    }
    environment.insert(key.into(), value.into());
    Ok(())
}

/// Preserve the invocation spelling (notably venv/bin/python) while resolving
/// PATH once. The same absolute path is recorded and executed.
pub fn resolve_program(
    program: &Path,
    cwd: &Path,
    environment: &BTreeMap<String, String>,
) -> io::Result<PathBuf> {
    let candidates =
        if program.is_absolute() {
            vec![program.to_path_buf()]
        } else if program.components().count() > 1 {
            vec![cwd.join(program)]
        } else {
            std::env::split_paths(environment.get("PATH").ok_or_else(|| {
                io::Error::other("PATH is required for a relative worker program")
            })?)
            .map(|dir| {
                if dir.is_absolute() {
                    dir.join(program)
                } else {
                    cwd.join(dir).join(program)
                }
            })
            .collect()
        };
    candidates
        .into_iter()
        .find(|p| fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0))
        .ok_or_else(|| {
            io::Error::other(format!("executable is unavailable: {}", program.display()))
        })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ExecutionRecord {
    pub schema_version: u32,
    pub environment_policy: String,
    pub invocation_path: PathBuf,
    pub program: FileIdentity,
    pub arguments: Vec<String>,
    pub working_directory: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub inputs: Vec<FileIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment_inventory: Option<ArtifactRef>,
    pub sha256: String,
}
impl ExecutionRecord {
    pub fn capture(
        program: &Path,
        args: &[OsString],
        cwd: &Path,
        environment: BTreeMap<String, String>,
        files: &[BuildFile],
        trees: &[BuildTree],
        environment_inventory: Option<&ArtifactRef>,
    ) -> io::Result<Self> {
        let working_directory = cwd.canonicalize()?;
        if !working_directory.is_dir() {
            return Err(io::Error::other("worker directory is not a directory"));
        }
        let invocation_path = resolve_program(program, &working_directory, &environment)?;
        let program = identify("worker-executable", &invocation_path)?;
        let arguments = args
            .iter()
            .map(|s| {
                s.to_str()
                    .filter(|s| !s.contains('\0'))
                    .map(str::to_owned)
                    .ok_or_else(|| io::Error::other("worker argument must be valid UTF-8"))
            })
            .collect::<io::Result<_>>()?;
        let mut inputs = files
            .iter()
            .map(|f| identify(&f.name, &f.path))
            .collect::<io::Result<Vec<_>>>()?;
        for tree in trees {
            inputs.extend(tree.identify()?);
        }
        if let Some(pin) = environment_inventory {
            for mut file in crate::environment_inventory::accept(pin)?.files {
                file.name = format!("environment/{}", file.name);
                inputs.push(file);
            }
        }
        inputs.sort_by(|a, b| a.name.cmp(&b.name));
        let mut names = BTreeSet::new();
        for f in &inputs {
            if f.name.is_empty() || !names.insert(&f.name) {
                return Err(io::Error::other("duplicate or empty execution input name"));
            }
        }
        let mut record = Self {
            schema_version: 1,
            environment_policy: "explicit-v1".into(),
            invocation_path,
            program,
            arguments,
            working_directory,
            environment,
            inputs,
            environment_inventory: environment_inventory.cloned(),
            sha256: String::new(),
        };
        record.sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&record)?));
        Ok(record)
    }
    pub fn verify(&self, files: &[BuildFile], trees: &[BuildTree]) -> io::Result<()> {
        let args: Vec<OsString> = self.arguments.iter().map(Into::into).collect();
        let current = Self::capture(
            &self.invocation_path,
            &args,
            &self.working_directory,
            self.environment.clone(),
            files,
            trees,
            self.environment_inventory.as_ref(),
        )?;
        if &current != self {
            return Err(io::Error::other(
                "worker launch inputs changed during execution",
            ));
        }
        Ok(())
    }
    /// Linux independent executable identity, while the worker waits for Start.
    pub fn verify_process(&self, pid: u32) -> io::Result<()> {
        let observed = fs::read_link(format!("/proc/{pid}/exe"))?;
        if observed != self.program.canonical_path {
            return Err(io::Error::other(
                "worker executable disagrees with launch identity",
            ));
        }
        Ok(())
    }
}
fn identify(name: &str, path: &Path) -> io::Result<FileIdentity> {
    let canonical_path = path.canonicalize()?;
    Ok(FileIdentity {
        name: name.into(),
        sha256: sha256(&canonical_path)?,
        canonical_path,
    })
}
