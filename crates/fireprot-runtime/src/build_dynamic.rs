//! Dynamic dependencies of explicitly trusted host toolchain ELF objects.
//! `ldd` describes load-time linkage, not libraries loaded later by plugins.
use crate::{artifacts::sha256, build::FileIdentity};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{self, Read},
    path::PathBuf,
    process::{Command, Stdio},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DynamicInputs {
    pub inspector: PathBuf,
    pub objects: Vec<PathBuf>,
}

impl DynamicInputs {
    pub fn identify(&self) -> io::Result<Vec<FileIdentity>> {
        if !self.inspector.is_absolute() || self.objects.is_empty() {
            return Err(io::Error::other("invalid dynamic dependency specification"));
        }
        let mut paths = BTreeSet::from([self.inspector.canonicalize()?]);
        for object in &self.objects {
            if !object.is_absolute() || !object.is_file() {
                return Err(io::Error::other(
                    "dynamic dependency input must be an absolute ELF file",
                ));
            }
            let canonical = object.canonicalize()?;
            let mut magic = [0; 4];
            File::open(&canonical)?.read_exact(&mut magic)?;
            if magic != *b"\x7fELF" {
                return Err(io::Error::other("dynamic dependency input is not ELF"));
            }
            let output = Command::new(&self.inspector)
                .arg(object)
                .env("LC_ALL", "C")
                .stdin(Stdio::null())
                .output()?;
            if !output.status.success() || !output.stderr.is_empty() {
                return Err(io::Error::other(format!(
                    "dependency inspector failed for {}: {}",
                    object.display(),
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
            let report = std::str::from_utf8(&output.stdout).map_err(io::Error::other)?;
            for dependency in parse_report(report)? {
                paths.insert(dependency.canonicalize()?);
            }
            paths.insert(canonical);
        }
        paths
            .into_iter()
            .map(|path| {
                Ok(FileIdentity {
                    name: format!("dynamic/{}", path.display()),
                    sha256: sha256(&path)?,
                    canonical_path: path,
                })
            })
            .collect()
    }
}

fn parse_report(report: &str) -> io::Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for line in report.lines().map(str::trim).filter(|s| !s.is_empty()) {
        if line.starts_with("linux-vdso.") || line.starts_with("linux-gate.") {
            continue;
        }
        let value = line.split_once(" => ").map_or(line, |(_, value)| value);
        let (path, address) = value.rsplit_once(" (").ok_or_else(|| {
            io::Error::other(format!("unresolved or malformed dependency: {line}"))
        })?;
        if !path.starts_with('/')
            || !address.ends_with(')')
            || !address.starts_with("0x")
            || !address[2..address.len() - 1]
                .chars()
                .all(|c| c.is_ascii_hexdigit())
            || address.len() < 4
        {
            return Err(io::Error::other(format!(
                "invalid dependency record: {line}"
            )));
        }
        paths.push(PathBuf::from(path));
    }
    if paths.is_empty() {
        return Err(io::Error::other(
            "dependency inspector returned no file dependencies",
        ));
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_paths_without_using_runtime_addresses() {
        let paths = parse_report("linux-vdso.so.1 (0x123)\nlibx.so => /opt/tool chain/libx.so (0x567)\n/lib64/ld-linux.so.2 (0x9ab)\n").unwrap();
        assert_eq!(
            paths,
            [
                PathBuf::from("/opt/tool chain/libx.so"),
                PathBuf::from("/lib64/ld-linux.so.2")
            ]
        );
        for bad in [
            "libx.so => not found",
            "",
            "statically linked",
            "libx.so => relative.so (0x123)",
            "/lib/a.so (bad)",
        ] {
            assert!(parse_report(bad).is_err(), "{bad}");
        }
    }
}
