//! Persisted acceptance evidence. Callers must pin the receipt independently;
//! hashes detect changed files, not a malicious replacement of the entire record.
use crate::{artifacts::verify_outputs, supervisor::RunReport};
use fireprot_core::request::ArtifactRef;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

const METADATA: &[&str] = &[
    "request.json",
    "provenance.json",
    "execution.json",
    "supervisor.json",
];

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceptanceReceipt {
    pub schema_version: u32,
    pub metadata: Vec<ArtifactRef>,
    pub outputs: Vec<ArtifactRef>,
}

pub struct VerifiedArchive {
    pub root: PathBuf,
    pub receipt: AcceptanceReceipt,
    pub outputs: Vec<PathBuf>,
}

pub(crate) fn json_identity(name: &str, value: &impl Serialize) -> io::Result<ArtifactRef> {
    Ok(ArtifactRef {
        path: name.into(),
        sha256: format!("{:x}", Sha256::digest(serde_json::to_vec_pretty(value)?)),
    })
}

/// Called only after process, isolation and output-content acceptance. The final
/// supervisor report is written afterwards; both files are required by readers.
pub(crate) fn seal(
    root: &Path,
    mut metadata: Vec<ArtifactRef>,
    report: &RunReport,
    outputs: &[ArtifactRef],
) -> io::Result<()> {
    verify_outputs(root, &metadata)?;
    verify_outputs(root, outputs)?;
    metadata.push(json_identity("supervisor.json", report)?);
    let receipt = AcceptanceReceipt {
        schema_version: 1,
        metadata,
        outputs: outputs.to_vec(),
    };
    validate_inventory(&receipt)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("accepted.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    file.sync_all()
}

fn validate_inventory(receipt: &AcceptanceReceipt) -> io::Result<()> {
    let names: BTreeSet<_> = receipt.metadata.iter().map(|a| a.path.as_path()).collect();
    let required: BTreeSet<_> = METADATA.iter().map(Path::new).collect();
    if receipt.schema_version != 1
        || receipt.metadata.len() != METADATA.len()
        || names != required
        || receipt.outputs.is_empty()
    {
        return Err(io::Error::other("invalid acceptance inventory"));
    }
    let mut paths = names;
    paths.insert(Path::new("accepted.json"));
    for output in &receipt.outputs {
        if !paths.insert(&output.path) {
            return Err(io::Error::other("duplicate or reserved output artifact"));
        }
    }
    Ok(())
}

/// Validate a pinned receipt and all files it binds, without reopening original
/// model environments or checkpoints. Model-content validation remains required.
pub fn verify(pin: &ArtifactRef) -> io::Result<VerifiedArchive> {
    let path = pin.path.canonicalize()?;
    let root = path
        .parent()
        .ok_or_else(|| io::Error::other("receipt has no parent"))?
        .to_path_buf();
    let bytes = fs::read(&path)?;
    if format!("{:x}", Sha256::digest(&bytes)) != pin.sha256 {
        return Err(io::Error::other("acceptance receipt SHA-256 mismatch"));
    }
    let receipt: AcceptanceReceipt = serde_json::from_slice(&bytes)?;
    validate_inventory(&receipt)?;
    verify_outputs(&root, &receipt.metadata)?;
    let outputs = verify_outputs(&root, &receipt.outputs)?;
    Ok(VerifiedArchive {
        root,
        receipt,
        outputs,
    })
}
