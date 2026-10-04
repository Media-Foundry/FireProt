//! Complete inference invocation records linked to accepted model archives.
use crate::{
    artifacts::{sha256, verify_outputs},
    batch::{BatchPlan, BatchSpec, JobSpec},
    build::BuildFile,
    execution::{self, ExecutionRecord},
    invocation,
    run_archive::{self, VerifiedArchive},
};
use fireprot_core::{Request, request::ArtifactRef};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::process::ExitStatusExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

const BOUNDARY: &str = "before-spawn-through-observed-exit-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Specification {
    pub schema_version: u32,
    pub timeout_seconds: u64,
    pub request: ArtifactRef,
    pub profile: ArtifactRef,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema_version: u32,
    pub boundary: String,
    pub elapsed_seconds: f64,
    pub exit_code: i32,
    pub specification: Specification,
    /// Exact effective control snapshots, relative to this record's directory.
    pub request: ArtifactRef,
    pub profile: ArtifactRef,
    pub launch: ExecutionRecord,
    /// Original location; collectors can supply a relocated receipt with this hash.
    pub accepted: ArtifactRef,
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

fn pin(path: PathBuf) -> io::Result<ArtifactRef> {
    Ok(ArtifactRef {
        sha256: sha256(&path)?,
        path,
    })
}

/// Execute this framework's `infer` subcommand. The caller must supply the actual
/// FireProt executable and rerun model-content acceptance in `accept`.
/// Preparation and post-exit validation are deliberately outside the timer.
pub fn run(
    specification: Specification,
    destination: &Path,
    executable: &Path,
    cancelled: &AtomicBool,
    accept: impl FnOnce(&ArtifactRef) -> io::Result<()>,
) -> io::Result<ArtifactRef> {
    let plan = BatchPlan::load(BatchSpec {
        schema_version: specification.schema_version,
        timeout_seconds: specification.timeout_seconds,
        jobs: vec![JobSpec {
            request: specification.request.clone(),
            profile: specification.profile.clone(),
        }],
    })?;
    let job = &plan.jobs()[0];
    let mut request = job.request().as_request().clone();
    // Canonical paths make the effective snapshot identical to the request the
    // child archives. Artifact hashing remains inside the child's preflight.
    request.source.root = request.source.root.canonicalize()?;
    request.input.path = request.input.path.canonicalize()?;
    for artifact in request
        .checkpoints
        .values_mut()
        .chain(request.resources.values_mut())
    {
        artifact.path = artifact.path.canonicalize()?;
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "measurement cancelled before preparation",
        ));
    }
    fs::create_dir(destination)?;
    let destination = destination.canonicalize()?;
    let request_path = destination.join("request.json");
    let profile_path = destination.join("profile.json");
    write_new(&request_path, &serde_json::to_vec_pretty(&request)?)?;
    write_new(&profile_path, job.profile())?;
    let files = [
        BuildFile {
            name: "request".into(),
            path: request_path.clone(),
        },
        BuildFile {
            name: "profile".into(),
            path: profile_path.clone(),
        },
    ];
    let launch = ExecutionRecord::capture(
        executable,
        &[
            "infer".into(),
            request_path.into_os_string(),
            profile_path.into_os_string(),
        ],
        &std::env::current_dir()?,
        execution::environment(std::env::vars_os(), &[])?,
        &files,
        &[],
        None,
    )?;
    let mut command = Command::new(&launch.invocation_path);
    command
        .args(&launch.arguments)
        .current_dir(&launch.working_directory)
        .env_clear()
        .envs(&launch.environment)
        .stdout(Stdio::from(File::create(destination.join("stdout.log"))?))
        .stderr(Stdio::from(File::create(destination.join("stderr.log"))?));
    let measurement = invocation::measure(
        &mut command,
        Duration::from_secs(specification.timeout_seconds),
        cancelled,
    )?;
    let finish = || -> io::Result<ArtifactRef> {
        if !measurement.succeeded() || cancelled.load(Ordering::Relaxed) {
            return Err(io::Error::other(
                "inference invocation failed, timed out or was cancelled",
            ));
        }
        launch.verify(&files, &[])?;
        let accepted = pin(request.output_dir.join("accepted.json"))?;
        let mut request_pin = pin(destination.join("request.json"))?;
        request_pin.path = "request.json".into();
        let mut profile_pin = pin(destination.join("profile.json"))?;
        profile_pin.path = "profile.json".into();
        let record = Record {
            schema_version: 1,
            boundary: BOUNDARY.into(),
            elapsed_seconds: measurement.elapsed.as_secs_f64(),
            exit_code: 0,
            specification,
            request: request_pin,
            profile: profile_pin,
            launch,
            accepted,
        };
        let archive = run_archive::verify(&record.accepted)?;
        validate_link(&record, &request, &archive)?;
        accept(&record.accepted)?;
        if cancelled.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "measurement cancelled during acceptance",
            ));
        }
        let path = destination.join("invocation.json");
        write_new(&path, &serde_json::to_vec_pretty(&record)?)?;
        pin(path)
    };
    match finish() {
        Ok(pin) => Ok(pin),
        Err(error) => {
            write_new(
                &destination.join("failure.json"),
                &serde_json::to_vec_pretty(&serde_json::json!({
                    "schema_version":1, "error":error.to_string(),
                    "elapsed_seconds":measurement.elapsed.as_secs_f64(),
                    "exit_code":measurement.status.code(), "signal":measurement.status.signal(),
                    "stop_reason":measurement.stop_reason.map(|r| format!("{r:?}")),
                }))?,
            )?;
            Err(error)
        }
    }
}

fn validate_link(record: &Record, request: &Request, archive: &VerifiedArchive) -> io::Result<()> {
    let child_request: Request =
        serde_json::from_slice(&fs::read(archive.root.join("request.json"))?)?;
    let child_launch: ExecutionRecord =
        serde_json::from_slice(&fs::read(archive.root.join("execution.json"))?)?;
    let supervisor = child_launch
        .inputs
        .iter()
        .find(|f| f.name == "supervisor")
        .ok_or_else(|| io::Error::other("missing child supervisor identity"))?;
    if request != &child_request
        || supervisor.sha256 != record.launch.program.sha256
        || supervisor.canonical_path != record.launch.program.canonical_path
    {
        return Err(io::Error::other(
            "invocation disagrees with accepted request or executable",
        ));
    }
    Ok(())
}

/// Reopen an independently pinned record and its control snapshots, then bind it
/// to an already verified model archive. No original executable or assets needed.
/// The caller still checks model content, successful supervision and isolation.
pub fn verify(
    pin: &ArtifactRef,
    accepted: &ArtifactRef,
    archive: &VerifiedArchive,
) -> io::Result<Record> {
    let path = pin.path.canonicalize()?;
    let root = path
        .parent()
        .ok_or_else(|| io::Error::other("missing invocation parent"))?;
    let mut bytes = Vec::new();
    File::open(&path)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != pin.sha256 {
        return Err(io::Error::other("invocation record size or hash mismatch"));
    }
    let record: Record = serde_json::from_slice(&bytes)?;
    if record.schema_version != 1
        || record.boundary != BOUNDARY
        || record.exit_code != 0
        || !record.elapsed_seconds.is_finite()
        || record.elapsed_seconds <= 0.0
        || record.specification.schema_version != 1
        || record.specification.timeout_seconds == 0
        || record.specification.timeout_seconds > 604800
        || record.elapsed_seconds >= record.specification.timeout_seconds as f64
        || record.accepted.sha256 != accepted.sha256
        || record.request.path != Path::new("request.json")
        || record.profile.path != Path::new("profile.json")
        || record.profile.sha256 != record.specification.profile.sha256
        || record.launch.schema_version != 1
        || record.launch.environment_policy != "explicit-v1"
        || record.launch.environment_inventory.is_some()
        || record.launch.inputs.len() != 2
        || record.launch.arguments.len() != 3
        || record.launch.arguments[0] != "infer"
    {
        return Err(io::Error::other(
            "invalid or mismatched complete invocation record",
        ));
    }
    let paths = verify_outputs(root, &[record.request.clone(), record.profile.clone()])?;
    for (name, pin, argument) in [
        ("request", &record.request, &record.launch.arguments[1]),
        ("profile", &record.profile, &record.launch.arguments[2]),
    ] {
        let input = record
            .launch
            .inputs
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| io::Error::other("missing invocation control identity"))?;
        if input.sha256 != pin.sha256 || input.canonical_path != Path::new(argument) {
            return Err(io::Error::other("invocation control identity mismatch"));
        }
    }
    let request: Request = serde_json::from_slice(&fs::read(&paths[0])?)?;
    validate_link(&record, &request, archive)?;
    Ok(record)
}

impl Record {
    /// Conditions that must remain equal across repeated complete invocations.
    /// Per-run snapshot paths/hashes and receipt identities intentionally vary.
    pub fn conditions(&self) -> serde_json::Value {
        serde_json::json!({
            "boundary":self.boundary, "timeout_seconds":self.specification.timeout_seconds,
            "program":self.launch.program, "invocation_path":self.launch.invocation_path,
            "working_directory":self.launch.working_directory, "environment":self.launch.environment,
            "profile_sha256":self.profile.sha256,
        })
    }
}
