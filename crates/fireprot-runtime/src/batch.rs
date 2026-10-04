//! Ordered model jobs with pinned request/profile snapshots and stop-on-failure.
use crate::supervisor::RunReport;
use fireprot_core::{Request, ValidatedRequest, protocol::Phase, request::ArtifactRef};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchSpec {
    pub schema_version: u32,
    pub timeout_seconds: u64,
    pub jobs: Vec<JobSpec>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JobSpec {
    pub request: ArtifactRef,
    pub profile: ArtifactRef,
}

pub struct LoadedJob {
    request: ValidatedRequest,
    profile: Vec<u8>,
    request_identity: ArtifactRef,
    profile_identity: ArtifactRef,
}
impl LoadedJob {
    pub fn request(&self) -> &ValidatedRequest {
        &self.request
    }
    pub fn profile(&self) -> &[u8] {
        &self.profile
    }
}

pub struct BatchPlan {
    jobs: Vec<LoadedJob>,
    timeout: Duration,
}
impl BatchPlan {
    /// Load every control file once before starting any model. Paths have the
    /// same caller-working-directory meaning as the single-request CLI.
    pub fn load(spec: BatchSpec) -> io::Result<Self> {
        if spec.schema_version != 1 || spec.jobs.is_empty() || spec.jobs.len() > 256 {
            return Err(io::Error::other("batch requires schema 1 and 1-256 jobs"));
        }
        if spec.timeout_seconds == 0 || spec.timeout_seconds > 7 * 24 * 3600 {
            return Err(io::Error::other("batch timeout must be 1-604800 seconds"));
        }
        let mut ids = BTreeSet::new();
        let mut outputs = BTreeSet::new();
        let mut jobs = Vec::new();
        for job in spec.jobs {
            let (request_identity, bytes) = pinned(job.request)?;
            let (profile_identity, profile) = pinned(job.profile)?;
            let mut request: Request = serde_json::from_slice(&bytes)?;
            let output = std::path::absolute(&request.output_dir)?;
            let name = output
                .file_name()
                .ok_or_else(|| io::Error::other("invalid batch output directory"))?;
            let parent = output
                .parent()
                .ok_or_else(|| io::Error::other("missing output parent"))?
                .canonicalize()?;
            let output = parent.join(name);
            if output.try_exists()?
                || !outputs.insert(output.clone())
                || !ids.insert(request.run_id.clone())
            {
                return Err(io::Error::other(
                    "batch run IDs and new output directories must be unique",
                ));
            }
            request.output_dir = output;
            let request = ValidatedRequest::try_from(request).map_err(io::Error::other)?;
            jobs.push(LoadedJob {
                request,
                profile,
                request_identity,
                profile_identity,
            });
        }
        Ok(Self {
            jobs,
            timeout: Duration::from_secs(spec.timeout_seconds),
        })
    }

    pub fn jobs(&self) -> &[LoadedJob] {
        &self.jobs
    }
}

fn pinned(mut artifact: ArtifactRef) -> io::Result<(ArtifactRef, Vec<u8>)> {
    artifact.path = artifact.path.canonicalize()?;
    let file = File::open(&artifact.path)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::other(
            "batch control input must be a regular file",
        ));
    }
    let mut bytes = Vec::new();
    file.take(8 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 || format!("{:x}", Sha256::digest(&bytes)) != artifact.sha256 {
        return Err(io::Error::other(format!(
            "batch control input exceeds 8 MiB or hash mismatch: {}",
            artifact.path.display()
        )));
    }
    Ok((artifact, bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchPhase {
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobPhase {
    Succeeded,
    Failed,
    Skipped,
}

#[derive(Debug, Serialize)]
pub struct JobReport {
    pub run_id: String,
    pub output_dir: PathBuf,
    pub request: ArtifactRef,
    pub profile: ArtifactRef,
    pub phase: JobPhase,
    pub report: Option<RunReport>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BatchReport {
    pub schema_version: u32,
    pub phase: BatchPhase,
    pub timeout_seconds: u64,
    pub execution_seconds: f64,
    pub jobs: Vec<JobReport>,
}

/// A callback composes the existing supervised model execution. Cancellation is
/// shared across jobs, including cancellation during a running model.
pub fn execute(
    plan: &BatchPlan,
    cancelled: &AtomicBool,
    run: impl FnMut(usize, &LoadedJob) -> Result<RunReport, String>,
) -> BatchReport {
    let started = Instant::now();
    let timed_out = AtomicBool::new(false);
    let mut report = std::thread::scope(|scope| {
        let (done, wait) = std::sync::mpsc::channel::<()>();
        let timed_out = &timed_out;
        scope.spawn(move || {
            if matches!(
                wait.recv_timeout(plan.timeout),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                timed_out.store(true, Ordering::Relaxed);
                cancelled.store(true, Ordering::Relaxed);
            }
        });
        let report = execute_jobs(plan, cancelled, run);
        drop(done);
        report
    });
    if timed_out.load(Ordering::Relaxed) {
        report.phase = BatchPhase::TimedOut;
    }
    report.execution_seconds = started.elapsed().as_secs_f64();
    report
}

fn execute_jobs(
    plan: &BatchPlan,
    cancelled: &AtomicBool,
    mut run: impl FnMut(usize, &LoadedJob) -> Result<RunReport, String>,
) -> BatchReport {
    let mut phase = BatchPhase::Succeeded;
    let mut jobs = Vec::new();
    for (index, job) in plan.jobs.iter().enumerate() {
        let request = job.request.as_request();
        let mut record = JobReport {
            run_id: request.run_id.clone(),
            output_dir: request.output_dir.clone(),
            request: job.request_identity.clone(),
            profile: job.profile_identity.clone(),
            phase: JobPhase::Skipped,
            report: None,
            error: None,
        };
        if phase != BatchPhase::Succeeded || cancelled.load(Ordering::Relaxed) {
            if phase == BatchPhase::Succeeded {
                phase = BatchPhase::Cancelled;
            }
            record.error = Some("not started after batch failure or cancellation".into());
        } else {
            match run(index, job) {
                Ok(report) => {
                    if report.run_id != request.run_id || report.phase != Phase::Succeeded {
                        record.phase = JobPhase::Failed;
                        record.error =
                            Some("model job failed or returned a mismatched run identity".into());
                        phase = BatchPhase::Failed;
                    } else {
                        record.phase = JobPhase::Succeeded;
                    }
                    record.report = Some(report);
                }
                Err(error) => {
                    record.phase = JobPhase::Failed;
                    record.error = Some(error);
                    phase = BatchPhase::Failed;
                }
            }
            if cancelled.load(Ordering::Relaxed) {
                phase = BatchPhase::Cancelled;
            }
        }
        jobs.push(record);
    }
    BatchReport {
        schema_version: 1,
        phase,
        timeout_seconds: plan.timeout.as_secs(),
        execution_seconds: 0.0,
        jobs,
    }
}

pub fn read_spec(path: &Path) -> io::Result<BatchSpec> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 8 * 1024 * 1024 {
        return Err(io::Error::other("batch manifest exceeds 8 MiB"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}
