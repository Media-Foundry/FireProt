//! Explicit paired execution plans built on the shared batch lifecycle.
use crate::{
    artifacts::sha256,
    batch::{BatchPlan, BatchSpec, JobSpec, LoadedJob},
    measured_run,
};
use fireprot_core::request::ArtifactRef;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Baseline,
    Candidate,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairSpec {
    pub first: Arm,
    pub baseline: JobSpec,
    pub candidate: JobSpec,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Specification {
    pub schema_version: u32,
    pub timeout_seconds: u64,
    pub invocation_timeout_seconds: u64,
    pub pairs: Vec<PairSpec>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Slot {
    pub pair: usize,
    pub arm: Arm,
}

pub struct Plan {
    specification: Specification,
    batch: BatchPlan,
    slots: Vec<Slot>,
}

impl Plan {
    pub fn load(specification: Specification) -> io::Result<Self> {
        if specification.schema_version != 1
            || specification.pairs.is_empty()
            || specification.pairs.len() > 128
            || specification.pairs[0].first != Arm::Baseline
            || specification.invocation_timeout_seconds == 0
            || specification.invocation_timeout_seconds > 604800
        {
            return Err(io::Error::other(
                "experiment requires 1-128 pairs, an initial baseline and a 1-604800 second invocation deadline",
            ));
        }
        let mut jobs = Vec::new();
        let mut slots = Vec::new();
        for (pair, spec) in specification.pairs.iter().enumerate() {
            let order = if spec.first == Arm::Baseline {
                [Arm::Baseline, Arm::Candidate]
            } else {
                [Arm::Candidate, Arm::Baseline]
            };
            for arm in order {
                let job = match arm {
                    Arm::Baseline => &spec.baseline,
                    Arm::Candidate => &spec.candidate,
                };
                jobs.push(JobSpec {
                    request: job.request.clone(),
                    profile: job.profile.clone(),
                });
                slots.push(Slot { pair, arm });
            }
        }
        let batch = BatchPlan::load(BatchSpec {
            schema_version: 1,
            timeout_seconds: specification.timeout_seconds,
            jobs,
        })?;
        let reference = batch.jobs()[0].request().as_request();
        let mut arms: [Option<&LoadedJob>; 2] = [None, None];
        for (job, slot) in batch.jobs().iter().zip(&slots) {
            let mut request = job.request().as_request().clone();
            request.run_id.clone_from(&reference.run_id);
            request.output_dir.clone_from(&reference.output_dir);
            request.backend = reference.backend;
            if &request != reference {
                return Err(io::Error::other("experiment scientific conditions differ"));
            }
            let index = usize::from(slot.arm == Arm::Candidate);
            if let Some(first) = arms[index] {
                if first.request().as_request().backend != job.request().as_request().backend
                    || first.profile() != job.profile()
                {
                    return Err(io::Error::other(
                        "backend or profile bytes change within an experiment arm",
                    ));
                }
            } else {
                arms[index] = Some(job);
            }
        }
        Ok(Self {
            specification,
            batch,
            slots,
        })
    }

    /// Check every model profile before creating snapshots or executing a job.
    pub fn prepare(
        self,
        destination: &Path,
        validate: impl Fn(&[u8]) -> io::Result<()>,
    ) -> io::Result<Prepared> {
        for job in self.batch.jobs() {
            validate(job.profile())?;
        }
        let path = std::path::absolute(destination)?;
        let name = path
            .file_name()
            .ok_or_else(|| io::Error::other("invalid experiment destination"))?;
        let root = path
            .parent()
            .ok_or_else(|| io::Error::other("missing destination parent"))?
            .canonicalize()?
            .join(name);
        if self
            .batch
            .jobs()
            .iter()
            .any(|j| j.request().as_request().output_dir == root)
        {
            return Err(io::Error::other(
                "experiment destination aliases a model output",
            ));
        }
        fs::create_dir(&root)?;
        let root = root.canonicalize()?;
        let specification_pin = write_record(&root, "execution-plan.json", &self.specification)?;
        let mut invocations = Vec::new();
        for (index, job) in self.batch.jobs().iter().enumerate() {
            let request = write_record(
                &root,
                &format!("request-{index:03}.json"),
                job.request().as_request(),
            )?;
            let profile_path = root.join(format!("profile-{index:03}.json"));
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&profile_path)?;
            file.write_all(job.profile())?;
            file.sync_all()?;
            let profile = ArtifactRef {
                sha256: sha256(&profile_path)?,
                path: profile_path,
            };
            invocations.push(measured_run::Specification {
                schema_version: 1,
                timeout_seconds: self.specification.invocation_timeout_seconds,
                request,
                profile,
            });
        }
        Ok(Prepared {
            plan: self,
            root,
            specification_pin,
            invocations,
        })
    }
}

pub struct Prepared {
    plan: Plan,
    root: PathBuf,
    specification_pin: ArtifactRef,
    invocations: Vec<measured_run::Specification>,
}

impl Prepared {
    pub fn batch(&self) -> &BatchPlan {
        &self.plan.batch
    }
    pub fn slots(&self) -> &[Slot] {
        &self.plan.slots
    }
    pub fn invocations(&self) -> &[measured_run::Specification] {
        &self.invocations
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn specification_pin(&self) -> &ArtifactRef {
        &self.specification_pin
    }
    pub fn write_record(&self, name: &str, value: &impl Serialize) -> io::Result<ArtifactRef> {
        write_record(&self.root, name, value)
    }
}

fn write_record(root: &Path, name: &str, value: &impl Serialize) -> io::Result<ArtifactRef> {
    if name.is_empty()
        || Path::new(name).components().count() != 1
        || !matches!(
            Path::new(name).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        return Err(io::Error::other(
            "experiment record requires a single filename",
        ));
    }
    let path = root.join(name);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    Ok(ArtifactRef {
        sha256: sha256(&path)?,
        path,
    })
}
