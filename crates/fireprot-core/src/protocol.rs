//! Versioned worker messages and a supervisor-owned lifecycle.
//!
//! A worker can announce outputs, but only the Rust supervisor can accept them
//! after checking process exit, artifacts and continuous GPU isolation.
use std::{collections::BTreeSet, path::Component};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    SCHEMA_VERSION, ValidatedRequest,
    request::{ArtifactRef, validate_artifact},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceIdentity {
    pub worker_pid: u32,
    /// Ordinal inside the worker's restricted HIP visibility, normally zero.
    pub logical_device: u32,
    pub pci_address: String,
    pub architecture: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerEvent {
    Ready {
        device: DeviceIdentity,
    },
    Progress {
        stage: String,
    },
    /// Paths must be relative to this run's output directory. The runtime also
    /// verifies filesystem containment, content and hashes before acceptance.
    Outputs {
        artifacts: Vec<ArtifactRef>,
    },
    Failed {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Envelope {
    pub schema_version: u32,
    pub run_id: String,
    /// Zero-based, strictly sequential within this worker process.
    pub sequence: u64,
    pub payload: WorkerEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    AwaitingReady,
    AwaitingStart,
    Running,
    AwaitingVerification,
    Succeeded,
    Failed,
}

/// Not deserializable: these are observations produced by the supervisor,
/// never fields accepted from a worker's completion message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionChecks {
    pub exit_code: Option<i32>,
    pub verified_artifacts: usize,
    pub isolation_samples: usize,
    pub external_pids: Vec<u32>,
    pub monitor_errors: Vec<String>,
}

#[derive(Debug)]
pub struct Session {
    run_id: String,
    worker_pid: u32,
    phase: Phase,
    next_sequence: u64,
    device: Option<DeviceIdentity>,
    artifacts: Vec<ArtifactRef>,
    failure: Option<String>,
}

impl Session {
    pub fn new(request: &ValidatedRequest, worker_pid: u32) -> Result<Self, ProtocolError> {
        if worker_pid == 0 {
            return Err(ProtocolError::Identity);
        }
        Ok(Self {
            run_id: request.as_request().run_id.clone(),
            worker_pid,
            phase: Phase::AwaitingReady,
            next_sequence: 0,
            device: None,
            artifacts: Vec::new(),
            failure: None,
        })
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn device(&self) -> Option<&DeviceIdentity> {
        self.device.as_ref()
    }
    pub fn artifacts(&self) -> &[ArtifactRef] {
        &self.artifacts
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// Every invalid worker message permanently fails this session. The runtime
    /// must then stop the child and retain its diagnostic output.
    pub fn receive(&mut self, envelope: Envelope) -> Result<(), ProtocolError> {
        if matches!(self.phase, Phase::Succeeded | Phase::Failed) {
            return Err(ProtocolError::Terminal);
        }
        let result = self.receive_inner(envelope);
        if let Err(error) = &result {
            self.fail(error.to_string());
        }
        result
    }

    fn receive_inner(&mut self, envelope: Envelope) -> Result<(), ProtocolError> {
        if envelope.schema_version != SCHEMA_VERSION {
            return Err(ProtocolError::Schema);
        }
        if envelope.run_id != self.run_id {
            return Err(ProtocolError::RunId);
        }
        if envelope.sequence != self.next_sequence {
            return Err(ProtocolError::Sequence);
        }
        let next = self
            .next_sequence
            .checked_add(1)
            .ok_or(ProtocolError::Sequence)?;
        match envelope.payload {
            WorkerEvent::Ready { device } if self.phase == Phase::AwaitingReady => {
                if device.worker_pid != self.worker_pid
                    || device.logical_device != 0
                    || !valid_pci(&device.pci_address)
                    || device.architecture.trim().is_empty()
                {
                    return Err(ProtocolError::Identity);
                }
                self.device = Some(device);
                self.phase = Phase::AwaitingStart;
            }
            WorkerEvent::Progress { stage } if self.phase == Phase::Running => {
                if stage.trim().is_empty() {
                    return Err(ProtocolError::EmptyStage);
                }
            }
            WorkerEvent::Outputs { artifacts } if self.phase == Phase::Running => {
                if artifacts.is_empty() {
                    return Err(ProtocolError::Artifacts);
                }
                let mut paths = BTreeSet::new();
                for artifact in &artifacts {
                    validate_artifact(artifact).map_err(|_| ProtocolError::Artifacts)?;
                    if !artifact
                        .path
                        .components()
                        .all(|c| matches!(c, Component::Normal(_)))
                        || !paths.insert(artifact.path.clone())
                    {
                        return Err(ProtocolError::Artifacts);
                    }
                }
                self.artifacts = artifacts;
                self.phase = Phase::AwaitingVerification;
            }
            WorkerEvent::Failed { message } => {
                self.fail(if message.trim().is_empty() {
                    "worker failed without a diagnostic".into()
                } else {
                    message
                });
            }
            _ => return Err(ProtocolError::Order),
        }
        self.next_sequence = next;
        Ok(())
    }

    /// Called only after the supervisor verifies PCI-to-SMI identity and starts
    /// monitoring. Receiving Ready alone never authorizes model execution.
    pub fn authorize_start(&mut self) -> Result<(), ProtocolError> {
        if self.phase != Phase::AwaitingStart {
            return Err(ProtocolError::Order);
        }
        self.phase = Phase::Running;
        Ok(())
    }

    /// A clean process exit is necessary but insufficient for a successful run.
    pub fn finish(&mut self, checks: CompletionChecks) -> Result<(), ProtocolError> {
        if matches!(self.phase, Phase::Succeeded | Phase::Failed) {
            return Err(ProtocolError::Terminal);
        }
        let valid = self.phase == Phase::AwaitingVerification
            && checks.exit_code == Some(0)
            && checks.verified_artifacts == self.artifacts.len()
            && checks.isolation_samples > 0
            && checks.external_pids.is_empty()
            && checks.monitor_errors.is_empty();
        if !valid {
            self.fail(
                "incomplete output, failed process, artifact verification or GPU isolation".into(),
            );
            return Err(ProtocolError::Verification);
        }
        self.phase = Phase::Succeeded;
        Ok(())
    }

    /// Timeout, cancellation and transport errors are supervisor failures.
    pub fn abort(&mut self, reason: impl Into<String>) -> Result<(), ProtocolError> {
        if matches!(self.phase, Phase::Succeeded | Phase::Failed) {
            return Err(ProtocolError::Terminal);
        }
        self.fail(reason.into());
        Ok(())
    }

    fn fail(&mut self, message: String) {
        self.phase = Phase::Failed;
        self.failure = Some(message);
    }
}

fn valid_pci(value: &str) -> bool {
    let b = value.as_bytes();
    b.len() == 12
        && b[4] == b':'
        && b[7] == b':'
        && b[10] == b'.'
        && b.iter()
            .enumerate()
            .all(|(i, c)| matches!(i, 4 | 7 | 10) || c.is_ascii_hexdigit())
        && (b'0'..=b'7').contains(&b[11])
        && u8::from_str_radix(&value[8..10], 16).is_ok_and(|slot| slot <= 31)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProtocolError {
    #[error("worker protocol version mismatch")]
    Schema,
    #[error("worker run ID mismatch")]
    RunId,
    #[error("worker sequence is missing, duplicated or out of order")]
    Sequence,
    #[error("worker process or device identity is invalid")]
    Identity,
    #[error("worker event or supervisor command is invalid in this phase")]
    Order,
    #[error("worker progress stage must not be empty")]
    EmptyStage,
    #[error("worker artifacts must be nonempty, unique relative paths with valid hashes")]
    Artifacts,
    #[error("run failed final verification")]
    Verification,
    #[error("session is already terminal")]
    Terminal,
}
