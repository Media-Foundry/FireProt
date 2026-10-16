use std::{collections::BTreeMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::SCHEMA_VERSION;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Native,
    Fireprot,
    Chunked,
    Hip,
    HipFused,
    Triton,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Float32,
    Bfloat16,
    /// Preserve upstream mixed-precision boundaries, not an automatic cast.
    Native,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointRole {
    Model,
    LanguageModel,
    LanguageModelRegression,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourcePin {
    pub root: PathBuf,
    pub commit: String,
}

/// Keep model-specific cycle semantics explicit in the wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case", deny_unknown_fields)]
pub enum Workload {
    /// Standalone sequence encoder inference, with no folding or generation.
    Plm {
        model_name: String,
        precision: Precision,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        batch: Option<crate::plm_batch::Options>,
    },
    Protenix {
        model_name: String,
        /// Total trunk cycles, unlike the additional recycles below.
        cycles: u32,
        diffusion_steps: u32,
        samples: u32,
        precision: Precision,
    },
    Atlasfold {
        model_name: String,
        /// Additional passes; total passes equal recycles + 1.
        recycles: u32,
        diffusion_steps: u32,
        samples: u32,
        precision: Precision,
    },
    Openfold3 {
        model_name: String,
        /// Additional passes; total passes equal recycles + 1.
        recycles: u32,
        diffusion_steps: u32,
        samples: u32,
        precision: Precision,
        /// None preserves the upstream trainer default and legacy wire identity.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        deterministic: Option<bool>,
    },
    Esmfold2 {
        model_name: String,
        /// Upstream num_loops: the pinned release executes loops + 1 passes.
        loops: u32,
        /// Upstream schedule size before the native sigma cap. Record actual
        /// denoiser calls separately; do not relabel this as executed steps.
        sampling_steps: u32,
        samples: u32,
        precision: Precision,
        /// Native token-axis chunk size, identical in paired comparisons.
        chunk_size: u32,
        deterministic: bool,
    },
}

impl Workload {
    pub fn model_name(&self) -> &str {
        match self {
            Self::Plm { model_name, .. }
            | Self::Protenix { model_name, .. }
            | Self::Atlasfold { model_name, .. }
            | Self::Openfold3 { model_name, .. }
            | Self::Esmfold2 { model_name, .. } => model_name,
        }
    }

    pub fn trunk_passes(&self) -> Result<u32, RequestError> {
        match self {
            Self::Plm { .. } => Ok(1),
            Self::Protenix { cycles, .. } if *cycles > 0 => Ok(*cycles),
            Self::Protenix { .. } => Err(RequestError::ZeroCount("cycles")),
            Self::Atlasfold { recycles, .. } | Self::Openfold3 { recycles, .. } => {
                recycles.checked_add(1).ok_or(RequestError::CycleOverflow)
            }
            Self::Esmfold2 { loops, .. } => loops.checked_add(1).ok_or(RequestError::CycleOverflow),
        }
    }

    fn validate(&self, backend: Backend) -> Result<(), RequestError> {
        if self.model_name().trim().is_empty() {
            return Err(RequestError::EmptyField("model_name"));
        }
        self.trunk_passes()?;
        if let Self::Plm {
            model_name,
            precision,
            batch,
        } = self
        {
            if !matches!(
                model_name.as_str(),
                "esmc-fold2-6b" | "esm2-3b" | "esm3-open" | "atlaslm-3b"
            ) {
                return Err(RequestError::EmptyField(
                    "unsupported standalone pLM model_name",
                ));
            }
            if *precision != Precision::Native {
                return Err(RequestError::UnsupportedPrecision);
            }
            if let Some(options) = batch {
                options
                    .validate()
                    .map_err(|_| RequestError::EmptyField("invalid pLM batch options"))?;
            }
            if !matches!(backend, Backend::Native | Backend::Fireprot) {
                return Err(RequestError::UnsupportedBackend);
            }
            return Ok(());
        }
        if let Self::Esmfold2 {
            model_name,
            chunk_size,
            deterministic,
            ..
        } = self
        {
            if model_name != "biohub/ESMFold2" || *chunk_size == 0 {
                return Err(RequestError::Esmfold2Contract);
            }
            if backend == Backend::Fireprot && !deterministic {
                return Err(RequestError::Esmfold2Contract);
            }
        }
        let (steps, samples, precision, backends) = match self {
            Self::Plm { .. } => unreachable!("validated above"),
            Self::Protenix {
                diffusion_steps,
                samples,
                precision,
                ..
            } => (
                *diffusion_steps,
                *samples,
                *precision,
                &[Backend::Native, Backend::Fireprot][..],
            ),
            Self::Atlasfold {
                diffusion_steps,
                samples,
                precision,
                ..
            } => (
                *diffusion_steps,
                *samples,
                *precision,
                &[
                    Backend::Native,
                    Backend::Chunked,
                    Backend::Hip,
                    Backend::HipFused,
                ][..],
            ),
            Self::Openfold3 {
                diffusion_steps,
                samples,
                precision,
                ..
            } => (
                *diffusion_steps,
                *samples,
                *precision,
                &[Backend::Native, Backend::HipFused, Backend::Triton][..],
            ),
            Self::Esmfold2 {
                sampling_steps,
                samples,
                precision,
                ..
            } => (
                *sampling_steps,
                *samples,
                *precision,
                &[Backend::Native, Backend::Fireprot][..],
            ),
        };
        if steps == 0 {
            return Err(RequestError::ZeroCount("diffusion_steps"));
        }
        if samples == 0 {
            return Err(RequestError::ZeroCount("samples"));
        }
        if !backends.contains(&backend) {
            return Err(RequestError::UnsupportedBackend);
        }
        let valid_precision = match self {
            Self::Protenix { .. } => matches!(precision, Precision::Float32 | Precision::Bfloat16),
            Self::Plm { .. } | Self::Atlasfold { .. } | Self::Esmfold2 { .. } => {
                precision == Precision::Native
            }
            Self::Openfold3 { .. } => precision == Precision::Float32,
        };
        if !valid_precision {
            return Err(RequestError::UnsupportedPrecision);
        }
        if matches!(self, Self::Openfold3 { deterministic, .. }
            if backend == Backend::HipFused && *deterministic != Some(true))
        {
            return Err(RequestError::StrictDeterminismRequired);
        }
        Ok(())
    }
}

/// Environment and filesystem availability are checked by the runtime, after
/// this semantic validation and before executing any model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub run_id: String,
    pub workload: Workload,
    pub backend: Backend,
    pub seed: u64,
    /// Requested HIP visibility ordinal; never assumed to equal an SMI index.
    pub hip_device: u32,
    pub source: SourcePin,
    pub checkpoints: BTreeMap<CheckpointRole, ArtifactRef>,
    /// Named non-weight model assets, verified by the same preflight gate.
    #[serde(default)]
    pub resources: BTreeMap<String, ArtifactRef>,
    pub input: ArtifactRef,
    pub output_dir: PathBuf,
}

/// Construction goes through TryFrom so callers cannot bypass validation by
/// deserializing this type directly or mutating its inner request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct ValidatedRequest(Request);

impl ValidatedRequest {
    pub fn as_request(&self) -> &Request {
        &self.0
    }
    pub fn into_request(self) -> Request {
        self.0
    }
}

impl TryFrom<Request> for ValidatedRequest {
    type Error = RequestError;

    fn try_from(request: Request) -> Result<Self, Self::Error> {
        if request.schema_version != SCHEMA_VERSION {
            return Err(RequestError::Schema(request.schema_version));
        }
        if request.run_id.is_empty()
            || request.run_id.len() > 128
            || matches!(request.run_id.as_str(), "." | "..")
            || !request
                .run_id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_.-".contains(&c))
        {
            return Err(RequestError::RunId);
        }
        request.workload.validate(request.backend)?;
        if request.source.root.as_os_str().is_empty() {
            return Err(RequestError::EmptyField("source.root"));
        }
        if !hex_digest(&request.source.commit, 40) {
            return Err(RequestError::SourceCommit);
        }
        if request.output_dir.as_os_str().is_empty() {
            return Err(RequestError::EmptyField("output_dir"));
        }
        validate_artifact(&request.input)?;
        let roles: Vec<_> = request.checkpoints.keys().copied().collect();
        let required = match &request.workload {
            Workload::Atlasfold { .. } => {
                vec![CheckpointRole::Model, CheckpointRole::LanguageModel]
            }
            Workload::Protenix { model_name, .. } if model_name == "protenix_mini_esm_v0.5.0" => {
                vec![
                    CheckpointRole::Model,
                    CheckpointRole::LanguageModel,
                    CheckpointRole::LanguageModelRegression,
                ]
            }
            _ => vec![CheckpointRole::Model],
        };
        if roles != required {
            return Err(RequestError::CheckpointRoles);
        }
        for artifact in request.checkpoints.values() {
            validate_artifact(artifact)?;
        }
        for (name, artifact) in &request.resources {
            if name.is_empty() {
                return Err(RequestError::EmptyField("resource name"));
            }
            validate_artifact(artifact)?;
        }
        Ok(Self(request))
    }
}

fn hex_digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

pub(crate) fn validate_artifact(artifact: &ArtifactRef) -> Result<(), RequestError> {
    if artifact.path.as_os_str().is_empty() {
        return Err(RequestError::EmptyField("artifact.path"));
    }
    if !hex_digest(&artifact.sha256, 64) {
        return Err(RequestError::ArtifactDigest);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RequestError {
    #[error(
        "ESMFold2 requires biohub/ESMFold2, positive chunk_size, and deterministic FireProt execution"
    )]
    Esmfold2Contract,
    #[error("unsupported request schema version {0}")]
    Schema(u32),
    #[error("run_id must be 1–128 ASCII letters/digits/underscore/dot/hyphen, excluding . and ..")]
    RunId,
    #[error("{0} must not be empty")]
    EmptyField(&'static str),
    #[error("{0} must be greater than zero")]
    ZeroCount(&'static str),
    #[error("recycle count overflows the total trunk pass count")]
    CycleOverflow,
    #[error("backend is unsupported for this model family")]
    UnsupportedBackend,
    #[error("precision is unsupported for this model family")]
    UnsupportedPrecision,
    #[error("OpenFold3 hip_fused requires deterministic=true")]
    StrictDeterminismRequired,
    #[error("source commit must be 40 lowercase hexadecimal characters")]
    SourceCommit,
    #[error("artifact SHA-256 must be 64 lowercase hexadecimal characters")]
    ArtifactDigest,
    #[error(
        "checkpoint roles do not match the model; AtlasFold requires language_model; Mini-ESM also requires language_model_regression"
    )]
    CheckpointRoles,
}
