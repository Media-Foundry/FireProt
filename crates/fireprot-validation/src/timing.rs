//! Descriptive repeat statistics for explicitly supplied observations.
//! This does not establish model parity, isolation or authenticity of timings.
use crate::{Error, Result};
use fireprot_core::{Request, ValidatedRequest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimingScope {
    Model,
    WorkerRequest,
    Supervisor,
    EndToEnd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingObservation {
    pub request: Request,
    pub scope: TimingScope,
    /// Explicit cache preparation description; unknown states must not be pooled.
    pub cache_state: String,
    pub environment: BTreeMap<String, String>,
    pub seconds: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingSeries {
    pub schema_version: u32,
    pub observations: Vec<TimingObservation>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct TimingSummary {
    pub count: usize,
    pub min: f64,
    pub median: f64,
    /// Nearest rank: sorted[ceil(0.95 * count) - 1].
    pub p95: f64,
    pub max: f64,
    pub mean: f64,
}

impl TimingSeries {
    pub fn summarize(&self) -> Result<TimingSummary> {
        if self.schema_version != 1 || self.observations.is_empty() {
            return Err(Error::Schema("unsupported or empty timing series".into()));
        }
        let first = &self.observations[0];
        let mut identifiers = BTreeSet::new();
        for observation in &self.observations {
            ValidatedRequest::try_from(observation.request.clone())
                .map_err(|e| Error::Schema(e.to_string()))?;
            if !identifiers.insert(&observation.request.run_id) {
                return Err(Error::Schema("duplicate timing run ID".into()));
            }
            if observation.cache_state.trim().is_empty()
                || observation.environment.is_empty()
                || observation
                    .environment
                    .iter()
                    .any(|(k, v)| k.trim().is_empty() || v.trim().is_empty())
            {
                return Err(Error::Schema("missing timing conditions".into()));
            }
            // Only the run's storage/identity may vary within a repeat series.
            // Keep seed, physical selection, precision, source and all artifacts.
            let mut request = observation.request.clone();
            request.run_id.clone_from(&first.request.run_id);
            request.output_dir.clone_from(&first.request.output_dir);
            if request != first.request
                || observation.scope != first.scope
                || observation.cache_state != first.cache_state
                || observation.environment != first.environment
            {
                return Err(Error::Schema("incompatible timing conditions".into()));
            }
        }
        summarize(
            &self
                .observations
                .iter()
                .map(|o| o.seconds)
                .collect::<Vec<_>>(),
        )
    }
}

/// All observations are retained: no implicit warmup removal, trimming or outlier
/// rejection. Invalid/empty samples are errors, never an empty successful report.
pub fn summarize(seconds: &[f64]) -> Result<TimingSummary> {
    if seconds.is_empty() {
        return Err(Error::Schema("empty timing samples".into()));
    }
    if seconds.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::Schema("timings must be finite and positive".into()));
    }
    let mut ordered = seconds.to_vec();
    ordered.sort_by(f64::total_cmp);
    let count = ordered.len();
    let min = ordered[0];
    let max = ordered[count - 1];
    let middle = count / 2;
    let median = if count % 2 == 0 {
        ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) / 2.0
    } else {
        ordered[middle]
    };
    // Scaling bounds the sum and compensated addition retains small terms.
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in &ordered {
        let term = value / max - compensation;
        let next = sum + term;
        compensation = (next - sum) - term;
        sum = next;
    }
    let mean = (max * (sum / count as f64).min(1.0)).max(min);
    // ceil(95*n/100) == n - floor(n/20), without integer multiplication overflow.
    let p95 = ordered[count - count / 20 - 1];
    Ok(TimingSummary {
        count,
        min,
        median,
        p95,
        max,
        mean,
    })
}
