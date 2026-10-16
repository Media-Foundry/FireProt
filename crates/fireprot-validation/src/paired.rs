//! Paired descriptive performance comparison with explicit treatment changes.
//! Archive authentication and model correctness must be checked by the caller.
use crate::{
    Error, Result,
    timing::{TimingSeries, TimingSummary, summarize},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Difference {
    pub baseline: Option<Value>,
    pub candidate: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct Comparison {
    pub baseline: TimingSummary,
    pub candidate: TimingSummary,
    pub median_speedup: f64,
    pub p95_speedup: f64,
    /// Per-pair ratios; distinct from the ratio of whole-series medians.
    pub paired_speedups: Vec<f64>,
    pub paired_summary: TimingSummary,
    pub observed_trend: Trend,
    pub changes: BTreeMap<String, Difference>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Trend {
    InsufficientRepeats,
    FasterMedianAndP95,
    SlowerMedianAndP95,
    MixedOrUnchanged,
}

#[derive(Debug, Serialize)]
pub struct SampleComparison {
    pub sample: u32,
    pub policy: &'static str,
    pub coordinates: Option<crate::metrics::CoordinateMetrics>,
    pub confidence: Option<crate::metrics::ConfidenceMetrics>,
    pub exact_arrays: BTreeMap<String, bool>,
    /// Informational only: established Protenix gates do not require bitwise
    /// equality of these full floating confidence arrays.
    pub floating_confidence_diagnostics: BTreeMap<String, crate::array::ArrayComparison>,
    pub passed: bool,
}

/// Compare already content-validated predictions. Thresholds are fixed to the
/// existing model gates; callers cannot turn a failed run into a relaxed pass.
pub fn compare_prediction(
    workload: &fireprot_core::request::Workload,
    baseline: &crate::bundle::Prediction,
    baseline_arrays: &BTreeMap<String, crate::array::Array>,
    candidate: &crate::bundle::Prediction,
    candidate_arrays: &BTreeMap<String, crate::array::Array>,
) -> Result<SampleComparison> {
    use crate::{
        array::{Array, DType, Policy},
        metrics,
    };
    if baseline.sample != candidate.sample
        || !baseline.arrays.keys().eq(candidate.arrays.keys())
        || !baseline.arrays.keys().eq(baseline_arrays.keys())
        || !candidate.arrays.keys().eq(candidate_arrays.keys())
    {
        return Err(error("sample identity or output array names differ"));
    }
    let protenix = matches!(workload, fireprot_core::request::Workload::Protenix { .. });
    let mut output = SampleComparison {
        sample: baseline.sample,
        policy: if protenix {
            "protenix-established-1e-3-v1"
        } else {
            "all-arrays-exact-v1"
        },
        coordinates: None,
        confidence: None,
        exact_arrays: BTreeMap::new(),
        floating_confidence_diagnostics: BTreeMap::new(),
        passed: true,
    };
    for (key, desc) in &baseline.arrays {
        let other = &candidate.arrays[key];
        if desc.shape != other.shape
            || desc.storage_dtype != other.storage_dtype
            || desc.original_dtype != other.original_dtype
        {
            return Err(error("prediction native shape/dtype metadata differs"));
        }
        let restore = |array: &Array| -> Result<Array> {
            if array.dtype() != desc.storage_dtype
                || array.shape() != desc.shape
                || (desc.original_dtype != desc.storage_dtype
                    && !(desc.original_dtype == DType::Bfloat16
                        && desc.storage_dtype == DType::Uint16))
            {
                return Err(error("prediction tensor disagrees with descriptor"));
            }
            Array::new(
                desc.original_dtype,
                desc.shape.clone(),
                array.bytes().to_vec(),
            )
        };
        let a = restore(&baseline_arrays[key])?;
        let b = restore(&candidate_arrays[key])?;
        if protenix && key == "coordinates" {
            output.coordinates = Some(metrics::coordinates(&a.coordinates()?, &b.coordinates()?)?);
        } else {
            let comparison = crate::array::compare(&a, &b, Policy::Exact)?;
            if protenix
                && [
                    "confidence.atom_plddt",
                    "confidence.token_pair_pae",
                    "confidence.token_pair_pde",
                    "confidence.contact_probs",
                ]
                .contains(&key.as_str())
            {
                output
                    .floating_confidence_diagnostics
                    .insert(key.clone(), comparison);
            } else {
                output.passed &= comparison.passed;
                output.exact_arrays.insert(key.clone(), comparison.passed);
            }
        }
    }
    if protenix {
        let confidence = metrics::confidence(
            baseline
                .confidence
                .as_ref()
                .ok_or_else(|| error("missing reference confidence"))?,
            candidate
                .confidence
                .as_ref()
                .ok_or_else(|| error("missing candidate confidence"))?,
        )?;
        output.passed &= metrics::protenix_passes(
            output
                .coordinates
                .as_ref()
                .ok_or_else(|| error("missing coordinates"))?,
            &confidence,
            1e-3,
            1e-3,
        )?;
        output.confidence = Some(confidence);
    }
    Ok(output)
}

fn error(message: impl Into<String>) -> Error {
    Error::Schema(message.into())
}

fn parse(environment: &BTreeMap<String, String>, key: &str) -> Result<Value> {
    serde_json::from_str(
        environment
            .get(key)
            .ok_or_else(|| error(format!("missing {key} conditions")))?,
    )
    .map_err(|e| error(e.to_string()))
}

fn difference(
    key: String,
    a: Option<&Value>,
    b: Option<&Value>,
    out: &mut BTreeMap<String, Difference>,
) {
    if a != b {
        out.insert(
            key,
            Difference {
                baseline: a.cloned(),
                candidate: b.cloned(),
            },
        );
    }
}

fn named_inputs(launch: &Value) -> Result<BTreeMap<String, Value>> {
    let inputs = launch["inputs"]
        .as_array()
        .ok_or_else(|| error("missing launch input identities"))?;
    let mut result = BTreeMap::new();
    for input in inputs {
        let name = input["name"]
            .as_str()
            .ok_or_else(|| error("unnamed launch input"))?;
        if result.insert(name.into(), input.clone()).is_some() {
            return Err(error("duplicate launch input"));
        }
    }
    Ok(result)
}

/// Discover only permitted treatment differences. Scientific workload, random
/// state, source/resources, software versions and physical GPU remain equal.
pub fn differences(
    baseline: &TimingSeries,
    candidate: &TimingSeries,
) -> Result<BTreeMap<String, Difference>> {
    differences_with_monitor_treatment(baseline, candidate, false)
}

/// Opt-in replacement of an existing AMD SMI monitor, for E2E experiments only.
/// This does not establish monitor equivalence; independent qualification is
/// required. Exact observed identities must still be declared by the caller.
pub fn differences_with_monitor_treatment(
    baseline: &TimingSeries,
    candidate: &TimingSeries,
    monitor_treatment: bool,
) -> Result<BTreeMap<String, Difference>> {
    baseline.summarize()?;
    candidate.summarize()?;
    if baseline.observations.len() != candidate.observations.len() {
        return Err(error("unequal pair counts"));
    }
    let mut ids = BTreeSet::new();
    for observation in baseline.observations.iter().chain(&candidate.observations) {
        if !ids.insert(&observation.request.run_id) {
            return Err(error("run reused across comparison arms"));
        }
    }
    let a = &baseline.observations[0];
    let b = &candidate.observations[0];
    if monitor_treatment && a.scope != crate::timing::TimingScope::EndToEnd {
        return Err(error("monitor treatment requires end-to-end timing"));
    }
    let mut request = b.request.clone();
    request.run_id.clone_from(&a.request.run_id);
    request.output_dir.clone_from(&a.request.output_dir);
    request.backend = a.request.backend;
    if request != a.request || a.scope != b.scope || a.cache_state != b.cache_state {
        return Err(error(
            "paired workload, seed, device, source, artifacts, cache or scope mismatch",
        ));
    }
    if !a.environment.keys().eq(b.environment.keys()) {
        return Err(error("comparison condition fields differ"));
    }
    for key in ["physical_device", "worker", "provenance_sha256"] {
        if !a.environment.contains_key(key) || a.environment.get(key) != b.environment.get(key) {
            return Err(error(format!("protected {key} conditions differ")));
        }
    }
    if a.environment.keys().any(|k| {
        ![
            "physical_device",
            "worker",
            "provenance_sha256",
            "launch",
            "invocation",
        ]
        .contains(&k.as_str())
    }) {
        return Err(error("unrecognized comparison condition"));
    }
    let mut changes = BTreeMap::new();
    difference(
        "backend".into(),
        Some(&json!(a.request.backend)),
        Some(&json!(b.request.backend)),
        &mut changes,
    );
    let mut la = parse(&a.environment, "launch")?;
    let mut lb = parse(&b.environment, "launch")?;
    let ai = named_inputs(&la)?;
    let bi = named_inputs(&lb)?;
    let ae = la["environment"]
        .as_object()
        .ok_or_else(|| error("missing launch environment"))?;
    let be = lb["environment"]
        .as_object()
        .ok_or_else(|| error("missing launch environment"))?;
    let env_keys: BTreeSet<_> = ae.keys().chain(be.keys()).collect();
    for key in env_keys {
        if ae.get(key) != be.get(key) {
            if key == "FIREPROT_AMD_SMI_PROTOCOL" && !monitor_treatment {
                return Err(error(
                    "monitor protocol change requires explicit E2E treatment",
                ));
            }
            if !key.starts_with("FIREPROT_") {
                return Err(error(format!(
                    "protected launch environment changed: {key}"
                )));
            }
            difference(
                format!("launch.environment.{key}"),
                ae.get(key),
                be.get(key),
                &mut changes,
            );
        }
    }
    let input_keys: BTreeSet<_> = ai.keys().chain(bi.keys()).collect();
    for key in input_keys {
        if ai.get(key) != bi.get(key) {
            let monitor_replacement = monitor_treatment
                && key == "amd-smi"
                && ai.contains_key(key)
                && bi.contains_key(key);
            if key != "rust-bridge" && !key.starts_with("amd-kernels/") && !monitor_replacement {
                return Err(error(format!("protected launch input changed: {key}")));
            }
            difference(
                format!("launch.inputs.{key}"),
                ai.get(key),
                bi.get(key),
                &mut changes,
            );
        }
    }
    for launch in [&mut la, &mut lb] {
        let object = launch
            .as_object_mut()
            .ok_or_else(|| error("invalid launch conditions"))?;
        object.remove("environment");
        object.remove("inputs");
    }
    if la != lb {
        return Err(error(
            "protected executable, arguments or working directory changed",
        ));
    }
    if a.environment.contains_key("invocation") {
        let mut ia = parse(&a.environment, "invocation")?;
        let mut ib = parse(&b.environment, "invocation")?;
        difference(
            "invocation.profile_sha256".into(),
            ia.get("profile_sha256"),
            ib.get("profile_sha256"),
            &mut changes,
        );
        for invocation in [&mut ia, &mut ib] {
            invocation
                .as_object_mut()
                .ok_or_else(|| error("invalid invocation conditions"))?
                .remove("profile_sha256");
        }
        if ia != ib {
            return Err(error("protected invocation conditions differ"));
        }
    }
    Ok(changes)
}

pub fn compare(
    baseline: &TimingSeries,
    candidate: &TimingSeries,
    expected: &BTreeMap<String, Difference>,
) -> Result<Comparison> {
    compare_with_monitor_treatment(baseline, candidate, expected, false)
}

/// Explicit E2E monitor treatment; default `compare` keeps monitors protected.
pub fn compare_with_monitor_treatment(
    baseline: &TimingSeries,
    candidate: &TimingSeries,
    expected: &BTreeMap<String, Difference>,
    monitor_treatment: bool,
) -> Result<Comparison> {
    let changes = differences_with_monitor_treatment(baseline, candidate, monitor_treatment)?;
    if &changes != expected {
        return Err(error(format!(
            "treatment declaration differs from observed changes: {}",
            json!(changes)
        )));
    }
    let baseline_summary = baseline.summarize()?;
    let candidate_summary = candidate.summarize()?;
    let ratio = |a: f64, b: f64| {
        let value = a / b;
        if !value.is_finite() || value <= 0.0 {
            Err(Error::Nonfinite)
        } else {
            Ok(value)
        }
    };
    let median_speedup = ratio(baseline_summary.median, candidate_summary.median)?;
    let p95_speedup = ratio(baseline_summary.p95, candidate_summary.p95)?;
    let paired_speedups: Vec<_> = baseline
        .observations
        .iter()
        .zip(&candidate.observations)
        .map(|(a, b)| ratio(a.seconds, b.seconds))
        .collect::<Result<_>>()?;
    let observed_trend = if paired_speedups.len() < 3 {
        Trend::InsufficientRepeats
    } else if median_speedup > 1.0 && p95_speedup > 1.0 {
        Trend::FasterMedianAndP95
    } else if median_speedup < 1.0 && p95_speedup < 1.0 {
        Trend::SlowerMedianAndP95
    } else {
        Trend::MixedOrUnchanged
    };
    Ok(Comparison {
        baseline: baseline_summary,
        candidate: candidate_summary,
        median_speedup,
        p95_speedup,
        paired_summary: summarize(&paired_speedups)?,
        paired_speedups,
        observed_trend,
        changes,
    })
}
