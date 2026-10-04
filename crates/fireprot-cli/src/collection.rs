//! Compose archived runtime evidence with model-content and timing validation.
use fireprot_core::{Request, ValidatedRequest, request::ArtifactRef};
use fireprot_runtime::{device::normalize_pci, measured_run, run_archive};
use fireprot_validation::{
    bundle,
    timing::{TimingObservation, TimingScope, TimingSeries, TimingSummary},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collection {
    schema_version: u32,
    scope: TimingScope,
    /// Caller declaration: cache preparation cannot be inferred after execution.
    cache_state: String,
    runs: Vec<ArtifactRef>,
    /// Required, in run order, only for complete-invocation E2E collection.
    #[serde(default)]
    invocations: Vec<ArtifactRef>,
}

impl Collection {
    pub fn single(pin: ArtifactRef) -> Self {
        Self {
            schema_version: 1,
            scope: TimingScope::Supervisor,
            cache_state: "content validation only; not a performance comparison".into(),
            runs: vec![pin],
            invocations: Vec::new(),
        }
    }
}

#[derive(Serialize)]
pub struct Collected {
    pub(crate) series: TimingSeries,
    summary: TimingSummary,
    pub(crate) receipts: Vec<ArtifactRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    invocations: Vec<ArtifactRef>,
}

pub fn collect(plan: Collection) -> Result<Collected> {
    if plan.schema_version != 1
        || plan.runs.is_empty()
        || plan.cache_state.trim().is_empty()
        || if plan.scope == TimingScope::EndToEnd {
            plan.invocations.len() != plan.runs.len()
        } else {
            !plan.invocations.is_empty()
        }
    {
        return Err(
            "invalid collection; E2E requires one pinned invocation record per accepted run".into(),
        );
    }
    let mut observations = Vec::new();
    let mut roots = BTreeSet::new();
    for (index, pin) in plan.runs.iter().enumerate() {
        let archive = run_archive::verify(pin)?;
        if !roots.insert(archive.root.clone()) {
            return Err("duplicate run directory".into());
        }
        let request: Request =
            serde_json::from_slice(&fs::read(archive.root.join("request.json"))?)?;
        let _: ValidatedRequest = request.clone().try_into()?;
        let report: Value =
            serde_json::from_slice(&fs::read(archive.root.join("supervisor.json"))?)?;
        let mut launch: Value =
            serde_json::from_slice(&fs::read(archive.root.join("execution.json"))?)?;
        if report["run_id"] != request.run_id
            || report["phase"] != "succeeded"
            || !report.get("failure").is_some_and(Value::is_null)
            || report["device_verified"] != true
            || report["isolation_samples"].as_u64().unwrap_or(0) == 0
            || report["worker_pid"].as_u64().unwrap_or(0) == 0
            || launch["schema_version"] != 1
            || launch["environment_policy"] != "explicit-v1"
            || !launch["sha256"].is_string()
            || report["execution_sha256"] != launch["sha256"]
        {
            return Err("run did not retain successful execution/isolation evidence".into());
        }
        let physical = &report["physical_device"];
        let announced = &report["announced_device"];
        let pci = normalize_pci(
            physical["pci_address"]
                .as_str()
                .ok_or("missing physical PCI")?,
        )?;
        if physical["hip_device"] != request.hip_device
            || physical["smi_device"].as_u64().is_none()
            || announced["worker_pid"] != report["worker_pid"]
            || announced["logical_device"] != 0
            || normalize_pci(
                announced["pci_address"]
                    .as_str()
                    .ok_or("missing announced PCI")?,
            )? != pci
            || physical["architecture"].as_str().is_none_or(str::is_empty)
            || physical["architecture"] != announced["architecture"]
        {
            return Err("inconsistent archived device mapping".into());
        }
        let elapsed = report["elapsed_seconds"]
            .as_f64()
            .ok_or("missing supervisor timing")?;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err("invalid supervisor timing".into());
        }
        // Allow an intact archive to be copied, without rewriting the stored request.
        let mut relocated = request.clone();
        relocated.output_dir = archive.root.clone();
        let content = bundle::validate(&relocated.try_into()?, &archive.outputs)?;
        let invocation = plan
            .invocations
            .get(index)
            .map(|invocation| measured_run::verify(invocation, pin, &archive))
            .transpose()?;
        let seconds = match plan.scope {
            TimingScope::Model => content.model_seconds,
            TimingScope::WorkerRequest => content.request_seconds,
            TimingScope::Supervisor => elapsed,
            TimingScope::EndToEnd => {
                let seconds = invocation
                    .as_ref()
                    .ok_or("missing invocation record")?
                    .elapsed_seconds;
                if seconds < elapsed {
                    return Err(
                        "complete invocation is shorter than its supervisor interval".into(),
                    );
                }
                seconds
            }
        };
        let environment = launch["environment"]
            .as_object_mut()
            .ok_or("missing launch environment")?;
        if environment
            .get("HIP_VISIBLE_DEVICES")
            .and_then(Value::as_str)
            != Some(&request.hip_device.to_string())
        {
            return Err("inconsistent archived visibility".into());
        }
        // These three values identify a particular invocation, not its conditions.
        for key in [
            "FIREPROT_REQUEST",
            "FIREPROT_OUTPUT_DIR",
            "FIREPROT_WORKER_SOCKET",
        ] {
            environment.remove(key);
        }
        launch
            .as_object_mut()
            .ok_or("invalid launch record")?
            .remove("sha256");
        // Fresh workers run inside their own output directory. Compare that
        // relationship using archived paths, without consulting the live FS.
        // An object marker cannot collide with a historical fixed path string.
        if launch["working_directory"].as_str() == request.output_dir.to_str() {
            launch["working_directory"] = serde_json::json!({"relative_to": "request.output_dir"});
        }
        let provenance = archive
            .receipt
            .metadata
            .iter()
            .find(|p| p.path == std::path::Path::new("provenance.json"))
            .ok_or("missing provenance pin")?;
        let mut environment = BTreeMap::from([
            ("launch".into(), serde_json::to_string(&launch)?),
            (
                "worker".into(),
                serde_json::to_string(&content.environment)?,
            ),
            ("physical_device".into(), serde_json::to_string(physical)?),
            ("provenance_sha256".into(), provenance.sha256.clone()),
        ]);
        if let Some(invocation) = invocation {
            environment.insert(
                "invocation".into(),
                serde_json::to_string(&invocation.conditions())?,
            );
        }
        observations.push(TimingObservation {
            request,
            scope: plan.scope,
            cache_state: plan.cache_state.clone(),
            seconds,
            environment,
        });
    }
    let series = TimingSeries {
        schema_version: 1,
        observations,
    };
    let summary = series.summarize()?;
    Ok(Collected {
        series,
        summary,
        receipts: plan.runs,
        invocations: plan.invocations,
    })
}
