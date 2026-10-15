//! Assemble a pinned model worker and the independent device monitor.
use crate::support::{Result, cancellation, read_json, read_request};
use fireprot_core::request::ArtifactRef;
use fireprot_runtime::{
    build::BuildFile,
    build_tree::BuildTree,
    device::AmdSmiMonitor,
    execution,
    preflight::{SourceOrigin, prepare},
    supervisor::{self, Limits, RunReport, WorkerCommand},
};
use fireprot_validation::bundle;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    python: PathBuf,
    workers: PathBuf,
    hip_probe: PathBuf,
    amd_smi: PathBuf,
    source_manifest: Option<ArtifactRef>,
    #[serde(default)]
    environment_inventory: Option<ArtifactRef>,
    #[serde(default)]
    environment: BTreeMap<String, String>,
}

pub(crate) fn validate_profile(bytes: &[u8]) -> Result<()> {
    let _: Execution = serde_json::from_slice(bytes)?;
    Ok(())
}

pub fn run(request: &Path, profile: &Path) -> Result<RunReport> {
    let request = read_request(request)?;
    let execution: Execution = read_json(profile)?;
    let cancelled = cancellation()?;
    run_loaded(request, execution, &cancelled)
}

/// Validate the dedicated command family and execute the same immutable request.
pub fn run_plm(request: &Path, profile: &Path, family: &str) -> Result<RunReport> {
    let request = read_request(request)?;
    let fireprot_core::request::Workload::Plm { model_name, .. } = &request.as_request().workload
    else {
        return Err("standalone pLM command requires a plm workload".into());
    };
    if (family == "atlaslm") != (model_name == "atlaslm-3b") {
        return Err("pLM command does not match the requested model family".into());
    }
    let execution: Execution = read_json(profile)?;
    let cancelled = cancellation()?;
    run_loaded(request, execution, &cancelled)
}

pub fn run_batch(path: &Path) -> Result<fireprot_runtime::batch::BatchReport> {
    use fireprot_runtime::batch::{self, BatchPlan};
    let cancelled = cancellation()?;
    let plan = BatchPlan::load(batch::read_spec(path)?)?;
    // Validate every profile before allowing the first model to start.
    let profiles: Vec<Execution> = plan
        .jobs()
        .iter()
        .map(|job| serde_json::from_slice(job.profile()))
        .collect::<std::result::Result<_, _>>()?;
    Ok(batch::execute(&plan, &cancelled, |index, job| {
        run_loaded(job.request().clone(), profiles[index].clone(), &cancelled)
            .map_err(|e| e.to_string())
    }))
}

fn run_loaded(
    request: fireprot_core::ValidatedRequest,
    execution: Execution,
    cancelled: &AtomicBool,
) -> Result<RunReport> {
    if cancelled.load(Ordering::Relaxed) {
        return Err("inference cancelled".into());
    }
    let origin = match execution.source_manifest {
        Some(manifest) => SourceOrigin::Exported { manifest },
        None => SourceOrigin::GitCheckout,
    };
    let prepared = prepare(request, origin)?;
    let workers = execution.workers.canonicalize()?;
    let root = prepared.request().as_request().source.root.clone();
    let framework = workers
        .parent()
        .ok_or("worker package needs a framework parent directory")?
        .to_path_buf();
    let pythonpath = std::env::join_paths([
        workers.clone(),
        root.join("src"),
        root.clone(),
        framework.clone(),
    ])?;
    let mut env: Vec<_> = execution
        .environment
        .into_iter()
        .map(|(k, v)| (k.into(), v.into()))
        .collect();
    env.push(("PYTHONPATH".into(), pythonpath));
    env.push(("PYTHONNOUSERSITE".into(), "1".into()));
    if execution.environment_inventory.is_some() {
        env.push(("PYTHONDONTWRITEBYTECODE".into(), "1".into()));
    }
    let program = std::path::absolute(execution.python)?;
    let host_environment = execution::environment(std::env::vars_os(), &env)?;
    let hip_probe = execution.hip_probe.canonicalize()?;
    let amd_smi = execution::resolve_program(
        &execution.amd_smi,
        &std::env::current_dir()?,
        &host_environment,
    )?;
    let mut identity_files = vec![
        BuildFile {
            name: "supervisor".into(),
            path: std::env::current_exe()?,
        },
        BuildFile {
            name: "worker-entry".into(),
            path: workers.join("fireprot_workers/__main__.py"),
        },
        BuildFile {
            name: "hip-probe".into(),
            path: hip_probe.clone(),
        },
        BuildFile {
            name: "amd-smi".into(),
            path: amd_smi.clone(),
        },
        BuildFile {
            name: "root-import-shim".into(),
            path: framework.join("fireprot/__init__.py"),
        },
    ];
    let venv = program
        .parent()
        .and_then(std::path::Path::parent)
        .map(|p| p.join("pyvenv.cfg"));
    if let Some(path) = venv.filter(|p| p.is_file()) {
        identity_files.push(BuildFile {
            name: "python-venv".into(),
            path,
        });
    }
    if prepared.request().as_request().backend != fireprot_core::request::Backend::Native {
        let path = env
            .iter()
            .find(|(key, _)| key == "FIREPROT_BRIDGE_LIBRARY")
            .map(|(_, v)| PathBuf::from(v))
            .unwrap_or_else(|| framework.join("target/release/lib_fireprot_bridge.so"));
        identity_files.push(BuildFile {
            name: "rust-bridge".into(),
            path,
        });
    }
    let identity_trees = vec![
        BuildTree {
            name: "workers".into(),
            path: workers.join("fireprot_workers"),
            suffixes: vec![".py".into()],
        },
        BuildTree {
            name: "tensor-boundary".into(),
            path: workers.join("fireprot"),
            suffixes: vec![".py".into()],
        },
        BuildTree {
            name: "amd-kernels".into(),
            path: framework.join("kernels/amd"),
            suffixes: vec![
                ".cpp".into(),
                ".cu".into(),
                ".hip".into(),
                ".hip".into(),
                ".h".into(),
                ".hpp".into(),
            ],
        },
    ];
    let worker = WorkerCommand {
        program,
        args: vec!["-m".into(), "fireprot_workers".into()],
        // The supervisor creates this fresh directory before resolving the
        // execution identity. Upstream relative caches must be run-local.
        working_directory: prepared.request().as_request().output_dir.clone(),
        environment: env,
        identity_files,
        identity_trees,
        environment_inventory: execution.environment_inventory,
    };
    let mut monitor = AmdSmiMonitor::new(hip_probe, amd_smi, Duration::from_secs(10));
    let report = supervisor::run(
        &prepared,
        &worker,
        &mut monitor,
        &Limits::default(),
        cancelled,
        |paths| {
            bundle::validate_current(prepared.request(), paths)
                .map(|_| ())
                .map_err(|e| e.to_string())
        },
    )?;

    Ok(report)
}
