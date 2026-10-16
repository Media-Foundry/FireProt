use crate::artifacts::verify_outputs;
use crate::command::ProcessGroup;
use crate::preflight::PreparedRequest;
use crate::{
    build::BuildFile, build_tree::BuildTree, device::PhysicalDevice, execution::ExecutionRecord,
};
use fireprot_core::protocol::{CompletionChecks, DeviceIdentity, Envelope, Phase, Session};
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsString,
    fs::{self, File},
    io::{self, Read, Write},
    os::unix::{
        net::{UnixListener, UnixStream},
        process::CommandExt,
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

/// Independent host observations. Implementations must fail on missing or
/// malformed device/process records and bound the duration of each probe.
pub trait DeviceMonitor {
    fn physical_device(&self) -> Option<PhysicalDevice> {
        None
    }
    fn verify_device(&mut self, hip_device: u32, device: &DeviceIdentity) -> Result<(), String>;
    /// Return unexpected PIDs on the verified physical device, excluding only
    /// the launched worker and explicitly verified descendants.
    fn sample(&mut self, worker_pid: u32) -> Result<Vec<u32>, String>;
}

pub struct WorkerCommand {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub working_directory: PathBuf,
    pub environment: Vec<(OsString, OsString)>,
    pub identity_files: Vec<BuildFile>,
    pub identity_trees: Vec<BuildTree>,
    pub environment_inventory: Option<fireprot_core::request::ArtifactRef>,
}

pub struct Limits {
    pub timeout: Duration,
    pub poll_interval: Duration,
    pub monitor_interval: Duration,
    pub max_frame_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(7200),
            poll_interval: Duration::from_millis(10),
            monitor_interval: Duration::from_millis(250),
            max_frame_bytes: 1024 * 1024,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunReport {
    pub run_id: String,
    pub phase: Phase,
    pub worker_pid: u32,
    pub elapsed_seconds: f64,
    pub isolation_samples: usize,
    pub failure: Option<String>,
    pub announced_device: Option<DeviceIdentity>,
    pub device_verified: bool,
    pub physical_device: Option<PhysicalDevice>,
    pub execution_sha256: String,
}

/// Supervise one fresh output directory after mandatory provenance preflight.
/// `validate_content` must check
/// model-required arrays, shapes, original dtypes and finiteness; hashes alone
/// are insufficient. Failures after spawn are retained in supervisor.json.
pub fn run(
    prepared: &PreparedRequest,
    worker: &WorkerCommand,
    monitor: &mut dyn DeviceMonitor,
    limits: &Limits,
    cancelled: &AtomicBool,
    validate_content: impl FnOnce(&[PathBuf]) -> Result<(), String>,
) -> io::Result<RunReport> {
    if limits.timeout.is_zero()
        || limits.poll_interval.is_zero()
        || limits.monitor_interval.is_zero()
        || limits.max_frame_bytes == 0
    {
        return Err(io::Error::other("runtime limits must be positive"));
    }
    let request = prepared.request();
    let r = request.as_request();
    fs::create_dir(&r.output_dir)?; // Never overwrite or reuse a prior run.
    let output = r.output_dir.canonicalize()?;
    let request_path = output.join("request.json");
    fs::write(&request_path, serde_json::to_vec_pretty(request)?)?;
    fs::write(
        output.join("provenance.json"),
        serde_json::to_vec_pretty(prepared.provenance())?,
    )?;
    let channel = tempfile::tempdir()?;
    let socket_path = channel.path().join("worker.sock");
    let listener = UnixListener::bind(&socket_path)?;
    listener.set_nonblocking(true)?;
    let mut environment = crate::execution::environment(std::env::vars_os(), &worker.environment)?;
    for key in [
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ] {
        environment.remove(key);
    }
    environment.insert("HIP_VISIBLE_DEVICES".into(), r.hip_device.to_string());
    for (key, path) in [
        ("FIREPROT_WORKER_SOCKET", &socket_path),
        ("FIREPROT_REQUEST", &request_path),
        ("FIREPROT_OUTPUT_DIR", &output),
    ] {
        environment.insert(
            key.into(),
            path.to_str()
                .ok_or_else(|| io::Error::other("worker runtime paths must be UTF-8"))?
                .into(),
        );
    }
    let execution = ExecutionRecord::capture(
        &worker.program,
        &worker.args,
        &worker.working_directory,
        environment,
        &worker.identity_files,
        &worker.identity_trees,
        worker.environment_inventory.as_ref(),
    )?;
    fs::write(
        output.join("execution.json"),
        serde_json::to_vec_pretty(&execution)?,
    )?;
    let mut command = Command::new(&execution.invocation_path);
    command
        .args(&worker.args)
        .current_dir(&execution.working_directory)
        .env_clear()
        .envs(&execution.environment)
        .stdin(Stdio::null())
        .stdout(Stdio::from(File::create(output.join("stdout.log"))?))
        .stderr(Stdio::from(File::create(output.join("stderr.log"))?))
        .process_group(0);
    let started = Instant::now();
    #[cfg(target_os = "linux")]
    let mut affinity = crate::cpu_affinity::ScopedAffinity::enter(
        execution
            .environment
            .get(crate::cpu_affinity::ENV)
            .map(String::as_str),
    )?;
    let spawned = command.spawn().map(|child| ProcessGroup(child, false));
    // Restore even when spawn fails. A post-spawn restoration error follows the
    // normal failure/report path, which stops the child group before returning.
    #[cfg(target_os = "linux")]
    let restoration = affinity
        .restore()
        .map_err(|error| format!("restoring launch thread CPU affinity failed: {error}"));
    #[cfg(not(target_os = "linux"))]
    let restoration: Result<(), String> = Ok(());
    let mut child = spawned?;
    let pid = child.0.id();
    let mut session = Session::new(request, pid).map_err(io::Error::other)?;
    let mut samples = 0;
    let mut device_verified = false;
    let outcome = restoration.and_then(|()| {
        drive(
            &mut child,
            &listener,
            &mut session,
            monitor,
            limits,
            cancelled,
            r.hip_device,
            started,
            &mut samples,
            &execution,
            worker,
            &mut device_verified,
        )
    });
    // Stop the whole group before inspecting output files on either path.
    child.stop();
    let verification = outcome.and_then(|exit_code| {
        execution
            .verify(&worker.identity_files, &worker.identity_trees)
            .map_err(|e| e.to_string())?;
        if exit_code != Some(0) || session.phase() != Phase::AwaitingVerification {
            return Err("worker exited unsuccessfully or without predictions".into());
        }
        let paths = verify_outputs(&output, session.artifacts()).map_err(|e| e.to_string())?;
        validate_content(&paths)?;
        check_limits(cancelled, started, limits)?;
        session
            .finish(CompletionChecks {
                exit_code,
                verified_artifacts: paths.len(),
                isolation_samples: samples,
                external_pids: vec![],
                monitor_errors: vec![],
            })
            .map_err(|e| e.to_string())
    });
    if let Err(error) = verification {
        if session.phase() != Phase::Failed {
            let _ = session.abort(error);
        }
    }
    let mut report = RunReport {
        run_id: r.run_id.clone(),
        phase: session.phase(),
        worker_pid: pid,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        isolation_samples: samples,
        failure: session.failure().map(str::to_owned),
        announced_device: session.device().cloned(),
        device_verified,
        physical_device: monitor.physical_device(),
        execution_sha256: execution.sha256.clone(),
    };
    if report.phase == Phase::Succeeded {
        let metadata = vec![
            crate::run_archive::json_identity("request.json", request)?,
            crate::run_archive::json_identity("provenance.json", prepared.provenance())?,
            crate::run_archive::json_identity("execution.json", &execution)?,
        ];
        if let Err(error) =
            crate::run_archive::seal(&output, metadata, &report, session.artifacts())
        {
            report.phase = Phase::Failed;
            report.failure = Some(format!("acceptance archive failed: {error}"));
        }
    }
    write_report(&output, &report)?;
    Ok(report)
}

fn write_report(output: &Path, report: &RunReport) -> io::Result<()> {
    let pending = output.join(".supervisor.json.tmp");
    fs::write(&pending, serde_json::to_vec_pretty(report)?)?;
    fs::rename(pending, output.join("supervisor.json"))
}

fn check_limits(cancelled: &AtomicBool, started: Instant, limits: &Limits) -> Result<(), String> {
    if cancelled.load(Ordering::Relaxed) {
        return Err("run cancelled".into());
    }
    if started.elapsed() >= limits.timeout {
        return Err("worker timeout".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn drive(
    child: &mut ProcessGroup,
    listener: &UnixListener,
    session: &mut Session,
    monitor: &mut dyn DeviceMonitor,
    limits: &Limits,
    cancelled: &AtomicBool,
    hip_device: u32,
    started: Instant,
    samples: &mut usize,
    execution: &ExecutionRecord,
    worker: &WorkerCommand,
    device_verified: &mut bool,
) -> Result<Option<i32>, String> {
    let mut stream: Option<UnixStream> = None;
    let mut pending = Vec::new();
    let mut eof = false;
    let mut last_sample = Instant::now();
    loop {
        check_limits(cancelled, started, limits)?;
        if stream.is_none() {
            match listener.accept() {
                Ok((socket, _)) => {
                    socket.set_nonblocking(true).map_err(|e| e.to_string())?;
                    stream = Some(socket);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e.to_string()),
            }
        }
        if let Some(socket) = stream.as_mut() {
            // Bound work per tick so floods cannot starve monitoring/cancellation.
            let mut buffer = [0u8; 8192];
            match socket.read(&mut buffer) {
                Ok(0) => eof = true,
                Ok(n) => pending.extend_from_slice(&buffer[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e.to_string()),
            }
            while let Some(end) = pending.iter().position(|&b| b == b'\n') {
                if end > limits.max_frame_bytes {
                    return Err("worker frame too large".into());
                }
                let event: Envelope = serde_json::from_slice(&pending[..end])
                    .map_err(|e| format!("invalid worker frame: {e}"))?;
                pending.drain(..=end);
                session.receive(event).map_err(|e| e.to_string())?;
                if session.phase() == Phase::Failed {
                    return Err(session.failure().unwrap_or("worker failed").into());
                }
                if session.phase() == Phase::AwaitingStart {
                    execution
                        .verify_process(child.0.id())
                        .map_err(|e| e.to_string())?;
                    execution
                        .verify(&worker.identity_files, &worker.identity_trees)
                        .map_err(|e| e.to_string())?;
                    monitor.verify_device(hip_device, session.device().unwrap())?;
                    *device_verified = true;
                    sample(monitor, child.0.id(), samples)?;
                    last_sample = Instant::now();
                    check_limits(cancelled, started, limits)?;
                    session.authorize_start().map_err(|e| e.to_string())?;
                    socket
                        .write_all(b"{\"schema_version\":1,\"command\":\"start\"}\n")
                        .map_err(|e| e.to_string())?;
                }
            }
            if pending.len() > limits.max_frame_bytes {
                return Err("worker frame too large".into());
            }
            if eof && !pending.is_empty() {
                return Err("unterminated worker frame".into());
            }
        }
        if matches!(
            session.phase(),
            Phase::Running | Phase::AwaitingVerification
        ) && last_sample.elapsed() >= limits.monitor_interval
        {
            sample(monitor, child.0.id(), samples)?;
            last_sample = Instant::now();
        }
        if let Some(status) = child.0.try_wait().map_err(|e| e.to_string())? {
            // Drain the protocol stream after exit. Buffered frames may remain.
            if stream.is_none() || eof {
                if session.phase() == Phase::AwaitingVerification {
                    sample(monitor, child.0.id(), samples)?;
                }
                return Ok(status.code());
            }
        }
        thread::sleep(limits.poll_interval);
    }
}

fn sample(monitor: &mut dyn DeviceMonitor, pid: u32, samples: &mut usize) -> Result<(), String> {
    let external = monitor.sample(pid)?;
    *samples += 1;
    if !external.is_empty() {
        return Err(format!("external GPU processes: {external:?}"));
    }
    Ok(())
}
