//! Bounded subprocesses for host probes; stdout and stderr never share a pipe.
use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use std::{
    io::{self, Read, Seek, SeekFrom},
    os::unix::process::CommandExt,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

pub(crate) struct ProcessGroup(pub(crate) Child, pub(crate) bool);
impl ProcessGroup {
    pub(crate) fn stop(&mut self) {
        if self.1 {
            return;
        }
        self.1 = true;
        // process_group(0) makes the child's PID its PGID. Kill descendants too,
        // including descendants left behind by an already exited worker.
        let _ = killpg(Pid::from_raw(self.0.id() as i32), Signal::SIGKILL);
        let _ = self.0.wait();
    }
}
impl Drop for ProcessGroup {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Execute a host probe with bounded time and retained-output size. Commands
/// are direct argv invocations; no shell or inherited GPU visibility is used.
pub fn output(command: &mut Command, timeout: Duration, max_bytes: u64) -> io::Result<Vec<u8>> {
    if timeout.is_zero() || max_bytes == 0 {
        return Err(io::Error::other("invalid probe limits"));
    }
    let mut stdout = tempfile::tempfile()?;
    let mut stderr = tempfile::tempfile()?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout.try_clone()?))
        .stderr(Stdio::from(stderr.try_clone()?))
        .process_group(0);
    let mut child = ProcessGroup(command.spawn()?, false);
    let start = Instant::now();
    let status = loop {
        if stdout.metadata()?.len() > max_bytes || stderr.metadata()?.len() > max_bytes {
            return Err(io::Error::other("probe output exceeds limit"));
        }
        if start.elapsed() >= timeout {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "probe timed out"));
        }
        if let Some(status) = child.0.try_wait()? {
            break status;
        }
        thread::sleep(Duration::from_millis(10));
    };
    child.stop();
    if stdout.metadata()?.len() > max_bytes || stderr.metadata()?.len() > max_bytes {
        return Err(io::Error::other("probe output exceeds limit"));
    }
    if !status.success() {
        stderr.seek(SeekFrom::Start(0))?;
        let mut diagnostic = String::new();
        stderr.take(max_bytes).read_to_string(&mut diagnostic)?;
        return Err(io::Error::other(format!(
            "probe failed ({status}): {diagnostic}"
        )));
    }
    stdout.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    stdout.take(max_bytes).read_to_end(&mut bytes)?;
    Ok(bytes)
}
