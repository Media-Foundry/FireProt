//! Wall-clock measurement of a complete, cooperatively supervised invocation.
//!
//! The child must handle SIGTERM by cleaning up its own worker process groups.
//! A deadline requests cancellation; it is not a hard bound on cleanup time.
//! Do not use this interface for arbitrary commands or unconsumed output pipes.
use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use std::{
    io,
    os::unix::process::CommandExt,
    process::{Child, Command, ExitStatus, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Cancelled,
    Deadline,
}

#[derive(Debug)]
pub struct Measurement {
    pub status: ExitStatus,
    /// Before spawn through observed exit, including cancellation cleanup.
    pub elapsed: Duration,
    /// An interrupted invocation never becomes a successful measurement merely
    /// because a cooperative child exits with status zero.
    pub stop_reason: Option<StopReason>,
}

impl Measurement {
    pub fn succeeded(&self) -> bool {
        self.status.success() && self.stop_reason.is_none()
    }
}

struct SupervisedChild {
    child: Child,
    reaped: bool,
}

impl SupervisedChild {
    fn cancel(&self) -> io::Result<()> {
        match kill(Pid::from_raw(self.child.id() as i32), Signal::SIGTERM) {
            Ok(()) | Err(nix::errno::Errno::ESRCH) => Ok(()),
            Err(error) => Err(io::Error::from_raw_os_error(error as i32)),
        }
    }
}

impl Drop for SupervisedChild {
    fn drop(&mut self) {
        if !self.reaped {
            // Killing only the supervisor's process group could orphan workers
            // in their separately created groups. Wait for cooperative cleanup.
            let _ = self.cancel();
            let _ = self.child.wait();
        }
    }
}

/// Measure one trusted supervisor command, preserving its exit status.
/// Configure stdout/stderr as files or inherited streams before calling.
/// A failed spawn or pre-launch cancellation yields no timing observation.
pub fn measure(
    command: &mut Command,
    deadline: Duration,
    cancelled: &AtomicBool,
) -> io::Result<Measurement> {
    if deadline.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "zero invocation deadline",
        ));
    }
    if cancelled.load(Ordering::Relaxed) {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "invocation cancelled before spawn",
        ));
    }
    command.stdin(Stdio::null()).process_group(0);
    let start = Instant::now();
    let mut child = SupervisedChild {
        child: command.spawn()?,
        reaped: false,
    };
    let mut stop_reason = None;
    loop {
        // Cancellation wins over a simultaneous observed successful exit.
        if stop_reason.is_none() {
            stop_reason = if cancelled.load(Ordering::Relaxed) {
                Some(StopReason::Cancelled)
            } else if start.elapsed() >= deadline {
                Some(StopReason::Deadline)
            } else {
                None
            };
            if stop_reason.is_some() {
                child.cancel()?;
            }
        }
        if let Some(status) = child.child.try_wait()? {
            let elapsed = start.elapsed();
            child.reaped = true;
            return Ok(Measurement {
                status,
                elapsed,
                stop_reason,
            });
        }
        thread::sleep(Duration::from_millis(1));
    }
}
