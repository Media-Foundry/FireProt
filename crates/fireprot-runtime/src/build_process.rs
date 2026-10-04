//! Shared guarded compiler lifecycle for Torch helpers and direct HIP commands.
use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    os::{
        fd::OwnedFd,
        unix::{net::UnixStream, process::CommandExt},
    },
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
struct CompilerGroup(Child, UnixStream);
impl Drop for CompilerGroup {
    fn drop(&mut self) {
        // Also stop descendants left behind by a failed or prematurely exited helper.
        let _ = killpg(Pid::from_raw(self.0.id() as i32), Signal::SIGKILL);
        let _ = self.0.wait();
        let _ = self.1.shutdown(std::net::Shutdown::Both);
    }
}

fn log_tail(path: &Path) -> String {
    let read = || -> io::Result<String> {
        let mut file = File::open(path)?;
        let start = file.metadata()?.len().saturating_sub(16 * 1024);
        file.seek(SeekFrom::Start(start))?;
        let mut bytes = Vec::new();
        file.take(16 * 1024).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    };
    read().unwrap_or_else(|error| format!("compiler log unavailable: {error}"))
}

pub(crate) struct CompilerCommand<'a> {
    pub guard: &'a Path,
    pub program: &'a Path,
    pub arguments: Vec<OsString>,
    pub directory: &'a Path,
    pub environment: &'a BTreeMap<String, Option<String>>,
    pub timeout: Duration,
}
impl CompilerCommand<'_> {
    pub fn run(self, check: &mut dyn FnMut() -> io::Result<()>) -> io::Result<()> {
        let log_path = self.directory.join("compiler.log");
        let log = File::create(&log_path)?;
        let (lifetime, child_lifetime) = UnixStream::pair()?;
        let status_path = self.directory.join("compiler-status.json");
        let mut command = Command::new(self.guard);
        command
            .arg(self.program)
            .args(&self.arguments)
            .arg(&status_path)
            .current_dir(self.directory)
            .env_clear()
            .envs(
                self.environment
                    .iter()
                    .filter_map(|(key, value)| value.as_ref().map(|v| (key, v))),
            )
            .stdin(Stdio::from(OwnedFd::from(child_lifetime)))
            .stdout(log.try_clone()?)
            .stderr(log)
            .process_group(0);
        let mut group = CompilerGroup(command.spawn()?, lifetime);
        drop(command); // Close the parent copy of the child socket endpoint.
        let started = Instant::now();
        let result = loop {
            if let Err(error) = check() {
                break Err(error);
            }
            match fs::read(&status_path) {
                Ok(bytes) => {
                    #[derive(Deserialize)]
                    #[serde(deny_unknown_fields)]
                    struct Status {
                        success: bool,
                        code: Option<i32>,
                    }
                    let status: Status = serde_json::from_slice(&bytes)?;
                    break if status.success && status.code == Some(0) {
                        Ok(())
                    } else {
                        Err(io::Error::other(format!(
                            "native compiler exited with {:?}",
                            status.code
                        )))
                    };
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => (),
                Err(e) => break Err(e),
            }
            if let Some(status) = group.0.try_wait()? {
                break Err(io::Error::other(format!(
                    "compiler guard exited without status: {status}"
                )));
            }
            if started.elapsed() >= self.timeout {
                break Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "native compiler timed out",
                ));
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        drop(group);
        if let Err(error) = result {
            return Err(io::Error::new(
                error.kind(),
                format!(
                    "{error}\nCompiler log (last 16 KiB):\n{}",
                    log_tail(&log_path)
                ),
            ));
        }
        Ok(())
    }
}
