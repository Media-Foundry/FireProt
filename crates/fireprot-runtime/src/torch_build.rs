//! Rust orchestration of a declared Torch compiler helper. The helper owns only
//! the Torch compiler API; Rust owns inputs, environment, lifetime and publication.
use crate::{
    artifacts::sha256,
    build::{BuildIdentity, BuildSpec},
    build_cache::{CachedBuild, get_or_build_checked},
};
use nix::{
    sys::signal::{Signal, killpg},
    unistd::Pid,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    os::{
        fd::OwnedFd,
        unix::{net::UnixStream, process::CommandExt},
    },
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const RECIPE_KEY: &str = "fireprot_torch_process";

/// Roles refer to named files in `build.files`, never untracked executable paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TorchBuildSpec {
    pub build: BuildSpec,
    pub python: String,
    pub helper: String,
    pub guard: String,
    pub lock_timeout_ms: u64,
    pub sources: Vec<String>,
    /// Applies to the compiler process, not dependency scans or cache-lock waits.
    pub timeout_ms: u64,
}

#[derive(Serialize)]
struct CompilerRecipe<'a> {
    name: &'a str,
    sources: Vec<PathBuf>,
    build_directory: &'a Path,
    extra_cflags: &'a [String],
    extra_cuda_cflags: &'a [String],
}

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

impl TorchBuildSpec {
    /// Bind ordered roles and orchestration version to the existing build digest.
    /// Existing callback-built artifacts are intentionally different cache entries.
    pub fn identify(&self) -> io::Result<BuildIdentity> {
        if self.timeout_ms == 0
            || self.lock_timeout_ms == 0
            || self.sources.is_empty()
            || self.build.arguments.contains_key(RECIPE_KEY)
            || !self.build.arguments.contains_key("cxx")
            || !self.build.arguments.contains_key("hip")
            || !self
                .build
                .package
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || !self
                .build
                .package
                .starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        {
            return Err(io::Error::other(
                "invalid Torch process build specification",
            ));
        }
        for (key, value) in &self.build.environment {
            if key.is_empty()
                || key.contains(['=', '\0'])
                || value.as_ref().is_some_and(|v| v.contains('\0'))
            {
                return Err(io::Error::other("invalid compiler environment"));
            }
        }
        let python = self.python_path()?;
        let mut spec = self.build.clone();
        spec.arguments.insert(
            RECIPE_KEY.into(),
            vec![
                "2".into(),
                self.python.clone(),
                self.helper.clone(),
                self.guard.clone(),
                self.lock_timeout_ms.to_string(),
                serde_json::to_string(&self.sources)?,
                self.timeout_ms.to_string(),
                python
                    .to_str()
                    .ok_or_else(|| io::Error::other("Python path must be UTF-8"))?
                    .into(),
            ],
        );
        let identity = spec.identify()?;
        self.input(&identity, &self.python)?;
        self.input(&identity, &self.helper)?;
        self.input(&identity, &self.guard)?;
        let mut names = BTreeSet::new();
        for source in &self.sources {
            let path = self.input(&identity, source)?;
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| io::Error::other("source needs a UTF-8 filename"))?;
            if !matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("cu" | "hip" | "cpp" | "cc" | "cxx" | "c")
            ) || !names.insert(name)
            {
                return Err(io::Error::other("unsupported or duplicate source filename"));
            }
        }
        Ok(identity)
    }

    fn input<'a>(&self, identity: &'a BuildIdentity, name: &str) -> io::Result<&'a Path> {
        identity
            .files
            .iter()
            .find(|input| input.name == name)
            .map(|input| input.canonical_path.as_path())
            .ok_or_else(|| io::Error::other(format!("undeclared Torch build input: {name}")))
    }

    fn python_path(&self) -> io::Result<&Path> {
        let path = self
            .build
            .files
            .iter()
            .find(|f| f.name == self.python)
            .ok_or_else(|| io::Error::other("undeclared Python executable"))?
            .path
            .as_path();
        if !path.is_absolute() {
            return Err(io::Error::other("Python executable path must be absolute"));
        }
        Ok(path)
    }

    pub fn run(&self, root: &Path) -> io::Result<CachedBuild> {
        self.run_checked(root, &mut || Ok(()))
    }

    pub fn run_checked(
        &self,
        root: &Path,
        check: &mut dyn FnMut() -> io::Result<()>,
    ) -> io::Result<CachedBuild> {
        check()?;
        let identity = self.identify()?;
        check()?;
        let name = identity.cache_name()?;
        get_or_build_checked(
            root,
            &identity,
            Duration::from_millis(self.lock_timeout_ms),
            check,
            |directory, check| {
                let mut sources = Vec::new();
                for role in &self.sources {
                    let input = self.input(&identity, role)?;
                    let target = directory.join(input.file_name().expect("validated source name"));
                    fs::copy(input, &target)?;
                    let expected = identity
                        .files
                        .iter()
                        .find(|f| &f.name == role)
                        .expect("validated input");
                    if sha256(&target)? != expected.sha256 {
                        return Err(io::Error::other(
                            "source changed while copying into staging",
                        ));
                    }
                    sources.push(target);
                }
                let recipe = CompilerRecipe {
                    name: &name,
                    sources,
                    build_directory: directory,
                    extra_cflags: &identity.arguments["cxx"],
                    extra_cuda_cflags: &identity.arguments["hip"],
                };
                let recipe_path = directory.join("recipe.json");
                fs::write(&recipe_path, serde_json::to_vec(&recipe)?)?;
                let log_path = directory.join("compiler.log");
                let log = File::create(&log_path)?;
                let python = self.python_path()?;
                if python.canonicalize()? != self.input(&identity, &self.python)? {
                    return Err(io::Error::other("Python executable link changed"));
                }
                // Preserve the venv invocation path: resolving this symlink before
                // exec would silently select the base interpreter's environment.
                let (lifetime, child_lifetime) = UnixStream::pair()?;
                let status_path = directory.join("compiler-status.json");
                let mut command = Command::new(self.input(&identity, &self.guard)?);
                command
                    .arg(python)
                    .arg(self.input(&identity, &self.helper)?)
                    .arg(&recipe_path)
                    .arg(&status_path)
                    .current_dir(directory)
                    .env_clear()
                    .envs(
                        identity
                            .environment
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
                                    "Torch compiler exited with {:?}",
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
                    if started.elapsed() >= Duration::from_millis(self.timeout_ms) {
                        break Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "Torch compiler timed out",
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
                Ok(vec![format!("{name}.so")])
            },
        )
    }
}
