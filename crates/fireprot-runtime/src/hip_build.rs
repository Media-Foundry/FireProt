//! Direct native shared-library builds with declared compiler and linker inputs.
use crate::{
    artifacts::sha256,
    build::{BuildIdentity, BuildSpec},
    build_cache::{CachedBuild, get_or_build_checked},
    build_process::CompilerCommand,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, ffi::OsString, fs, io, path::Path, time::Duration};
const RECIPE_KEY: &str = "fireprot_hip_process";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HipBuildSpec {
    pub build: BuildSpec,
    pub compiler: String,
    pub guard: String,
    pub sources: Vec<String>,
    pub timeout_ms: u64,
    pub lock_timeout_ms: u64,
}
impl HipBuildSpec {
    fn declared_path(&self, name: &str) -> io::Result<&Path> {
        let file = self
            .build
            .files
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| io::Error::other(format!("undeclared native input: {name}")))?;
        if !file.path.is_absolute() {
            return Err(io::Error::other("native input paths must be absolute"));
        }
        Ok(&file.path)
    }
    pub fn identify(&self) -> io::Result<BuildIdentity> {
        if self.timeout_ms == 0
            || self.lock_timeout_ms == 0
            || self.sources.is_empty()
            || self.build.arguments.contains_key(RECIPE_KEY)
            || !self.build.arguments.contains_key("compile")
            || !self.build.arguments.contains_key("link")
        {
            return Err(io::Error::other("invalid direct HIP build recipe"));
        }
        for (key, value) in &self.build.environment {
            if key.is_empty()
                || key.contains(['=', '\0'])
                || value.as_ref().is_some_and(|v| v.contains('\0'))
            {
                return Err(io::Error::other("invalid compiler environment"));
            }
        }
        for include in self.build.arguments.get("include").into_iter().flatten() {
            let path = Path::new(include);
            if !path.is_absolute()
                || !self
                    .build
                    .trees
                    .iter()
                    .any(|tree| tree.path == path && tree.suffixes.is_empty())
            {
                return Err(io::Error::other(
                    "include path requires a complete declared build tree",
                ));
            }
        }
        let compiler = self.declared_path(&self.compiler)?;
        self.declared_path(&self.guard)?;
        let mut names = BTreeSet::new();
        for source in &self.sources {
            let path = self.declared_path(source)?;
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .ok_or_else(|| io::Error::other("invalid source filename"))?;
            if !matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("cpp" | "cc" | "cxx" | "c" | "cu" | "hip")
            ) || !names.insert(name)
            {
                return Err(io::Error::other("unsupported or duplicate source filename"));
            }
        }
        let mut spec = self.build.clone();
        spec.arguments.insert(
            RECIPE_KEY.into(),
            vec![
                "1".into(),
                self.compiler.clone(),
                self.guard.clone(),
                serde_json::to_string(&self.sources)?,
                self.timeout_ms.to_string(),
                self.lock_timeout_ms.to_string(),
                compiler
                    .to_str()
                    .ok_or_else(|| io::Error::other("compiler path must be UTF-8"))?
                    .into(),
            ],
        );
        spec.identify()
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
                let mut arguments: Vec<OsString> = identity.arguments["compile"]
                    .iter()
                    .map(Into::into)
                    .collect();
                for include in identity.arguments.get("include").into_iter().flatten() {
                    arguments.push(format!("-I{include}").into());
                }
                for role in &self.sources {
                    let original = self.declared_path(role)?;
                    let input = identity
                        .files
                        .iter()
                        .find(|f| &f.name == role)
                        .expect("validated role");
                    let target =
                        directory.join(original.file_name().expect("validated source filename"));
                    fs::copy(original, &target)?;
                    if sha256(&target)? != input.sha256 {
                        return Err(io::Error::other(
                            "source changed while copying into staging",
                        ));
                    }
                    arguments.push(target.into());
                }
                arguments.extend(identity.arguments["link"].iter().map(OsString::from));
                let library = format!("{name}.so");
                arguments.push("-o".into());
                arguments.push(directory.join(&library).into());
                let program = self.declared_path(&self.compiler)?;
                let compiler = identity
                    .files
                    .iter()
                    .find(|f| f.name == self.compiler)
                    .expect("validated compiler");
                if program.canonicalize()? != compiler.canonical_path {
                    return Err(io::Error::other("compiler executable link changed"));
                }
                let guard = identity
                    .files
                    .iter()
                    .find(|f| f.name == self.guard)
                    .expect("validated guard");
                // Record the actual argv in staging; do not interpolate a shell command.
                let argv: Vec<_> = arguments
                    .iter()
                    .map(|a| {
                        a.to_str()
                            .ok_or_else(|| io::Error::other("compiler argument must be UTF-8"))
                    })
                    .collect::<io::Result<_>>()?;
                fs::write(
                    directory.join("command.json"),
                    serde_json::to_vec(&serde_json::json!({"program":program,"arguments":argv}))?,
                )?;
                CompilerCommand {
                    guard: &guard.canonical_path,
                    program,
                    arguments,
                    directory,
                    environment: &identity.environment,
                    timeout: Duration::from_millis(self.timeout_ms),
                }
                .run(check)?;
                Ok(vec![library])
            },
        )
    }
}
