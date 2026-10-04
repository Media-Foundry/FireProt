#![cfg(unix)]
use fireprot_runtime::{
    build::{BuildFile, BuildSpec},
    torch_build::TorchBuildSpec,
};
use std::{collections::BTreeMap, fs, path::Path};

fn fixture(root: &Path, body: &str) -> TorchBuildSpec {
    fs::write(root.join("source.cu"), "source payload").unwrap();
    fs::write(root.join("helper.py"), format!("import json, os, pathlib, sys, time\nr=json.load(open(sys.argv[1]))\np=pathlib.Path(r['build_directory'])\n{body}\n")).unwrap();
    TorchBuildSpec {
        build: BuildSpec {
            schema_version: 1,
            package: "test_torch".into(),
            target: "gfx90a".into(),
            files: vec![
                BuildFile {
                    name: "python".into(),
                    path: "/usr/bin/python3".into(),
                },
                BuildFile {
                    name: "helper".into(),
                    path: root.join("helper.py"),
                },
                BuildFile {
                    name: "guard".into(),
                    path: env!("CARGO_BIN_EXE_fireprot-compiler-guard").into(),
                },
                BuildFile {
                    name: "source".into(),
                    path: root.join("source.cu"),
                },
            ],
            trees: vec![],
            dynamic_inputs: None,
            arguments: BTreeMap::from([
                ("cxx".into(), vec!["-O3".into()]),
                ("hip".into(), vec!["--offload-arch=gfx90a".into()]),
            ]),
            environment: BTreeMap::from([
                ("FIREPROT_BUILD_TEST_VALUE".into(), Some("recorded".into())),
                ("HOME".into(), None),
            ]),
            versions: BTreeMap::from([("recipe".into(), "fixture".into())]),
        },
        python: "python".into(),
        helper: "helper".into(),
        guard: "guard".into(),
        lock_timeout_ms: 3000,
        sources: vec!["source".into()],
        timeout_ms: 3000,
    }
}
const SUCCESS: &str = "assert os.environ.get('HOME') is None\nassert os.environ['FIREPROT_BUILD_TEST_VALUE']=='recorded'\nassert r['extra_cflags']==['-O3']\nassert r['extra_cuda_cflags']==['--offload-arch=gfx90a']\nassert pathlib.Path(r['sources'][0]).parent==p\n(p/(r['name']+'.so')).write_bytes(pathlib.Path(r['sources'][0]).read_bytes())";

#[test]
fn process_builds_with_recorded_inputs_and_reuses_verified_cache() {
    let dir = tempfile::tempdir().unwrap();
    let spec = fixture(dir.path(), SUCCESS);
    let cache = dir.path().join("cache");
    let result = spec.run(&cache).unwrap();
    assert!(!result.cache_hit);
    assert!(result.directory.join("compiler.log").is_file());
    assert_eq!(
        fs::read(result.directory.join(format!(
            "{}.so",
            result.receipt.identity.cache_name().unwrap()
        )))
        .unwrap(),
        b"source payload"
    );
    assert!(spec.run(&cache).unwrap().cache_hit);
    fs::write(dir.path().join("source.cu"), "changed source").unwrap();
    let next = spec.run(&cache).unwrap();
    assert!(!next.cache_hit);
    assert_ne!(result.directory, next.directory);
}

#[test]
fn failure_and_timeout_preserve_diagnostics_without_publishing() {
    for (body, timeout, expected) in [
        (
            "print('compiler diagnostic',flush=True)\nsys.exit(9)",
            3000,
            "compiler diagnostic",
        ),
        (
            "print('before timeout',flush=True)\ntime.sleep(30)",
            200,
            "before timeout",
        ),
        ("pass", 3000, "No such file"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut spec = fixture(dir.path(), body);
        spec.timeout_ms = timeout;
        let cache = dir.path().join("cache");
        let name = spec.identify().unwrap().cache_name().unwrap();
        let error = spec.run(&cache).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        if timeout == 200 {
            assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        }
        assert!(!cache.join(name).exists());
        assert!(!fs::read_dir(cache).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".building-")
        }));
    }
}

#[test]
fn roles_source_order_and_timeout_are_bound_to_identity() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = fixture(dir.path(), SUCCESS);
    let original = spec.identify().unwrap().sha256;
    spec.timeout_ms += 1;
    assert_ne!(spec.identify().unwrap().sha256, original);
    spec.helper = "not-declared".into();
    assert!(spec.identify().is_err());
    spec.helper = "helper".into();
    spec.sources.push("source".into());
    assert!(spec.identify().is_err());
    spec.sources.pop();
    spec.build
        .arguments
        .insert("fireprot_torch_process".into(), vec![]);
    assert!(spec.identify().is_err());
}

#[test]
fn timeout_stops_compiler_descendants() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("escaped-write");
    let body = format!(
        "import subprocess\nsubprocess.Popen([sys.executable,'-c',{}])\nprint('child started',flush=True)\ntime.sleep(30)",
        serde_json::to_string(&format!(
            "import time,pathlib; time.sleep(1); pathlib.Path({}).write_text('escaped')",
            serde_json::to_string(&marker).unwrap()
        ))
        .unwrap()
    );
    let mut spec = fixture(dir.path(), &body);
    spec.timeout_ms = 200;
    assert_eq!(
        spec.run(&dir.path().join("cache")).unwrap_err().kind(),
        std::io::ErrorKind::TimedOut
    );
    std::thread::sleep(std::time::Duration::from_millis(1200));
    assert!(!marker.exists());
}

#[test]
fn interpreter_symlink_spelling_is_preserved_and_part_of_identity() {
    let dir = tempfile::tempdir().unwrap();
    let alias = dir.path().join("venv-python");
    std::os::unix::fs::symlink("/usr/bin/python3", &alias).unwrap();
    let body = format!(
        "assert sys.executable == {}\n{SUCCESS}",
        serde_json::to_string(&alias).unwrap()
    );
    let mut spec = fixture(dir.path(), &body);
    let original = spec.identify().unwrap().sha256;
    spec.build.files[0].path = alias;
    assert_ne!(spec.identify().unwrap().sha256, original);
    spec.run(&dir.path().join("cache")).unwrap();
}

#[test]
fn build_host_child() {
    let Some(root) = std::env::var_os("FIREPROT_TEST_BUILD_HOST") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let spec: TorchBuildSpec =
        serde_json::from_slice(&fs::read(root.join("spec.json")).unwrap()).unwrap();
    spec.run(&root.join("cache")).unwrap();
}

#[test]
fn killing_host_stops_compiler_group_and_never_publishes() {
    use std::{
        process::{Command, Stdio},
        time::{Duration, Instant},
    };
    let dir = tempfile::tempdir().unwrap();
    let ready = dir.path().join("ready.json");
    let marker = dir.path().join("escaped-write");
    let body = format!(
        "import subprocess\nc=subprocess.Popen([sys.executable,'-c',{}])\npathlib.Path({}).write_text(json.dumps([os.getpid(),os.getpgrp(),c.pid]))\ntime.sleep(30)",
        serde_json::to_string(&format!(
            "import time,pathlib; time.sleep(2); pathlib.Path({}).write_text('escaped')",
            serde_json::to_string(&marker).unwrap()
        ))
        .unwrap(),
        serde_json::to_string(&ready).unwrap()
    );
    let mut spec = fixture(dir.path(), &body);
    spec.timeout_ms = 30000;
    let name = spec.identify().unwrap().cache_name().unwrap();
    fs::write(
        dir.path().join("spec.json"),
        serde_json::to_vec(&spec).unwrap(),
    )
    .unwrap();
    let mut host = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "build_host_child"])
        .env("FIREPROT_TEST_BUILD_HOST", dir.path())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ready.exists() {
        if host.try_wait().unwrap().is_some() || Instant::now() > deadline {
            let _ = host.kill();
            let _ = host.wait();
            panic!("compiler did not start");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let pids: Vec<i32> = serde_json::from_slice(&fs::read(&ready).unwrap()).unwrap();
    host.kill().unwrap();
    host.wait().unwrap();
    std::thread::sleep(Duration::from_millis(2300));
    let escaped = marker.exists();
    // Clean the deliberately reproduced old implementation leak before asserting.
    let _ = nix::sys::signal::killpg(
        nix::unistd::Pid::from_raw(pids[1]),
        nix::sys::signal::Signal::SIGKILL,
    );
    assert!(!escaped, "compiler descendant survived its host");
    assert!(!dir.path().join("cache").join(name).exists());
}

#[test]
fn cancellation_during_compilation_stops_process_and_returns_original_reason() {
    use std::{
        io,
        time::{Duration, Instant},
    };
    let dir = tempfile::tempdir().unwrap();
    let spec = fixture(dir.path(), "time.sleep(30)");
    let started = Instant::now();
    let e = spec
        .run_checked(&dir.path().join("cache"), &mut || {
            if started.elapsed() > Duration::from_millis(250) {
                Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "requested cancellation",
                ))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::Interrupted);
    assert!(e.to_string().contains("requested cancellation"));
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn include_paths_reach_compiler_and_require_complete_inventories() {
    use fireprot_runtime::build_tree::BuildTree;
    let dir = tempfile::tempdir().unwrap();
    let include = dir.path().join("include");
    fs::create_dir(&include).unwrap();
    fs::write(include.join("kernel.h"), "header").unwrap();
    let mut spec = fixture(
        dir.path(),
        &format!(
            "assert r['extra_include_paths']==[{}]\nassert pathlib.Path(r['extra_include_paths'][0],'kernel.h').read_text()\n{SUCCESS}",
            serde_json::to_string(&include).unwrap()
        ),
    );
    spec.build
        .arguments
        .insert("include".into(), vec![include.to_str().unwrap().into()]);
    assert!(spec.identify().is_err());
    spec.build.trees.push(BuildTree {
        name: "include".into(),
        path: include.clone(),
        suffixes: vec![".h".into()],
    });
    assert!(spec.identify().is_err());
    spec.build.trees[0].suffixes.clear();
    let first = spec.run(&dir.path().join("cache")).unwrap();
    fs::write(include.join("new.h"), "new dependency").unwrap();
    let second = spec.run(&dir.path().join("cache")).unwrap();
    assert!(!second.cache_hit);
    assert_ne!(first.directory, second.directory);
}
