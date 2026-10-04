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
