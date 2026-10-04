#![cfg(unix)]
use fireprot_runtime::{
    build::{BuildFile, BuildSpec},
    build_tree::BuildTree,
    hip_build::HipBuildSpec,
};
use std::{
    collections::BTreeMap,
    fs, io,
    os::unix::fs::PermissionsExt,
    path::Path,
    time::{Duration, Instant},
};
fn fixture(root: &Path) -> HipBuildSpec {
    fs::write(
        root.join("source.c"),
        "#include <value.h>\nint answer(void) { return VALUE; }\n",
    )
    .unwrap();
    let include = root.join("include");
    fs::create_dir(&include).unwrap();
    fs::write(include.join("value.h"), "#define VALUE 42\n").unwrap();
    HipBuildSpec {
        build: BuildSpec {
            schema_version: 1,
            package: "host_test".into(),
            target: "host".into(),
            files: vec![
                BuildFile {
                    name: "compiler".into(),
                    path: "/usr/bin/cc".into(),
                },
                BuildFile {
                    name: "guard".into(),
                    path: env!("CARGO_BIN_EXE_fireprot-compiler-guard").into(),
                },
                BuildFile {
                    name: "source".into(),
                    path: root.join("source.c"),
                },
            ],
            trees: vec![BuildTree {
                name: "headers".into(),
                path: include.clone(),
                suffixes: vec![],
            }],
            dynamic_inputs: None,
            arguments: BTreeMap::from([
                (
                    "compile".into(),
                    vec!["-O2".into(), "-fPIC".into(), "-shared".into()],
                ),
                ("include".into(), vec![include.to_str().unwrap().into()]),
                ("link".into(), vec![]),
            ]),
            environment: BTreeMap::from([("PATH".into(), Some("/usr/bin:/bin".into()))]),
            versions: BTreeMap::from([
                ("recipe".into(), "test".into()),
                ("abi".into(), "c".into()),
            ]),
        },
        compiler: "compiler".into(),
        guard: "guard".into(),
        sources: vec!["source".into()],
        timeout_ms: 5000,
        lock_timeout_ms: 5000,
    }
}
#[test]
fn real_shared_library_tracks_headers_and_records_argv() {
    let dir = tempfile::tempdir().unwrap();
    let spec = fixture(dir.path());
    let cache = dir.path().join("cache");
    let first = spec.run(&cache).unwrap();
    assert!(!first.cache_hit);
    let name = first.receipt.artifacts.keys().next().unwrap();
    assert_eq!(
        &fs::read(first.directory.join(name)).unwrap()[..4],
        b"\x7fELF"
    );
    let argv: serde_json::Value =
        serde_json::from_slice(&fs::read(first.directory.join("command.json")).unwrap()).unwrap();
    assert_eq!(argv["program"], "/usr/bin/cc");
    assert_eq!(argv["arguments"][0], "-O2");
    assert!(spec.run(&cache).unwrap().cache_hit);
    fs::write(dir.path().join("include/value.h"), "#define VALUE 43\n").unwrap();
    let changed = spec.run(&cache).unwrap();
    assert!(!changed.cache_hit);
    assert_ne!(changed.directory, first.directory);
}
#[test]
fn undeclared_roles_headers_and_failed_compilers_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = fixture(dir.path());
    spec.guard = "missing".into();
    assert!(spec.identify().is_err());
    spec.guard = "guard".into();
    spec.build.trees[0].suffixes = vec![".h".into()];
    assert!(spec.identify().is_err());
    spec.build.trees[0].suffixes.clear();
    fs::write(dir.path().join("source.c"), "invalid C source;").unwrap();
    let name = spec.identify().unwrap().cache_name().unwrap();
    let cache = dir.path().join("cache");
    let e = spec.run(&cache).unwrap_err();
    assert!(e.to_string().contains("error:"));
    assert!(!cache.join(name).exists());
}
#[test]
fn compiler_timeout_and_cancellation_do_not_publish() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = fixture(dir.path());
    let compiler = dir.path().join("compiler");
    fs::write(
        &compiler,
        "#!/bin/sh\necho compiler-waiting >&2\nsleep 30\n",
    )
    .unwrap();
    fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
    spec.build.files[0].path = compiler;
    spec.timeout_ms = 200;
    let error = spec.run(&dir.path().join("cache")).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert!(error.to_string().contains("compiler-waiting"));
    spec.timeout_ms = 5000;
    let started = Instant::now();
    let error = spec
        .run_checked(&dir.path().join("cache"), &mut || {
            if started.elapsed() > Duration::from_millis(200) {
                Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::Interrupted);
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn c_abi_build_cannot_be_published_as_python_extension() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = fixture(dir.path());
    spec.build.versions.insert("torch".into(), "fixture".into());
    spec.build.versions.insert("hip".into(), "fixture".into());
    let build = spec.run(&dir.path().join("cache")).unwrap();
    let destination = dir.path().join("published");
    let error =
        fireprot_runtime::kernel_bundle::publish(&build.directory, &destination).unwrap_err();
    assert!(error.to_string().contains("c_abi"));
    assert!(!destination.exists());
}

#[test]
fn c_abi_bundle_is_typed_and_independent_of_build_inputs() {
    use fireprot_runtime::{
        artifacts::sha256,
        kernel_bundle::{KernelCompatibility, accept, publish},
    };
    let dir = tempfile::tempdir().unwrap();
    let mut spec = fixture(dir.path());
    spec.build.versions.extend([
        ("torch".into(), "fixture".into()),
        ("hip".into(), "fixture".into()),
        ("c_abi".into(), "answer-v1".into()),
    ]);
    let build = spec.run(&dir.path().join("cache")).unwrap();
    let bundle = publish(&build.directory, &dir.path().join("bundle")).unwrap();
    assert!(
        matches!(&bundle.compatibility, KernelCompatibility::C { c_abi, .. } if c_abi == "answer-v1")
    );
    fs::remove_dir_all(dir.path().join("cache")).unwrap();
    fs::remove_file(dir.path().join("source.c")).unwrap();
    fs::remove_dir_all(dir.path().join("include")).unwrap();
    let accepted = serde_json::to_value(
        accept(
            &bundle.directory,
            &bundle.manifest_sha256,
            &bundle.compatibility,
        )
        .unwrap(),
    )
    .unwrap();
    assert!(accepted.get("module_name").is_none());
    let library = accepted["library"].as_str().unwrap();
    assert_eq!(&fs::read(library).unwrap()[..4], b"\x7fELF");
    let mut incompatible = bundle.compatibility.clone();
    if let KernelCompatibility::C { c_abi, .. } = &mut incompatible {
        *c_abi = "answer-v2".into();
    }
    assert!(accept(&bundle.directory, &bundle.manifest_sha256, &incompatible).is_err());
    let path = bundle.directory.join("kernel.json");
    let original: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(original["schema_version"], 2);
    for (field, value) in [
        ("schema_version", serde_json::json!(1)),
        ("module_name", serde_json::json!("pretend_python")),
    ] {
        let mut manifest = original.clone();
        manifest[field] = value;
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(
            accept(
                &bundle.directory,
                &sha256(&path).unwrap(),
                &bundle.compatibility
            )
            .is_err()
        );
    }
    let mut ambiguous = serde_json::to_value(&bundle.compatibility).unwrap();
    ambiguous["python_soabi"] = "cpython-test".into();
    assert!(serde_json::from_value::<KernelCompatibility>(ambiguous).is_err());
}
