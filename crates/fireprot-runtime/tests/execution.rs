#![cfg(unix)]
use fireprot_runtime::{
    build::BuildFile,
    build_tree::BuildTree,
    execution::{ExecutionRecord, environment, resolve_program},
};
use std::{collections::BTreeMap, ffi::OsString, fs};

#[test]
fn ambient_environment_is_bounded_and_overrides_are_explicit() {
    let ambient = vec![
        ("PATH".into(), "/usr/bin:/bin".into()),
        ("HOME".into(), "/fixture".into()),
        ("UNRELATED_TOKEN".into(), "fixture-secret".into()),
        ("FIREPROT_DISABLE_CK_EXTENSION".into(), "1".into()),
    ];
    let result = environment(
        ambient,
        &[
            ("HOME".into(), "/chosen".into()),
            ("FIREPROT_TEST_SETTING".into(), "explicit".into()),
        ],
    )
    .unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result["HOME"], "/chosen");
    assert!(!result.contains_key("UNRELATED_TOKEN"));
    assert!(!result.contains_key("FIREPROT_DISABLE_CK_EXTENSION"));
    assert!(
        environment(
            Vec::<(OsString, OsString)>::new(),
            &[("BAD=KEY".into(), "value".into())]
        )
        .is_err()
    );
}

#[test]
fn execution_tracks_content_topology_and_virtual_environment_invocation() {
    let root = tempfile::tempdir().unwrap();
    let bin = root.path().join("venv/bin");
    fs::create_dir_all(&bin).unwrap();
    let python = bin.join("python");
    std::os::unix::fs::symlink("/usr/bin/python3", &python).unwrap();
    let package = root.path().join("workers");
    fs::create_dir(&package).unwrap();
    fs::write(package.join("worker.py"), "first").unwrap();
    let cfg = root.path().join("venv/pyvenv.cfg");
    fs::write(&cfg, "home = /usr/bin\n").unwrap();
    let files = vec![BuildFile {
        name: "venv".into(),
        path: cfg.clone(),
    }];
    let trees = vec![BuildTree {
        name: "workers".into(),
        path: package.clone(),
        suffixes: vec![".py".into()],
    }];
    let env = BTreeMap::from([("PATH".into(), bin.to_str().unwrap().into())]);
    assert_eq!(
        resolve_program(std::path::Path::new("python"), root.path(), &env).unwrap(),
        python
    );
    let record = ExecutionRecord::capture(
        &python,
        &["-m".into(), "workers".into()],
        root.path(),
        env,
        &files,
        &trees,
        None,
    )
    .unwrap();
    assert_eq!(record.invocation_path, python);
    assert_eq!(
        record.program.canonical_path,
        std::path::Path::new("/usr/bin/python3")
            .canonicalize()
            .unwrap()
    );
    record.verify(&files, &trees).unwrap();
    fs::write(package.join("ignored.pyc"), "cache").unwrap();
    record.verify(&files, &trees).unwrap();
    fs::write(package.join("added.py"), "new input").unwrap();
    assert!(record.verify(&files, &trees).is_err());
    fs::remove_file(package.join("added.py")).unwrap();
    fs::write(package.join("worker.py"), "other").unwrap();
    assert!(record.verify(&files, &trees).is_err());
    fs::write(package.join("worker.py"), "first").unwrap();
    fs::write(&cfg, "home = /different\n").unwrap();
    assert!(record.verify(&files, &trees).is_err());
    assert!(record.verify_process(std::process::id()).is_err());
}
