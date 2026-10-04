use fireprot_core::request::ArtifactRef;
use fireprot_runtime::{
    artifacts::sha256,
    build::BuildFile,
    build_tree::BuildTree,
    environment_inventory::{InventorySpec, accept},
    execution::ExecutionRecord,
};
use std::{collections::BTreeMap, fs, path::Path};

#[test]
fn pinned_environment_rejects_changes_before_capture_and_later_verification() {
    let tmp = tempfile::tempdir().unwrap();
    let packages = tmp.path().join("site-packages");
    fs::create_dir(&packages).unwrap();
    let module = packages.join("model.py");
    fs::write(&module, "original").unwrap();
    let spec = InventorySpec {
        schema_version: 1,
        files: vec![],
        trees: vec![BuildTree {
            name: "packages".into(),
            path: packages.clone(),
            suffixes: vec![],
        }],
    };
    let manifest = tmp.path().join("environment.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&spec.capture().unwrap()).unwrap(),
    )
    .unwrap();
    let pin = ArtifactRef {
        path: manifest.clone(),
        sha256: sha256(&manifest).unwrap(),
    };
    let capture = || {
        ExecutionRecord::capture(
            Path::new("/usr/bin/python3"),
            &[],
            tmp.path(),
            BTreeMap::new(),
            &[],
            &[],
            Some(&pin),
        )
    };
    let record = capture().unwrap();
    assert_eq!(record.inputs[0].name, "environment/packages/model.py");
    record.verify(&[], &[]).unwrap();
    fs::write(&module, "modified").unwrap();
    assert!(capture().is_err());
    assert!(record.verify(&[], &[]).is_err());
    fs::write(&module, "original").unwrap();
    fs::write(packages.join("new.pth"), "injected path").unwrap();
    assert!(record.verify(&[], &[]).is_err());
    fs::remove_file(packages.join("new.pth")).unwrap();
    record.verify(&[], &[]).unwrap();
    fs::write(&manifest, "{}").unwrap();
    assert!(record.verify(&[], &[]).is_err());
}

#[test]
fn environment_rejects_ambiguous_names_and_retargeted_inputs() {
    let tmp = tempfile::tempdir().unwrap();
    let a = tmp.path().join("a");
    let b = tmp.path().join("b");
    fs::write(&a, "same").unwrap();
    fs::write(&b, "same").unwrap();
    let link = tmp.path().join("active");
    std::os::unix::fs::symlink(&a, &link).unwrap();
    let mut spec = InventorySpec {
        schema_version: 1,
        files: vec![BuildFile {
            name: "library".into(),
            path: link.clone(),
        }],
        trees: vec![],
    };
    let p = tmp.path().join("environment.json");
    fs::write(&p, serde_json::to_vec(&spec.capture().unwrap()).unwrap()).unwrap();
    let pin = ArtifactRef {
        sha256: sha256(&p).unwrap(),
        path: p,
    };
    accept(&pin).unwrap();
    fs::remove_file(&link).unwrap();
    std::os::unix::fs::symlink(&b, &link).unwrap();
    assert!(accept(&pin).is_err());
    spec.files.push(spec.files[0].clone());
    assert!(spec.capture().is_err());
    spec.files.clear();
    assert!(spec.capture().is_err());
}

#[cfg(unix)]
#[test]
fn overlapping_roots_keep_every_alias_and_legacy_inventory_identity() {
    use fireprot_runtime::build::FileIdentity;
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let packages = tmp.path().join("packages");
    fs::create_dir(&packages).unwrap();
    let module = packages.join("model.py");
    fs::write(&module, "model source").unwrap();
    let alias = tmp.path().join("linked-packages");
    symlink(&packages, &alias).unwrap();
    let spec = InventorySpec {
        schema_version: 1,
        files: vec![BuildFile {
            name: "explicit".into(),
            path: alias.join("model.py"),
        }],
        trees: vec![
            BuildTree {
                name: "first".into(),
                path: packages,
                suffixes: vec![],
            },
            BuildTree {
                name: "second".into(),
                path: alias,
                suffixes: vec![],
            },
        ],
    };
    let captured = spec.capture().unwrap();
    let expected: Vec<_> = ["explicit", "first/model.py", "second/model.py"]
        .into_iter()
        .map(|name| FileIdentity {
            name: name.into(),
            canonical_path: module.canonicalize().unwrap(),
            sha256: sha256(&module).unwrap(),
        })
        .collect();
    assert_eq!(captured.files, expected);
    let manifest = tmp.path().join("inventory.json");
    fs::write(&manifest, serde_json::to_vec(&captured).unwrap()).unwrap();
    let pin = ArtifactRef {
        sha256: sha256(&manifest).unwrap(),
        path: manifest,
    };
    accept(&pin).unwrap();
    fs::write(&module, "new contents").unwrap();
    assert!(accept(&pin).is_err());
}
