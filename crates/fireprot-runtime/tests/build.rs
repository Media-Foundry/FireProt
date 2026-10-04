use fireprot_runtime::build::{BuildFile, BuildSpec};
use std::collections::BTreeMap;

fn fixture(root: &std::path::Path) -> BuildSpec {
    for (name, content) in [
        ("kernel.cu", "source"),
        ("kernel.h", "header"),
        ("hipcc", "compiler"),
    ] {
        std::fs::write(root.join(name), content).unwrap();
    }
    BuildSpec {
        schema_version: 1,
        dynamic_inputs: None,
        trees: vec![],
        package: "fireprot_pair".into(),
        target: "gfx90a".into(),
        files: ["kernel.cu", "kernel.h", "hipcc"]
            .into_iter()
            .map(|name| BuildFile {
                name: name.into(),
                path: root.join(name),
            })
            .collect(),
        arguments: BTreeMap::from([(
            "hip".into(),
            vec!["-O3".into(), "--offload-arch=gfx90a".into()],
        )]),
        environment: BTreeMap::from([("CXX".into(), None)]),
        versions: BTreeMap::from([
            ("recipe".into(), "1".into()),
            ("torch".into(), "test".into()),
        ]),
    }
}

#[test]
fn every_effective_build_input_changes_identity() {
    let dir = tempfile::tempdir().unwrap();
    let spec = fixture(dir.path());
    let base = spec.identify().unwrap();
    base.verify().unwrap();
    assert_eq!(
        base,
        serde_json::from_str(&serde_json::to_string(&base).unwrap()).unwrap()
    );
    assert_eq!(
        base.cache_name().unwrap(),
        format!("fireprot_pair_{}", base.sha256)
    );
    for index in 0..spec.files.len() {
        let path = &spec.files[index].path;
        let original = std::fs::read(path).unwrap();
        std::fs::write(path, "changed").unwrap();
        assert_ne!(spec.identify().unwrap().sha256, base.sha256);
        assert!(base.verify().is_err());
        std::fs::write(path, original).unwrap();
    }
    let mut changed = spec.clone();
    changed.arguments.get_mut("hip").unwrap().reverse();
    assert_ne!(changed.identify().unwrap().sha256, base.sha256);
    changed = spec.clone();
    changed
        .environment
        .insert("CXX".into(), Some(String::new()));
    assert_ne!(changed.identify().unwrap().sha256, base.sha256);
    changed = spec.clone();
    changed.versions.insert("recipe".into(), "2".into());
    assert_ne!(changed.identify().unwrap().sha256, base.sha256);
    changed = spec.clone();
    changed.files.reverse();
    assert_ne!(changed.identify().unwrap().sha256, base.sha256);
    changed = spec.clone();
    changed.target = "gfx942".into();
    assert_ne!(changed.identify().unwrap().sha256, base.sha256);
    let mut forged = base.clone();
    forged.files[0].sha256 = "0".repeat(64);
    assert!(forged.verify().is_err());
    assert!(forged.cache_name().is_err());
    std::fs::remove_file(&spec.files[1].path).unwrap();
    assert!(base.verify().is_err());
}

#[test]
fn ambiguous_or_incomplete_inputs_fail_closed() {
    let dir = tempfile::tempdir().unwrap();
    let spec = fixture(dir.path());
    let mut invalid = spec.clone();
    invalid.files.push(invalid.files[0].clone());
    assert!(invalid.identify().is_err());
    invalid = spec.clone();
    invalid.package = "../escape".into();
    assert!(invalid.identify().is_err());
    invalid = spec.clone();
    invalid.schema_version = 2;
    assert!(invalid.identify().is_err());
    invalid = spec.clone();
    invalid.files[0].path = dir.path().to_owned();
    assert!(invalid.identify().is_err());
    invalid = spec.clone();
    invalid.versions.clear();
    assert!(invalid.identify().is_err());
    invalid = spec;
    invalid.files.clear();
    assert!(invalid.identify().is_err());
}
