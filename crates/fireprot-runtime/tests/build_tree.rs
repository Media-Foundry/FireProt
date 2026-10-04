use fireprot_runtime::{
    build::{BuildFile, BuildSpec},
    build_tree::BuildTree,
};
use std::{collections::BTreeMap, fs};

fn setup(root: &std::path::Path) -> BuildSpec {
    fs::write(root.join("source.cu"), "source").unwrap();
    fs::create_dir(root.join("include")).unwrap();
    fs::write(root.join("include/a.h"), "a").unwrap();
    BuildSpec {
        schema_version: 1,
        dynamic_inputs: None,
        package: "test".into(),
        target: "gfx90a".into(),
        files: vec![BuildFile {
            name: "source".into(),
            path: root.join("source.cu"),
        }],
        trees: vec![BuildTree {
            name: "headers".into(),
            path: root.join("include"),
            suffixes: vec![],
        }],
        arguments: BTreeMap::new(),
        environment: BTreeMap::new(),
        versions: BTreeMap::from([("recipe".into(), "1".into())]),
    }
}

#[test]
fn directory_membership_and_content_participate_in_verification() {
    let dir = tempfile::tempdir().unwrap();
    let spec = setup(dir.path());
    let old = spec.identify().unwrap();
    old.verify().unwrap();
    fs::write(dir.path().join("include/b.h"), "b").unwrap();
    assert!(old.verify().is_err());
    let new = spec.identify().unwrap();
    assert_ne!(new.sha256, old.sha256);
    assert_eq!(
        new.tree_files
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>(),
        ["headers/a.h", "headers/b.h"]
    );
    fs::remove_file(dir.path().join("include/b.h")).unwrap();
    assert!(new.verify().is_err());
    old.verify().unwrap();
    fs::write(dir.path().join("include/a.h"), "edited").unwrap();
    assert!(old.verify().is_err());
}

#[test]
fn explicit_suffix_selection_excludes_generated_bytecode() {
    let dir = tempfile::tempdir().unwrap();
    let mut spec = setup(dir.path());
    spec.trees[0].suffixes = vec![".h".into()];
    let old = spec.identify().unwrap();
    fs::write(dir.path().join("include/helper.pyc"), "generated").unwrap();
    old.verify().unwrap();
    spec.trees.push(spec.trees[0].clone());
    assert!(spec.identify().is_err());
}

#[cfg(unix)]
#[test]
fn follows_linked_directories_but_rejects_cycles_and_retargeting() {
    use std::os::unix::fs::symlink;
    let dir = tempfile::tempdir().unwrap();
    let spec = setup(dir.path());
    fs::create_dir(dir.path().join("external")).unwrap();
    fs::write(dir.path().join("external/b.h"), "b").unwrap();
    let link = dir.path().join("include/linked");
    symlink(dir.path().join("external"), &link).unwrap();
    let old = spec.identify().unwrap();
    assert_eq!(old.tree_files.len(), 2);
    fs::remove_file(&link).unwrap();
    symlink(dir.path().join("include"), &link).unwrap();
    assert!(spec.identify().unwrap_err().to_string().contains("cycle"));
    assert!(old.verify().is_err());
    fs::remove_file(&link).unwrap();
    symlink(dir.path().join("missing"), &link).unwrap();
    assert!(spec.identify().is_err());
}

#[test]
fn pre_tree_schema_receipt_keeps_its_cache_identity() {
    let receipt: fireprot_runtime::build::BuildIdentity = serde_json::from_str(r#"{"schema_version":1,"package":"legacy","target":"gfx90a","files":[{"name":"source","canonical_path":"/tmp/legacy.cu","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"arguments":{},"environment":{},"versions":{"recipe":"1"},"sha256":"221280e9c569a490332d7b0b507ad97bad633a71e18fcc355bbc808dc7e43110"}"#).unwrap();
    assert_eq!(
        receipt.cache_name().unwrap(),
        "legacy_221280e9c569a490332d7b0b507ad97bad633a71e18fcc355bbc808dc7e43110"
    );
    assert!(receipt.trees.is_empty());
    assert!(receipt.tree_files.is_empty());
}
