#![cfg(unix)]
use fireprot_runtime::{
    artifacts::sha256,
    build::{BuildFile, BuildSpec},
    build_cache::get_or_build,
    kernel_bundle::{KernelCompatibility, accept, publish},
};
use std::{collections::BTreeMap, fs};

fn fixture(root: &std::path::Path) -> std::path::PathBuf {
    let source = root.join("kernel.cu");
    fs::write(&source, "source fixture").unwrap();
    let spec = BuildSpec {
        schema_version: 1,
        package: "test_kernel".into(),
        target: "gfx90a".into(),
        files: vec![BuildFile {
            name: "source".into(),
            path: source,
        }],
        trees: vec![],
        dynamic_inputs: None,
        arguments: BTreeMap::new(),
        environment: BTreeMap::new(),
        versions: BTreeMap::from([
            ("torch".into(), "test-torch".into()),
            ("hip".into(), "test-hip".into()),
            ("soabi".into(), "test-abi".into()),
        ]),
    };
    let identity = spec.identify().unwrap();
    let filename = format!("{}.so", identity.cache_name().unwrap());
    get_or_build(&root.join("cache"), &identity, |directory| {
        fs::write(directory.join(&filename), "fixture library bytes")?;
        Ok(vec![filename])
    })
    .unwrap()
    .directory
}

#[test]
fn deployment_acceptance_needs_no_build_inputs_and_never_overwrites() {
    let root = tempfile::tempdir().unwrap();
    let cache = fixture(root.path());
    let destination = root.path().join("published");
    let published = publish(&cache, &destination).unwrap();
    assert!(publish(&cache, &destination).is_err());
    fs::remove_file(root.path().join("kernel.cu")).unwrap();
    fs::remove_dir_all(root.path().join("cache")).unwrap();
    let accepted = accept(
        &destination,
        &published.manifest_sha256,
        &published.compatibility,
    )
    .unwrap();
    let report = serde_json::to_value(accepted).unwrap();
    let path = std::path::Path::new(report["library"].as_str().unwrap());
    assert_eq!(fs::read(path).unwrap(), b"fixture library bytes");
    assert_eq!(
        report["module_name"].as_str().unwrap(),
        path.file_stem().unwrap()
    );
    let mut incompatible = published.compatibility.clone();
    if let KernelCompatibility::Python { target, .. } = &mut incompatible {
        *target = "gfx942".into();
    }
    assert!(accept(&destination, &published.manifest_sha256, &incompatible).is_err());
    incompatible = published.compatibility.clone();
    if let KernelCompatibility::Python { torch, .. } = &mut incompatible {
        *torch = "another-torch".into();
    }
    assert!(accept(&destination, &published.manifest_sha256, &incompatible).is_err());
    assert!(accept(&destination, &"0".repeat(64), &published.compatibility).is_err());
}

#[test]
fn rejects_corrupt_library_receipt_and_inconsistent_re_pinned_manifest() {
    let root = tempfile::tempdir().unwrap();
    let cache = fixture(root.path());
    let published = publish(&cache, &root.path().join("published")).unwrap();
    let manifest_path = published.directory.join("kernel.json");
    let original = fs::read(&manifest_path).unwrap();
    let mut manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
    let library = published
        .directory
        .join(manifest["library"]["path"].as_str().unwrap());
    let bytes = fs::read(&library).unwrap();
    fs::write(&library, "corrupt").unwrap();
    assert!(
        accept(
            &published.directory,
            &published.manifest_sha256,
            &published.compatibility
        )
        .is_err()
    );
    fs::write(&library, bytes).unwrap();
    let receipt = published.directory.join("build-receipt.json");
    let receipt_bytes = fs::read(&receipt).unwrap();
    fs::write(&receipt, "{}").unwrap();
    assert!(
        accept(
            &published.directory,
            &published.manifest_sha256,
            &published.compatibility
        )
        .is_err()
    );
    fs::write(&receipt, receipt_bytes).unwrap();
    // Even an explicitly re-pinned manifest must agree with its build receipt.
    manifest["build_identity_sha256"] = "bad".into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        accept(
            &published.directory,
            &sha256(&manifest_path).unwrap(),
            &published.compatibility
        )
        .is_err()
    );
    manifest = serde_json::from_slice(&original).unwrap();
    manifest["library"]["path"] = "../outside.so".into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        accept(
            &published.directory,
            &sha256(&manifest_path).unwrap(),
            &published.compatibility
        )
        .is_err()
    );
}

#[test]
fn rejects_corrupt_source_cache_and_symlink_bundle_artifacts() {
    let root = tempfile::tempdir().unwrap();
    let cache = fixture(root.path());
    let published = publish(&cache, &root.path().join("published")).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(published.directory.join("kernel.json")).unwrap())
            .unwrap();
    let name = manifest["library"]["path"].as_str().unwrap();
    let library = published.directory.join(name);
    fs::remove_file(&library).unwrap();
    std::os::unix::fs::symlink(cache.join(name), &library).unwrap();
    assert!(
        accept(
            &published.directory,
            &published.manifest_sha256,
            &published.compatibility
        )
        .is_err()
    );
    fs::write(cache.join(name), "corrupt cache").unwrap();
    let destination = root.path().join("rejected");
    assert!(publish(&cache, &destination).is_err());
    assert!(!destination.exists());
}
