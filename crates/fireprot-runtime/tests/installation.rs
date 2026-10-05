use fireprot_core::request::ArtifactRef;
use fireprot_runtime::{
    artifacts::sha256,
    installation::{InstallSpec, install, verify},
};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::{PermissionsExt, symlink},
    path::Path,
};

fn fixture(root: &Path) -> InstallSpec {
    for name in [
        "python/fireprot/__init__.py",
        "python/fireprot_workers/__main__.py",
        "kernels/amd/example.hip",
        "fireprot/__init__.py",
        "README.md",
        "Cargo.toml",
        "Cargo.lock",
        "LICENSE",
        "THIRD_PARTY.md",
        "licenses/Apache-2.0.txt",
        "licenses/CK-MIT.txt",
        "licenses/PyTorch-BSD.txt",
    ] {
        let p = root.join(name);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, name).unwrap();
    }
    fs::create_dir_all(root.join("python/fireprot/__pycache__")).unwrap();
    fs::write(
        root.join("python/fireprot/__pycache__/cached.pyc"),
        "ignored",
    )
    .unwrap();
    fs::create_dir_all(root.join("runtime")).unwrap();
    fs::write(root.join("runtime/weight.pt"), "not a distribution file").unwrap();
    let binary = root.join("binary");
    fs::write(&binary, "test executable").unwrap();
    let artifact = ArtifactRef {
        sha256: sha256(&binary).unwrap(),
        path: binary,
    };
    InstallSpec {
        schema_version: 1,
        source_root: root.to_owned(),
        cli: artifact.clone(),
        bridge: artifact.clone(),
        compiler_guard: artifact.clone(),
        hip_probe: artifact,
        cache_root: root.join("external-cache"),
        profiles: BTreeMap::from([(
            "native".into(),
            serde_json::json!({"python":"/external/python","workers":"/old/python","hip_probe":"/old/probe","amd_smi":"/usr/bin/amd-smi","source_manifest":null,"environment":{"MODEL_CACHE":"/models/cache"}}),
        )]),
    }
}

#[test]
fn independent_installation_retains_profile_and_excludes_runtime() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("source");
    let spec = fixture(&source);
    let dst = tmp.path().join("installed");
    let result = install(&spec, &dst).unwrap();
    assert!(!dst.join("runtime").exists());
    assert!(
        !dst.join("lib/fireprot/python/fireprot/__pycache__")
            .exists()
    );
    let profile: serde_json::Value =
        serde_json::from_slice(&fs::read(dst.join("profiles/native.json")).unwrap()).unwrap();
    assert_eq!(profile["python"], "/external/python");
    assert_eq!(profile["environment"]["MODEL_CACHE"], "/models/cache");
    assert_eq!(
        profile["workers"],
        dst.join("lib/fireprot/python").to_str().unwrap()
    );
    assert_eq!(
        profile["environment"]["FIREPROT_BUILD_GUARD"],
        dst.join("bin/fireprot-compiler-guard").to_str().unwrap()
    );
    fs::remove_dir_all(source).unwrap();
    for notice in [
        "LICENSE",
        "THIRD_PARTY.md",
        "licenses/Apache-2.0.txt",
        "licenses/CK-MIT.txt",
        "licenses/PyTorch-BSD.txt",
    ] {
        assert_eq!(
            fs::read_to_string(dst.join("share/fireprot").join(notice)).unwrap(),
            notice
        );
    }
    verify(&dst, &result.manifest_sha256).unwrap();
    assert!(install(&spec, &dst).is_err());
    fs::write(
        dst.join("lib/fireprot/python/fireprot/__init__.py"),
        "changed",
    )
    .unwrap();
    assert!(verify(&dst, &result.manifest_sha256).is_err());
}

#[test]
fn reject_extra_files_symlinks_permissions_and_wrong_digest() {
    let tmp = tempfile::tempdir().unwrap();
    let spec = fixture(&tmp.path().join("source"));
    let dst = tmp.path().join("installed");
    let result = install(&spec, &dst).unwrap();
    assert!(verify(&dst, &"0".repeat(64)).is_err());
    fs::write(dst.join("extra"), "unexpected").unwrap();
    assert!(verify(&dst, &result.manifest_sha256).is_err());
    fs::remove_file(dst.join("extra")).unwrap();
    let cli = dst.join("bin/fireprot");
    fs::set_permissions(&cli, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(verify(&dst, &result.manifest_sha256).is_err());
    fs::set_permissions(&cli, fs::Permissions::from_mode(0o755)).unwrap();
    symlink(&spec.cli.path, dst.join("escape")).unwrap();
    assert!(verify(&dst, &result.manifest_sha256).is_err());
}

#[test]
fn failed_installation_does_not_publish_or_replace() {
    let tmp = tempfile::tempdir().unwrap();
    let mut spec = fixture(&tmp.path().join("source"));
    let dst = tmp.path().join("installed");
    spec.cli.sha256 = "0".repeat(64);
    assert!(install(&spec, &dst).is_err());
    assert!(!dst.exists());
    spec.cli.sha256 = sha256(&spec.cli.path).unwrap();
    symlink(
        &spec.cli.path,
        spec.source_root.join("python/fireprot/escape.py"),
    )
    .unwrap();
    assert!(install(&spec, &dst).is_err());
    assert!(!dst.exists());
    fs::remove_file(spec.source_root.join("python/fireprot/escape.py")).unwrap();
    spec.profiles
        .insert("../escape".into(), serde_json::json!({}));
    assert!(install(&spec, &dst).is_err());
    assert!(!dst.exists());
    fs::create_dir(&dst).unwrap();
    fs::write(dst.join("keep"), "untouched").unwrap();
    assert!(install(&spec, &dst).is_err());
    assert_eq!(fs::read_to_string(dst.join("keep")).unwrap(), "untouched");
}
