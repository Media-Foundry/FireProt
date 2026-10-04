use fireprot_core::request::ArtifactRef;
use fireprot_runtime::artifacts::{sha256, verify_outputs};

#[test]
fn hashes_are_standard_and_changes_are_detected() {
    let d = tempfile::tempdir().unwrap();
    let file = d.path().join("data");
    std::fs::write(&file, b"abc").unwrap();
    let hash = sha256(&file).unwrap();
    assert_eq!(
        hash,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let a = ArtifactRef {
        path: "data".into(),
        sha256: hash,
    };
    assert!(verify_outputs(d.path(), std::slice::from_ref(&a)).is_ok());
    std::fs::write(file, b"changed").unwrap();
    assert!(verify_outputs(d.path(), &[a]).is_err());
}
#[test]
fn escaping_paths_and_symlinks_fail_even_with_matching_hash() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("outputs");
    std::fs::create_dir(&root).unwrap();
    let outside = d.path().join("outside");
    std::fs::write(&outside, b"abc").unwrap();
    let hash = sha256(&outside).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
    for path in ["../outside", "link", "", "/etc/hostname"] {
        assert!(
            verify_outputs(
                &root,
                &[ArtifactRef {
                    path: path.into(),
                    sha256: hash.clone()
                }]
            )
            .is_err()
        );
    }
    assert!(sha256(&root).is_err());
}
