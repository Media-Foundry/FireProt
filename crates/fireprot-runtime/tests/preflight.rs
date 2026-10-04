use fireprot_core::{Request, ValidatedRequest, request::ArtifactRef};
use fireprot_runtime::{artifacts::sha256, preflight::*};
use serde_json::json;
use std::{fs, path::PathBuf, process::Command};
struct Fixture {
    dir: tempfile::TempDir,
    root: PathBuf,
    commit: String,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("source");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("model.py"), b"source\n").unwrap();
        let mut f = Self {
            dir,
            root,
            commit: String::new(),
        };
        f.git(&["init", "-q"]);
        f.git(&["add", "model.py"]);
        f.git(&[
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ]);
        f.commit = f.git(&["rev-parse", "HEAD"]).trim().into();
        fs::write(f.dir.path().join("input"), b"input").unwrap();
        fs::write(f.dir.path().join("weights"), b"weights").unwrap();
        f
    }
    fn git(&self, args: &[&str]) -> String {
        let o = Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(args)
            .output()
            .unwrap();
        assert!(o.status.success(), "{:?}", o);
        String::from_utf8(o.stdout).unwrap()
    }
    fn request(&self) -> Request {
        let input = self.dir.path().join("input");
        let weight = self.dir.path().join("weights");
        serde_json::from_value(json!({"schema_version":1,"run_id":"preflight","backend":"native","seed":101,"hip_device":7,
            "workload":{"model":"openfold3","model_name":"openbind-0","recycles":3,"diffusion_steps":200,"samples":1,"precision":"float32"},
            "source":{"root":self.root,"commit":self.commit},"checkpoints":{"model":{"path":weight,"sha256":sha256(&weight).unwrap()}},
            "input":{"path":input,"sha256":sha256(&input).unwrap()},"output_dir":self.dir.path().join("output")})).unwrap()
    }
    fn export(&self) -> ArtifactRef {
        let manifest = export_source(&self.root, &self.commit).unwrap();
        let path = self.dir.path().join("source-manifest.json");
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        ArtifactRef {
            sha256: sha256(&path).unwrap(),
            path,
        }
    }
}
fn checked(r: Request) -> ValidatedRequest {
    ValidatedRequest::try_from(r).unwrap()
}
#[test]
fn pins_actual_source_input_and_weights() {
    let f = Fixture::new();
    let p = prepare(checked(f.request()), SourceOrigin::GitCheckout).unwrap();
    assert_eq!(p.provenance().source.commit, f.commit);
    assert_eq!(p.provenance().source.files.len(), 1);
    assert_eq!(p.provenance().input.bytes, 5);
    assert_eq!(p.provenance().checkpoints.values().next().unwrap().bytes, 7);
    assert!(!p.request().as_request().output_dir.exists());
}
#[test]
fn wrong_commit_and_hidden_dirty_source_fail() {
    let f = Fixture::new();
    let mut r = f.request();
    r.source.commit = "0".repeat(40);
    assert!(prepare(checked(r), SourceOrigin::GitCheckout).is_err());
    f.git(&["update-index", "--assume-unchanged", "model.py"]);
    fs::write(f.root.join("model.py"), b"changed\n").unwrap();
    assert!(f.git(&["status", "--porcelain"]).is_empty());
    assert!(
        prepare(checked(f.request()), SourceOrigin::GitCheckout)
            .err()
            .unwrap()
            .contains("differ from commit")
    );
}
#[test]
fn dirty_staged_or_missing_source_is_rejected() {
    let f = Fixture::new();
    fs::write(f.root.join("model.py"), b"changed").unwrap();
    assert!(export_source(&f.root, &f.commit).is_err());
    f.git(&["add", "model.py"]);
    assert!(export_source(&f.root, &f.commit).is_err());
    fs::remove_file(f.root.join("model.py")).unwrap();
    assert!(export_source(&f.root, &f.commit).is_err());
}
#[test]
fn corrupted_input_or_any_checkpoint_fails() {
    let f = Fixture::new();
    let mut r = f.request();
    r.input.sha256 = "0".repeat(64);
    assert!(prepare(checked(r), SourceOrigin::GitCheckout).is_err());
    let mut r = f.request();
    r.checkpoints.values_mut().next().unwrap().sha256 = "0".repeat(64);
    assert!(prepare(checked(r), SourceOrigin::GitCheckout).is_err());
    let mut r = f.request();
    r.input.path = f.dir.path().join("missing");
    assert!(prepare(checked(r), SourceOrigin::GitCheckout).is_err());
}
#[test]
fn exported_tree_needs_no_git_but_requires_matching_bytes() {
    let f = Fixture::new();
    let manifest = f.export();
    let destination = f.dir.path().join("remote");
    fs::create_dir(&destination).unwrap();
    fs::copy(f.root.join("model.py"), destination.join("model.py")).unwrap();
    let mut r = f.request();
    r.source.root = destination.clone();
    let p = prepare(
        checked(r.clone()),
        SourceOrigin::Exported {
            manifest: manifest.clone(),
        },
    )
    .unwrap();
    assert!(p.provenance().exported_manifest.is_some());
    fs::write(destination.join("model.py"), b"different").unwrap();
    assert!(prepare(checked(r), SourceOrigin::Exported { manifest }).is_err());
}
#[test]
fn manifest_is_pinned_and_paths_cannot_escape() {
    let f = Fixture::new();
    let manifest = f.export();
    let mut bytes: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest.path).unwrap()).unwrap();
    bytes["commit"] = json!("0".repeat(40));
    fs::write(&manifest.path, serde_json::to_vec(&bytes).unwrap()).unwrap();
    assert!(
        prepare(
            checked(f.request()),
            SourceOrigin::Exported {
                manifest: manifest.clone()
            }
        )
        .is_err()
    );
    let manifest = ArtifactRef {
        sha256: sha256(&manifest.path).unwrap(),
        ..manifest
    };
    assert!(prepare(checked(f.request()), SourceOrigin::Exported { manifest }).is_err());
    let manifest = f.export();
    let mut m: SourceManifest = serde_json::from_slice(&fs::read(&manifest.path).unwrap()).unwrap();
    let entry = m.files.remove(&PathBuf::from("model.py")).unwrap();
    m.files.insert("../input".into(), entry);
    fs::write(&manifest.path, serde_json::to_vec(&m).unwrap()).unwrap();
    let manifest = ArtifactRef {
        sha256: sha256(&manifest.path).unwrap(),
        ..manifest
    };
    assert!(prepare(checked(f.request()), SourceOrigin::Exported { manifest }).is_err());
}
