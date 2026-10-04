#![cfg(unix)]
use fireprot_runtime::{
    build::{BuildFile, BuildSpec},
    build_cache::get_or_build,
};
use std::{
    collections::BTreeMap,
    fs, io,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
};

fn identity(root: &std::path::Path) -> fireprot_runtime::build::BuildIdentity {
    let source = root.join("source.cu");
    fs::write(&source, "original").unwrap();
    BuildSpec {
        schema_version: 1,
        dynamic_inputs: None,
        trees: vec![],
        package: "test_kernel".into(),
        target: "gfx90a".into(),
        files: vec![BuildFile {
            name: "source".into(),
            path: source,
        }],
        arguments: BTreeMap::new(),
        environment: BTreeMap::new(),
        versions: BTreeMap::from([("recipe".into(), "1".into())]),
    }
    .identify()
    .unwrap()
}

fn compile(root: &std::path::Path) -> io::Result<Vec<String>> {
    fs::write(root.join("kernel.so"), b"fake library")?;
    Ok(vec!["kernel.so".into()])
}

#[test]
fn concurrent_builders_publish_once_and_corruption_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let id = identity(dir.path());
    let cache = dir.path().join("cache");
    let barrier = Arc::new(Barrier::new(4));
    let calls = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let (id, cache, barrier, calls) =
                (id.clone(), cache.clone(), barrier.clone(), calls.clone());
            std::thread::spawn(move || {
                barrier.wait();
                get_or_build(&cache, &id, |path| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    compile(path)
                })
                .unwrap()
                .cache_hit
            })
        })
        .collect();
    let hits = threads
        .into_iter()
        .map(|t| usize::from(t.join().unwrap()))
        .sum::<usize>();
    assert_eq!(hits, 3);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let built = cache.join(id.cache_name().unwrap());
    fs::write(built.join("kernel.so"), "corrupt").unwrap();
    assert!(
        get_or_build(&cache, &id, |_| panic!(
            "corrupted entry must not rebuild silently"
        ))
        .is_err()
    );
    assert_eq!(
        fs::read_to_string(built.join("kernel.so")).unwrap(),
        "corrupt"
    );
}

#[test]
fn failures_and_changed_inputs_never_publish() {
    let dir = tempfile::tempdir().unwrap();
    let id = identity(dir.path());
    let cache = dir.path().join("cache");
    assert!(
        get_or_build(&cache, &id, |path| {
            compile(path)?;
            Err(io::Error::other("compiler failed"))
        })
        .is_err()
    );
    assert!(!cache.join(id.cache_name().unwrap()).exists());
    assert!(
        get_or_build(&cache, &id, |path| {
            fs::write(&id.files[0].canonical_path, "changed while compiling")?;
            compile(path)
        })
        .is_err()
    );
    assert!(!cache.join(id.cache_name().unwrap()).exists());
    fs::write(&id.files[0].canonical_path, "original").unwrap();
    assert!(!get_or_build(&cache, &id, compile).unwrap().cache_hit);
    assert!(
        get_or_build(&cache, &id, |_| panic!("cache must be reused"))
            .unwrap()
            .cache_hit
    );
}

#[test]
fn unsafe_or_missing_artifacts_and_receipts_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let id = identity(dir.path());
    let cache = dir.path().join("cache");
    for names in [
        vec![],
        vec!["../source.cu".into()],
        vec!["missing.so".into()],
        vec!["kernel.so".into(), "kernel.so".into()],
    ] {
        assert!(
            get_or_build(&cache, &id, |path| {
                compile(path)?;
                Ok(names)
            })
            .is_err()
        );
    }
    assert!(
        get_or_build(&cache, &id, |path| {
            std::os::unix::fs::symlink(&id.files[0].canonical_path, path.join("kernel.so"))?;
            Ok(vec!["kernel.so".into()])
        })
        .is_err()
    );
    let built = get_or_build(&cache, &id, compile).unwrap();
    fs::remove_file(built.directory.join("fireprot-build.json")).unwrap();
    assert!(get_or_build(&cache, &id, |_| panic!("missing receipt must fail")).is_err());
}

#[test]
fn cache_child_process() {
    let Some(root) = std::env::var_os("FIREPROT_TEST_CACHE_CHILD_ROOT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let id: fireprot_runtime::build::BuildIdentity =
        serde_json::from_slice(&fs::read(root.join("identity.json")).unwrap()).unwrap();
    get_or_build(&root.join("cache"), &id, |path| {
        compile(path)?;
        fs::write(root.join("child-building"), "ready")?;
        loop {
            std::thread::park();
        }
    })
    .unwrap();
}

#[test]
fn killed_process_releases_lock_without_publishing_partial_build() {
    let dir = tempfile::tempdir().unwrap();
    let id = identity(dir.path());
    fs::write(
        dir.path().join("identity.json"),
        serde_json::to_vec(&id).unwrap(),
    )
    .unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "cache_child_process", "--nocapture"])
        .env("FIREPROT_TEST_CACHE_CHILD_ROOT", dir.path())
        .stdout(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !dir.path().join("child-building").exists() {
        if child.try_wait().unwrap().is_some() || std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("child never acquired build lock");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    let cache = dir.path().join("cache");
    assert!(!cache.join(id.cache_name().unwrap()).exists());
    // A stale temporary directory may remain. It must never count as a hit.
    assert!(!get_or_build(&cache, &id, compile).unwrap().cache_hit);
}

#[test]
fn waiting_for_lock_can_timeout_or_cancel_without_touching_owner() {
    use fireprot_runtime::build_cache::get_or_build_checked;
    use nix::fcntl::{Flock, FlockArg};
    use std::time::{Duration, Instant};
    let dir = tempfile::tempdir().unwrap();
    let id = identity(dir.path());
    let cache = dir.path().join("cache");
    fs::create_dir(&cache).unwrap();
    let file = fs::File::create(cache.join(format!("{}.lock", id.cache_name().unwrap()))).unwrap();
    let owner = Flock::lock(file, FlockArg::LockExclusive).unwrap();
    let e = get_or_build_checked(
        &cache,
        &id,
        Duration::from_millis(80),
        &mut || Ok(()),
        |_, _| panic!("locked build must not run"),
    )
    .unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::TimedOut);
    let start = Instant::now();
    let e = get_or_build_checked(
        &cache,
        &id,
        Duration::from_secs(5),
        &mut || {
            if start.elapsed() > Duration::from_millis(60) {
                Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"))
            } else {
                Ok(())
            }
        },
        |_, _| panic!("cancelled build must not run"),
    )
    .unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::Interrupted);
    assert!(start.elapsed() < Duration::from_secs(1));
    drop(owner);
    assert!(!get_or_build(&cache, &id, compile).unwrap().cache_hit);
}

#[test]
fn cancellation_after_compilation_cannot_publish() {
    use fireprot_runtime::build_cache::get_or_build_checked;
    use std::{cell::Cell, time::Duration};
    let dir = tempfile::tempdir().unwrap();
    let id = identity(dir.path());
    let cancel = Cell::new(false);
    let cache = dir.path().join("cache");
    let e = get_or_build_checked(
        &cache,
        &id,
        Duration::from_secs(1),
        &mut || {
            if cancel.get() {
                Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"))
            } else {
                Ok(())
            }
        },
        |path, _| {
            let result = compile(path);
            cancel.set(true);
            result
        },
    )
    .unwrap_err();
    assert_eq!(e.kind(), io::ErrorKind::Interrupted);
    assert!(!cache.join(id.cache_name().unwrap()).exists());
    assert!(!get_or_build(&cache, &id, compile).unwrap().cache_hit);
}
