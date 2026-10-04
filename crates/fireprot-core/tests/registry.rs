use fireprot_core::registry::*;
use std::collections::BTreeMap;
fn signature() -> OperationSignature {
    OperationSignature {
        operation: "triangle".into(),
        device_type: "hip".into(),
        architecture: "gfx90a".into(),
        dtype: "float32".into(),
        shape: vec![1000, 1000, 128],
        strides: vec![128000, 128, 1],
        layout: "contiguous".into(),
        mode: "inference".into(),
        mask: "none".into(),
        inplace: false,
        attributes: BTreeMap::new(),
    }
}
fn implementation(name: &str, priority: i32) -> Implementation {
    Implementation {
        operation: "triangle".into(),
        name: name.into(),
        version: "1".into(),
        priority,
    }
}
#[test]
fn priority_and_ties_are_deterministic() {
    let mut r = Registry::default();
    r.register(implementation("z", 10), |_| Ok(())).unwrap();
    r.register(implementation("a", 10), |_| Ok(())).unwrap();
    r.register(implementation("slow", 1), |_| Ok(())).unwrap();
    assert_eq!(r.resolve(&signature()).implementation.unwrap().name, "a");
    assert_eq!(r.count("triangle", "a"), 1);
}
#[test]
fn missing_and_rejected_candidates_fall_back() {
    let mut r = Registry::default();
    assert!(r.resolve(&signature()).implementation.is_none());
    r.register(implementation("wrong-arch", 10), |_| {
        Err("architecture mismatch".into())
    })
    .unwrap();
    let s = r.resolve(&signature());
    assert!(s.implementation.is_none());
    assert_eq!(s.rejected[0].reason, "architecture mismatch");
    assert_eq!(r.count("triangle", "native"), 2);
}
#[test]
fn registration_and_removal_invalidate_cached_decisions() {
    let mut r = Registry::default();
    r.register(implementation("old", 1), |_| Ok(())).unwrap();
    r.resolve(&signature());
    assert!(r.resolve(&signature()).cache_hit);
    r.register(implementation("new", 2), |_| Ok(())).unwrap();
    let s = r.resolve(&signature());
    assert!(!s.cache_hit);
    assert_eq!(s.implementation.unwrap().name, "new");
    r.unregister("triangle", "new").unwrap();
    let s = r.resolve(&signature());
    assert!(!s.cache_hit);
    assert_eq!(s.implementation.unwrap().name, "old");
}
#[test]
fn exact_strides_and_attributes_are_part_of_cache_identity() {
    let mut r = Registry::default();
    r.register(implementation("packed", 1), |s| {
        if s.strides.last() == Some(&1) && s.attributes.is_empty() {
            Ok(())
        } else {
            Err("layout or environment mismatch".into())
        }
    })
    .unwrap();
    let mut s = signature();
    assert!(r.resolve(&s).implementation.is_some());
    s.strides[2] = 1000;
    assert!(r.resolve(&s).implementation.is_none());
    let mut s = signature();
    s.attributes.insert("torch".into(), "different".into());
    assert!(r.resolve(&s).implementation.is_none());
}
#[test]
fn errors_and_panics_do_not_select_an_implementation() {
    let mut r = Registry::default();
    r.register(implementation("broken", 1), |_| panic!("probe failed"))
        .unwrap();
    let s = r.resolve(&signature());
    assert!(s.implementation.is_none());
    assert_eq!(s.rejected[0].reason, "capability predicate panicked");
}
#[test]
fn duplicate_registration_and_missing_removal_are_errors() {
    let mut r = Registry::default();
    r.register(implementation("one", 1), |_| Ok(())).unwrap();
    assert_eq!(
        r.register(implementation("one", 2), |_| Ok(())),
        Err(RegistryError::Duplicate)
    );
    assert_eq!(
        r.unregister("triangle", "absent"),
        Err(RegistryError::Unknown)
    );
    r.clear_runtime_state();
    assert!(!r.resolve(&signature()).cache_hit);
    assert_eq!(r.count("triangle", "one"), 1);
}

#[test]
fn manifest_retains_history_until_reset_and_keeps_registration() {
    let mut r = Registry::default();
    r.resolve(&signature());
    r.register(implementation("fast", 1), |_| Ok(())).unwrap();
    // Registration invalidates the cache, not the audit history.
    assert_eq!(r.manifest().decisions[0].selected, "native");
    r.resolve(&signature());
    r.resolve(&signature());
    let manifest = r.manifest();
    assert_eq!(manifest.implementations["triangle"][0].name, "fast");
    assert_eq!(manifest.selection_counts[0].implementation, "fast");
    assert_eq!(manifest.selection_counts[0].count, 2);
    assert_eq!(manifest.selection_counts[1].implementation, "native");
    assert_eq!(manifest.selection_counts[1].count, 1);
    assert_eq!(manifest.decisions.len(), 1);
    assert_eq!(manifest.decisions[0].selected, "fast");
    r.clear_runtime_state();
    assert!(r.manifest().decisions.is_empty());
    assert!(r.manifest().selection_counts.is_empty());
    assert_eq!(r.manifest().implementations, manifest.implementations);
    assert!(!r.resolve(&signature()).cache_hit);
}
