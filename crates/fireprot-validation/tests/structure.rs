use fireprot_validation::structure;
const CIF: &[u8] = include_bytes!("fixtures/prediction.cif");

#[test]
fn malformed_mandatory_fields_fail_without_poisoning_later_reads() {
    let text = std::str::from_utf8(CIF).unwrap();
    for (before, after) in [
        ("ATOM N N .", "ATOM ? N ."),
        ("ATOM N N .", "ATOM N ? ."),
        ("0 1.25 2 3", "0 ? 2 3"),
        ("0 1.25 2 3", "0 1.25 ? 3"),
        ("0 1.25 2 3", "0 1.25 2 ?"),
        ("1 4 1.0", "1 ? 1.0"),
    ] {
        assert!(structure::parse(text.replace(before, after).as_bytes()).is_err());
        assert_eq!(structure::parse(CIF).unwrap().atoms.len(), 4);
    }
}

#[test]
fn atom_identity_includes_chemical_element() {
    let original = structure::parse(CIF).unwrap();
    let changed = structure::parse(
        std::str::from_utf8(CIF)
            .unwrap()
            .replace("ATOM N N .", "ATOM C N .")
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(original.coordinates, changed.coordinates);
    assert_ne!(original.atoms, changed.atoms);
    assert_eq!(original.atoms[0].element, "N");
}
#[test]
fn prediction_reader_preserves_float32_coordinates_and_atom_identity() {
    let s = structure::parse(CIF).unwrap();
    assert_eq!(
        s.coordinates,
        vec![[1.25, 2., 3.], [4., 5., 6.], [-1., 0., 2.], [1., 1., 0.]]
    );
    assert_eq!(s.atoms[0].serial, 1);
    assert_eq!(s.atoms[0].atom_name, "N");
    assert_eq!(s.atoms[0].chain, "A");
}
#[test]
fn rejects_invalid_empty_nonfinite_alternate_and_multiple_model_structures() {
    let text = std::str::from_utf8(CIF).unwrap();
    for bad in [
        "not a CIF".into(),
        "data_empty\n_entry.id empty\n".into(),
        text.replace("ATOM N N .", "ATOM N N A"),
        text.replace("1.25", "NaN"),
        text.replace("1 4 1.0", "2 4 1.0"),
        text.replace("1 4 1.0", "1 3 1.0"),
        text.replace("1.25 2 3", "1.25 2"),
    ] {
        assert!(structure::parse(bad.as_bytes()).is_err(), "{bad}");
    }
}
#[test]
fn legacy_protenix_gate_checks_atom_identity_and_confidence() {
    let d = tempfile::tempdir().unwrap();
    let r = d.path().join("r.cif");
    let c = d.path().join("c.cif");
    let rc = d.path().join("r.json");
    let cc = d.path().join("c.json");
    std::fs::write(&r, CIF).unwrap();
    std::fs::write(&c, CIF).unwrap();
    std::fs::write(&rc, r#"{"plddt":80.0}"#).unwrap();
    std::fs::write(&cc, r#"{"plddt":80.0}"#).unwrap();
    assert!(
        structure::compare_protenix(&r, &rc, &c, &cc)
            .unwrap()
            .passed
    );
    std::fs::write(&cc, r#"{"plddt":81.0}"#).unwrap();
    assert!(
        !structure::compare_protenix(&r, &rc, &c, &cc)
            .unwrap()
            .passed
    );
    std::fs::write(
        &c,
        std::str::from_utf8(CIF).unwrap().replace("GLY A", "GLY B"),
    )
    .unwrap();
    assert!(structure::compare_protenix(&r, &rc, &c, &cc).is_err());
}
