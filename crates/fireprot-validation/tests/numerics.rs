use fireprot_validation::{
    array::{Array, DType, Policy, compare},
    metrics::{confidence, coordinates, protenix_passes},
    numpy::{compare_archives, ground_truth, read_npy, read_npz},
};
use serde_json::json;
fn f64array(shape: Vec<usize>, v: &[f64]) -> Array {
    Array::new(
        DType::Float64,
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
    .unwrap()
}
#[test]
fn array_identity_includes_shape_dtype_and_signed_zero() {
    let a = f64array(vec![2], &[0.0, 1.0]);
    let b = f64array(vec![2], &[-0.0, 1.0]);
    assert!(!compare(&a, &b, Policy::Exact).unwrap().passed);
    assert!(
        compare(&a, &b, Policy::Absolute { tolerance: 0.0 })
            .unwrap()
            .passed
    );
    assert!(compare(&a, &f64array(vec![1, 2], &[0.0, 1.0]), Policy::Exact).is_err());
    let c = Array::new(DType::Int64, vec![2], a.bytes().to_vec()).unwrap();
    assert!(compare(&a, &c, Policy::Exact).is_err());
    assert!(compare(&c, &c, Policy::Absolute { tolerance: 0.0 }).is_err());
}
#[test]
fn invalid_shapes_nonfinite_values_and_tolerances_fail() {
    for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Array::new(DType::Float64, vec![1], v.to_le_bytes().to_vec()).is_err());
    }
    assert!(Array::new(DType::Float32, vec![0], vec![]).is_err());
    assert!(Array::new(DType::Float32, vec![usize::MAX, 2], vec![]).is_err());
    assert!(Array::new(DType::Float32, vec![2], vec![0; 4]).is_err());
    assert!(Array::new(DType::Bool, vec![1], vec![2]).is_err());
    let a = f64array(vec![1], &[1.]);
    for tolerance in [-1., f64::NAN, f64::INFINITY] {
        assert!(compare(&a, &a, Policy::Absolute { tolerance }).is_err());
    }
}
#[test]
fn half_precision_is_decoded_without_losing_original_dtype() {
    let a = Array::new(DType::Bfloat16, vec![2], vec![0x80, 0x3f, 0, 0xc0]).unwrap();
    assert_eq!(a.values().unwrap(), vec![1., -2.]);
    let b = Array::new(DType::Float16, vec![3], vec![0, 0x3c, 0, 0xc0, 1, 0]).unwrap();
    assert_eq!(b.values().unwrap(), vec![1., -2., 2f64.powi(-24)]);
    assert!(Array::new(DType::Bfloat16, vec![1], vec![0x80, 0x7f]).is_err());
    assert!(Array::new(DType::Float16, vec![1], vec![0, 0x7c]).is_err());
}
#[test]
fn kabsch_matches_numpy_fixture() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/alignment.json")).unwrap();
    let r: Vec<[f64; 3]> = serde_json::from_value(f["reference"].clone()).unwrap();
    let c: Vec<[f64; 3]> = serde_json::from_value(f["candidate"].clone()).unwrap();
    let m = coordinates(&r, &c).unwrap();
    assert!(
        (m.raw_rmsd_angstrom - f["metrics"]["raw_rmsd_angstrom"].as_f64().unwrap()).abs() < 1e-12
    );
    assert!(
        (m.aligned_rmsd_angstrom - f["metrics"]["aligned_rmsd_angstrom"].as_f64().unwrap()).abs()
            < 1e-12
    );
    assert!(
        (m.max_abs_coordinate_delta_angstrom
            - f["metrics"]["max_abs_coordinate_delta_angstrom"]
                .as_f64()
                .unwrap())
        .abs()
            < 1e-12
    );
}
#[test]
fn rotations_pass_but_reflections_do_not() {
    let r = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
    let c: Vec<_> = r
        .iter()
        .map(|p| [p[1] + 4., -p[0] - 2., p[2] + 7.])
        .collect();
    assert!(coordinates(&r, &c).unwrap().aligned_rmsd_angstrom < 1e-12);
    let reflected: Vec<_> = r.iter().map(|p| [-p[0], p[1], p[2]]).collect();
    assert!(coordinates(&r, &reflected).unwrap().aligned_rmsd_angstrom > 0.1);
}
#[test]
fn degenerate_and_invalid_coordinates() {
    assert!(
        coordinates(&[[1., 2., 3.]], &[[8., 9., 10.]])
            .unwrap()
            .aligned_rmsd_angstrom
            < 1e-12
    );
    assert!(
        coordinates(&[[1., 1., 1.]; 4], &[[2., 3., 4.]; 4])
            .unwrap()
            .aligned_rmsd_angstrom
            < 1e-12
    );
    assert!(coordinates(&[], &[]).is_err());
    assert!(coordinates(&[[0.; 3]], &[[0.; 3]; 2]).is_err());
    assert!(coordinates(&[[f64::NAN, 0., 0.]], &[[0.; 3]]).is_err());
}
#[test]
fn confidence_checks_every_numeric_leaf_and_schema() {
    let r = json!({"a":1.,"nested":[{"b":2.}],"ignored":"text"});
    let c = json!({"a":1.25,"nested":[{"b":2.5}],"ignored":"changed"});
    let m = confidence(&r, &c).unwrap();
    assert_eq!(m.numeric_leaf_count, 2);
    assert_eq!(m.max_abs_delta, 0.5);
    assert_eq!(m.mean_abs_delta, 0.375);
    let coord = coordinates(&[[0.; 3]], &[[0.; 3]]).unwrap();
    assert!(!protenix_passes(&coord, &m, 1e-3, 1e-3).unwrap());
    let missing = confidence(&r, &json!({"a":1.})).unwrap();
    assert!(!protenix_passes(&coord, &missing, 1e-3, 1e-3).unwrap());
    assert!(confidence(&json!({}), &json!({})).is_err());
    let collision = confidence(&json!({"0":1}), &json!([1])).unwrap();
    assert!(!collision.missing_from_candidate.is_empty());
}
#[test]
fn numpy_endian_fortran_and_ground_truth_mask() {
    let a = read_npy(include_bytes!("fixtures/fortran_be.npy")).unwrap();
    assert_eq!(a.shape(), &[2, 3, 4]);
    assert_eq!(
        a.values().unwrap(),
        (0..24).map(|x| x as f64).collect::<Vec<_>>()
    );
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mapping.npz");
    let mapping = read_npz(&p, 10000).unwrap();
    assert_eq!(
        ground_truth(&mapping).unwrap(),
        vec![[0., 0., 0.], [0., 2., 0.], [0., 0., 3.]]
    );
    assert!(
        compare_archives(&mapping, &mapping, Policy::Exact)
            .unwrap()
            .values()
            .all(|v| v.passed)
    );
    let mut missing = mapping.clone();
    missing.remove("mask");
    assert!(ground_truth(&missing).is_err());
    assert!(compare_archives(&mapping, &missing, Policy::Exact).is_err());
    assert!(read_npz(&p, 10).is_err());
    let truncated = &include_bytes!("fixtures/fortran_be.npy")[..150];
    assert!(read_npy(truncated).is_err());
}
#[test]
fn duplicate_archive_members_are_rejected() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/duplicate.npz");
    assert!(read_npz(&p, 10000).is_err());
}
