use super::*;
use crate::structure::AtomIdentity;

fn f32_array(shape: Vec<usize>, data: &[f32]) -> Array {
    Array::new(
        DType::Float32,
        shape,
        data.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn fixture() -> (BTreeMap<String, Array>, Structure) {
    let xyz = vec![1.23456f32; 15];
    let mut arrays = BTreeMap::from([
        ("coordinates".into(), f32_array(vec![5, 3], &xyz)),
        ("plddt".into(), f32_array(vec![1], &[0.8])),
        ("plddt_per_atom".into(), f32_array(vec![8], &[0.8; 8])),
        (
            "atom_mask".into(),
            Array::new(DType::Bool, vec![8], vec![1, 1, 1, 1, 1, 0, 0, 0]).unwrap(),
        ),
        ("pae".into(), f32_array(vec![1, 1], &[1.])),
        ("pde".into(), f32_array(vec![1, 1], &[1.])),
        ("pair_chains_iptm".into(), f32_array(vec![1, 1], &[0.8])),
        ("ptm".into(), f32_array(vec![], &[0.8])),
        ("iptm".into(), f32_array(vec![], &[0.])),
    ]);
    arrays.insert(
        "token_to_atoms".into(),
        Array::new(
            DType::Int32,
            vec![1, 2],
            [0i32, 5].iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
        .unwrap(),
    );
    for name in ["residue_index", "entity_id"] {
        arrays.insert(
            name.into(),
            Array::new(DType::Int64, vec![1], 0i64.to_le_bytes().to_vec()).unwrap(),
        );
    }
    let atoms = ["N", "CA", "C", "O", "CB"]
        .iter()
        .enumerate()
        .map(|(i, n)| AtomIdentity {
            serial: i + 1,
            chain: "A".into(),
            residue: 1,
            insertion: None,
            residue_name: "ALA".into(),
            atom_name: (*n).into(),
            element: n[..1].into(),
        })
        .collect();
    (
        arrays,
        Structure {
            atoms,
            coordinates: vec![[f64::from(1.235f32); 3]; 5],
        },
    )
}

#[test]
fn native_cif_rounding_preserves_unrounded_output_identity() {
    let (arrays, structure) = fixture();
    validate_structure("A", &arrays, &structure).unwrap();
    assert_ne!(
        arrays["coordinates"].coordinates().unwrap(),
        structure.coordinates
    );
}

#[test]
fn malformed_atom_padding_is_rejected() {
    for bytes in [vec![1, 1, 1, 1, 0, 0, 0, 0], vec![0, 1, 1, 1, 1, 1, 0, 0]] {
        let (mut arrays, structure) = fixture();
        arrays.insert(
            "atom_mask".into(),
            Array::new(DType::Bool, vec![8], bytes).unwrap(),
        );
        assert!(validate_structure("A", &arrays, &structure).is_err());
    }
}

#[test]
fn native_biotite_decimal_intermediate_is_preserved() {
    for (raw, written) in [
        (-18.454_5_f32, -18.454_f32),
        (23.425_5, 23.425),
        (-22.180_5, -22.180),
        (-19.208_5, -19.209),
        (-36.810_5, -36.810),
        (11.121_5, 11.121),
        (-43.596_5, -43.596),
        (-56.357_5, -56.358),
        (-52.022_5, -52.023),
        (34.732_5, 34.733),
        (33.054_5, 33.054),
    ] {
        let (mut arrays, mut structure) = fixture();
        arrays.insert("coordinates".into(), f32_array(vec![5, 3], &[raw; 15]));
        structure.coordinates = vec![[f64::from(written); 3]; 5];
        validate_structure("A", &arrays, &structure).unwrap();
    }
}

#[test]
fn wrong_sequence_partition_identity_and_coordinates_fail() {
    for mode in 0..6 {
        let (mut arrays, mut structure) = fixture();
        match mode {
            0 => structure.atoms[4].atom_name = "CA".into(),
            1 => structure.coordinates[0][0] += 0.001,
            2 => structure.atoms[0].chain = "B".into(),
            3 => {
                arrays.insert(
                    "token_to_atoms".into(),
                    Array::new(
                        DType::Int32,
                        vec![1, 2],
                        [0i32, 4].iter().flat_map(|x| x.to_le_bytes()).collect(),
                    )
                    .unwrap(),
                );
            }
            4 => {
                arrays.remove("pde");
            }
            _ => structure.atoms[0].residue_name = "GLY".into(),
        }
        assert!(
            validate_structure("A", &arrays, &structure).is_err(),
            "{mode}"
        );
    }
    let (arrays, structure) = fixture();
    assert!(validate_structure("X", &arrays, &structure).is_err());
    assert!(validate_structure("G", &arrays, &structure).is_err());
}
