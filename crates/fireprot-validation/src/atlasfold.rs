//! AtlasFold monomer atom14-to-CIF correspondence for the pinned upstream writer.
//!
//! The caller must independently bind `sequence` to the request input. This
//! primitive alone does not authenticate the sequence or establish run acceptance.
use crate::{
    Error, Result,
    array::{Array, DType},
    structure::Structure,
};

/// Require every written atom, in upstream order, with its serialized coordinate.
/// Unused atom14 slots have no CIF counterpart; their values are still covered by
/// the separate complete-array comparison. No structural tolerance is introduced.
pub fn validate_atom14(sequence: &str, coordinates: &Array, structure: &Structure) -> Result<()> {
    if sequence.is_empty()
        || coordinates.dtype() != DType::Float32
        || coordinates.shape() != [sequence.len(), 14, 3]
        || structure.atoms.len() != structure.coordinates.len()
    {
        return Err(Error::Schema(
            "AtlasFold atom14 shape/dtype mismatch".into(),
        ));
    }
    let values = coordinates.values()?;
    let mut index = 0;
    for (residue, aa) in sequence.bytes().enumerate() {
        let (residue_name, names) = residue_atoms(aa)?;
        for (slot, name) in names.split_ascii_whitespace().enumerate() {
            let atom = structure
                .atoms
                .get(index)
                .ok_or_else(|| Error::Schema("missing AtlasFold CIF atom".into()))?;
            if atom.serial != index + 1
                || atom.chain != "A"
                || atom.residue != (residue + 1) as isize
                || atom.insertion.is_some()
                || atom.residue_name != residue_name
                || atom.atom_name != name
                || atom.element != name[..1]
            {
                return Err(Error::Schema(format!(
                    "AtlasFold CIF identity mismatch at atom {}",
                    index + 1
                )));
            }
            for axis in 0..3 {
                let expected = serialized_coordinate(values[(residue * 14 + slot) * 3 + axis])?;
                if structure.coordinates[index][axis] != expected {
                    return Err(Error::Schema(format!(
                        "AtlasFold CIF coordinate mismatch at atom {}, axis {axis}",
                        index + 1
                    )));
                }
            }
            index += 1;
        }
    }
    if index != structure.atoms.len() {
        return Err(Error::Schema("extra AtlasFold CIF atoms".into()));
    }
    Ok(())
}

// AtlasFold 8ab3aca0, common/residue_constants.py. X is UNK with five atoms.
fn residue_atoms(aa: u8) -> Result<(&'static str, &'static str)> {
    Ok(match aa {
        b'A' => ("ALA", "N CA C O CB"),
        b'R' => ("ARG", "N CA C O CB CG CD NE CZ NH1 NH2"),
        b'N' => ("ASN", "N CA C O CB CG OD1 ND2"),
        b'D' => ("ASP", "N CA C O CB CG OD1 OD2"),
        b'C' => ("CYS", "N CA C O CB SG"),
        b'Q' => ("GLN", "N CA C O CB CG CD OE1 NE2"),
        b'E' => ("GLU", "N CA C O CB CG CD OE1 OE2"),
        b'G' => ("GLY", "N CA C O"),
        b'H' => ("HIS", "N CA C O CB CG ND1 CD2 CE1 NE2"),
        b'I' => ("ILE", "N CA C O CB CG1 CG2 CD1"),
        b'L' => ("LEU", "N CA C O CB CG CD1 CD2"),
        b'K' => ("LYS", "N CA C O CB CG CD CE NZ"),
        b'M' => ("MET", "N CA C O CB CG SD CE"),
        b'F' => ("PHE", "N CA C O CB CG CD1 CD2 CE1 CE2 CZ"),
        b'P' => ("PRO", "N CA C O CB CG CD"),
        b'S' => ("SER", "N CA C O CB OG"),
        b'T' => ("THR", "N CA C O CB OG1 CG2"),
        b'W' => ("TRP", "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2"),
        b'Y' => ("TYR", "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH"),
        b'V' => ("VAL", "N CA C O CB CG1 CG2"),
        b'X' => ("UNK", "N CA C O CB"),
        _ => return Err(Error::Schema("invalid AtlasFold sequence residue".into())),
    })
}

fn serialized_coordinate(value: f64) -> Result<f64> {
    if !value.is_finite() || f64::from(value as f32) != value {
        return Err(Error::Schema(
            "AtlasFold public coordinates must be finite FP32".into(),
        ));
    }
    // Gemmi 0.7.5 uses stb_sprintf %.9g on double coordinates, with decimal
    // midpoint ties away from zero. Rust's formatter rounds ties to even.
    // A fixed 149 fractional digits represents every FP32 value exactly,
    // including the smallest subnormal (2^-149), without scaled-float rounding.
    let decimal = format!("{:.149}", value.abs());
    let point = decimal.find('.').expect("fixed decimal point");
    let digits: Vec<u8> = decimal.bytes().filter(|&b| b != b'.').collect();
    let Some(first) = digits.iter().position(|&b| b != b'0') else {
        return Ok(value);
    };
    let mut significand = 0u32;
    for i in first..first + 9 {
        significand = significand * 10 + u32::from(digits.get(i).copied().unwrap_or(b'0') - b'0');
    }
    if digits.get(first + 9).is_some_and(|&b| b >= b'5') {
        significand += 1;
    }
    let exponent = point as i32 - first as i32 - 9;
    let sign = if value.is_sign_negative() { "-" } else { "" };
    let intermediate: f64 = format!("{sign}{significand}e{exponent}")
        .parse()
        .map_err(|_| Error::Schema("invalid Gemmi coordinate serialization".into()))?;
    // AtlasFold formats the parsed Gemmi number using Python .3f, then the
    // shared CIF reader converts to FP32. Rust .3 agrees with Python's ties-even.
    let rounded: f64 = format!("{intermediate:.3}")
        .parse()
        .map_err(|_| Error::Schema("invalid AtlasFold coordinate serialization".into()))?;
    let result = f64::from(rounded as f32);
    if !result.is_finite() {
        return Err(Error::Nonfinite);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{numpy, structure};
    use std::path::Path;
    const SEQUENCE: &str = "ACDEFGHIKLMNPQRSTVWYX";
    fn fixture() -> (Array, Structure) {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/atlasfold_atom14");
        let coordinates = numpy::read_npz(&root.join("coordinates.npz"), 1024 * 1024)
            .unwrap()
            .remove("coordinates")
            .unwrap();
        (
            coordinates,
            structure::read(&root.join("prediction.cif")).unwrap(),
        )
    }
    #[test]
    fn upstream_writer_all_residues_and_rounding_boundaries() {
        let (coordinates, structure) = fixture();
        assert_eq!(structure.atoms.len(), 172);
        validate_atom14(SEQUENCE, &coordinates, &structure).unwrap();
        // Direct .3f gives 0.012; Gemmi first serializes this FP32 as 0.0115.
        assert_eq!(
            serialized_coordinate(f64::from(0.0115_f32)).unwrap(),
            f64::from(0.011_f32)
        );
        assert_eq!(
            serialized_coordinate(1000000.125).unwrap(),
            f64::from("1000000.13".parse::<f32>().unwrap())
        );
        assert_eq!(
            serialized_coordinate(f64::from(f32::from_bits(1))).unwrap(),
            0.0
        );
        assert_eq!(
            serialized_coordinate(f64::from(f32::MAX)).unwrap(),
            f64::from(f32::MAX)
        );
    }
    #[test]
    fn serialization_matches_upstream_for_random_bits_and_decimal_boundaries() {
        let rows: Vec<[u32; 2]> = serde_json::from_str(include_str!(
            "../../../tests/fixtures/atlasfold_atom14/rounding.json"
        ))
        .unwrap();
        assert_eq!(rows.len(), 10087);
        for [input, expected] in rows {
            let actual = serialized_coordinate(f64::from(f32::from_bits(input))).unwrap() as f32;
            assert_eq!(actual.to_bits(), expected, "input bits {input:08x}");
        }
    }
    #[test]
    fn rejects_wrong_atoms_sequence_and_coordinates() {
        for mutation in 0..10 {
            let (coordinates, mut structure) = fixture();
            match mutation {
                0 => {
                    structure.atoms[0].atom_name = "CB".into();
                }
                1 => {
                    structure.atoms[0].residue_name = "GLY".into();
                }
                2 => {
                    structure.atoms[0].chain = "B".into();
                }
                3 => {
                    structure.atoms[0].residue = 2;
                }
                4 => {
                    structure.atoms[0].element = "C".into();
                }
                5 => {
                    structure.atoms[0].insertion = Some("A".into());
                }
                6 => {
                    structure.atoms[0].serial = 2;
                }
                7 => {
                    structure.coordinates[0][0] += 0.001;
                }
                8 => {
                    structure.atoms.pop();
                    structure.coordinates.pop();
                }
                9 => {
                    structure.atoms.push(structure.atoms[0].clone());
                    structure.coordinates.push(structure.coordinates[0]);
                }
                _ => unreachable!(),
            }
            assert!(
                validate_atom14(SEQUENCE, &coordinates, &structure).is_err(),
                "mutation {mutation}"
            );
        }
        let (coordinates, structure) = fixture();
        for sequence in [
            "",
            "BCDEFGHIKLMNPQRSTVWYX",
            "XCDEFGHIKLMNPQRSTVWYA",
            "acdefghiklmnpqrstvwyx",
        ] {
            assert!(validate_atom14(sequence, &coordinates, &structure).is_err());
        }
    }
    #[test]
    fn unused_slots_are_omitted_but_written_slots_are_required() {
        let (coordinates, structure) = fixture();
        let mut bytes = coordinates.bytes().to_vec();
        let offset = 5 * 3 * 4; // ALA slot 5 is not a physical atom.
        bytes[offset..offset + 4].copy_from_slice(&999f32.to_le_bytes());
        let changed = Array::new(DType::Float32, coordinates.shape().to_vec(), bytes).unwrap();
        validate_atom14(SEQUENCE, &changed, &structure).unwrap();
        assert!(
            validate_atom14(
                SEQUENCE,
                &Array::new(
                    DType::Float64,
                    coordinates.shape().to_vec(),
                    vec![0; coordinates.bytes().len() * 2]
                )
                .unwrap(),
                &structure
            )
            .is_err()
        );
    }
}
