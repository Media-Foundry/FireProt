//! ESMFold2 monomer outputs: preserve raw FP32 coordinates and verify native CIF rounding.
use crate::{
    Error, Result,
    array::{Array, DType},
    structure::Structure,
};
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
#[path = "esmfold2_tests.rs"]
mod tests;

fn invalid(message: &str) -> Error {
    Error::Schema(format!("ESMFold2: {message}"))
}

pub(crate) fn validate_arrays(arrays: &BTreeMap<String, Array>) -> Result<()> {
    let get = |name: &str| {
        arrays
            .get(name)
            .ok_or_else(|| invalid(&format!("missing {name}")))
    };
    let coords = get("coordinates")?;
    let plddt = get("plddt")?;
    if coords.dtype() != DType::Float32
        || coords.shape().len() != 2
        || coords.shape()[1] != 3
        || coords.shape()[0] == 0
        || plddt.shape().len() != 1
        || plddt.shape()[0] == 0
    {
        return Err(invalid("coordinate/token shape mismatch"));
    }
    let (atoms, tokens) = (coords.shape()[0], plddt.shape()[0]);
    let mask = get("atom_mask")?;
    if mask.dtype() != DType::Bool
        || mask.shape().len() != 1
        || mask.bytes().len() < atoms
        || mask
            .bytes()
            .iter()
            .enumerate()
            .any(|(i, &v)| v != u8::from(i < atoms))
    {
        return Err(invalid(
            "atom mask must retain real atoms followed by padding",
        ));
    }
    for (name, shape) in [
        ("plddt", vec![tokens]),
        ("plddt_per_atom", mask.shape().to_vec()),
        ("pae", vec![tokens, tokens]),
        ("pde", vec![tokens, tokens]),
        ("pair_chains_iptm", vec![1, 1]),
    ] {
        let a = get(name)?;
        if !a.dtype().is_float() || a.shape() != shape {
            return Err(invalid("confidence shape/dtype mismatch"));
        }
        a.values()?;
    }
    for name in ["ptm", "iptm"] {
        let a = get(name)?;
        if !a.dtype().is_float() || a.bytes().len() != a.dtype().width() {
            return Err(invalid("invalid scalar confidence"));
        }
        a.values()?;
    }
    for (name, dtype, shape) in [
        ("token_to_atoms", DType::Int32, vec![tokens, 2]),
        ("residue_index", DType::Int64, vec![tokens]),
        ("entity_id", DType::Int64, vec![tokens]),
    ] {
        let a = get(name)?;
        if a.dtype() != dtype || a.shape() != shape {
            return Err(invalid("invalid token metadata"));
        }
    }
    coords.values()?;
    Ok(())
}

pub(crate) fn validate_structure(
    sequence: &str,
    arrays: &BTreeMap<String, Array>,
    structure: &Structure,
) -> Result<()> {
    validate_arrays(arrays)?;
    let coords = arrays["coordinates"].coordinates()?;
    let spans: Vec<i32> = arrays["token_to_atoms"]
        .bytes()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
        .collect();
    if sequence.len() != arrays["plddt"].shape()[0] || coords.len() != structure.atoms.len() {
        return Err(invalid("input/structure length mismatch"));
    }
    let mut cursor = 0;
    for (residue, aa) in sequence.bytes().enumerate() {
        // Modified/unknown residues need a separate CCD-aware contract.
        if aa == b'X' {
            return Err(invalid(
                "only the 20 standard monomer residues are integrated",
            ));
        }
        let (resname, atom_names) = crate::atlasfold::residue_atoms(aa)?;
        let expected: BTreeSet<_> = atom_names.split_ascii_whitespace().collect();
        let end = cursor + expected.len();
        if end > coords.len()
            || i64::from(spans[2 * residue]) != cursor as i64
            || i64::from(spans[2 * residue + 1]) != end as i64
        {
            return Err(invalid("token atom partition differs from sequence"));
        }
        let mut seen = BTreeSet::new();
        for (i, xyz) in coords.iter().enumerate().take(end).skip(cursor) {
            let a = &structure.atoms[i];
            if a.serial != i + 1
                || a.chain != "A"
                || a.residue != (residue + 1) as isize
                || a.insertion.is_some()
                || a.residue_name != resname
                || !expected.contains(a.atom_name.as_str())
                || !seen.insert(a.atom_name.as_str())
                || a.element != a.atom_name[..1]
            {
                return Err(invalid("CIF atom identity mismatch"));
            }
            for (axis, value) in xyz.iter().enumerate() {
                // Biotite first stores FP32 as its shortest round-trip decimal.
                // Biohub then parses that text as f64 and applies Python .3f.
                // Direct FP32->f64 promotion differs at decimal half boundaries.
                let intermediate: f64 = (*value as f32)
                    .to_string()
                    .parse()
                    .map_err(|_| invalid("invalid intermediate coordinate"))?;
                let serialized: f32 = format!("{intermediate:.3}")
                    .parse()
                    .map_err(|_| invalid("invalid coordinate"))?;
                if f64::from(serialized) != structure.coordinates[i][axis] {
                    return Err(invalid("CIF differs from raw coordinate serialization"));
                }
            }
        }
        if seen != expected {
            return Err(invalid("missing CIF atoms"));
        }
        cursor = end;
    }
    if cursor != coords.len() {
        return Err(invalid("extra CIF atoms"));
    }
    Ok(())
}
