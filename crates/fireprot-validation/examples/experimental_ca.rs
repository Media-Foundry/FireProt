//! CA-only experimental structure diagnostics; no inference acceptance-policy change.
use fireprot_validation::{array::DType, metrics, numpy, structure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}
fn lddt(reference: &[[f64; 3]], candidate: &[[f64; 3]]) -> Value {
    let mut scores = vec![0.0; reference.len()];
    let mut counts = vec![0usize; reference.len()];
    for i in 0..reference.len() {
        for j in i + 1..reference.len() {
            let d = distance(&reference[i], &reference[j]);
            if d < 15.0 {
                let error = (d - distance(&candidate[i], &candidate[j])).abs();
                let score = [0.5, 1., 2., 4.].iter().filter(|&&t| error < t).count() as f64 / 4.;
                scores[i] += score;
                scores[j] += score;
                counts[i] += 1;
                counts[j] += 1;
            }
        }
    }
    let local: Vec<Option<f64>> = scores
        .iter()
        .zip(&counts)
        .map(|(s, n)| if *n > 0 { Some(s / *n as f64) } else { None })
        .collect();
    let valid: Vec<_> = local.iter().flatten().copied().collect();
    json!({"mean": if valid.is_empty(){None}else{Some(valid.iter().sum::<f64>()/valid.len() as f64)},
        "per_residue":local, "neighbor_counts":counts,
        "definition":"CA-only per-residue mean; experimental distances <15 A; strict error thresholds 0.5,1,2,4 A; no sequence realignment"})
}
fn hash(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 6 {
        return Err(
            "Usage: experimental_ca GT_DIRECTORY PRED.npz PRED.cif OUTPUT.json CHAIN".into(),
        );
    }
    let root = Path::new(&args[1]);
    let metadata: Value = serde_json::from_slice(&fs::read(root.join("metadata.json"))?)?;
    for name in ["input.json", "gt_mapping.npz"] {
        if metadata["artifacts"][name].as_str() != Some(hash(&root.join(name))?.as_str()) {
            return Err(format!("Experimental artifact hash mismatch: {name}").into());
        }
    }
    let truth = numpy::read_npz(&root.join("gt_mapping.npz"), 64 * 1024 * 1024)?;
    let gt = numpy::ground_truth(&truth)?;
    let mask = truth.get("mask").ok_or("missing mask")?;
    if mask.bytes().iter().any(|&x| x > 1) {
        return Err("Invalid boolean mask".into());
    }
    let residues = metadata["residues"]
        .as_array()
        .ok_or("missing residue mapping")?;
    if residues.len() != mask.bytes().len() {
        return Err("Mapping length mismatch".into());
    }
    let ids = truth.get("label_seq_id").ok_or("missing label_seq_id")?;
    // This extractor's declared mapping is little-endian int32, consecutive 1..N.
    if ids.dtype() != DType::Int32
        || ids.shape() != [residues.len()]
        || ids.bytes().len() != 4 * residues.len()
    {
        return Err("Invalid sequence ID length".into());
    }
    for (i, bytes) in ids.bytes().chunks_exact(4).enumerate() {
        if i32::from_le_bytes(bytes.try_into()?) != i as i32 + 1
            || residues[i]["label_seq_id"].as_u64() != Some(i as u64 + 1)
        {
            return Err("Nonconsecutive experimental sequence mapping".into());
        }
    }
    let archive = numpy::read_npz(Path::new(&args[2]), 512 * 1024 * 1024)?;
    let coordinates = archive
        .get("coordinates")
        .ok_or("missing prediction coordinates")?
        .coordinates()?;
    let parsed = structure::read(Path::new(&args[3]))?;
    if coordinates.len() != parsed.atoms.len()
        || coordinates
            .iter()
            .flatten()
            .zip(parsed.coordinates.iter().flatten())
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 0.002)
    {
        return Err("NPZ/CIF binding mismatch".into());
    }
    let mut mapped = vec![None; residues.len()];
    for (atom, xyz) in parsed.atoms.iter().zip(&coordinates) {
        if atom.chain == args[5] && atom.atom_name == "CA" && atom.element == "C" {
            let i = usize::try_from(atom.residue)?
                .checked_sub(1)
                .ok_or("Invalid residue number")?;
            if i >= residues.len()
                || atom.insertion.is_some()
                || residues[i]["comp_id"].as_str() != Some(atom.residue_name.as_str())
                || mapped[i].is_some()
            {
                return Err("CA sequence identity mismatch or duplicate".into());
            }
            mapped[i] = Some(*xyz);
        }
    }
    let prediction: Vec<_> = mapped
        .iter()
        .zip(mask.bytes())
        .filter(|(_, m)| **m == 1)
        .map(|(p, _)| p.ok_or("Missing experimental residue in prediction"))
        .collect::<Result<_, _>>()?;
    let result = json!({"scope":"Experimental CA-only structure diagnostics; development sample, not an independent test set",
        "pdb_id":metadata["pdb_id"], "experimental_chain":metadata["label_asym_id"], "prediction_chain":args[5],
        "gt_mapping_sha256":hash(&root.join("gt_mapping.npz"))?, "metadata_sha256":hash(&root.join("metadata.json"))?,
        "prediction_npz_sha256":hash(Path::new(&args[2]))?, "prediction_cif_sha256":hash(Path::new(&args[3]))?,
        "ca_count":gt.len(), "npz_cif_binding_checked":true,"sequence_identity_checked":true,
        "coordinates":metrics::coordinates(&gt,&prediction)?, "ca_lddt":lddt(&gt,&prediction)});
    fs::write(&args[4], serde_json::to_string_pretty(&result)? + "\n")?;
    println!(
        "{}",
        json!({"ca_count":gt.len(),"coordinates":result["coordinates"],"ca_lddt":result["ca_lddt"]["mean"]})
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rigid_motion_and_internal_deformation() {
        let a = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
        let mut b: Vec<_> = a
            .iter()
            .map(|p| [-p[1] + 20., p[0] - 30., p[2] + 10.])
            .collect();
        assert!(metrics::coordinates(&a, &b).unwrap().aligned_rmsd_angstrom < 1e-10);
        assert_eq!(lddt(&a, &b)["mean"], 1.0);
        b[3][2] += 3.;
        assert!(lddt(&a, &b)["mean"].as_f64().unwrap() < 1.0);
        assert!(metrics::coordinates(&a, &b).unwrap().aligned_rmsd_angstrom > 0.1);
    }
}
