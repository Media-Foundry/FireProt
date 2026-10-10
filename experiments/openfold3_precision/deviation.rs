//! Diagnostic structure/confidence comparison; deliberately no acceptance threshold.
use fireprot_validation::{metrics, numpy::read_npz, structure};
use nalgebra::{Matrix3, Vector3, linalg::SVD};
use serde_json::{Value, json};
use std::{collections::BTreeMap, error::Error, path::Path};

fn aligned_distances(reference: &[[f64; 3]], candidate: &[[f64; 3]]) -> Vec<f64> {
    assert!(!reference.is_empty() && reference.len() == candidate.len());
    let n = reference.len() as f64;
    let mean = |points: &[[f64; 3]]| {
        points.iter().fold(Vector3::zeros(), |sum, point| {
            sum + Vector3::from(*point) / n
        })
    };
    let mr = mean(reference);
    let mc = mean(candidate);
    let mut covariance = Matrix3::zeros();
    for (r, c) in reference.iter().zip(candidate) {
        covariance += (Vector3::from(*c) - mc) * (Vector3::from(*r) - mr).transpose() / n;
    }
    let rotation = if covariance.norm() == 0.0 {
        Matrix3::identity()
    } else {
        let svd = SVD::try_new(covariance, true, true, 1e-14, 1000).expect("SVD failed");
        let mut u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        if (u * vt).determinant() < 0.0 {
            u.set_column(2, &(-u.column(2)));
        }
        u * vt
    };
    reference
        .iter()
        .zip(candidate)
        .map(|(r, c)| {
            (rotation.transpose() * (Vector3::from(*c) - mc) - (Vector3::from(*r) - mr)).norm()
        })
        .collect()
}

fn distribution(values: &[f64]) -> Value {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        let p = q * (sorted.len() - 1) as f64;
        let a = p.floor() as usize;
        sorted[a] + (sorted[p.ceil() as usize] - sorted[a]) * p.fract()
    };
    json!({"count": values.len(), "mean": values.iter().sum::<f64>() / values.len() as f64,
        "rms": (values.iter().map(|x| x*x).sum::<f64>() / values.len() as f64).sqrt(),
        "p50": quantile(0.5), "p90": quantile(0.9), "p95": quantile(0.95),
        "p99": quantile(0.99), "max": sorted.last().unwrap()})
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "Usage: deviation REFERENCE.npz REFERENCE.cif CANDIDATE.npz CANDIDATE.cif".into(),
        );
    }
    let reference = read_npz(Path::new(&args[1]), 512 * 1024 * 1024)?;
    let candidate = read_npz(Path::new(&args[3]), 512 * 1024 * 1024)?;
    let rs = structure::read(Path::new(&args[2]))?;
    let cs = structure::read(Path::new(&args[4]))?;
    if rs.atoms != cs.atoms || reference.keys().ne(candidate.keys()) {
        return Err("Atom identities or array sets differ".into());
    }
    let r = reference
        .get("coordinates")
        .ok_or("Missing coordinates")?
        .coordinates()?;
    let c = candidate
        .get("coordinates")
        .ok_or("Missing coordinates")?
        .coordinates()?;
    if r.len() != rs.atoms.len() || c.len() != cs.atoms.len() {
        return Err("CIF/NPZ atom count mismatch".into());
    }
    for (raw, written) in [(&r, &rs.coordinates), (&c, &cs.coordinates)] {
        // CIF writers may round; this binds atom order without replacing raw NPZ coordinates.
        if raw
            .iter()
            .flatten()
            .zip(written.iter().flatten())
            .any(|(a, b)| (a - b).abs() > 0.001)
        {
            return Err("CIF/NPZ coordinates differ beyond writer rounding".into());
        }
    }
    let all_atom = metrics::coordinates(&r, &c)?;
    let distances = aligned_distances(&r, &c);
    let measured_rmsd =
        (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    if (measured_rmsd - all_atom.aligned_rmsd_angstrom).abs() > 1e-9 {
        return Err("Alignment disagrees with maintained Rust metrics".into());
    }
    let ca: Vec<usize> = rs
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| a.atom_name == "CA")
        .map(|(i, _)| i)
        .collect();
    if ca.len() < 3 {
        return Err("Expected protein C-alpha atoms".into());
    }
    let rca: Vec<_> = ca.iter().map(|&i| r[i]).collect();
    let cca: Vec<_> = ca.iter().map(|&i| c[i]).collect();
    let ca_distances: Vec<_> = ca.iter().map(|&i| distances[i]).collect();
    let mut residue_distances: BTreeMap<(String, isize, Option<String>, String), Vec<f64>> =
        BTreeMap::new();
    for (a, &d) in rs.atoms.iter().zip(&distances) {
        residue_distances
            .entry((
                a.chain.clone(),
                a.residue,
                a.insertion.clone(),
                a.residue_name.clone(),
            ))
            .or_default()
            .push(d);
    }
    let mut residues: Vec<_> = residue_distances
        .into_iter()
        .map(|(key, d)| {
            let rmsd = (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt();
            (
                rmsd,
                json!({"chain":key.0,"residue":key.1,"insertion":key.2,"name":key.3,
                     "atom_count":d.len(),"aligned_rmsd_angstrom":rmsd}),
            )
        })
        .collect();
    residues.sort_by(|a, b| b.0.total_cmp(&a.0));
    let residue_rmsds: Vec<_> = residues.iter().map(|x| x.0).collect();
    let mut pair_distance_changes = Vec::new();
    for i in 0..ca.len() {
        for j in (i + 1)..ca.len() {
            let rd = (Vector3::from(rca[i]) - Vector3::from(rca[j])).norm();
            let cd = (Vector3::from(cca[i]) - Vector3::from(cca[j])).norm();
            pair_distance_changes.push((cd - rd).abs());
        }
    }
    let mut confidence = BTreeMap::new();
    for (name, a) in &reference {
        if name == "coordinates" {
            continue;
        }
        let b = &candidate[name];
        if a.shape() != b.shape() || a.dtype() != b.dtype() {
            return Err(format!("Array contract changed: {name}").into());
        }
        if a.dtype().is_float() {
            let av = a.values()?;
            let bv = b.values()?;
            let differences: Vec<_> = av.iter().zip(&bv).map(|(x, y)| (y - x).abs()).collect();
            confidence.insert(
                name.clone(),
                json!({"absolute_delta":distribution(&differences),
                "reference_mean":av.iter().sum::<f64>() / av.len() as f64,
                "candidate_mean":bv.iter().sum::<f64>() / bv.len() as f64,
                "bitwise_equal":a.bytes()==b.bytes()}),
            );
        } else {
            confidence.insert(name.clone(), json!({"bitwise_equal":a.bytes()==b.bytes()}));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"schema_version":1,
        "quality_verdict":null, "scope":"Prediction regression; no experimental structure truth or selected tolerance.",
        "all_atom":all_atom, "aligned_atom_displacement_angstrom":distribution(&distances),
        "ca_own_alignment":metrics::coordinates(&rca,&cca)?,
        "ca_after_all_atom_alignment_angstrom":distribution(&ca_distances),
        "residue_rmsd_after_all_atom_alignment_angstrom":distribution(&residue_rmsds),
        "largest_residue_deviations":residues.iter().take(20).map(|x| &x.1).collect::<Vec<_>>(),
        "all_ca_pair_distance_absolute_change_angstrom":distribution(&pair_distance_changes),
        "confidence":confidence}))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_translation_and_proper_rotation() {
        let r = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
        let c: Vec<_> = r
            .iter()
            .map(|p| [-p[1] + 20., p[0] - 7., p[2] + 5.])
            .collect();
        assert!(aligned_distances(&r, &c).iter().all(|x| *x < 1e-12));
    }

    #[test]
    fn preserves_local_distortion_and_rejects_reflection_as_rotation() {
        let r = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
        let c: Vec<_> = r.iter().map(|p| [-p[0], p[1], p[2]]).collect();
        assert!(aligned_distances(&r, &c).iter().any(|x| *x > 0.1));
        let mut c = r;
        c[0][0] += 0.2;
        assert!(aligned_distances(&r, &c).iter().any(|x| *x > 0.01));
    }
}
