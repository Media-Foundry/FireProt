//! Diagnostic Base regression report. Does not modify historical comparison policy.
use fireprot_validation::{metrics, numpy, structure};
use serde_json::json;
use std::{env, fs, path::Path};

fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn relative(reference: &[[f64; 3]], candidate: &[[f64; 3]]) -> serde_json::Value {
    let mut deltas = Vec::new();
    let mut local_scores = vec![0.0; reference.len()];
    let mut local_counts = vec![0usize; reference.len()];
    for i in 0..reference.len() {
        for j in (i + 1)..reference.len() {
            let r = distance(&reference[i], &reference[j]);
            let delta = (r - distance(&candidate[i], &candidate[j])).abs();
            deltas.push(delta);
            if r < 15.0 {
                let score =
                    [0.5, 1.0, 2.0, 4.0].iter().filter(|&&t| delta < t).count() as f64 / 4.0;
                local_scores[i] += score;
                local_scores[j] += score;
                local_counts[i] += 1;
                local_counts[j] += 1;
            }
        }
    }
    let rms = (deltas.iter().map(|d| d * d).sum::<f64>() / deltas.len() as f64).sqrt();
    deltas.sort_by(f64::total_cmp);
    let per_residue: Vec<Option<f64>> = local_scores
        .iter()
        .zip(&local_counts)
        .map(|(s, n)| if *n > 0 { Some(s / *n as f64) } else { None })
        .collect();
    let valid: Vec<f64> = per_residue.iter().flatten().copied().collect();
    json!({"pair_count":deltas.len(), "pair_distance_rms_delta_angstrom":rms,
        "pair_distance_p95_abs_delta_angstrom":deltas[(0.95*deltas.len() as f64).ceil() as usize-1],
        "pair_distance_max_abs_delta_angstrom":deltas.last(),
        "ca_lddt_to_reference":if valid.is_empty(){None}else{Some(valid.iter().sum::<f64>()/valid.len() as f64)},
        "ca_lddt_per_residue":per_residue, "neighbor_counts":local_counts,
        "definition":"CA-only per-residue mean; reference distance <15A; strict error thresholds 0.5,1,2,4A; no alignment or sequence realignment"})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err("Usage: base_drift REF.npz CAND.npz REF.cif CAND.cif OUTPUT.json".into());
    }
    let a = numpy::read_npz(Path::new(&args[1]), 512 * 1024 * 1024)?;
    let b = numpy::read_npz(Path::new(&args[2]), 512 * 1024 * 1024)?;
    let ac = a
        .get("coordinates")
        .ok_or("missing reference coordinates")?
        .coordinates()?;
    let bc = b
        .get("coordinates")
        .ok_or("missing candidate coordinates")?
        .coordinates()?;
    let sa = structure::read(Path::new(&args[3]))?;
    let sb = structure::read(Path::new(&args[4]))?;
    if sa.atoms != sb.atoms || ac.len() != sa.atoms.len() || bc.len() != sb.atoms.len() {
        return Err("Atom identity/count mismatch".into());
    }
    for (coordinates, parsed) in [(&ac, &sa.coordinates), (&bc, &sb.coordinates)] {
        if coordinates
            .iter()
            .flatten()
            .zip(parsed.iter().flatten())
            .any(|(a, b)| !a.is_finite() || (a - b).abs() > 0.002)
        {
            return Err("NPZ/CIF coordinate order or value mismatch".into());
        }
    }
    let ids: Vec<usize> = sa
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (a.atom_name == "CA" && a.element == "C").then_some(i))
        .collect();
    if ids.len() < 2 {
        return Err("At least two corresponding CA atoms required".into());
    }
    let ca: Vec<_> = ids.iter().map(|&i| ac[i]).collect();
    let cb: Vec<_> = ids.iter().map(|&i| bc[i]).collect();
    let full = metrics::coordinates(&ac, &bc)?;
    let ca_metrics = metrics::coordinates(&ca, &cb)?;
    let report = json!({"policy":"user-requested-Base-aligned-RMSD-0.1A-20261007",
        "scope":"Numerical regression to reference prediction; not experimental structural accuracy or a confidence gate",
        "coordinate_tolerance_angstrom":0.1,
        "coordinate_gate_passed":full.aligned_rmsd_angstrom<=0.1,
        "atom_identities_exact":true, "npz_cif_binding_checked":true,
        "all_atom":full, "ca":ca_metrics, "relative_ca_geometry":relative(&ca,&cb),
        "ca_identities":ids.iter().map(|&i| &sa.atoms[i]).collect::<Vec<_>>()});
    fs::write(&args[5], serde_json::to_string_pretty(&report)? + "\n")?;
    println!(
        "{}",
        json!({"coordinate_gate_passed":report["coordinate_gate_passed"],"all_atom":report["all_atom"],"ca":report["ca"]})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rigid_motion_does_not_count_as_relative_drift() {
        let a = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
        let b: Vec<_> = a
            .iter()
            .map(|p| [-p[1] + 20., p[0] - 30., p[2] + 10.])
            .collect();
        let m = metrics::coordinates(&a, &b).unwrap();
        assert!(m.raw_rmsd_angstrom > 10. && m.aligned_rmsd_angstrom < 1e-10);
        let r = relative(&a, &b);
        assert_eq!(r["pair_distance_max_abs_delta_angstrom"], 0.0);
        assert_eq!(r["ca_lddt_to_reference"], 1.0);
    }
    #[test]
    fn internal_displacement_is_detected_after_alignment() {
        let a = [[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
        let mut b = a;
        b[3][2] += 2.;
        assert!(metrics::coordinates(&a, &b).unwrap().aligned_rmsd_angstrom > 0.1);
        let r = relative(&a, &b);
        assert!(r["pair_distance_rms_delta_angstrom"].as_f64().unwrap() > 0.1);
        assert!(r["ca_lddt_to_reference"].as_f64().unwrap() < 1.0);
    }
}
