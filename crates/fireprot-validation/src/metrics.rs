use crate::{Error, Result, array::validate_tolerance};
use nalgebra::{Matrix3, Vector3, linalg::SVD};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
#[derive(Debug, Serialize)]
pub struct CoordinateMetrics {
    pub atom_count: usize,
    pub raw_rmsd_angstrom: f64,
    pub aligned_rmsd_angstrom: f64,
    pub max_abs_coordinate_delta_angstrom: f64,
}
/// Kabsch alignment permits rotations/translations, never reflections.
pub fn coordinates(reference: &[[f64; 3]], candidate: &[[f64; 3]]) -> Result<CoordinateMetrics> {
    if reference.is_empty() || reference.len() != candidate.len() {
        return Err(Error::Shape);
    }
    if reference
        .iter()
        .chain(candidate)
        .flatten()
        .any(|x| !x.is_finite())
    {
        return Err(Error::Nonfinite);
    }
    let n = reference.len() as f64;
    let mean = |x: &[[f64; 3]]| {
        x.iter()
            .fold(Vector3::zeros(), |a, b| a + Vector3::from(*b) / n)
    };
    let mr = mean(reference);
    let mc = mean(candidate);
    let mut cov = Matrix3::zeros();
    let mut raw = 0.0;
    let mut max: f64 = 0.0;
    for (r, c) in reference.iter().zip(candidate) {
        let r = Vector3::from(*r);
        let c = Vector3::from(*c);
        let d = c - r;
        raw += d.norm_squared() / n;
        max = max.max(d.abs().max());
        cov += (c - mc) * (r - mr).transpose() / n;
    }
    if !cov.iter().all(|x| x.is_finite()) || !raw.is_finite() {
        return Err(Error::Nonfinite);
    }
    let rotation = if cov.norm() == 0.0 {
        Matrix3::identity()
    } else {
        let svd = SVD::try_new(cov, true, true, 1e-14, 1000).ok_or(Error::Decomposition)?;
        let mut u = svd.u.ok_or(Error::Decomposition)?;
        let vt = svd.v_t.ok_or(Error::Decomposition)?;
        if (u * vt).determinant() < 0.0 {
            u.set_column(2, &(-u.column(2)));
        }
        u * vt
    };
    let aligned = reference
        .iter()
        .zip(candidate)
        .map(|(r, c)| {
            let d = rotation.transpose() * (Vector3::from(*c) - mc) - (Vector3::from(*r) - mr);
            d.norm_squared() / n
        })
        .sum::<f64>()
        .sqrt();
    if !aligned.is_finite() {
        return Err(Error::Nonfinite);
    }
    Ok(CoordinateMetrics {
        atom_count: reference.len(),
        raw_rmsd_angstrom: raw.sqrt(),
        aligned_rmsd_angstrom: aligned,
        max_abs_coordinate_delta_angstrom: max,
    })
}
fn flatten(v: &Value, path: String, out: &mut BTreeMap<String, f64>) -> Result<()> {
    match v {
        Value::Object(m) => {
            for (k, v) in m {
                flatten(
                    v,
                    format!("{path}/object/{}", k.replace('~', "~0").replace('/', "~1")),
                    out,
                )?;
            }
        }
        Value::Array(a) => {
            for (i, v) in a.iter().enumerate() {
                flatten(v, format!("{path}/array/{i}"), out)?;
            }
        }
        Value::Number(n) => {
            let x = n.as_f64().ok_or(Error::Nonfinite)?;
            if !x.is_finite() {
                return Err(Error::Nonfinite);
            }
            out.insert(path, x);
        }
        _ => {}
    }
    Ok(())
}
#[derive(Debug, Serialize)]
pub struct ConfidenceMetrics {
    pub numeric_leaf_count: usize,
    pub max_abs_delta: f64,
    pub mean_abs_delta: f64,
    pub missing_from_candidate: Vec<String>,
    pub extra_in_candidate: Vec<String>,
}
pub fn confidence(reference: &Value, candidate: &Value) -> Result<ConfidenceMetrics> {
    let mut r = BTreeMap::new();
    let mut c = BTreeMap::new();
    flatten(reference, String::new(), &mut r)?;
    flatten(candidate, String::new(), &mut c)?;
    if r.is_empty() || c.is_empty() {
        return Err(Error::Schema(
            "confidence must contain numeric leaves".into(),
        ));
    }
    let deltas: Vec<_> = r
        .iter()
        .filter_map(|(k, v)| c.get(k).map(|x| (x - v).abs()))
        .collect();
    if deltas.iter().any(|x| !x.is_finite()) {
        return Err(Error::Nonfinite);
    }
    let mean = if deltas.is_empty() {
        0.0
    } else {
        deltas.iter().map(|v| v / deltas.len() as f64).sum()
    };
    Ok(ConfidenceMetrics {
        numeric_leaf_count: deltas.len(),
        max_abs_delta: deltas.into_iter().fold(0.0, f64::max),
        mean_abs_delta: mean,
        missing_from_candidate: r.keys().filter(|k| !c.contains_key(*k)).cloned().collect(),
        extra_in_candidate: c.keys().filter(|k| !r.contains_key(*k)).cloned().collect(),
    })
}
pub fn protenix_passes(
    c: &CoordinateMetrics,
    q: &ConfidenceMetrics,
    coordinate_tolerance: f64,
    confidence_tolerance: f64,
) -> Result<bool> {
    validate_tolerance(coordinate_tolerance)?;
    validate_tolerance(confidence_tolerance)?;
    Ok(c.atom_count > 0
        && c.aligned_rmsd_angstrom.is_finite()
        && c.aligned_rmsd_angstrom <= coordinate_tolerance
        && q.numeric_leaf_count > 0
        && q.max_abs_delta.is_finite()
        && q.max_abs_delta <= confidence_tolerance
        && q.missing_from_candidate.is_empty()
        && q.extra_in_candidate.is_empty())
}
