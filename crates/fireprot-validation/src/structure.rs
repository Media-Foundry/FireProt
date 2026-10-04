//! Single-model prediction CIF validation and legacy-compatible coordinate reads.
use crate::{Error, Result, metrics};
use pdbtbx::{
    ContainsAtomConformer, ContainsAtomConformerResidue, ContainsAtomConformerResidueChain,
};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    fs,
    io::{BufReader, Cursor},
    path::Path,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AtomIdentity {
    pub serial: usize,
    pub chain: String,
    pub residue: isize,
    pub insertion: Option<String>,
    pub residue_name: String,
    pub atom_name: String,
    pub element: String,
}
#[derive(Debug)]
pub struct Structure {
    pub atoms: Vec<AtomIdentity>,
    /// Match the previous Biotite AtomArray FP32 coordinates, promoted to f64.
    pub coordinates: Vec<[f64; 3]>,
}

pub fn read(path: &Path) -> Result<Structure> {
    if fs::metadata(path)?.len() > 64 * 1024 * 1024 {
        return Err(Error::Schema("CIF exceeds 64 MiB limit".into()));
    }
    parse(&fs::read(path)?)
}

pub fn parse(bytes: &[u8]) -> Result<Structure> {
    if bytes.len() > 64 * 1024 * 1024 {
        return Err(Error::Schema("CIF exceeds 64 MiB limit".into()));
    }
    // pdbtbx 0.12 can panic on malformed mandatory atom fields. Keep all
    // parser-owned state inside this unwind boundary; never accept partial data.
    // Do not change the process-wide panic hook (other jobs may be running).
    let (pdb, warnings) = std::panic::catch_unwind(|| {
        pdbtbx::ReadOptions::new()
            .set_format(pdbtbx::Format::Mmcif)
            .set_level(pdbtbx::StrictnessLevel::Strict)
            .read_raw(BufReader::new(Cursor::new(bytes)))
    })
    .map_err(|_| Error::Schema("prediction CIF parser rejected malformed input".into()))?
    .map_err(|e| Error::Schema(format!("invalid prediction CIF: {e:?}")))?;
    if !warnings.is_empty() || pdb.model_count() != 1 || pdb.atom_count() == 0 {
        return Err(Error::Schema(
            "prediction CIF must contain one nonempty model without parser warnings".into(),
        ));
    }
    let mut rows = Vec::new();
    for h in pdb.atoms_with_hierarchy() {
        if h.conformer().alternative_location().is_some() {
            return Err(Error::Schema(
                "alternate conformers are outside the prediction CIF contract".into(),
            ));
        }
        let a = h.atom();
        let xyz = [a.x(), a.y(), a.z()].map(|x| f64::from(x as f32));
        if xyz.iter().any(|x| !x.is_finite()) {
            return Err(Error::Nonfinite);
        }
        rows.push((
            AtomIdentity {
                serial: a.serial_number(),
                chain: h.chain().id().into(),
                residue: h.residue().serial_number(),
                insertion: h.residue().insertion_code().map(Into::into),
                residue_name: h.conformer().name().into(),
                atom_name: a.name().into(),
                element: a
                    .element()
                    .ok_or_else(|| Error::Schema("missing CIF atom element".into()))?
                    .symbol()
                    .into(),
            },
            xyz,
        ));
    }
    rows.sort_by_key(|(a, _)| a.serial);
    let mut seen = BTreeSet::new();
    if rows.iter().any(|(a, _)| !seen.insert(a.serial)) {
        return Err(Error::Schema("duplicate CIF atom IDs".into()));
    }
    let (atoms, coordinates) = rows.into_iter().unzip();
    Ok(Structure { atoms, coordinates })
}

#[derive(Debug, Serialize)]
pub struct ProtenixComparison {
    pub coordinates: metrics::CoordinateMetrics,
    pub confidence: metrics::ConfidenceMetrics,
    pub coordinate_tolerance_angstrom: f64,
    pub confidence_tolerance: f64,
    pub passed: bool,
}

pub fn compare_protenix(
    reference: &Path,
    reference_confidence: &Path,
    candidate: &Path,
    candidate_confidence: &Path,
) -> Result<ProtenixComparison> {
    let r = read(reference)?;
    let c = read(candidate)?;
    if r.atoms != c.atoms {
        return Err(Error::Schema("CIF atom identities do not match".into()));
    }
    let coordinates = metrics::coordinates(&r.coordinates, &c.coordinates)?;
    let read_json = |path| {
        serde_json::from_slice(&fs::read(path)?)
            .map_err(|e| Error::Schema(format!("confidence JSON: {e}")))
    };
    let confidence = metrics::confidence(
        &read_json(reference_confidence)?,
        &read_json(candidate_confidence)?,
    )?;
    let passed = metrics::protenix_passes(&coordinates, &confidence, 1e-3, 1e-3)?;
    Ok(ProtenixComparison {
        coordinates,
        confidence,
        coordinate_tolerance_angstrom: 1e-3,
        confidence_tolerance: 1e-3,
        passed,
    })
}
