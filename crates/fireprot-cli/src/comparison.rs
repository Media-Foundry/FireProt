//! Compose authenticated archive collection, scientific gates and paired timing.
use crate::{
    collection::{self, Collected, Collection},
    support::Result,
};
use fireprot_runtime::run_archive;
use fireprot_validation::{bundle, numpy::read_npz, paired, structure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Specification {
    schema_version: u32,
    treatment_description: String,
    expected_changes: BTreeMap<String, paired::Difference>,
    baseline: Collection,
    candidate: Collection,
}

impl Specification {
    pub(crate) fn new(
        treatment_description: String,
        expected_changes: BTreeMap<String, paired::Difference>,
        baseline: Collection,
        candidate: Collection,
    ) -> Self {
        Self {
            schema_version: 1,
            treatment_description,
            expected_changes,
            baseline,
            candidate,
        }
    }
}

#[derive(Serialize)]
pub struct Pair {
    baseline_run_id: String,
    candidate_run_id: String,
    samples: Vec<paired::SampleComparison>,
    cif_atom_identities_checked: bool,
}

#[derive(Serialize)]
pub struct Report {
    schema_version: u32,
    treatment_description: String,
    baseline: Collected,
    candidate: Collected,
    pairs: Vec<Pair>,
    pub correctness_passed: bool,
    /// A failed numerical comparison never publishes even descriptive speedups.
    performance: Option<paired::Comparison>,
}

pub fn run(specification: Specification) -> Result<Report> {
    if specification.schema_version != 1 || specification.treatment_description.trim().is_empty() {
        return Err("comparison requires schema 1 and a treatment description".into());
    }
    let baseline = collection::collect(specification.baseline)?;
    let candidate = collection::collect(specification.candidate)?;
    let performance = paired::compare(
        &baseline.series,
        &candidate.series,
        &specification.expected_changes,
    )?;
    let mut pairs = Vec::new();
    let mut correctness_passed = true;
    for (index, (a, b)) in baseline
        .receipts
        .iter()
        .zip(&candidate.receipts)
        .enumerate()
    {
        let ar = run_archive::verify(a)?;
        let br = run_archive::verify(b)?;
        let mut aq = baseline.series.observations[index].request.clone();
        let mut bq = candidate.series.observations[index].request.clone();
        aq.output_dir = ar.root.clone();
        bq.output_dir = br.root.clone();
        let ab = bundle::validate(&aq.clone().try_into()?, &ar.outputs)?;
        let bb = bundle::validate(&bq.clone().try_into()?, &br.outputs)?;
        if ab.schema_version != bb.schema_version {
            return Err("paired bundle schema versions differ".into());
        }
        let mut samples = Vec::new();
        for ap in &ab.predictions {
            let bp = bb
                .predictions
                .iter()
                .find(|p| p.sample == ap.sample)
                .ok_or("missing paired sample")?;
            match (&ap.structure, &bp.structure) {
                (Some(a), Some(b)) => {
                    if structure::read(&ar.root.join(a))?.atoms
                        != structure::read(&br.root.join(b))?.atoms
                    {
                        return Err("paired CIF atom identities differ".into());
                    }
                }
                (None, None) => {}
                _ => return Err("paired structure coverage differs".into()),
            }
            let arrays_a = read_npz(&ar.root.join(&ap.archive), 512 * 1024 * 1024)?;
            let arrays_b = read_npz(&br.root.join(&bp.archive), 512 * 1024 * 1024)?;
            let comparison =
                paired::compare_prediction(&aq.workload, ap, &arrays_a, bp, &arrays_b)?;
            correctness_passed &= comparison.passed;
            samples.push(comparison);
        }
        pairs.push(Pair {
            baseline_run_id: aq.run_id,
            candidate_run_id: bq.run_id,
            samples,
            cif_atom_identities_checked: ab.schema_version >= 2,
        });
    }
    Ok(Report {
        schema_version: 1,
        treatment_description: specification.treatment_description,
        baseline,
        candidate,
        pairs,
        correctness_passed,
        performance: correctness_passed.then_some(performance),
    })
}
