//! Compose pinned paired scheduling with existing measurement and comparison gates.
use crate::{
    collection::{self, Collection},
    comparison, inference,
    support::{Result, cancellation, read_json},
};
use fireprot_core::request::ArtifactRef;
use fireprot_runtime::{
    artifacts::sha256,
    batch::{self, BatchPhase},
    experiment::{self, Arm, Slot},
    measured_run, run_archive,
    supervisor::RunReport,
};
use fireprot_validation::{paired::Difference, timing::TimingScope};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::BTreeMap, fs::OpenOptions, io::Write, path::Path, sync::atomic::Ordering,
    time::Instant,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Specification {
    schema_version: u32,
    execution: experiment::Specification,
    comparison: Settings,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    scope: TimingScope,
    cache_state: String,
    treatment_description: String,
    expected_changes: BTreeMap<String, Difference>,
}

#[derive(Serialize)]
struct MeasuredJob {
    index: usize,
    slot: Slot,
    invocation: ArtifactRef,
    accepted: ArtifactRef,
}

#[derive(Serialize)]
pub struct Outcome {
    pub phase: BatchPhase,
    report: ArtifactRef,
    comparison: Option<ArtifactRef>,
}

fn compare(
    settings: &Settings,
    a: &[&MeasuredJob],
    b: &[&MeasuredJob],
    baseline_repeat: bool,
) -> Result<comparison::Report> {
    let collection = |jobs: &[&MeasuredJob]| {
        Collection::from_runs(
            settings.scope,
            settings.cache_state.clone(),
            jobs.iter().map(|j| j.accepted.clone()).collect(),
            if settings.scope == TimingScope::EndToEnd {
                jobs.iter().map(|j| j.invocation.clone()).collect()
            } else {
                Vec::new()
            },
        )
    };
    comparison::run(comparison::Specification::new(
        if baseline_repeat {
            format!(
                "Baseline repeat consistency: {}",
                settings.treatment_description
            )
        } else {
            settings.treatment_description.clone()
        },
        if baseline_repeat {
            BTreeMap::new()
        } else {
            settings.expected_changes.clone()
        },
        collection(a),
        collection(b),
    ))
}

pub fn run(specification: Specification, destination: &Path) -> Result<Outcome> {
    if specification.schema_version != 1
        || specification.comparison.cache_state.trim().is_empty()
        || specification
            .comparison
            .treatment_description
            .trim()
            .is_empty()
    {
        return Err(
            "benchmark requires schema 1, explicit cache state and treatment description".into(),
        );
    }
    let cancelled = cancellation()?;
    let settings = specification.comparison;
    let prepared =
        experiment::Plan::load(specification.execution)?.prepare(destination, |bytes| {
            inference::validate_profile(bytes).map_err(|e| std::io::Error::other(e.to_string()))
        })?;
    let settings_pin = prepared.write_record("comparison-settings.json", &settings)?;
    let journal_path = prepared.root().join("order.jsonl");
    let mut journal = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&journal_path)?;
    let started = Instant::now();
    let mut jobs: Vec<MeasuredJob> = Vec::new();
    let mut checks: Vec<ArtifactRef> = Vec::new();
    let mut final_comparison = None;
    let executable = std::env::current_exe()?;
    let batch = batch::execute(prepared.batch(), &cancelled, |index, job| {
        let slot = prepared.slots()[index];
        let event = |journal: &mut std::fs::File,
                     kind: &str,
                     error: Option<&str>|
         -> std::io::Result<()> {
            serde_json::to_writer(
                &mut *journal,
                &json!({"index":index,"pair":slot.pair,"arm":slot.arm,
                "run_id":job.request().as_request().run_id,"event":kind,"offset_seconds":started.elapsed().as_secs_f64(),"error":error}),
            )?;
            journal.write_all(b"\n")?;
            journal.sync_all()
        };
        let mut execute = || -> Result<RunReport> {
            event(&mut journal, "started", None)?;
            let invocation = measured_run::run(
                prepared.invocations()[index].clone(),
                &prepared.root().join(format!("measurement-{index:03}")),
                &executable,
                &cancelled,
                |pin| {
                    collection::collect(Collection::single(pin.clone()))
                        .map(|_| ())
                        .map_err(|e| std::io::Error::other(e.to_string()))
                },
            )?;
            let record: measured_run::Record = read_json(&invocation.path)?;
            let archive = run_archive::verify(&record.accepted)?;
            let report: RunReport = read_json(&archive.root.join("supervisor.json"))?;
            jobs.push(MeasuredJob {
                index,
                slot,
                invocation,
                accepted: record.accepted,
            });
            event(&mut journal, "measured", None)?;
            // The first baseline is the declared reference. Every subsequent
            // job passes comparison against it before another job can start.
            if index > 0 {
                let check = compare(
                    &settings,
                    &[&jobs[0]],
                    &[jobs.last().ok_or("missing measured job")?],
                    slot.arm == Arm::Baseline,
                )?;
                checks.push(
                    prepared.write_record(&format!("reference-check-{index:03}.json"), &check)?,
                );
                if !check.correctness_passed {
                    return Err("reference correctness check failed".into());
                }
            }
            if index % 2 == 1 {
                let baseline = jobs
                    .iter()
                    .find(|j| j.slot.pair == slot.pair && j.slot.arm == Arm::Baseline)
                    .ok_or("missing pair baseline")?;
                let candidate = jobs
                    .iter()
                    .find(|j| j.slot.pair == slot.pair && j.slot.arm == Arm::Candidate)
                    .ok_or("missing pair candidate")?;
                let check = compare(&settings, &[baseline], &[candidate], false)?;
                checks.push(
                    prepared.write_record(&format!("pair-check-{:03}.json", slot.pair), &check)?,
                );
                if !check.correctness_passed {
                    return Err("paired correctness check failed".into());
                }
            }
            if index + 1 == prepared.batch().jobs().len() {
                let mut baseline: Vec<_> = jobs
                    .iter()
                    .filter(|j| j.slot.arm == Arm::Baseline)
                    .collect();
                let mut candidate: Vec<_> = jobs
                    .iter()
                    .filter(|j| j.slot.arm == Arm::Candidate)
                    .collect();
                baseline.sort_by_key(|j| j.slot.pair);
                candidate.sort_by_key(|j| j.slot.pair);
                let comparison = compare(&settings, &baseline, &candidate, false)?;
                if !comparison.correctness_passed {
                    return Err("aggregate correctness check failed".into());
                }
                if cancelled.load(Ordering::Relaxed) {
                    return Err("benchmark cancelled during final comparison".into());
                }
                final_comparison = Some(prepared.write_record("comparison.json", &comparison)?);
            }
            if cancelled.load(Ordering::Relaxed) {
                return Err("benchmark cancelled during verification".into());
            }
            event(&mut journal, "verified", None)?;
            Ok(report)
        };
        match execute() {
            Ok(report) => Ok(report),
            Err(error) => {
                let message = error.to_string();
                if let Err(journal_error) = event(&mut journal, "failed", Some(&message)) {
                    return Err(format!("{message}; journal write failed: {journal_error}"));
                }
                Err(message)
            }
        }
    });
    journal.sync_all()?;
    drop(journal);
    let order = ArtifactRef {
        sha256: sha256(&journal_path)?,
        path: journal_path,
    };
    if batch.phase != BatchPhase::Succeeded {
        final_comparison = None;
    }
    let record=prepared.write_record("benchmark.json",&json!({"schema_version":1,"phase":batch.phase,
        "execution_plan":prepared.specification_pin(),"comparison_settings":settings_pin,
        "order":order,"batch":batch,"measurements":jobs,"checks":checks,"comparison":final_comparison,
        "reference_policy":"first_baseline_anchor","model_residency":"fresh_process_per_job"}))?;
    Ok(Outcome {
        phase: batch.phase,
        report: record,
        comparison: final_comparison,
    })
}
