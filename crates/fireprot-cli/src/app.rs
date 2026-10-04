//! Compose command handlers. Inference setup, collection and parsing are separate.
use crate::{
    args::{self, Command},
    collection, inference,
    support::{Result, cancellation, check_build_cancelled, print_json, read_json, read_request},
};
use fireprot_core::protocol::Phase;
use fireprot_runtime::{
    hip_build::HipBuildSpec, kernel_bundle, preflight::export_source, torch_build::TorchBuildSpec,
};
use fireprot_validation::{
    array::Policy,
    numpy::{compare_archives, read_npz},
    timing::TimingSeries,
};
use serde_json::json;
use std::ffi::OsString;

pub fn run(arguments: impl IntoIterator<Item = OsString>) -> Result<()> {
    let arguments = arguments
        .into_iter()
        .map(|s| s.into_string().map_err(|_| "arguments must be UTF-8"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let command = args::parse(&arguments)?;
    execute(command)
}

fn execute(command: Command<'_>) -> Result<()> {
    match command {
        Command::Help(name) => print!("{}", args::help(name)),
        Command::Version => println!("fireprot {}", env!("CARGO_PKG_VERSION")),
        Command::Inspect(path) => print_json(&read_request(path)?)?,
        Command::Infer { request, profile } => {
            let report = inference::run(request, profile)?;
            print_json(&report)?;
            if report.phase != Phase::Succeeded {
                return Err("inference failed; inspect supervisor.json and stderr.log".into());
            }
        }
        Command::SummarizeTimings(path) => {
            let series: TimingSeries = read_json(path)?;
            let summary = series.summarize()?;
            print_json(&json!({"series":series,"summary":summary}))?;
        }
        Command::CollectTimings(path) => print_json(&collection::collect(read_json(path)?)?)?,
        Command::ExportSource { root, commit } => print_json(&export_source(root, commit)?)?,
        Command::CompareNpz {
            reference,
            candidate,
        } => {
            let reference = read_npz(reference, 512 * 1024 * 1024)?;
            let candidate = read_npz(candidate, 512 * 1024 * 1024)?;
            let result = compare_archives(&reference, &candidate, Policy::Exact)?;
            print_json(&result)?;
            if result.values().any(|r| !r.passed) {
                return Err("prediction comparison failed".into());
            }
        }
        Command::BuildTorch {
            specification,
            cache,
        } => {
            let spec: TorchBuildSpec = read_json(specification)?;
            let cancelled = cancellation()?;
            let result = spec.run_checked(cache, &mut || check_build_cancelled(&cancelled))?;
            print_json(
                &json!({"directory":result.directory,"cache_hit":result.cache_hit,"receipt":result.receipt}),
            )?;
        }
        Command::BuildHip {
            specification,
            cache,
        } => {
            let spec: HipBuildSpec = read_json(specification)?;
            let cancelled = cancellation()?;
            let result = spec.run_checked(cache, &mut || check_build_cancelled(&cancelled))?;
            print_json(
                &json!({"directory":result.directory,"cache_hit":result.cache_hit,"receipt":result.receipt}),
            )?;
        }
        Command::PublishKernel { cache, destination } => {
            print_json(&kernel_bundle::publish(cache, destination)?)?
        }
        Command::VerifyKernel {
            directory,
            digest,
            compatibility,
        } => {
            print_json(&kernel_bundle::accept(
                directory,
                digest,
                &read_json(compatibility)?,
            )?)?;
        }
    }
    Ok(())
}
