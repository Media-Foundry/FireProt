use fireprot_core::{Request, ValidatedRequest, request::ArtifactRef};
use fireprot_runtime::{
    device::AmdSmiMonitor,
    preflight::{SourceOrigin, export_source, prepare},
    supervisor::{self, Limits, WorkerCommand},
};
use fireprot_validation::{
    array::Policy,
    bundle,
    numpy::{compare_archives, read_npz},
};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    python: PathBuf,
    workers: PathBuf,
    hip_probe: PathBuf,
    amd_smi: PathBuf,
    source_manifest: Option<ArtifactRef>,
    #[serde(default)]
    environment: BTreeMap<String, String>,
}
fn read_request(path: &str) -> Result<ValidatedRequest, Box<dyn std::error::Error>> {
    let request: Request = serde_json::from_slice(&fs::read(path)?)?;
    Ok(request.try_into()?)
}
fn execute() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice(){
        ["inspect",request]=>{let r=read_request(request)?;println!("{}",serde_json::to_string_pretty(&r)?);},
        ["export-source",root,commit]=>println!("{}",serde_json::to_string_pretty(&export_source(std::path::Path::new(root),commit)?)?),
        ["compare-npz",reference,candidate]=>{
            let r=read_npz(std::path::Path::new(reference),512*1024*1024)?;let c=read_npz(std::path::Path::new(candidate),512*1024*1024)?;
            let result=compare_archives(&r,&c,Policy::Exact)?;println!("{}",serde_json::to_string_pretty(&result)?);
            if result.values().any(|r|!r.passed){return Err("prediction comparison failed".into());}
        },
        ["build-torch",specification,cache]=>{
            let spec:fireprot_runtime::torch_build::TorchBuildSpec=serde_json::from_slice(&fs::read(specification)?)?;
            let result=spec.run(std::path::Path::new(cache))?;
            println!("{}",serde_json::to_string_pretty(&serde_json::json!({"directory":result.directory,"cache_hit":result.cache_hit,"receipt":result.receipt}))?);
        },
        ["publish-kernel",cache,destination]=>println!("{}",serde_json::to_string_pretty(&fireprot_runtime::kernel_bundle::publish(std::path::Path::new(cache),std::path::Path::new(destination))?)?),
        ["verify-kernel",directory,digest,compatibility]=>{
            let expected=serde_json::from_slice(&fs::read(compatibility)?)?;
            println!("{}",serde_json::to_string_pretty(&fireprot_runtime::kernel_bundle::accept(std::path::Path::new(directory),digest,&expected)?)?);
        },
        ["infer",request,execution]=>{
            let request=read_request(request)?;let execution:Execution=serde_json::from_slice(&fs::read(execution)?)?;
            let origin=match execution.source_manifest{Some(manifest)=>SourceOrigin::Exported{manifest},None=>SourceOrigin::GitCheckout};
            let cancelled=Arc::new(AtomicBool::new(false));
            signal_hook::flag::register(signal_hook::consts::SIGINT,Arc::clone(&cancelled))?;
            signal_hook::flag::register(signal_hook::consts::SIGTERM,Arc::clone(&cancelled))?;
            let prepared=prepare(request,origin)?;
            let workers=execution.workers.canonicalize()?;
            let root=prepared.request().as_request().source.root.clone();
            let framework=workers.parent().ok_or("worker package needs a framework parent directory")?.to_path_buf();
            let pythonpath=std::env::join_paths([workers.clone(),root.join("src"),root.clone(),framework])?;
            let mut env:Vec<_>=execution.environment.into_iter().map(|(k,v)|(k.into(),v.into())).collect();
            env.push(("PYTHONPATH".into(),pythonpath));env.push(("PYTHONNOUSERSITE".into(),"1".into()));
            let worker=WorkerCommand{program:std::path::absolute(execution.python)?,args:vec!["-m".into(),"fireprot_workers".into()],working_directory:workers,environment:env};
            let mut monitor=AmdSmiMonitor::new(execution.hip_probe.canonicalize()?,execution.amd_smi,Duration::from_secs(10));
            let report=supervisor::run(&prepared,&worker,&mut monitor,&Limits::default(),&cancelled,|paths|bundle::validate(prepared.request(),paths).map(|_|()).map_err(|e|e.to_string()))?;
            println!("{}",serde_json::to_string_pretty(&report)?);
            if report.phase!=fireprot_core::protocol::Phase::Succeeded{return Err("inference failed; inspect supervisor.json and stderr.log".into());}
        },
        _=>return Err("usage: fireprot inspect REQUEST | infer REQUEST EXECUTION | export-source ROOT COMMIT | compare-npz REFERENCE CANDIDATE | build-torch SPECIFICATION CACHE | publish-kernel CACHE DESTINATION | verify-kernel DIRECTORY MANIFEST_SHA256 COMPATIBILITY_JSON".into())
    }
    Ok(())
}
fn main() {
    if let Err(error) = execute() {
        eprintln!("fireprot: {error}");
        std::process::exit(1);
    }
}
