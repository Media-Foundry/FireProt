use fireprot_core::{Request, ValidatedRequest, request::ArtifactRef};
use fireprot_runtime::{
    build::BuildFile,
    build_tree::BuildTree,
    device::AmdSmiMonitor,
    execution,
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
        ["summarize-timings",path]=>{
            let series:fireprot_validation::timing::TimingSeries=serde_json::from_slice(&fs::read(path)?)?;
            let summary=series.summarize()?;
            println!("{}",serde_json::to_string_pretty(&serde_json::json!({"series":series,"summary":summary}))?);
        },
        ["export-source",root,commit]=>println!("{}",serde_json::to_string_pretty(&export_source(std::path::Path::new(root),commit)?)?),
        ["compare-npz",reference,candidate]=>{
            let r=read_npz(std::path::Path::new(reference),512*1024*1024)?;let c=read_npz(std::path::Path::new(candidate),512*1024*1024)?;
            let result=compare_archives(&r,&c,Policy::Exact)?;println!("{}",serde_json::to_string_pretty(&result)?);
            if result.values().any(|r|!r.passed){return Err("prediction comparison failed".into());}
        },
        ["build-torch",specification,cache]=>{
            let spec:fireprot_runtime::torch_build::TorchBuildSpec=serde_json::from_slice(&fs::read(specification)?)?;
            let cancelled=Arc::new(AtomicBool::new(false));
            signal_hook::flag::register(signal_hook::consts::SIGINT,Arc::clone(&cancelled))?;
            signal_hook::flag::register(signal_hook::consts::SIGTERM,Arc::clone(&cancelled))?;
            let result=spec.run_checked(std::path::Path::new(cache), &mut || {
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {Err(std::io::Error::new(std::io::ErrorKind::Interrupted,"build cancelled"))} else {Ok(())}
            })?;
            println!("{}",serde_json::to_string_pretty(&serde_json::json!({"directory":result.directory,"cache_hit":result.cache_hit,"receipt":result.receipt}))?);
        },
        ["build-hip",specification,cache]=>{
            let spec:fireprot_runtime::hip_build::HipBuildSpec=serde_json::from_slice(&fs::read(specification)?)?;
            let cancelled=Arc::new(AtomicBool::new(false));
            signal_hook::flag::register(signal_hook::consts::SIGINT,Arc::clone(&cancelled))?;
            signal_hook::flag::register(signal_hook::consts::SIGTERM,Arc::clone(&cancelled))?;
            let result=spec.run_checked(std::path::Path::new(cache), &mut || {
                if cancelled.load(std::sync::atomic::Ordering::Relaxed) {Err(std::io::Error::new(std::io::ErrorKind::Interrupted,"build cancelled"))} else {Ok(())}
            })?;
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
            let pythonpath=std::env::join_paths([workers.clone(),root.join("src"),root.clone(),framework.clone()])?;
            let mut env:Vec<_>=execution.environment.into_iter().map(|(k,v)|(k.into(),v.into())).collect();
            env.push(("PYTHONPATH".into(),pythonpath));env.push(("PYTHONNOUSERSITE".into(),"1".into()));
            let program=std::path::absolute(execution.python)?;
            let host_environment=execution::environment(std::env::vars_os(),&env)?;
            let hip_probe=execution.hip_probe.canonicalize()?;
            let amd_smi=execution::resolve_program(&execution.amd_smi,&std::env::current_dir()?,&host_environment)?;
            let mut identity_files=vec![
                BuildFile{name:"supervisor".into(),path:std::env::current_exe()?},
                BuildFile{name:"worker-entry".into(),path:workers.join("fireprot_workers/__main__.py")},
                BuildFile{name:"hip-probe".into(),path:hip_probe.clone()},
                BuildFile{name:"amd-smi".into(),path:amd_smi.clone()},
                BuildFile{name:"root-import-shim".into(),path:framework.join("fireprot/__init__.py")},
            ];
            let venv=program.parent().and_then(std::path::Path::parent).map(|p|p.join("pyvenv.cfg"));
            if let Some(path)=venv.filter(|p|p.is_file()) {identity_files.push(BuildFile{name:"python-venv".into(),path});}
            if prepared.request().as_request().backend!=fireprot_core::request::Backend::Native {
                let path=env.iter().find(|(key,_)|key=="FIREPROT_BRIDGE_LIBRARY").map(|(_,v)|PathBuf::from(v)).unwrap_or_else(||framework.join("target/release/lib_fireprot_bridge.so"));
                identity_files.push(BuildFile{name:"rust-bridge".into(),path});
            }
            let identity_trees=vec![
                BuildTree{name:"workers".into(),path:workers.join("fireprot_workers"),suffixes:vec![".py".into()]},
                BuildTree{name:"tensor-boundary".into(),path:workers.join("fireprot"),suffixes:vec![".py".into()]},
                BuildTree{name:"amd-kernels".into(),path:framework.join("kernels/amd"),suffixes:vec![".cpp".into(),".cu".into(),".hip".into(),".h".into(),".hpp".into()]},
            ];
            let worker=WorkerCommand{program,args:vec!["-m".into(),"fireprot_workers".into()],working_directory:workers,environment:env,identity_files,identity_trees};
            let mut monitor=AmdSmiMonitor::new(hip_probe,amd_smi,Duration::from_secs(10));
            let report=supervisor::run(&prepared,&worker,&mut monitor,&Limits::default(),&cancelled,|paths|bundle::validate(prepared.request(),paths).map(|_|()).map_err(|e|e.to_string()))?;
            println!("{}",serde_json::to_string_pretty(&report)?);
            if report.phase!=fireprot_core::protocol::Phase::Succeeded{return Err("inference failed; inspect supervisor.json and stderr.log".into());}
        },
        _=>return Err("usage: fireprot inspect REQUEST | summarize-timings SERIES_JSON | infer REQUEST EXECUTION | export-source ROOT COMMIT | compare-npz REFERENCE CANDIDATE | build-torch SPECIFICATION CACHE | build-hip SPECIFICATION CACHE | publish-kernel CACHE DESTINATION | verify-kernel DIRECTORY MANIFEST_SHA256 COMPATIBILITY_JSON".into())
    }
    Ok(())
}
fn main() {
    if let Err(error) = execute() {
        eprintln!("fireprot: {error}");
        std::process::exit(1);
    }
}
