//! Isolated same-GCD throughput experiment; not maintained infer acceptance.
//! Rust owns timing, process launch, cohort monitoring and exact-output checks.
extern crate serde_json;
use serde_json::{json, Value};
use std::{collections::BTreeSet, fs, path::Path, process::{Child, Command, Stdio}, thread, time::{Duration, Instant}};

fn read(p: &str) -> Value { serde_json::from_slice(&fs::read(p).unwrap()).unwrap() }
fn write(p: &str, v: &Value) { fs::write(p, serde_json::to_vec_pretty(v).unwrap()).unwrap(); }
fn observe(allowed: &BTreeSet<u32>) -> Value {
    let p = Command::new("/usr/bin/amd-smi").args(["process", "--json"]).output().unwrap();
    assert!(p.status.success());
    let v: Value = serde_json::from_slice(&p.stdout).unwrap();
    let row = v.as_array().unwrap().iter().find(|r| r["gpu"].as_u64()==Some(5)).unwrap();
    for entry in row["process_list"].as_array().unwrap() {
        let info = &entry["process_info"];
        if info.as_str()==Some("No running processes detected") { continue; }
        let pid = info["pid"].as_u64().or_else(|| info["pid"].as_str().unwrap().parse().ok()).unwrap() as u32;
        assert!(allowed.contains(&pid), "Unexpected GPU process {pid}");
    }
    v
}
fn main() {
    let path = std::env::args().nth(1).expect("plan.json required");
    let plan = read(&path);
    let root = plan["output"].as_str().unwrap();
    assert!(!Path::new(root).exists()); fs::create_dir(root).unwrap();
    // Physical mapping is checked against SMI rather than assuming equal HIP/SMI ordinals.
    let smi = Command::new("/usr/bin/amd-smi").args(["list", "--json"]).output().unwrap();
    assert!(smi.status.success());
    let devices: Value = serde_json::from_slice(&smi.stdout).unwrap();
    assert!(devices.as_array().unwrap().iter().any(|r| r["gpu"]==5 && r["bdf"]=="0000:93:00.0"));
    let idle = observe(&BTreeSet::new()); write(&format!("{root}/idle.json"), &idle);
    let mut all = Vec::new();
    for group in plan["groups"].as_array().unwrap() {
        let mode = group["mode"].as_str().unwrap(); assert!(mode=="serial" || mode=="concurrent");
        let jobs = group["jobs"].as_array().unwrap(); assert_eq!(jobs.len(),2);
        let mut children: Vec<(Child, Instant, Value)> = Vec::new();
        let mut finished = Vec::new(); let mut next=0; let mut samples=0;
        let started=Instant::now();
        while next<jobs.len() || !children.is_empty() {
            while next<jobs.len() && (mode=="concurrent" || children.is_empty()) {
                let job=jobs[next].clone(); let q=read(job["request"].as_str().unwrap());
                assert_eq!(q["hip_device"],7); assert_eq!(q["workload"]["samples"],1);
                let mut cmd=Command::new(plan["python"].as_str().unwrap());
                cmd.arg(plan["member"].as_str().unwrap()).env_clear();
                for (key,val) in plan["environment"].as_object().unwrap() { cmd.env(key,val.as_str().unwrap()); }
                cmd.env("HIP_VISIBLE_DEVICES","7").env("FIREPROT_REQUEST",job["request"].as_str().unwrap())
                    .env("FIREPROT_OUTPUT_DIR",q["output_dir"].as_str().unwrap())
                    .env_remove("FIREPROT_WORKER_SOCKET")
                    .current_dir(plan["cwd"].as_str().unwrap()).stdin(Stdio::null())
                    .stdout(fs::File::create(job["stdout"].as_str().unwrap()).unwrap())
                    .stderr(fs::File::create(job["stderr"].as_str().unwrap()).unwrap());
                children.push((cmd.spawn().unwrap(),Instant::now(),job)); next+=1;
            }
            // Preserve ownership of every child on monitoring failure; kill only our cohort.
            let allowed=children.iter().map(|x| x.0.id()).collect();
            let observed=std::panic::catch_unwind(|| observe(&allowed));
            if observed.is_err() || started.elapsed().as_secs()>1800 {
                for (child,_,_) in &mut children { let _=child.kill(); let _=child.wait(); }
                panic!("Cohort monitoring failed or group timed out");
            }
            samples+=1;
            let mut i=0;
            while i<children.len() {
                if let Some(status)=children[i].0.try_wait().unwrap() {
                    let (_,clock,job)=children.remove(i);
                    if !status.success() {
                        for (child,_,_) in &mut children { let _=child.kill(); let _=child.wait(); }
                        panic!("Batch member failed: {job}");
                    }
                    finished.push(json!({"job":job,"elapsed_seconds":clock.elapsed().as_secs_f64()}));
                } else { i+=1; }
            }
            if next<jobs.len() || !children.is_empty() { thread::sleep(Duration::from_millis(250)); }
        }
        let wall=started.elapsed().as_secs_f64();
        for member in &finished {
            let q=read(member["job"]["request"].as_str().unwrap()); let out=q["output_dir"].as_str().unwrap();
            let check=Command::new(plan["cli"].as_str().unwrap()).args(["compare-npz",plan["reference_npz"].as_str().unwrap(),&format!("{out}/sample-0.npz")]).output().unwrap();
            assert!(check.status.success()); let exact:Value=serde_json::from_slice(&check.stdout).unwrap();
            assert_eq!(exact.as_object().unwrap().len(),9); assert!(exact.as_object().unwrap().values().all(|x|x["bitwise_equal"]==true && x["passed"]==true));
            write(&format!("{out}/exact.json"),&exact);
            let pred=read(&format!("{out}/predictions.json")); let reference=read(plan["reference_predictions"].as_str().unwrap());
            assert_eq!(pred["predictions"][0]["confidence"],reference["predictions"][0]["confidence"]);
        }
        all.push(json!({"mode":mode,"group":group["name"],"jobs":finished,"wall_seconds":wall,"proteins_per_hour":7200.0/wall,"monitor_samples":samples}));
        write(&format!("{root}/progress.json"),&json!(all));
    }
    write(&format!("{root}/completed.json"),&json!({"scope":"Two independent fresh-process requests on one GCD, serial versus concurrent; includes Python startup and model loading, excludes maintained Rust preflight. Not a tensor-batched model or maintained E2E receipt.","groups":all}));
}
