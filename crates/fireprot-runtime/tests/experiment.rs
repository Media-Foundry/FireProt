#![cfg(unix)]
use fireprot_runtime::{
    artifacts::sha256,
    experiment::{Arm, Plan},
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn specification(root: &Path) -> Value {
    let mut jobs = Vec::new();
    let profile = root.join("profile.json");
    fs::write(&profile, b"{\"fixture\":true}").unwrap();
    for index in 0..4 {
        let mut request: Value = serde_json::from_str(include_str!(
            "../../../configs/requests/protenix-n1000-native.json"
        ))
        .unwrap();
        request["run_id"] = json!(format!("experiment-{index}"));
        request["output_dir"] = json!(root.join(format!("output-{index}")));
        let p = root.join(format!("request-{index}.json"));
        fs::write(&p, serde_json::to_vec(&request).unwrap()).unwrap();
        jobs.push(json!({"request":{"path":p,"sha256":sha256(&p).unwrap()},"profile":{"path":profile,"sha256":sha256(&profile).unwrap()}}));
    }
    json!({"schema_version":1,"timeout_seconds":600,"invocation_timeout_seconds":60,
        "pairs":[{"first":"baseline","baseline":jobs[0],"candidate":jobs[1]},
                 {"first":"candidate","baseline":jobs[2],"candidate":jobs[3]}]})
}

#[test]
fn exact_order_and_snapshots_survive_original_control_replacement() {
    let d = tempfile::tempdir().unwrap();
    let value = specification(d.path());
    let plan = Plan::load(serde_json::from_value(value).unwrap()).unwrap();
    fs::write(d.path().join("profile.json"), b"invalid replacement").unwrap();
    fs::remove_file(d.path().join("request-0.json")).unwrap();
    let prepared = plan
        .prepare(&d.path().join("experiment"), |bytes| {
            assert_eq!(
                serde_json::from_slice::<Value>(bytes)?,
                json!({"fixture":true})
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(
        prepared
            .slots()
            .iter()
            .map(|s| (s.pair, s.arm))
            .collect::<Vec<_>>(),
        vec![
            (0, Arm::Baseline),
            (0, Arm::Candidate),
            (1, Arm::Candidate),
            (1, Arm::Baseline)
        ]
    );
    assert_eq!(
        prepared
            .batch()
            .jobs()
            .iter()
            .map(|j| j.request().as_request().run_id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "experiment-0",
            "experiment-1",
            "experiment-3",
            "experiment-2"
        ]
    );
    for invocation in prepared.invocations() {
        assert_eq!(
            sha256(&invocation.profile.path).unwrap(),
            invocation.profile.sha256
        );
        assert_eq!(
            sha256(&invocation.request.path).unwrap(),
            invocation.request.sha256
        );
    }
    assert!(
        prepared
            .write_record("../outside.json", &json!({}))
            .is_err()
    );
    assert!(
        prepared
            .write_record("execution-plan.json", &json!({}))
            .is_err()
    );
}

#[test]
fn invalid_scientific_conditions_profiles_order_and_destinations_reject_before_execution() {
    for mode in [
        "seed",
        "backend",
        "profile",
        "order",
        "deadline",
        "duplicate",
        "destination",
        "profile_validator",
    ] {
        let d = tempfile::tempdir().unwrap();
        let mut value = specification(d.path());
        match mode {
            "order" => value["pairs"][0]["first"] = json!("candidate"),
            "deadline" => value["invocation_timeout_seconds"] = json!(0),
            "duplicate" => value["pairs"][1]["baseline"] = value["pairs"][0]["baseline"].clone(),
            "profile" => {
                let p = d.path().join("different.json");
                fs::write(&p, b"{}").unwrap();
                value["pairs"][1]["baseline"]["profile"] =
                    json!({"path":p,"sha256":sha256(&p).unwrap()});
            }
            "seed" | "backend" => {
                let p = d.path().join("request-2.json");
                let mut r: Value = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
                if mode == "seed" {
                    r["seed"] = json!(999);
                } else {
                    r["backend"] = json!("fireprot");
                }
                fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
                value["pairs"][1]["baseline"]["request"]["sha256"] = sha256(&p).unwrap().into();
            }
            _ => {}
        }
        let plan = Plan::load(serde_json::from_value(value).unwrap());
        if mode == "destination" {
            assert!(
                plan.unwrap()
                    .prepare(&d.path().join("output-0"), |_| Ok(()))
                    .is_err()
            );
        } else if mode == "profile_validator" {
            assert!(
                plan.unwrap()
                    .prepare(&d.path().join("experiment"), |_| Err(
                        std::io::Error::other("profile invalid")
                    ))
                    .is_err()
            );
            assert!(!d.path().join("experiment").exists());
        } else {
            assert!(plan.is_err(), "{mode}");
        }
        assert!(!d.path().join("output-0").exists());
    }
}
