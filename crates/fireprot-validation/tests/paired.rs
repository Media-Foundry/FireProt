use fireprot_validation::{
    paired::{self, Trend},
    timing::{TimingObservation, TimingScope, TimingSeries},
};
use serde_json::json;
use std::collections::BTreeMap;

fn series(prefix: &str, times: &[f64]) -> TimingSeries {
    let request: fireprot_core::Request = serde_json::from_str(include_str!(
        "../../../configs/requests/atlasfold-n1000-native.json"
    ))
    .unwrap();
    TimingSeries { schema_version: 1, observations: times.iter().enumerate().map(|(i,&seconds)| {
        let mut request = request.clone();
        request.run_id = format!("{prefix}-{i}"); request.output_dir = format!("outputs/{prefix}-{i}").into();
        TimingObservation { request, scope: TimingScope::Model, cache_state: "synthetic fixture".into(), seconds,
            environment: BTreeMap::from([
                ("physical_device".into(), json!({"pci":"0000:93:00.0"}).to_string()),
                ("worker".into(), json!({"python":"fixture"}).to_string()),
                ("provenance_sha256".into(), "fixture".into()),
                ("launch".into(), json!({"program":"same","environment":{"HIP_VISIBLE_DEVICES":"7"},"inputs":[{"name":"supervisor","sha256":"same"}]}).to_string()),
            ]),
        }
    }).collect() }
}

#[test]
fn ratios_trends_and_small_sample_limits_are_explicit() {
    let a = series("a", &[10., 20., 30.]);
    for (times, trend, median, p95) in [
        ([5., 10., 15.], Trend::FasterMedianAndP95, 2., 2.),
        ([20., 40., 60.], Trend::SlowerMedianAndP95, 0.5, 0.5),
        ([5., 10., 60.], Trend::MixedOrUnchanged, 2., 0.5),
    ] {
        let r = paired::compare(&a, &series("b", &times), &BTreeMap::new()).unwrap();
        assert_eq!(r.observed_trend, trend);
        assert_eq!(r.median_speedup, median);
        assert_eq!(r.p95_speedup, p95);
        assert_eq!(r.paired_speedups.len(), 3);
    }
    let r = paired::compare(&series("a", &[10.]), &series("b", &[1.]), &BTreeMap::new()).unwrap();
    assert_eq!(r.observed_trend, Trend::InsufficientRepeats);
    assert!(
        paired::compare(
            &series("a", &[f64::MAX]),
            &series("b", &[f64::from_bits(1)]),
            &BTreeMap::new()
        )
        .is_err()
    );
}

#[test]
fn scientific_conditions_and_reused_runs_cannot_be_declared_away() {
    let a = series("a", &[10.]);
    for index in 0..7 {
        let mut b = series("b", &[5.]);
        let o = &mut b.observations[0];
        match index {
            0 => o.request.seed += 1,
            1 => o.request.hip_device += 1,
            2 => o.scope = TimingScope::EndToEnd,
            3 => o.cache_state = "cold".into(),
            4 => o.request.run_id = a.observations[0].request.run_id.clone(),
            5 => {
                o.environment
                    .insert("worker".into(), json!({"python":"different"}).to_string());
            }
            _ => {
                let mut l: serde_json::Value =
                    serde_json::from_str(&o.environment["launch"]).unwrap();
                l["environment"]["PYTHONHASHSEED"] = json!("different");
                o.environment.insert("launch".into(), l.to_string());
            }
        }
        assert!(paired::differences(&a, &b).is_err(), "{index}");
    }
    assert!(paired::differences(&a, &series("b", &[1., 2.])).is_err());
}

#[test]
fn declared_changes_must_match_exact_values_and_input_names_are_not_array_indices() {
    let a = series("a", &[10.]);
    let mut b = series("b", &[5.]);
    b.observations[0].request.backend = fireprot_core::request::Backend::HipFused;
    b.observations[0].environment.insert(
        "launch".into(),
        json!({
            "program":"same","environment":{"HIP_VISIBLE_DEVICES":"7","FIREPROT_TEST_SWITCH":"1"},
            "inputs":[{"name":"rust-bridge","sha256":"added"},{"name":"supervisor","sha256":"same"}]
        })
        .to_string(),
    );
    let differences = paired::differences(&a, &b).unwrap();
    assert_eq!(differences.len(), 3);
    assert!(paired::compare(&a, &b, &BTreeMap::new()).is_err());
    assert!(paired::compare(&a, &b, &differences).is_ok());
    let mut wrong = differences;
    wrong
        .get_mut("launch.environment.FIREPROT_TEST_SWITCH")
        .unwrap()
        .candidate = Some(json!("0"));
    assert!(paired::compare(&a, &b, &wrong).is_err());
}

#[test]
fn model_gates_keep_full_array_diagnostics_separate_from_protenix_acceptance() {
    use fireprot_validation::{
        array::{Array, DType},
        bundle::{Prediction, TensorDescriptor},
    };
    let array = |shape: Vec<usize>, values: &[f32]| {
        Array::new(
            DType::Float32,
            shape,
            values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )
        .unwrap()
    };
    let mut a: BTreeMap<String, Array> = BTreeMap::from([
        (
            "coordinates".into(),
            array(
                vec![4, 3],
                &[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
            ),
        ),
        ("confidence.atom_plddt".into(), array(vec![4], &[80.; 4])),
        (
            "input.esm_token_embedding".into(),
            array(vec![1, 2], &[1., 2.]),
        ),
    ]);
    let prediction = Prediction {
        input_id: None,
        sample: 0,
        archive: "unused".into(),
        structure: None,
        confidence: Some(json!({"plddt":80.,"ptm":0.9})),
        arrays: a
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    TensorDescriptor {
                        shape: v.shape().to_vec(),
                        storage_dtype: v.dtype(),
                        original_dtype: v.dtype(),
                    },
                )
            })
            .collect(),
    };
    let workload:fireprot_core::request::Workload=serde_json::from_value(json!({"model":"protenix","model_name":"fixture","cycles":4,"diffusion_steps":1,"samples":1,"precision":"float32"})).unwrap();
    let mut b = a.clone();
    b.insert(
        "confidence.atom_plddt".into(),
        array(vec![4], &[80.0001; 4]),
    );
    let result = paired::compare_prediction(&workload, &prediction, &a, &prediction, &b).unwrap();
    assert!(result.passed);
    assert!(!result.floating_confidence_diagnostics["confidence.atom_plddt"].bitwise_equal);
    b.insert(
        "input.esm_token_embedding".into(),
        array(vec![1, 2], &[1., 3.]),
    );
    assert!(
        !paired::compare_prediction(&workload, &prediction, &a, &prediction, &b)
            .unwrap()
            .passed
    );
    b = a.clone();
    b.insert(
        "coordinates".into(),
        array(
            vec![4, 3],
            &[0., 0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 1.],
        ),
    );
    assert!(
        !paired::compare_prediction(&workload, &prediction, &a, &prediction, &b)
            .unwrap()
            .passed
    );
    a.remove("coordinates");
    assert!(paired::compare_prediction(&workload, &prediction, &a, &prediction, &b).is_err());
}
