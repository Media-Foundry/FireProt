use fireprot_validation::timing::{TimingObservation, TimingScope, TimingSeries, summarize};
use std::collections::BTreeMap;

fn series() -> TimingSeries {
    let request = serde_json::from_str(include_str!(
        "../../../configs/requests/atlasfold-n1000-native.json"
    ))
    .unwrap();
    let first = TimingObservation {
        request,
        scope: TimingScope::Model,
        cache_state: "published kernels; model resident; one warmup excluded upstream".into(),
        environment: BTreeMap::from([("fixture".into(), "not a measured model run".into())]),
        seconds: 1.0,
    };
    let mut second = first.clone();
    second.request.run_id = "repeat-2".into();
    second.request.output_dir = "outputs/repeat-2".into();
    second.seconds = 3.0;
    TimingSeries {
        schema_version: 1,
        observations: vec![first, second],
    }
}

#[test]
fn historical_statistics_and_nearest_rank_boundaries() {
    let r = summarize(&[9.0, 1.0, 4.0, 2.0]).unwrap();
    assert_eq!(
        (r.count, r.min, r.median, r.p95, r.max, r.mean),
        (4, 1.0, 3.0, 9.0, 9.0, 4.0)
    );
    let values: Vec<_> = (1..=20).map(f64::from).collect();
    assert_eq!(summarize(&values).unwrap().p95, 19.0);
    assert_eq!(summarize(&values[..19]).unwrap().p95, 19.0);
    assert_eq!(summarize(&[2.0, 5.0, 1.0]).unwrap().median, 2.0);
    assert_eq!(series().summarize().unwrap().mean, 2.0);
}

#[test]
fn invalid_values_fail_and_finite_extremes_do_not_overflow() {
    assert!(summarize(&[]).is_err());
    for bad in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(summarize(&[1.0, bad]).is_err());
    }
    for value in [f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)] {
        let r = summarize(&[value, value]).unwrap();
        assert_eq!(r.mean, value);
        assert_eq!(r.median, value);
    }
}

#[test]
fn never_pool_different_workloads_cache_environment_or_seeds() {
    let changes: Vec<fn(&mut TimingSeries)> = vec![
        |s| s.schema_version = 2,
        |s| s.observations[1].request.run_id = s.observations[0].request.run_id.clone(),
        |s| s.observations[1].request.seed += 1,
        |s| s.observations[1].request.hip_device += 1,
        |s| s.observations[1].request.backend = fireprot_core::request::Backend::Hip,
        |s| s.observations[1].scope = TimingScope::EndToEnd,
        |s| s.observations[1].cache_state = "cold".into(),
        |s| s.observations[1].environment.clear(),
        |s| {
            s.observations[1]
                .environment
                .insert("fixture".into(), "different".into());
        },
        |s| s.observations[1].request.input.sha256 = "0".repeat(64),
        |s| s.observations[1].request.schema_version = 99,
    ];
    for change in changes {
        let mut input = series();
        change(&mut input);
        assert!(input.summarize().is_err());
    }
    let mut value = serde_json::to_value(series()).unwrap();
    value["observations"][0]["hidden_warmup"] = true.into();
    assert!(serde_json::from_value::<TimingSeries>(value).is_err());
}
