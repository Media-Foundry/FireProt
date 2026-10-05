use fireprot_core::request::{
    ArtifactRef, Backend, CheckpointRole, Precision, SourcePin, Workload,
};
use fireprot_core::{Request, RequestError, ValidatedRequest};
use std::collections::BTreeMap;

fn artifact() -> ArtifactRef {
    ArtifactRef {
        path: "/assets/model.pt".into(),
        sha256: "a".repeat(64),
    }
}
fn request(workload: Workload) -> Request {
    let mut checkpoints = BTreeMap::from([(CheckpointRole::Model, artifact())]);
    if matches!(workload, Workload::Atlasfold { .. }) {
        checkpoints.insert(CheckpointRole::LanguageModel, artifact());
    }
    Request {
        schema_version: 1,
        run_id: "long-1536-seed102".into(),
        workload,
        backend: Backend::Native,
        seed: 102,
        hip_device: 7,
        source: SourcePin {
            root: "/models/upstream".into(),
            commit: "b".repeat(40),
        },
        checkpoints,
        resources: BTreeMap::new(),
        input: artifact(),
        output_dir: "/outputs/new-run".into(),
    }
}
fn workloads() -> [Workload; 3] {
    [
        Workload::Protenix {
            model_name: "protenix_mini_default_v0.5.0".into(),
            cycles: 4,
            diffusion_steps: 1,
            samples: 1,
            precision: Precision::Float32,
        },
        Workload::Atlasfold {
            model_name: "atlasfold-260703".into(),
            recycles: 4,
            diffusion_steps: 100,
            samples: 1,
            precision: Precision::Native,
        },
        Workload::Openfold3 {
            model_name: "openbind-0".into(),
            recycles: 3,
            diffusion_steps: 200,
            samples: 1,
            precision: Precision::Float32,
            deterministic: None,
        },
    ]
}

#[test]
fn model_semantics_survive_wire_round_trip() {
    for (workload, passes) in workloads().into_iter().zip([4, 5, 4]) {
        let r = request(workload);
        let json = serde_json::to_string(&r).unwrap();
        let decoded: Request = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, r);
        let validated = ValidatedRequest::try_from(decoded).unwrap();
        assert_eq!(
            validated.as_request().workload.trunk_passes().unwrap(),
            passes
        );
    }
}

#[test]
fn explicit_zero_recycles_means_one_pass_but_zero_cycles_is_invalid() {
    let mut values = workloads();
    if let Workload::Protenix { cycles, .. } = &mut values[0] {
        *cycles = 0;
    }
    if let Workload::Atlasfold { recycles, .. } = &mut values[1] {
        *recycles = 0;
    }
    if let Workload::Openfold3 { recycles, .. } = &mut values[2] {
        *recycles = 0;
    }
    assert_eq!(
        ValidatedRequest::try_from(request(values[0].clone())),
        Err(RequestError::ZeroCount("cycles"))
    );
    for workload in &values[1..] {
        let r = ValidatedRequest::try_from(request(workload.clone())).unwrap();
        assert_eq!(r.as_request().workload.trunk_passes().unwrap(), 1);
    }
}

#[test]
fn atlasfold_cannot_silently_omit_language_model() {
    let mut r = request(workloads()[1].clone());
    r.checkpoints.remove(&CheckpointRole::LanguageModel);
    assert_eq!(
        ValidatedRequest::try_from(r),
        Err(RequestError::CheckpointRoles)
    );
}

#[test]
fn unrecognized_schema_and_fields_are_rejected() {
    let mut r = request(workloads()[0].clone());
    r.schema_version = 2;
    assert_eq!(
        ValidatedRequest::try_from(r.clone()),
        Err(RequestError::Schema(2))
    );
    let mut json = serde_json::to_value(r).unwrap();
    json["silently_reduce_cycles"] = true.into();
    assert!(serde_json::from_value::<Request>(json).is_err());
}

#[test]
fn family_specific_backend_and_precision_are_enforced() {
    let mut r = request(workloads()[2].clone());
    r.backend = Backend::Fireprot;
    assert_eq!(
        ValidatedRequest::try_from(r),
        Err(RequestError::UnsupportedBackend)
    );
    let mut w = workloads()[1].clone();
    if let Workload::Atlasfold { precision, .. } = &mut w {
        *precision = Precision::Bfloat16;
    }
    assert_eq!(
        ValidatedRequest::try_from(request(w)),
        Err(RequestError::UnsupportedPrecision)
    );
}

#[test]
fn counts_cannot_overflow_or_disappear() {
    let mut w = workloads()[2].clone();
    if let Workload::Openfold3 { recycles, .. } = &mut w {
        *recycles = u32::MAX;
    }
    assert_eq!(
        ValidatedRequest::try_from(request(w)),
        Err(RequestError::CycleOverflow)
    );
    for mut w in workloads() {
        match &mut w {
            Workload::Protenix { samples, .. }
            | Workload::Atlasfold { samples, .. }
            | Workload::Openfold3 { samples, .. } => *samples = 0,
        }
        assert_eq!(
            ValidatedRequest::try_from(request(w)),
            Err(RequestError::ZeroCount("samples"))
        );
    }
}

#[test]
fn invalid_provenance_and_path_like_ids_are_rejected() {
    let mut r = request(workloads()[0].clone());
    r.input.sha256 = "not-a-digest".into();
    assert_eq!(
        ValidatedRequest::try_from(r),
        Err(RequestError::ArtifactDigest)
    );
    for id in ["../escape", ".", "", "a/b"] {
        let mut r = request(workloads()[0].clone());
        r.run_id = id.into();
        assert_eq!(ValidatedRequest::try_from(r), Err(RequestError::RunId));
    }
}

#[test]
fn openfold3_determinism_is_explicit_and_preserves_legacy_wire_identity() {
    let old: serde_json::Value = serde_json::from_str(include_str!(
        "../../../configs/requests/openfold3-n1000-native.json"
    ))
    .unwrap();
    let decoded: Request = serde_json::from_value(old.clone()).unwrap();
    let encoded = serde_json::to_value(&decoded).unwrap();
    assert_eq!(encoded["workload"], old["workload"]);
    for flag in [false, true] {
        let mut explicit = old.clone();
        explicit["workload"]["deterministic"] = flag.into();
        let decoded: Request = serde_json::from_value(explicit.clone()).unwrap();
        ValidatedRequest::try_from(decoded.clone()).unwrap();
        assert_eq!(
            serde_json::to_value(decoded).unwrap()["workload"],
            explicit["workload"]
        );
    }
    let mut invalid = old;
    invalid["workload"]["deterministic"] = "true".into();
    assert!(serde_json::from_value::<Request>(invalid).is_err());
}

#[test]
fn openfold3_hip_fused_requires_explicit_strict_determinism() {
    for flag in [None, Some(false), Some(true)] {
        let mut r = request(workloads()[2].clone());
        if let Workload::Openfold3 { deterministic, .. } = &mut r.workload {
            *deterministic = flag;
        }
        r.backend = Backend::HipFused;
        if flag == Some(true) {
            let accepted = ValidatedRequest::try_from(r.clone()).unwrap();
            assert_eq!(accepted.as_request(), &r);
        } else {
            assert_eq!(
                ValidatedRequest::try_from(r),
                Err(RequestError::StrictDeterminismRequired)
            );
        }
    }
}

#[test]
fn mini_esm_requires_all_three_checkpoint_roles_without_changing_workload() {
    let mut w = workloads()[0].clone();
    if let Workload::Protenix { model_name, .. } = &mut w {
        *model_name = "protenix_mini_esm_v0.5.0".into();
    }
    let mut r = request(w);
    assert_eq!(
        ValidatedRequest::try_from(r.clone()),
        Err(RequestError::CheckpointRoles)
    );
    r.checkpoints
        .insert(CheckpointRole::LanguageModel, artifact());
    assert_eq!(
        ValidatedRequest::try_from(r.clone()),
        Err(RequestError::CheckpointRoles)
    );
    r.checkpoints
        .insert(CheckpointRole::LanguageModelRegression, artifact());
    let decoded: Request = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    let validated = ValidatedRequest::try_from(decoded).unwrap();
    assert_eq!(validated.as_request(), &r);
    assert_eq!(validated.as_request().workload.trunk_passes().unwrap(), 4);
    r.workload = workloads()[0].clone();
    assert_eq!(
        ValidatedRequest::try_from(r),
        Err(RequestError::CheckpointRoles)
    );
}
