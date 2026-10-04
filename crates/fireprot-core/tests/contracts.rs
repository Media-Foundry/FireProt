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
    r.backend = Backend::HipFused;
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
