use fireprot_core::{Request, ValidatedRequest};
use fireprot_validation::{
    atlasfold::input_sequence,
    bundle::{validate, validate_current},
    numpy::read_npz,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
};
const SEQUENCE: &str = "ACDEFGHIKLMNPQRSTVWYX";

fn setup(json_input: bool, samples: usize) -> (tempfile::TempDir, Request, Value, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let input = if json_input {
        serde_json::to_vec(&json!([{"name":"fixture","sequences":[{"proteinChain":{"sequence":SEQUENCE,"count":1}}]}])).unwrap()
    } else {
        format!(">fixture\r\n{SEQUENCE}\r\n").into_bytes()
    };
    let mut r: Request = serde_json::from_str(include_str!(
        "../../../configs/requests/atlasfold-n1000-native.json"
    ))
    .unwrap();
    r.run_id = "fixture".into();
    r.output_dir = root.clone();
    r.input.path = root.join(if json_input {
        "deleted-original.json"
    } else {
        "deleted-original.fa"
    });
    r.input.sha256 = format!("{:x}", Sha256::digest(&input));
    if let fireprot_core::request::Workload::Atlasfold { samples: count, .. } = &mut r.workload {
        *count = samples as u32;
    }
    fs::write(root.join("snapshot"), input).unwrap();
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/atlasfold_atom14");
    let mut source =
        zip::ZipArchive::new(fs::File::open(fixture.join("coordinates.npz")).unwrap()).unwrap();
    let mut coordinates = Vec::new();
    source
        .by_name("coordinates.npy")
        .unwrap()
        .read_to_end(&mut coordinates)
        .unwrap();
    let mut predictions = Vec::new();
    let mut verified = vec![root.join("predictions.json"), root.join("snapshot")];
    for sample in 0..samples {
        let archive = format!("sample-{sample}.npz");
        let structure = format!("sample-{sample}.cif");
        let mut zip = zip::ZipWriter::new(fs::File::create(root.join(&archive)).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("coordinates.npy", options).unwrap();
        zip.write_all(&coordinates).unwrap();
        for (name, shape, count) in [
            ("plddt", "(21,)", 21),
            ("pae", "(21, 21)", 441),
            ("ptm", "()", 1),
        ] {
            zip.start_file(format!("{name}.npy"), options).unwrap();
            let mut header =
                format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
            while (header.len() + 11) % 64 != 0 {
                header.push(' ');
            }
            header.push('\n');
            zip.write_all(b"\x93NUMPY\x01\x00").unwrap();
            zip.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
            zip.write_all(header.as_bytes()).unwrap();
            zip.write_all(&vec![0; count * 4]).unwrap();
        }
        zip.finish().unwrap();
        fs::copy(fixture.join("prediction.cif"), root.join(&structure)).unwrap();
        let arrays: serde_json::Map<String, Value> = read_npz(&root.join(&archive), 1024 * 1024)
            .unwrap()
            .into_iter()
            .map(|(n, a)| {
                (
                    n,
                    json!({"shape":a.shape(),"storage_dtype":a.dtype(),"original_dtype":a.dtype()}),
                )
            })
            .collect();
        predictions.push(json!({"sample":sample,"archive":archive,"structure":structure,"arrays":arrays,"confidence":null}));
        verified.extend([root.join(archive), root.join(structure)]);
    }
    let b = json!({"schema_version":3,"run_id":r.run_id,"seed":r.seed,"workload":r.workload,"predictions":predictions,"input_snapshot":"snapshot","model_seconds":1.,"request_seconds":2.,"environment":{"python":"fixture","torch":"fixture","hip":"fixture"}});
    (dir, r, b, verified)
}
fn save(r: &Request, b: &Value) {
    fs::write(
        r.output_dir.join("predictions.json"),
        serde_json::to_vec(b).unwrap(),
    )
    .unwrap();
}

#[test]
fn bound_sequence_accepts_every_sample_without_original_input_and_after_relocation() {
    for json_input in [true, false] {
        let (dir, mut r, b, mut v) = setup(json_input, 2);
        save(&r, &b);
        assert!(!r.input.path.exists());
        validate_current(&ValidatedRequest::try_from(r.clone()).unwrap(), &v).unwrap();
        let relocation = tempfile::tempdir().unwrap();
        let moved = relocation.path().join("archive");
        fs::rename(dir.path(), &moved).unwrap();
        v = v
            .iter()
            .map(|p| moved.join(p.file_name().unwrap()))
            .collect();
        r.output_dir = moved;
        validate(&r.try_into().unwrap(), &v).unwrap();
    }
}

#[test]
fn binding_and_each_samples_structure_are_mandatory() {
    for mutation in [
        "missing",
        "unverified",
        "escape",
        "digest",
        "different_sequence",
        "second_sample",
        "old_schema_with_snapshot",
        "non_atlas_schema3",
    ] {
        let (_dir, mut r, mut b, mut v) = setup(true, 2);
        match mutation {
            "missing" => {
                b.as_object_mut().unwrap().remove("input_snapshot");
            }
            "unverified" => {
                v.remove(1);
            }
            "escape" => b["input_snapshot"] = json!("../snapshot"),
            "digest" => fs::write(r.output_dir.join("snapshot"), b"[]").unwrap(),
            "different_sequence" => {
                let bytes = fs::read(r.output_dir.join("snapshot")).unwrap();
                let text = String::from_utf8(bytes)
                    .unwrap()
                    .replace(SEQUENCE, "XCDEFGHIKLMNPQRSTVWYA");
                fs::write(r.output_dir.join("snapshot"), &text).unwrap();
                r.input.sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
            }
            "second_sample" => {
                let p = r.output_dir.join("sample-1.cif");
                let text = fs::read_to_string(&p).unwrap().replace("ALA", "GLY");
                fs::write(p, text).unwrap();
            }
            "old_schema_with_snapshot" => b["schema_version"] = json!(2),
            "non_atlas_schema3" => {
                r.workload = fireprot_core::request::Workload::Openfold3 {
                    model_name: "openbind-0".into(),
                    recycles: 3,
                    diffusion_steps: 200,
                    samples: 2,
                    precision: fireprot_core::request::Precision::Float32,
                    deterministic: None,
                };
                r.checkpoints
                    .remove(&fireprot_core::request::CheckpointRole::LanguageModel);
                b["workload"] = serde_json::to_value(&r.workload).unwrap();
            }
            _ => unreachable!(),
        }
        save(&r, &b);
        assert!(validate(&r.try_into().unwrap(), &v).is_err(), "{mutation}");
    }
}

#[test]
fn old_atlas_archives_keep_historical_scope_but_cannot_enter_current_inference() {
    let (_dir, r, mut b, v) = setup(true, 1);
    b["schema_version"] = json!(2);
    b.as_object_mut().unwrap().remove("input_snapshot");
    save(&r, &b);
    let r: ValidatedRequest = r.try_into().unwrap();
    assert!(validate(&r, &v).is_ok());
    assert!(validate_current(&r, &v).is_err());
}

#[test]
fn parser_rejects_multimers_invalid_residues_and_missing_fields() {
    for bytes in [
        b"".as_slice(),
        b">a\nACD\n>b\nEFG",
        b">a\nacd",
        b">a\nACB",
        b">\nACD",
        b"\xff",
    ] {
        assert!(input_sequence(bytes, false).is_err());
    }
    for text in [
        "[]",
        "{}",
        r#"[{"name":"a","sequences":[{"proteinChain":{"sequence":"ACD","count":2}}]}]"#,
        r#"[{"name":"a","sequences":[{"proteinChain":{"sequence":"ACD"}},{"proteinChain":{"sequence":"EFG"}}]}]"#,
    ] {
        assert!(input_sequence(text.as_bytes(), true).is_err());
    }
    assert_eq!(
        input_sequence(b" >name \rACD\r\nEFX\n", false).unwrap(),
        "ACDEFX"
    );
}

#[test]
fn fasta_strip_matches_python_for_ascii_separator_whitespace() {
    assert_eq!(
        input_sequence(b"\x1f>name\x1f\rACD\x1f\nEFX\x1f", false).unwrap(),
        "ACDEFX"
    );
}
