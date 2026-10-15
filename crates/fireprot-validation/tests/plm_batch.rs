use fireprot_core::{Request, ValidatedRequest};
use fireprot_validation::bundle::validate_current;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::PathBuf};

fn archive(path: &std::path::Path, tokens: &[i64]) {
    let mut writer = zip::ZipWriter::new(fs::File::create(path).unwrap());
    for (name, dtype, shape, data) in [
        (
            "tokens",
            "<i8",
            format!("(1, {})", tokens.len()),
            tokens
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        (
            "embeddings",
            "<u2",
            format!("(1, {}, 2560)", tokens.len()),
            vec![0u8; tokens.len() * 2560 * 2],
        ),
    ] {
        writer
            .start_file(
                format!("{name}.npy"),
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
        let mut header =
            format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}");
        while (header.len() + 11) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        writer.write_all(b"\x93NUMPY\x01\x00").unwrap();
        writer
            .write_all(&(header.len() as u16).to_le_bytes())
            .unwrap();
        writer.write_all(header.as_bytes()).unwrap();
        writer.write_all(&data).unwrap();
    }
    writer.finish().unwrap();
}

fn setup() -> (tempfile::TempDir, Request, Value, Vec<PathBuf>) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let input = serde_json::to_vec(&json!({"schema_version":1,"sequences":[
        {"id":"long","sequence":"AC"},{"id":"short","sequence":"D"}]}))
    .unwrap();
    let mut request: Request = serde_json::from_str(include_str!(
        "../../../configs/requests/plm-esmc-fold2-6b-n1000-native.json"
    ))
    .unwrap();
    request.output_dir = root.clone();
    request.input.path = root.join("input.json");
    request.input.sha256 = format!("{:x}", Sha256::digest(&input));
    if let fireprot_core::request::Workload::Plm { batch, .. } = &mut request.workload {
        *batch = Some(fireprot_core::plm_batch::Options {
            items: 2,
            max_batch_size: 2,
            max_padded_tokens: 8,
            grouping: fireprot_core::plm_batch::Grouping::LengthSorted,
        });
    }
    fs::write(root.join("snapshot.json"), input).unwrap();
    let mut verified = vec![root.join("predictions.json"), root.join("snapshot.json")];
    let mut predictions = Vec::new();
    for (index, (id, tokens)) in [("long", vec![0, 5, 23, 2]), ("short", vec![0, 13, 2])]
        .into_iter()
        .enumerate()
    {
        let filename = format!("sample-{index}.npz");
        let path = root.join(&filename);
        archive(&path, &tokens);
        verified.push(path);
        predictions.push(json!({"sample":index,"input_id":id,"archive":filename,"confidence":null,"arrays":{
            "tokens":{"shape":[1,tokens.len()],"storage_dtype":"int64","original_dtype":"int64"},
            "embeddings":{"shape":[1,tokens.len(),2560],"storage_dtype":"uint16","original_dtype":"bfloat16"}}}));
    }
    let bundle = json!({"schema_version":6,"run_id":request.run_id,"seed":request.seed,"workload":request.workload,
        "input_snapshot":"snapshot.json","batch_groups":[[1,0]],"predictions":predictions,"model_seconds":1.0,
        "request_seconds":2.0,"environment":{"torch":"fixture","hip":"fixture","python":"fixture"}});
    (dir, request, bundle, verified)
}

#[test]
fn batch_binds_all_trimmed_outputs_to_the_rust_plan() {
    let (_dir, request, bundle, verified) = setup();
    fs::write(&verified[0], serde_json::to_vec(&bundle).unwrap()).unwrap();
    validate_current(&ValidatedRequest::try_from(request).unwrap(), &verified).unwrap();
}

#[test]
fn rejects_missing_swapped_unbound_and_padded_batch_outputs() {
    for change in [
        "missing", "id", "order", "group", "snapshot", "padding", "tokens", "schema",
    ] {
        let (_dir, request, mut bundle, verified) = setup();
        match change {
            "missing" => {
                bundle["predictions"].as_array_mut().unwrap().pop();
            }
            "id" => bundle["predictions"][0]["input_id"] = json!("short"),
            "order" => bundle["predictions"].as_array_mut().unwrap().swap(0, 1),
            "group" => bundle["batch_groups"] = json!([[0, 1]]),
            "snapshot" => {
                fs::write(&verified[1], b"{}").unwrap();
            }
            "padding" => {
                archive(&verified[3], &[0, 13, 2, 1]);
                bundle["predictions"][1]["arrays"]["tokens"]["shape"] = json!([1, 4]);
                bundle["predictions"][1]["arrays"]["embeddings"]["shape"] = json!([1, 4, 2560]);
            }
            "tokens" => archive(&verified[3], &[0, 5, 2]),
            "schema" => bundle["schema_version"] = json!(5),
            _ => unreachable!(),
        }
        fs::write(&verified[0], serde_json::to_vec(&bundle).unwrap()).unwrap();
        assert!(
            validate_current(&ValidatedRequest::try_from(request).unwrap(), &verified).is_err(),
            "{change}"
        );
    }
}
