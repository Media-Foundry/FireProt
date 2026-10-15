//! Model output content gate, independent of worker success declarations.
use crate::{
    Error, Result,
    array::{Array, DType},
    metrics::confidence,
    numpy::read_npz,
};
use fireprot_core::{ValidatedRequest, request::Workload};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Read,
    path::{Component, PathBuf},
};
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorDescriptor {
    pub shape: Vec<usize>,
    pub original_dtype: DType,
    pub storage_dtype: DType,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub sample: u32,
    pub archive: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure: Option<PathBuf>,
    pub arrays: BTreeMap<String, TensorDescriptor>,
    pub confidence: Option<serde_json::Value>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub schema_version: u32,
    pub run_id: String,
    pub seed: u64,
    pub workload: Workload,
    pub predictions: Vec<Prediction>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_snapshot: Option<PathBuf>,
    pub model_seconds: f64,
    pub request_seconds: f64,
    pub environment: BTreeMap<String, String>,
}
fn schema(message: &str) -> Error {
    Error::Schema(message.into())
}
/// Every referenced file must be in the supervisor's independently hash-verified
/// artifact list. Only native BF16 represented as raw uint16 storage may differ
/// from original dtype; conversions such as BF16 -> FP32 are rejected.
pub fn validate(request: &ValidatedRequest, verified: &[PathBuf]) -> Result<Bundle> {
    let r = request.as_request();
    let root = r.output_dir.canonicalize()?;
    let manifest = root.join("predictions.json").canonicalize()?;
    if !verified.contains(&manifest) {
        return Err(schema("missing verified prediction manifest"));
    }
    let bundle: Bundle =
        serde_json::from_slice(&fs::read(manifest)?).map_err(|e| Error::Schema(e.to_string()))?;
    if !matches!(bundle.schema_version, 1..=4)
        || bundle.run_id != r.run_id
        || bundle.seed != r.seed
        || bundle.workload != r.workload
    {
        return Err(schema("prediction identity/workload mismatch"));
    }
    if !bundle.model_seconds.is_finite()
        || bundle.model_seconds <= 0.0
        || !bundle.request_seconds.is_finite()
        || bundle.request_seconds < bundle.model_seconds
    {
        return Err(schema("invalid model/request timing"));
    }
    for name in ["torch", "hip", "python"] {
        if bundle.environment.get(name).is_none_or(|x| x.is_empty()) {
            return Err(schema("missing model environment identity"));
        }
    }
    let samples = match r.workload {
        Workload::Protenix { samples, .. }
        | Workload::Atlasfold { samples, .. }
        | Workload::Openfold3 { samples, .. }
        | Workload::Esmfold2 { samples, .. } => samples,
    };
    if bundle.predictions.len() != samples as usize {
        return Err(schema("prediction sample count mismatch"));
    }
    let sequence = if matches!(bundle.schema_version, 3 | 4) {
        if !matches!(
            (&r.workload, bundle.schema_version),
            (Workload::Atlasfold { .. }, 3) | (Workload::Esmfold2 { .. }, 4)
        ) {
            return Err(schema("input binding schema does not match the model"));
        }
        let relative = bundle
            .input_snapshot
            .as_ref()
            .ok_or_else(|| schema("missing AtlasFold input snapshot"))?;
        if relative.as_os_str().is_empty()
            || !relative
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(schema("invalid input snapshot path"));
        }
        let path = root.join(relative).canonicalize()?;
        if !path.starts_with(&root) || !verified.contains(&path) {
            return Err(schema("unverified input snapshot"));
        }
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 16 * 1024 * 1024 {
            return Err(schema("AtlasFold input exceeds 16 MiB"));
        }
        if format!("{:x}", Sha256::digest(&bytes)) != r.input.sha256 {
            return Err(schema("input snapshot differs from the request input pin"));
        }
        Some(crate::atlasfold::input_sequence(
            &bytes,
            r.input.path.extension().is_some_and(|e| e == "json"),
        )?)
    } else {
        if bundle.input_snapshot.is_some() {
            return Err(schema("input snapshot requires bundle schema 3 or 4"));
        }
        None
    };
    let mut ids = BTreeSet::new();
    let mut archives = BTreeSet::new();
    let mut structures = BTreeSet::new();
    for prediction in &bundle.predictions {
        if prediction.sample >= samples || !ids.insert(prediction.sample) {
            return Err(schema("duplicate or invalid sample ID"));
        }
        if prediction.archive.as_os_str().is_empty()
            || !prediction
                .archive
                .components()
                .all(|c| matches!(c, Component::Normal(_)))
        {
            return Err(schema("invalid prediction archive path"));
        }
        let path = root.join(&prediction.archive).canonicalize()?;
        if !path.starts_with(&root) || !verified.contains(&path) || !archives.insert(path.clone()) {
            return Err(schema("unverified or reused prediction archive"));
        }
        let stored = read_npz(&path, 512 * 1024 * 1024)?;
        if !stored.keys().eq(prediction.arrays.keys()) {
            return Err(schema("tensor descriptor names mismatch"));
        }
        let mut arrays = BTreeMap::new();
        for (name, array) in stored {
            let desc = &prediction.arrays[&name];
            if desc.shape != array.shape() || desc.storage_dtype != array.dtype() {
                return Err(schema("tensor descriptor shape/storage dtype mismatch"));
            }
            let array = if desc.original_dtype == desc.storage_dtype {
                array
            } else if desc.original_dtype == DType::Bfloat16 && desc.storage_dtype == DType::Uint16
            {
                Array::new(DType::Bfloat16, desc.shape.clone(), array.bytes().to_vec())?
            } else {
                return Err(schema("lossy or undeclared tensor dtype conversion"));
            };
            arrays.insert(name, array);
        }
        validate_model(&r.workload, &arrays, prediction.confidence.as_ref())?;
        if bundle.schema_version >= 2 {
            let file = prediction
                .structure
                .as_ref()
                .ok_or_else(|| schema("missing sample structure"))?;
            if file.as_os_str().is_empty()
                || !file.components().all(|c| matches!(c, Component::Normal(_)))
            {
                return Err(schema("invalid structure path"));
            }
            let file = root.join(file).canonicalize()?;
            if !file.starts_with(&root)
                || !verified.contains(&file)
                || !structures.insert(file.clone())
            {
                return Err(schema("unverified or reused structure"));
            }
            let structure = crate::structure::read(&file)?;
            if structure
                .atoms
                .iter()
                .enumerate()
                .any(|(i, a)| a.serial != i + 1)
            {
                return Err(schema(
                    "prediction CIF atom IDs must be consecutive from one",
                ));
            }
            let coords = &arrays["coordinates"];
            match &r.workload {
                Workload::Esmfold2 { .. } => {
                    crate::esmfold2::validate_structure(
                        sequence
                            .as_deref()
                            .ok_or_else(|| schema("missing ESMFold2 input binding"))?,
                        &arrays,
                        &structure,
                    )?;
                }
                Workload::Atlasfold { .. } => {
                    if let Some(sequence) = &sequence {
                        crate::atlasfold::validate_atom14(sequence, coords, &structure)?;
                    } else if structure.atoms.len() > coords.shape()[0] * 14 {
                        return Err(schema("AtlasFold CIF atom count exceeds atom14 storage"));
                    }
                }
                _ => {
                    if structure.coordinates != coords.coordinates()? {
                        return Err(schema("CIF coordinates differ from the sample tensor"));
                    }
                }
            }
        }
    }
    if bundle.schema_version >= 2
        && verified
            .iter()
            .filter(|p| p.extension().is_some_and(|e| e == "cif"))
            .count()
            != structures.len()
    {
        return Err(schema("unassociated CIF artifact"));
    }
    Ok(bundle)
}

/// AtlasFold requires schema 3 input-bound atom14/CIF correspondence. Other
/// models require schema 2. Older schemas remain readable for archive replay.
pub fn validate_current(request: &ValidatedRequest, verified: &[PathBuf]) -> Result<Bundle> {
    let bundle = validate(request, verified)?;
    let expected = match request.as_request().workload {
        Workload::Atlasfold { .. } => 3,
        Workload::Esmfold2 { .. } => 4,
        _ => 2,
    };
    if bundle.schema_version != expected {
        return Err(schema(&format!(
            "current inference requires output bundle schema {expected}"
        )));
    }
    Ok(bundle)
}
fn validate_model(
    workload: &Workload,
    arrays: &BTreeMap<String, Array>,
    summary: Option<&serde_json::Value>,
) -> Result<()> {
    let coords = arrays
        .get("coordinates")
        .ok_or_else(|| schema("missing coordinates"))?;
    if !coords.dtype().is_float() {
        return Err(schema("coordinates must be floating point"));
    }
    match workload {
        Workload::Esmfold2 { .. } => crate::esmfold2::validate_arrays(arrays)?,
        Workload::Atlasfold { .. } => {
            for key in ["plddt", "pae", "ptm"] {
                floating(arrays, key)?;
            }
            let plddt = &arrays["plddt"];
            let pae = &arrays["pae"];
            if coords.shape().len() != 3
                || coords.shape()[1..] != [14, 3]
                || plddt.shape() != [coords.shape()[0]]
                || pae.shape() != [coords.shape()[0], coords.shape()[0]]
                || arrays["ptm"].bytes().len() != arrays["ptm"].dtype().width()
            {
                return Err(schema("AtlasFold confidence shape mismatch"));
            }
        }
        Workload::Openfold3 { .. } => {
            for key in [
                "confidence.plddt",
                "confidence.pde",
                "confidence.pae",
                "confidence.ptm",
                "confidence.iptm",
                "confidence.gpde",
                "confidence.disorder",
                "confidence.has_clash",
                "confidence.sample_ranking_score",
                "confidence.chain_ptm.1",
            ] {
                floating(arrays, key)?;
            }
            let atoms = coords.coordinates()?.len();
            if arrays["confidence.plddt"].shape() != [atoms] {
                return Err(schema("OpenFold3 atom confidence shape mismatch"));
            }
            let pae = arrays["confidence.pae"].shape();
            if pae.len() != 2 || pae[0] != pae[1] || arrays["confidence.pde"].shape() != pae {
                return Err(schema("OpenFold3 pair confidence shape mismatch"));
            }
        }
        Workload::Protenix {
            cycles, model_name, ..
        } => {
            let atoms = coords.coordinates()?.len();
            let summary = summary.ok_or_else(|| schema("missing Protenix confidence JSON"))?;
            confidence(summary, summary)?;
            for key in ["plddt", "gpde", "ptm", "iptm", "ranking_score"] {
                if summary
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .is_none()
                {
                    return Err(schema("missing Protenix summary confidence field"));
                }
            }
            if summary["num_recycles"].as_f64() != Some(f64::from(*cycles)) {
                return Err(schema("Protenix confidence cycle count mismatch"));
            }
            for key in [
                "confidence.atom_plddt",
                "confidence.token_pair_pae",
                "confidence.token_pair_pde",
                "confidence.contact_probs",
            ] {
                floating(arrays, key)?;
            }
            let pae = arrays["confidence.token_pair_pae"].shape();
            if model_name == "protenix_mini_esm_v0.5.0" {
                let esm = floating(arrays, "input.esm_token_embedding")?;
                if pae.len() != 2 || esm.dtype() != DType::Float32 || esm.shape() != [pae[0], 2560]
                {
                    return Err(schema("Mini-ESM input embedding shape/dtype mismatch"));
                }
            }
            if arrays["confidence.atom_plddt"].shape() != [atoms]
                || pae.len() != 2
                || pae[0] != pae[1]
                || arrays["confidence.token_pair_pde"].shape() != pae
                || arrays["confidence.contact_probs"].shape() != pae
            {
                return Err(schema("Protenix confidence shape mismatch"));
            }
            for (key, count) in [
                ("confidence.atom_to_token_idx", atoms),
                ("confidence.atom_is_polymer", atoms),
                ("confidence.token_asym_id", pae[0]),
                ("confidence.token_has_frame", pae[0]),
            ] {
                if arrays.get(key).is_none_or(|a| a.shape() != [count]) {
                    return Err(schema("Protenix atom/token metadata shape mismatch"));
                }
            }
        }
    }
    Ok(())
}

fn floating<'a>(arrays: &'a BTreeMap<String, Array>, key: &str) -> Result<&'a Array> {
    let a = arrays
        .get(key)
        .ok_or_else(|| schema(&format!("missing required output {key}")))?;
    if !a.dtype().is_float() {
        return Err(schema("confidence array must be floating point"));
    }
    Ok(a)
}

#[cfg(test)]
mod protenix_tests {
    use super::*;
    use fireprot_core::request::Precision;
    use serde_json::json;

    #[test]
    fn protenix_requires_complete_native_confidence_and_requested_cycles() {
        let workload = Workload::Protenix {
            model_name: "fixture".into(),
            cycles: 4,
            diffusion_steps: 1,
            samples: 1,
            precision: Precision::Float32,
        };
        let mut arrays = BTreeMap::new();
        for (name, shape) in [
            ("coordinates", vec![3, 3]),
            ("confidence.atom_plddt", vec![3]),
            ("confidence.token_pair_pae", vec![2, 2]),
            ("confidence.token_pair_pde", vec![2, 2]),
            ("confidence.contact_probs", vec![2, 2]),
            ("confidence.atom_to_token_idx", vec![3]),
            ("confidence.atom_is_polymer", vec![3]),
            ("confidence.token_asym_id", vec![2]),
            ("confidence.token_has_frame", vec![2]),
        ] {
            let bytes = vec![0; shape.iter().product::<usize>() * 4];
            arrays.insert(
                name.into(),
                Array::new(DType::Float32, shape, bytes).unwrap(),
            );
        }
        let mut summary =
            json!({"plddt":80.,"gpde":1.,"ptm":0.9,"iptm":0.,"ranking_score":0.9,"num_recycles":4});
        assert!(validate_model(&workload, &arrays, Some(&summary)).is_ok());
        summary["num_recycles"] = json!(3);
        assert!(validate_model(&workload, &arrays, Some(&summary)).is_err());
        summary["num_recycles"] = json!(4);
        let original_plddt = arrays
            .insert(
                "confidence.atom_plddt".into(),
                Array::new(DType::Float32, vec![2], vec![0; 8]).unwrap(),
            )
            .unwrap();
        assert!(validate_model(&workload, &arrays, Some(&summary)).is_err());
        arrays.insert("confidence.atom_plddt".into(), original_plddt);
        assert!(validate_model(&workload, &arrays, Some(&summary)).is_ok());
        arrays.remove("confidence.token_pair_pae");
        assert!(validate_model(&workload, &arrays, Some(&summary)).is_err());
    }
}

#[cfg(test)]
mod mini_esm_tests {
    use super::*;
    use fireprot_core::request::Precision;
    use serde_json::json;

    #[test]
    fn esm_model_cannot_be_accepted_without_complete_fp32_language_features() {
        let workload = Workload::Protenix {
            model_name: "protenix_mini_esm_v0.5.0".into(),
            cycles: 4,
            diffusion_steps: 1,
            samples: 1,
            precision: Precision::Float32,
        };
        let mut arrays = BTreeMap::new();
        for (name, shape) in [
            ("coordinates", vec![3, 3]),
            ("confidence.atom_plddt", vec![3]),
            ("confidence.token_pair_pae", vec![2, 2]),
            ("confidence.token_pair_pde", vec![2, 2]),
            ("confidence.contact_probs", vec![2, 2]),
            ("confidence.atom_to_token_idx", vec![3]),
            ("confidence.atom_is_polymer", vec![3]),
            ("confidence.token_asym_id", vec![2]),
            ("confidence.token_has_frame", vec![2]),
        ] {
            let bytes = vec![0; shape.iter().product::<usize>() * 4];
            arrays.insert(
                name.into(),
                Array::new(DType::Float32, shape, bytes).unwrap(),
            );
        }
        let summary =
            json!({"plddt":80.,"gpde":1.,"ptm":0.9,"iptm":0.,"ranking_score":0.9,"num_recycles":4});
        assert!(validate_model(&workload, &arrays, Some(&summary)).is_err());
        for (dtype, shape, passes) in [
            (DType::Float32, vec![2, 2560], true),
            (DType::Float32, vec![1, 2560], false),
            (DType::Float32, vec![2, 1280], false),
            (DType::Bfloat16, vec![2, 2560], false),
        ] {
            let bytes = vec![0; shape.iter().product::<usize>() * dtype.width()];
            arrays.insert(
                "input.esm_token_embedding".into(),
                Array::new(dtype, shape, bytes).unwrap(),
            );
            assert_eq!(
                validate_model(&workload, &arrays, Some(&summary)).is_ok(),
                passes
            );
        }
    }
}
