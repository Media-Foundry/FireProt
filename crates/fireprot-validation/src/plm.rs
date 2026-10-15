//! Standalone sequence outputs. Retain BOS/EOS, native dtype and exact token IDs.
use crate::{
    Error, Result,
    array::{Array, DType},
};
use std::collections::BTreeMap;

pub(crate) fn validate(
    model: &str,
    sequence: &str,
    arrays: &BTreeMap<String, Array>,
) -> Result<()> {
    let fail = || Error::Schema("standalone pLM shape/dtype/token contract mismatch".into());
    let (width, vocab, dtype, logits) = match model {
        "esmc-fold2-6b" => (2560, 64, DType::Bfloat16, false),
        "esm2-3b" => (2560, 33, DType::Float32, true),
        "esm3-open" => (1536, 64, DType::Bfloat16, true),
        "atlaslm-3b" => (2304, 64, DType::Float32, true),
        _ => return Err(fail()),
    };
    if sequence.is_empty() || sequence.len() > 1000 || arrays.len() != if logits { 3 } else { 2 } {
        return Err(fail());
    }
    let embeddings = arrays.get("embeddings").ok_or_else(fail)?;
    let tokens = arrays.get("tokens").ok_or_else(fail)?;
    let length = sequence.len() + 2;
    if embeddings.shape() != [1, length, width]
        || embeddings.dtype() != dtype
        || tokens.shape() != [1, length]
        || tokens.dtype() != DType::Int64
    {
        return Err(fail());
    }
    // Shared native ESM vocabulary prefix. AtlasLM uses the same IDs for these residues.
    let alphabet = b"LAGVSERTIDPKQNFYMHWCX";
    let mut expected = vec![0i64];
    for residue in sequence.bytes() {
        expected.push(
            4 + alphabet
                .iter()
                .position(|&c| c == residue)
                .ok_or_else(fail)? as i64,
        );
    }
    expected.push(2);
    if !tokens
        .bytes()
        .chunks_exact(8)
        .zip(expected)
        .all(|(b, id)| i64::from_le_bytes(b.try_into().unwrap()) == id)
    {
        return Err(fail());
    }
    if logits {
        let value = arrays.get("logits").ok_or_else(fail)?;
        if value.shape() != [1, length, vocab]
            || value.dtype()
                != if model == "atlaslm-3b" {
                    DType::Bfloat16
                } else {
                    dtype
                }
        {
            return Err(fail());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> BTreeMap<String, Array> {
        BTreeMap::from([
            (
                "tokens".into(),
                Array::new(
                    DType::Int64,
                    vec![1, 4],
                    [0i64, 5, 23, 2]
                        .into_iter()
                        .flat_map(i64::to_le_bytes)
                        .collect(),
                )
                .unwrap(),
            ),
            (
                "embeddings".into(),
                Array::new(DType::Bfloat16, vec![1, 4, 2560], vec![0; 4 * 2560 * 2]).unwrap(),
            ),
        ])
    }
    #[test]
    fn binds_native_embeddings_to_residues_and_special_tokens() {
        assert!(validate("esmc-fold2-6b", "AC", &fixture()).is_ok());
        assert!(validate("esmc-fold2-6b", "CA", &fixture()).is_err());
        assert!(validate("esmc-fold2-6b", "A", &fixture()).is_err());
        assert!(validate("esm2-3b", "AC", &fixture()).is_err());
        let mut bad = fixture();
        bad.insert(
            "logits".into(),
            Array::new(DType::Bfloat16, vec![1, 4, 64], vec![0; 512]).unwrap(),
        );
        assert!(validate("esmc-fold2-6b", "AC", &bad).is_err());
        bad.remove("logits");
        bad.insert(
            "embeddings".into(),
            Array::new(DType::Float32, vec![1, 4, 2560], vec![0; 4 * 2560 * 4]).unwrap(),
        );
        assert!(validate("esmc-fold2-6b", "AC", &bad).is_err());
    }
}
