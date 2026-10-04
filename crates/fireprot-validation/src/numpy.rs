//! NPY/NPZ loading with explicit dtype, layout and allocation bounds; no pickle.
use crate::{
    Error, Result,
    array::{Array, ArrayComparison, DType, Policy, compare},
};
use npyz::{NpyHeader, Order};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Cursor, Read, Seek, SeekFrom},
    path::Path,
};

pub fn read_npy(bytes: &[u8]) -> Result<Array> {
    let mut reader = Cursor::new(bytes);
    let header = NpyHeader::from_reader(&mut reader)?;
    let npyz::DType::Plain(ty) = header.dtype() else {
        return Err(Error::Schema("structured/object dtype unsupported".into()));
    };
    let descr = ty.to_string();
    let dtype = match &descr[1..] {
        "f2" => DType::Float16,
        "f4" => DType::Float32,
        "f8" => DType::Float64,
        "i1" => DType::Int8,
        "i2" => DType::Int16,
        "i4" => DType::Int32,
        "i8" => DType::Int64,
        "u1" => DType::Uint8,
        "u2" => DType::Uint16,
        "u4" => DType::Uint32,
        "u8" => DType::Uint64,
        "b1" => DType::Bool,
        _ => return Err(Error::Schema(format!("unsupported NPY dtype {descr}"))),
    };
    let shape: Vec<usize> = header
        .shape()
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| Error::Shape))
        .collect::<Result<_>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(Error::Shape)?;
    let width = dtype.width();
    let raw = &bytes[reader.position() as usize..];
    if count == 0 || count.checked_mul(width) != Some(raw.len()) {
        return Err(Error::Shape);
    }
    let mut canonical = vec![0; raw.len()];
    for i in 0..count {
        let source = if header.order() == Order::Fortran {
            let mut remaining = i;
            let mut index = 0;
            let mut stride = count;
            for &dim in shape.iter().rev() {
                stride /= dim;
                index += (remaining % dim) * stride;
                remaining /= dim;
            }
            index
        } else {
            i
        };
        let chunk = &mut canonical[i * width..(i + 1) * width];
        chunk.copy_from_slice(&raw[source * width..(source + 1) * width]);
        if descr.starts_with('>') {
            chunk.reverse();
        }
    }
    Array::new(dtype, shape, canonical)
}
pub fn read_npz(path: &Path, max_uncompressed_bytes: u64) -> Result<BTreeMap<String, Array>> {
    let mut zip =
        zip::ZipArchive::new(File::open(path)?).map_err(|e| Error::Schema(e.to_string()))?;
    verify_central_directory_count(path, zip.central_directory_start(), zip.len())?;
    let mut arrays = BTreeMap::new();
    let mut total = 0u64;
    for i in 0..zip.len() {
        let entry = zip.by_index(i).map_err(|e| Error::Schema(e.to_string()))?;
        let name = entry
            .name()
            .strip_suffix(".npy")
            .ok_or_else(|| Error::Schema("non-NPY member in NPZ".into()))?
            .to_owned();
        if name.is_empty() || arrays.contains_key(&name) {
            return Err(Error::Schema("empty or duplicate NPZ key".into()));
        }
        total = total.checked_add(entry.size()).ok_or(Error::Shape)?;
        if total > max_uncompressed_bytes {
            return Err(Error::Schema("NPZ exceeds allocation limit".into()));
        }
        let expected = entry.size();
        let mut bytes = Vec::new();
        entry
            .take(expected.saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 != expected {
            return Err(Error::Shape);
        }
        arrays.insert(name, read_npy(&bytes)?);
    }
    if arrays.is_empty() {
        return Err(Error::Schema("empty NPZ archive".into()));
    }
    Ok(arrays)
}
pub fn compare_archives(
    reference: &BTreeMap<String, Array>,
    candidate: &BTreeMap<String, Array>,
    policy: Policy,
) -> Result<BTreeMap<String, ArrayComparison>> {
    if reference.is_empty() || !reference.keys().eq(candidate.keys()) {
        return Err(Error::Schema("output array keys mismatch or empty".into()));
    }
    reference
        .iter()
        .map(|(key, array)| Ok((key.clone(), compare(array, &candidate[key], policy)?)))
        .collect()
}
/// Experimental ground truth is exclusively coordinates[mask], never reference.
pub fn ground_truth(mapping: &BTreeMap<String, Array>) -> Result<Vec<[f64; 3]>> {
    let coordinates = mapping
        .get("coordinates")
        .ok_or_else(|| Error::Schema("missing coordinates".into()))?
        .coordinates()?;
    let mask = mapping
        .get("mask")
        .ok_or_else(|| Error::Schema("missing experimental mask".into()))?;
    if mask.dtype() != DType::Bool || mask.shape() != [coordinates.len()] {
        return Err(Error::Shape);
    }
    let result: Vec<_> = coordinates
        .into_iter()
        .zip(mask.bytes())
        .filter_map(|(c, &m)| (m == 1).then_some(c))
        .collect();
    if result.is_empty() {
        return Err(Error::Shape);
    }
    Ok(result)
}

// zip 2.x indexes entries by name and can hide duplicate central-directory rows.
// Count the raw standard central headers before trusting that deduplicated view.
fn verify_central_directory_count(path: &Path, start: u64, expected: usize) -> Result<()> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start))?;
    for _ in 0..expected {
        let mut header = [0u8; 46];
        file.read_exact(&mut header)?;
        if &header[..4] != b"PK\x01\x02" {
            return Err(Error::Schema("invalid ZIP central directory".into()));
        }
        let variable_bytes = [28, 30, 32]
            .into_iter()
            .map(|i| u16::from_le_bytes([header[i], header[i + 1]]) as i64)
            .sum();
        file.seek(SeekFrom::Current(variable_bytes))?;
    }
    let mut next = [0u8; 4];
    file.read_exact(&mut next)?;
    if &next == b"PK\x01\x02" {
        return Err(Error::Schema("duplicate ZIP member names".into()));
    }
    if &next != b"PK\x05\x06" && &next != b"PK\x06\x06" {
        return Err(Error::Schema("unsupported ZIP directory terminator".into()));
    }
    Ok(())
}
