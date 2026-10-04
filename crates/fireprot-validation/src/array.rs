use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Float16,
    Bfloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
}
impl DType {
    pub fn width(self) -> usize {
        match self {
            Self::Float16 | Self::Bfloat16 | Self::Int16 | Self::Uint16 => 2,
            Self::Float32 | Self::Int32 | Self::Uint32 => 4,
            Self::Float64 | Self::Int64 | Self::Uint64 => 8,
            _ => 1,
        }
    }
    pub fn is_float(self) -> bool {
        matches!(
            self,
            Self::Float16 | Self::Bfloat16 | Self::Float32 | Self::Float64
        )
    }
}
/// Canonical little-endian, C-order bytes. Shape and dtype remain part of identity.
#[derive(Debug, Clone)]
pub struct Array {
    dtype: DType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}
impl Array {
    pub fn new(dtype: DType, shape: Vec<usize>, bytes: Vec<u8>) -> Result<Self> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or(Error::Shape)?;
        if count == 0 || count.checked_mul(dtype.width()) != Some(bytes.len()) {
            return Err(Error::Shape);
        }
        let result = Self {
            dtype,
            shape,
            bytes,
        };
        if dtype.is_float() && result.values()?.iter().any(|x| !x.is_finite()) {
            return Err(Error::Nonfinite);
        }
        if dtype == DType::Bool && result.bytes.iter().any(|&x| x > 1) {
            return Err(Error::Schema("invalid boolean byte".into()));
        }
        Ok(result)
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn values(&self) -> Result<Vec<f64>> {
        if !self.dtype.is_float() {
            return Err(Error::Schema("floating array required".into()));
        }
        Ok(self
            .bytes
            .chunks_exact(self.dtype.width())
            .map(|b| match self.dtype {
                DType::Float32 => f32::from_le_bytes(b.try_into().unwrap()) as f64,
                DType::Float64 => f64::from_le_bytes(b.try_into().unwrap()),
                DType::Bfloat16 => {
                    f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16) as f64
                }
                DType::Float16 => {
                    let bits = u16::from_le_bytes(b.try_into().unwrap());
                    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
                    let e = (bits >> 10) & 31;
                    let m = (bits & 1023) as f64;
                    sign * match e {
                        0 => 2f64.powi(-14) * m / 1024.0,
                        31 => {
                            if m == 0.0 {
                                f64::INFINITY
                            } else {
                                f64::NAN
                            }
                        }
                        _ => 2f64.powi(e as i32 - 15) * (1.0 + m / 1024.0),
                    }
                }
                _ => unreachable!(),
            })
            .collect())
    }
    pub fn coordinates(&self) -> Result<Vec<[f64; 3]>> {
        if self.shape.len() != 2 || self.shape[1] != 3 {
            return Err(Error::Shape);
        }
        Ok(self
            .values()?
            .chunks_exact(3)
            .map(|x| [x[0], x[1], x[2]])
            .collect())
    }
}
#[derive(Debug, Clone, Copy)]
pub enum Policy {
    Exact,
    Absolute { tolerance: f64 },
}
#[derive(Debug, Serialize)]
pub struct ArrayComparison {
    pub bitwise_equal: bool,
    pub max_abs_delta: Option<f64>,
    pub passed: bool,
}
pub fn compare(reference: &Array, candidate: &Array, policy: Policy) -> Result<ArrayComparison> {
    if reference.dtype != candidate.dtype || reference.shape != candidate.shape {
        return Err(Error::Schema("array shape/dtype mismatch".into()));
    }
    let exact = reference.bytes == candidate.bytes;
    let delta = if reference.dtype.is_float() {
        let value = reference
            .values()?
            .iter()
            .zip(candidate.values()?)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if !value.is_finite() {
            return Err(Error::Nonfinite);
        }
        Some(value)
    } else {
        None
    };
    let passed = match policy {
        Policy::Exact => exact,
        Policy::Absolute { tolerance } => {
            validate_tolerance(tolerance)?;
            delta.ok_or_else(|| {
                Error::Schema("approximate integer comparison is unsupported".into())
            })? <= tolerance
        }
    };
    Ok(ArrayComparison {
        bitwise_equal: exact,
        max_abs_delta: delta,
        passed,
    })
}
pub(crate) fn validate_tolerance(t: f64) -> Result<()> {
    if t.is_finite() && t >= 0.0 {
        Ok(())
    } else {
        Err(Error::Tolerance)
    }
}
