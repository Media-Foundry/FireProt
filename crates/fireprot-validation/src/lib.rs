//! Model-independent numeric validation with explicit comparison policies.
pub mod array;
pub mod atlasfold;
pub mod bundle;
mod esmfold2;
pub mod metrics;
pub mod numpy;
pub mod paired;
pub mod structure;
pub mod timing;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid, empty or mismatched array shape/byte count")]
    Shape,
    #[error("nonfinite numeric value or arithmetic overflow")]
    Nonfinite,
    #[error("schema error: {0}")]
    Schema(String),
    #[error("tolerance must be finite and nonnegative")]
    Tolerance,
    #[error("coordinate SVD did not converge")]
    Decomposition,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

mod plm;
