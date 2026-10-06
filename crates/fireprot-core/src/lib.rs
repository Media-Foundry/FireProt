//! Model-independent contracts. This crate neither imports model runtimes nor
//! initializes a GPU. A valid request describes work; it does not prove that a
//! particular machine, checkpoint, or accelerated implementation can execute it.
pub mod protocol;
pub mod registry;
pub mod request;
pub mod workspace;

pub use request::{Request, RequestError, ValidatedRequest};

/// Wire contract version. Unknown versions are rejected before launching a worker.
pub const SCHEMA_VERSION: u32 = 1;
