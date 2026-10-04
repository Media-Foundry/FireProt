//! Linux/Unix model-worker supervision. Model execution remains in pinned workers.
pub mod artifacts;
#[cfg(unix)]
pub mod supervisor;
