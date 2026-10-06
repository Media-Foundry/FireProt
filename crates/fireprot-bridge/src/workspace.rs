//! Tensor owners acknowledge state changes through this additive policy API.
use fireprot_core::workspace::{Binding, WorkspaceError, WorkspaceLease as Lease};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};

fn error(error: WorkspaceError) -> PyErr {
    match error {
        WorkspaceError::Budget | WorkspaceError::Overflow => {
            PyValueError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// The core checks the actual calling Rust thread, not a caller-supplied token.
/// PyO3 exclusive borrows reject concurrent/reentrant mutation. No GPU objects
/// are retained here; release_after_drain is an acknowledgment by the caller.
#[pyclass(module = "_fireprot_bridge")]
pub struct WorkspaceLease {
    core: Lease,
}

#[pymethods]
impl WorkspaceLease {
    #[new]
    fn new(shape: Vec<u64>, element_bytes: u64, budget_bytes: u64) -> PyResult<Self> {
        Ok(Self {
            core: Lease::new(&shape, element_bytes, budget_bytes).map_err(error)?,
        })
    }

    fn reserve(&mut self, device: u32, stream: u64) -> PyResult<()> {
        self.core.reserve(Binding { device, stream }).map_err(error)
    }

    fn allocation_committed(&mut self) -> PyResult<()> {
        self.core.allocation_committed().map_err(error)
    }

    fn rollback_reservation(&mut self) -> PyResult<()> {
        self.core.rollback_reservation().map_err(error)
    }

    fn begin_use(&mut self, device: u32, stream: u64) -> PyResult<()> {
        self.core
            .begin_use(Binding { device, stream })
            .map_err(error)
    }

    fn finish_use(&mut self) -> PyResult<()> {
        self.core.finish_use().map_err(error)
    }

    fn fail(&mut self, reason: String) -> PyResult<()> {
        self.core.fail(reason).map_err(error)
    }

    fn begin_close(&mut self) -> PyResult<()> {
        self.core.begin_close().map_err(error)
    }

    fn release_after_drain(&mut self) -> PyResult<()> {
        self.core.release_after_drain().map_err(error)
    }

    fn manifest(&self) -> PyResult<String> {
        serde_json::to_string(self.core.receipt()).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
