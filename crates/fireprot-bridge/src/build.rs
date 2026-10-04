//! Slow build operations release the GIL, including waiting for another builder.
use fireprot_runtime::{build::BuildSpec, build_cache::get_or_build};
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use std::{cell::Cell, io, path::PathBuf};

thread_local! { static BUILDING: Cell<bool> = const { Cell::new(false) }; }

struct BuildGuard;
impl Drop for BuildGuard {
    fn drop(&mut self) {
        BUILDING.with(|value| value.set(false));
    }
}

/// The callback receives a staging directory and unique module/cache name,
/// and returns the flat filenames of finished libraries. It never decides hits.
#[pyfunction]
pub fn cached_build(
    py: Python<'_>,
    specification: &str,
    root: PathBuf,
    callback: Py<PyAny>,
) -> PyResult<String> {
    if !callback.bind(py).is_callable() {
        return Err(PyTypeError::new_err("build callback must be callable"));
    }
    let specification: BuildSpec = serde_json::from_str(specification)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if BUILDING.with(|value| value.replace(true)) {
        return Err(PyRuntimeError::new_err(
            "build callbacks must not reenter cached_build",
        ));
    }
    let _guard = BuildGuard;
    py.detach(move || {
        let identity = specification
            .identify()
            .map_err(|error| PyOSError::new_err(error.to_string()))?;
        let name = identity
            .cache_name()
            .map_err(|error| PyOSError::new_err(error.to_string()))?;
        let mut callback_error = None;
        let result = get_or_build(&root, &identity, |directory| {
            let result = Python::attach(|py| {
                callback
                    .call1(py, (directory.to_path_buf(), name.clone()))?
                    .extract::<Vec<String>>(py)
            });
            result.map_err(|error| {
                callback_error = Some(error);
                io::Error::other("Python build callback failed")
            })
        });
        if let Some(error) = callback_error {
            return Err(error);
        }
        let result = result.map_err(|error| PyOSError::new_err(error.to_string()))?;
        serde_json::to_string(&serde_json::json!({
            "directory": result.directory,
            "cache_hit": result.cache_hit,
            "receipt": result.receipt,
        }))
        .map_err(|error| PyValueError::new_err(error.to_string()))
    })
}
