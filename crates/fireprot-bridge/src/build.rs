//! Slow build operations release the GIL, including waiting for another builder.
use fireprot_runtime::{build::BuildSpec, build_cache::get_or_build_checked};
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use std::{cell::Cell, io, path::PathBuf, time::Duration};

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
        let mut signal_error = None;
        let result = get_or_build_checked(
            &root,
            &identity,
            Duration::from_secs(600),
            &mut || {
                Python::attach(|py| py.check_signals()).map_err(|error| {
                    signal_error = Some(error);
                    io::Error::new(io::ErrorKind::Interrupted, "Python build interrupted")
                })
            },
            |directory, _| {
                let result = Python::attach(|py| {
                    callback
                        .call1(py, (directory.to_path_buf(), name.clone()))?
                        .extract::<Vec<String>>(py)
                });
                result.map_err(|error| {
                    callback_error = Some(error);
                    io::Error::other("Python build callback failed")
                })
            },
        );
        if let Some(error) = callback_error.or(signal_error) {
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

/// Validate a published artifact without consulting compiler inputs or tools.
#[pyfunction]
pub fn accept_kernel(
    py: Python<'_>,
    directory: PathBuf,
    digest: String,
    compatibility: &str,
) -> PyResult<String> {
    let expected: fireprot_runtime::kernel_bundle::KernelCompatibility =
        serde_json::from_str(compatibility)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    py.detach(move || {
        let accepted = fireprot_runtime::kernel_bundle::accept(&directory, &digest, &expected)
            .map_err(|error| PyOSError::new_err(error.to_string()))?;
        serde_json::to_string(&accepted).map_err(|error| PyValueError::new_err(error.to_string()))
    })
}

/// Compile a declared Torch recipe under Rust process/cache ownership.
#[pyfunction]
pub fn build_torch(py: Python<'_>, specification: &str, root: PathBuf) -> PyResult<String> {
    let specification: fireprot_runtime::torch_build::TorchBuildSpec =
        serde_json::from_str(specification)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if BUILDING.with(|value| value.replace(true)) {
        return Err(PyRuntimeError::new_err(
            "build callbacks must not reenter build_torch",
        ));
    }
    let _guard = BuildGuard;
    py.detach(move || {
        let mut signal_error = None;
        let result = specification.run_checked(&root, &mut || {
            Python::attach(|py| py.check_signals()).map_err(|error| {
                signal_error = Some(error);
                io::Error::new(io::ErrorKind::Interrupted, "Python build interrupted")
            })
        });
        if let Some(error) = signal_error {
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
