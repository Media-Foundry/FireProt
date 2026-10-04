//! CPython stable-ABI bridge: metadata and capability callbacks only. Device
//! tensors and implementation functions stay in the upstream model process.
mod build;

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use fireprot_core::registry::{Implementation, OperationSignature, Registry, RegistryError};
use pyo3::{
    exceptions::{PyException, PyKeyError, PyTypeError, PyValueError},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyTuple,
};

struct Callback {
    function: Py<PyAny>,
}

fn capability(
    callback: &Callback,
    signature: &OperationSignature,
    interruption: &Mutex<Option<PyErr>>,
) -> Result<(), String> {
    let serialized = serde_json::to_string(signature).map_err(|error| error.to_string())?;
    Python::attach(|py| {
        let check = || -> PyResult<Result<(), String>> {
            let value = callback.function.call1(py, (serialized,))?;
            let value = value.bind(py);
            let (supported, reason) = if let Ok(tuple) = value.cast::<PyTuple>() {
                if tuple.len() != 2 {
                    return Err(PyValueError::new_err(
                        "capability tuple must have two items",
                    ));
                }
                (
                    tuple.get_item(0)?.is_truthy()?,
                    tuple.get_item(1)?.str()?.to_str()?.to_owned(),
                )
            } else {
                (value.is_truthy()?, "capability returned false".into())
            };
            Ok(if supported { Ok(()) } else { Err(reason) })
        };
        check().unwrap_or_else(|error| {
            if error.is_instance_of::<PyException>(py) {
                Err(format!("capability error: {error}"))
            } else {
                // KeyboardInterrupt/SystemExit must escape dispatch. This slot
                // is consumed immediately by resolve_checked before caching.
                *interruption.lock().expect("interruption lock poisoned") = Some(error);
                Err("capability interrupted".into())
            }
        })
    })
}

fn registry_error(error: RegistryError) -> PyErr {
    match error {
        RegistryError::Unknown => PyKeyError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

/// Python owns synchronization across threads. PyO3 exclusive borrows reject
/// reentrant mutation during a capability callback. Callbacks are visited by
/// CPython GC; the core closures hold weak references, avoiding hidden cycles.
#[pyclass(module = "_fireprot_bridge")]
#[derive(Default)]
struct OperatorRegistry {
    core: Registry,
    callbacks: BTreeMap<(String, String), Arc<Callback>>,
    interruption: Arc<Mutex<Option<PyErr>>>,
}

#[pymethods]
impl OperatorRegistry {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn register(&mut self, py: Python<'_>, metadata: &str, supports: Py<PyAny>) -> PyResult<()> {
        let metadata: Implementation = serde_json::from_str(metadata)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if !supports.bind(py).is_callable() {
            return Err(PyTypeError::new_err("supports must be callable"));
        }
        let key = (metadata.operation.clone(), metadata.name.clone());
        let callback = Arc::new(Callback { function: supports });
        let weak = Arc::downgrade(&callback);
        let interruption = Arc::clone(&self.interruption);
        self.core
            .register(metadata, move |signature| {
                let callback = weak.upgrade().ok_or("capability callback was released")?;
                capability(&callback, signature, &interruption)
            })
            .map_err(registry_error)?;
        self.callbacks.insert(key, callback);
        Ok(())
    }

    fn unregister(&mut self, operation: &str, name: &str) -> PyResult<()> {
        self.core
            .unregister(operation, name)
            .map_err(registry_error)?;
        self.callbacks.remove(&(operation.into(), name.into()));
        Ok(())
    }

    fn resolve(&mut self, signature: &str) -> PyResult<String> {
        let signature: OperationSignature = serde_json::from_str(signature)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let selection = self.core.resolve_checked(&signature, || {
            self.interruption
                .lock()
                .expect("interruption lock poisoned")
                .take()
                .map_or(Ok(()), Err)
        })?;
        serde_json::to_string(&selection).map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn manifest(&self) -> PyResult<String> {
        serde_json::to_string(&self.core.manifest())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn clear_runtime_state(&mut self) {
        self.core.clear_runtime_state();
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for callback in self.callbacks.values() {
            visit.call(&callback.function)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.core = Registry::default();
        self.callbacks.clear();
    }
}

#[pymodule]
fn _fireprot_bridge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ABI_VERSION", 1)?;
    module.add_class::<OperatorRegistry>()?;
    module.add_function(wrap_pyfunction!(build::cached_build, module)?)?;
    module.add_function(wrap_pyfunction!(build::accept_kernel, module)?)?;
    module.add_function(wrap_pyfunction!(build::build_torch, module)?)?;
    Ok(())
}
