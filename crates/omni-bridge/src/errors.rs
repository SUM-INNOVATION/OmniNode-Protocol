use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::create_exception;

create_exception!(omninode, PyOmniError, PyException, "Base exception for OmniNode errors.");
create_exception!(omninode, PyStoreError, PyOmniError, "Exception for shard storage errors.");

pub fn anyhow_to_pyerr<T>(r: anyhow::Result<T>) -> PyResult<T> {
    r.map_err(|e: anyhow::Error| PyOmniError::new_err(e.to_string()))
}
