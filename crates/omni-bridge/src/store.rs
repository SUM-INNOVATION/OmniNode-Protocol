//! #[pyclass] wrapper for OmniStore.

use std::path::Path;

use pyo3::prelude::*;

use omni_store::OmniStore;
use omni_types::config::StoreConfig;

use crate::errors::PyStoreError;
use crate::shard_view::PyShardView;
use crate::types::{PyModelManifest, PyStoreConfig};

#[pyclass(name = "OmniStore")]
pub struct PyOmniStore {
    inner: OmniStore,
}

#[pymethods]
impl PyOmniStore {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyStoreConfig>) -> PyResult<Self> {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        let store = OmniStore::new(cfg).map_err(|e| PyStoreError::new_err(e.to_string()))?;
        Ok(Self { inner: store })
    }

    /// Ingest a GGUF model: parse, chunk, store shards, build manifest.
    fn ingest_model(&self, path: &str) -> PyResult<PyModelManifest> {
        let manifest = self
            .inner
            .ingest_model(Path::new(path))
            .map_err(|e| PyStoreError::new_err(e.to_string()))?;
        Ok(PyModelManifest { inner: manifest })
    }

    /// Check whether a shard exists locally.
    fn has_shard(&self, cid: &str) -> bool {
        self.inner.has_shard(cid)
    }

    /// Zero-copy memory-mapped view of a local shard.
    fn mmap_shard(&self, cid: &str) -> PyResult<PyShardView> {
        let mmap = self
            .inner
            .mmap_shard(cid)
            .map_err(|e| PyStoreError::new_err(e.to_string()))?;
        Ok(PyShardView::new(mmap, cid.to_string()))
    }

    /// Read shard into bytes (copies data — use mmap_shard for zero-copy).
    fn get_shard(&self, cid: &str) -> PyResult<Vec<u8>> {
        self.inner
            .local
            .get(cid)
            .map_err(|e| PyStoreError::new_err(e.to_string()))
    }

    /// Root directory of the shard store.
    #[getter]
    fn store_dir(&self) -> String {
        self.inner.config.store_dir.display().to_string()
    }

    fn __repr__(&self) -> String {
        format!("OmniStore(dir='{}')", self.inner.config.store_dir.display())
    }
}
