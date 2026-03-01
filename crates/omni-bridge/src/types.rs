//! #[pyclass] wrappers for omni-types structs.

use std::path::PathBuf;

use pyo3::prelude::*;

use omni_types::config::{NetConfig, StoreConfig};
use omni_types::model::{LayerRange, ModelManifest, ShardDescriptor};

// ── NetConfig ────────────────────────────────────────────────────────────────

#[pyclass(name = "NetConfig")]
#[derive(Clone)]
pub struct PyNetConfig {
    pub inner: NetConfig,
}

#[pymethods]
impl PyNetConfig {
    #[new]
    #[pyo3(signature = (listen_port=None))]
    fn new(listen_port: Option<u16>) -> Self {
        let mut cfg = NetConfig::default();
        if let Some(p) = listen_port {
            cfg.listen_port = p;
        }
        Self { inner: cfg }
    }

    #[getter]
    fn listen_port(&self) -> u16 {
        self.inner.listen_port
    }

    fn __repr__(&self) -> String {
        format!("NetConfig(listen_port={})", self.inner.listen_port)
    }
}

// ── StoreConfig ──────────────────────────────────────────────────────────────

#[pyclass(name = "StoreConfig")]
#[derive(Clone)]
pub struct PyStoreConfig {
    pub inner: StoreConfig,
}

#[pymethods]
impl PyStoreConfig {
    #[new]
    #[pyo3(signature = (store_dir=None, layers_per_shard=None, max_shard_msg_bytes=None))]
    fn new(
        store_dir: Option<String>,
        layers_per_shard: Option<u32>,
        max_shard_msg_bytes: Option<usize>,
    ) -> Self {
        let mut cfg = StoreConfig::default();
        if let Some(d) = store_dir {
            cfg.store_dir = PathBuf::from(d);
        }
        if let Some(l) = layers_per_shard {
            cfg.layers_per_shard = l;
        }
        if let Some(m) = max_shard_msg_bytes {
            cfg.max_shard_msg_bytes = m;
        }
        Self { inner: cfg }
    }

    #[getter]
    fn store_dir(&self) -> String {
        self.inner.store_dir.display().to_string()
    }

    #[getter]
    fn layers_per_shard(&self) -> u32 {
        self.inner.layers_per_shard
    }

    #[getter]
    fn max_shard_msg_bytes(&self) -> usize {
        self.inner.max_shard_msg_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "StoreConfig(store_dir='{}', layers_per_shard={}, max_shard_msg_bytes={})",
            self.inner.store_dir.display(),
            self.inner.layers_per_shard,
            self.inner.max_shard_msg_bytes,
        )
    }
}

// ── LayerRange ───────────────────────────────────────────────────────────────

#[pyclass(name = "LayerRange")]
#[derive(Clone)]
pub struct PyLayerRange {
    pub inner: LayerRange,
}

#[pymethods]
impl PyLayerRange {
    #[new]
    fn new(start: u32, end: u32) -> Self {
        Self {
            inner: LayerRange { start, end },
        }
    }

    #[getter]
    fn start(&self) -> u32 {
        self.inner.start
    }

    #[getter]
    fn end(&self) -> u32 {
        self.inner.end
    }

    fn __len__(&self) -> usize {
        self.inner.len() as usize
    }

    fn __repr__(&self) -> String {
        format!("LayerRange(start={}, end={})", self.inner.start, self.inner.end)
    }
}

// ── ShardDescriptor ──────────────────────────────────────────────────────────

#[pyclass(name = "ShardDescriptor")]
#[derive(Clone)]
pub struct PyShardDescriptor {
    pub inner: ShardDescriptor,
}

#[pymethods]
impl PyShardDescriptor {
    #[getter]
    fn shard_index(&self) -> u32 {
        self.inner.shard_index
    }

    #[getter]
    fn cid(&self) -> &str {
        &self.inner.cid
    }

    #[getter]
    fn layer_range(&self) -> PyLayerRange {
        PyLayerRange {
            inner: self.inner.layer_range,
        }
    }

    #[getter]
    fn includes_embedding(&self) -> bool {
        self.inner.includes_embedding
    }

    #[getter]
    fn includes_output_head(&self) -> bool {
        self.inner.includes_output_head
    }

    #[getter]
    fn size_bytes(&self) -> u64 {
        self.inner.size_bytes
    }

    #[getter]
    fn blake3_hash(&self) -> &str {
        &self.inner.blake3_hash
    }

    fn __repr__(&self) -> String {
        format!(
            "ShardDescriptor(index={}, cid='{}', layers={}-{}, size={})",
            self.inner.shard_index,
            self.inner.cid,
            self.inner.layer_range.start,
            self.inner.layer_range.end,
            self.inner.size_bytes,
        )
    }
}

// ── ModelManifest ────────────────────────────────────────────────────────────

#[pyclass(name = "ModelManifest")]
#[derive(Clone)]
pub struct PyModelManifest {
    pub inner: ModelManifest,
}

#[pymethods]
impl PyModelManifest {
    #[getter]
    fn model_name(&self) -> &str {
        &self.inner.model_name
    }

    #[getter]
    fn model_hash(&self) -> &str {
        &self.inner.model_hash
    }

    #[getter]
    fn architecture(&self) -> &str {
        &self.inner.architecture
    }

    #[getter]
    fn total_layers(&self) -> u32 {
        self.inner.total_layers
    }

    #[getter]
    fn quantization(&self) -> &str {
        &self.inner.quantization
    }

    #[getter]
    fn total_size_bytes(&self) -> u64 {
        self.inner.total_size_bytes
    }

    #[getter]
    fn gguf_version(&self) -> u32 {
        self.inner.gguf_version
    }

    #[getter]
    fn shards(&self) -> Vec<PyShardDescriptor> {
        self.inner
            .shards
            .iter()
            .map(|s| PyShardDescriptor { inner: s.clone() })
            .collect()
    }

    /// Serialize to JSON string.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Deserialize from JSON string.
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        let inner: ModelManifest = serde_json::from_str(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelManifest(name='{}', arch='{}', layers={}, shards={})",
            self.inner.model_name,
            self.inner.architecture,
            self.inner.total_layers,
            self.inner.shards.len(),
        )
    }

    fn __len__(&self) -> usize {
        self.inner.shards.len()
    }
}
