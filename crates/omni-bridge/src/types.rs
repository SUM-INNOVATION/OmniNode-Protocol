//! #[pyclass] wrappers for omni-types structs.

use std::path::PathBuf;

use pyo3::prelude::*;

use omni_types::config::{NetConfig, StoreConfig};
use omni_types::model::{LayerRange, ModelManifest, ShardDescriptor};

// ── NetConfig ────────────────────────────────────────────────────────────────

#[pyclass(name = "NetConfig", from_py_object)]
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

#[pyclass(name = "StoreConfig", from_py_object)]
#[derive(Clone)]
pub struct PyStoreConfig {
    pub inner: StoreConfig,
}

#[pymethods]
impl PyStoreConfig {
    #[new]
    #[pyo3(signature = (store_dir=None, layers_per_shard=None, max_shard_msg_bytes=None, max_shard_total_bytes=None))]
    fn new(
        store_dir: Option<String>,
        layers_per_shard: Option<u32>,
        max_shard_msg_bytes: Option<usize>,
        max_shard_total_bytes: Option<u64>,
    ) -> PyResult<Self> {
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
        if let Some(t) = max_shard_total_bytes {
            cfg.max_shard_total_bytes = t;
        }

        // One validation entry point, shared with the Rust constructors, so a
        // caller that changes only `max_shard_msg_bytes` still gets the whole
        // configuration checked rather than just the field it touched.
        cfg.validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        Ok(Self { inner: cfg })
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

    #[getter]
    fn max_shard_total_bytes(&self) -> u64 {
        self.inner.max_shard_total_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "StoreConfig(store_dir='{}', layers_per_shard={}, max_shard_msg_bytes={}, max_shard_total_bytes={})",
            self.inner.store_dir.display(),
            self.inner.layers_per_shard,
            self.inner.max_shard_msg_bytes,
            self.inner.max_shard_total_bytes,
        )
    }
}

// ── LayerRange ───────────────────────────────────────────────────────────────

#[pyclass(name = "LayerRange", skip_from_py_object)]
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

#[pyclass(name = "ShardDescriptor", skip_from_py_object)]
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

#[pyclass(name = "ModelManifest", skip_from_py_object)]
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
#[cfg(test)]
mod tests {
    //! Constructor, validation and JSON behaviour of the config and model
    //! wrappers. Only `is_err()` is asserted on refusals: formatting a `PyErr`
    //! needs a Python interpreter, which these tests do not have. The Python
    //! suite (`tests/python/`) checks the exception types and messages.

    use super::*;
    use crate::test_support::ok;
    use omni_types::model::{LayerRange, ModelManifest, ShardDescriptor};

    #[test]
    fn net_config_defaults_and_port_override() {
        let d = PyNetConfig::new(None);
        assert_eq!(d.listen_port(), NetConfig::default().listen_port);
        let p = PyNetConfig::new(Some(4242));
        assert_eq!(p.listen_port(), 4242);
        assert_eq!(p.__repr__(), "NetConfig(listen_port=4242)");
    }

    #[test]
    fn store_config_applies_every_override() {
        let c = ok(
            PyStoreConfig::new(Some("/tmp/omni-x".into()), Some(2), Some(1024), Some(4096)),
            "valid configuration",
        );
        assert_eq!(c.store_dir(), "/tmp/omni-x");
        assert_eq!(
            (
                c.layers_per_shard(),
                c.max_shard_msg_bytes(),
                c.max_shard_total_bytes()
            ),
            (2, 1024, 4096)
        );
        assert!(c.__repr__().contains("store_dir='/tmp/omni-x'"));
    }

    #[test]
    fn store_config_refuses_zero_limits_through_the_shared_validator() {
        assert!(PyStoreConfig::new(None, None, Some(0), None).is_err());
        assert!(PyStoreConfig::new(None, None, None, Some(0)).is_err());
        // The defaults themselves validate.
        assert!(PyStoreConfig::new(None, None, None, None).is_ok());
    }

    #[test]
    fn layer_range_length_is_inclusive() {
        let r = PyLayerRange::new(3, 7);
        assert_eq!((r.start(), r.end(), r.__len__()), (3, 7, 5));
        assert_eq!(PyLayerRange::new(4, 4).__len__(), 1);
        assert_eq!(r.__repr__(), "LayerRange(start=3, end=7)");
    }

    fn manifest() -> ModelManifest {
        ModelManifest {
            model_name: "m".into(),
            model_hash: "h".repeat(64),
            architecture: "llama".into(),
            total_layers: 2,
            quantization: "F16".into(),
            total_size_bytes: 128,
            gguf_version: 3,
            shards: vec![ShardDescriptor {
                shard_index: 0,
                cid: "bafy0".into(),
                layer_range: LayerRange { start: 0, end: 1 },
                includes_embedding: true,
                includes_output_head: true,
                size_bytes: 128,
                blake3_hash: "b".repeat(64),
                snip_v2: None,
            }],
            snip_v2: None,
        }
    }

    #[test]
    fn manifest_json_round_trips_through_the_wrapper() {
        let m = PyModelManifest { inner: manifest() };
        let json = ok(m.to_json(), "serialises");
        let back = ok(PyModelManifest::from_json(&json), "parses");
        assert_eq!(back.model_name(), "m");
        assert_eq!(back.architecture(), "llama");
        assert_eq!(
            (back.total_layers(), back.gguf_version(), back.__len__()),
            (2, 3, 1)
        );
        let s = &back.shards()[0];
        assert_eq!(
            (s.shard_index(), s.cid(), s.size_bytes()),
            (0, "bafy0", 128)
        );
        assert_eq!((s.layer_range().start(), s.layer_range().end()), (0, 1));
        assert!(s.includes_embedding() && s.includes_output_head());
        assert_eq!(back.to_json().ok(), Some(json));
    }

    #[test]
    fn manifest_from_json_refuses_malformed_input() {
        assert!(PyModelManifest::from_json("").is_err());
        assert!(PyModelManifest::from_json("{\"model_name\": 5}").is_err());
        assert!(PyModelManifest::from_json("[]").is_err());
    }
}
