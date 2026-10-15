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
#[cfg(test)]
mod tests {
    //! The store wrapper end to end on a synthetic two-block GGUF: ingestion,
    //! shard presence, copied reads, zero-copy views and refusals. Every store
    //! lives in its own temporary directory, never the default under `$HOME`.
    //! Refusals are checked with `is_err()` only (no Python interpreter here);
    //! the Python suite checks their exception types.

    use std::path::PathBuf;

    use super::*;
    use crate::test_support::ok;

    /// A temporary directory removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let n = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let p =
                std::env::temp_dir().join(format!("omni-bridge-{tag}-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn put_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// GGUF v3 with two blocks, one 1-D F16 tensor each: 64 bytes of 0xAA
    /// for block 0, then 64 bytes of 0xBB for block 1.
    fn synthetic_gguf() -> Vec<u8> {
        let mut raw = Vec::new();
        raw.extend_from_slice(b"GGUF");
        raw.extend_from_slice(&3u32.to_le_bytes());
        raw.extend_from_slice(&2u64.to_le_bytes()); // tensors
        raw.extend_from_slice(&1u64.to_le_bytes()); // metadata entries
        put_str(&mut raw, "general.architecture");
        raw.extend_from_slice(&8u32.to_le_bytes()); // STRING
        put_str(&mut raw, "llama");
        for (name, offset) in [("blk.0.attn_q.weight", 0u64), ("blk.1.attn_q.weight", 64)] {
            put_str(&mut raw, name);
            raw.extend_from_slice(&1u32.to_le_bytes()); // 1-D
            raw.extend_from_slice(&32u64.to_le_bytes()); // 32 F16 elements
            raw.extend_from_slice(&1u32.to_le_bytes()); // F16
            raw.extend_from_slice(&offset.to_le_bytes());
        }
        while raw.len() % 32 != 0 {
            raw.push(0);
        }
        raw.extend_from_slice(&[0xAA; 64]);
        raw.extend_from_slice(&[0xBB; 64]);
        raw
    }

    fn store_in(dir: &TempDir) -> PyOmniStore {
        let cfg = StoreConfig {
            store_dir: dir.0.join("store"),
            layers_per_shard: 1,
            ..StoreConfig::default()
        };
        ok(
            PyOmniStore::new(Some(PyStoreConfig { inner: cfg })),
            "store opens",
        )
    }

    fn shard_files(dir: &TempDir) -> usize {
        std::fs::read_dir(dir.0.join("store"))
            .map(|it| {
                it.filter(|e| {
                    e.as_ref()
                        .map(|e| e.path().extension().is_some_and(|x| x == "shard"))
                        .unwrap_or(false)
                })
                .count()
            })
            .unwrap_or(0)
    }

    #[test]
    fn ingest_writes_one_shard_per_block_with_exactly_its_bytes() {
        let dir = TempDir::new("ingest");
        let model = dir.0.join("model.gguf");
        std::fs::write(&model, synthetic_gguf()).unwrap();
        let store = store_in(&dir);
        assert_eq!(store.store_dir(), dir.0.join("store").display().to_string());

        let m = ok(store.ingest_model(model.to_str().unwrap()), "ingests");
        let m = &m.inner;
        assert_eq!(
            (m.architecture.as_str(), m.gguf_version, m.shards.len()),
            ("llama", 3, 2)
        );
        for (i, (s, byte)) in m.shards.iter().zip([0xAAu8, 0xBB]).enumerate() {
            assert_eq!(s.shard_index, i as u32);
            assert_eq!(
                (s.layer_range.start, s.layer_range.end),
                (i as u32, i as u32)
            );
            assert_eq!(s.size_bytes, 64);
            assert!(store.has_shard(&s.cid));
            let bytes = ok(store.get_shard(&s.cid), "shard reads");
            assert_eq!(bytes, vec![byte; 64], "shard {i} holds exactly its tensor");
            assert!(store.mmap_shard(&s.cid).is_ok(), "shard {i} maps");
        }
        assert_ne!(m.shards[0].cid, m.shards[1].cid);
        assert_eq!(shard_files(&dir), 2);
        // The manifest is written beside the shards.
        assert!(dir.0.join("store").join("manifest.cbor").is_file());
    }

    #[test]
    fn missing_shards_are_absent_and_refused() {
        let dir = TempDir::new("missing");
        let store = store_in(&dir);
        let cid = "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku";
        assert!(!store.has_shard(cid));
        assert!(store.get_shard(cid).is_err());
        assert!(store.mmap_shard(cid).is_err());
    }

    #[test]
    fn a_malformed_model_is_refused_and_writes_no_shard() {
        let dir = TempDir::new("malformed");
        let store = store_in(&dir);
        let bad = dir.0.join("bad.gguf");
        std::fs::write(&bad, b"GGUF\x03\x00\x00\x00truncated").unwrap();
        assert!(store.ingest_model(bad.to_str().unwrap()).is_err());
        let not_gguf = dir.0.join("not.gguf");
        std::fs::write(&not_gguf, b"definitely not a model").unwrap();
        assert!(store.ingest_model(not_gguf.to_str().unwrap()).is_err());
        assert!(store
            .ingest_model(dir.0.join("absent.gguf").to_str().unwrap())
            .is_err());
        assert_eq!(shard_files(&dir), 0);
    }

    #[test]
    fn an_invalid_configuration_is_refused_before_a_store_is_created() {
        let dir = TempDir::new("invalid");
        let root = dir.0.join("never-created");
        // A configuration that bypassed the Python constructor's validation
        // is still refused by the store, before anything is created on disk.
        let cfg = StoreConfig {
            store_dir: root.clone(),
            max_shard_msg_bytes: 0,
            ..StoreConfig::default()
        };
        assert!(PyOmniStore::new(Some(PyStoreConfig { inner: cfg })).is_err());
        assert!(!root.exists());
    }
}
