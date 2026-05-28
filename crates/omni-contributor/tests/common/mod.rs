//! Test helpers — in-memory SNIP adapter for integration tests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use omni_store::snip_v2::SnipV2Error;
use omni_store::SnipV2Adapter;
use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectId, SnipV2ObjectRef};

/// In-memory SnipV2Adapter. `ingest_public` reads the file's bytes,
/// uses BLAKE3 of the bytes as the SNIP V2 ID (deterministic stable
/// identifier for opaque-bytes case), and stores. `download_public`
/// looks up by ID and writes to the requested path.
pub struct MockSnipStore {
    inner: Mutex<HashMap<[u8; 32], Vec<u8>>>,
}

impl MockSnipStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// `#[allow(dead_code)]` because some integration test crates
    /// (e.g. `handoff_integration`) compile the same `common`
    /// module but don't exercise this helper directly. Other
    /// test crates do.
    #[allow(dead_code)]
    pub fn insert_bytes(&self, bytes: &[u8]) -> SnipV2ObjectId {
        let id = *blake3::hash(bytes).as_bytes();
        self.inner.lock().unwrap().insert(id, bytes.to_vec());
        SnipV2ObjectId::from_bytes(id)
    }

    /// Returns the number of objects currently stored. Tests use this
    /// to assert that `run_job` does NOT publish anything when it
    /// refuses a job (e.g., bad runner accounting): no orphan SNIP
    /// objects. `#[allow(dead_code)]` because this helper is used
    /// only from `verify_negative_cases.rs`; other test files compile
    /// the same `common` module and would warn without the allow.
    #[allow(dead_code)]
    pub fn object_count(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

impl SnipV2Adapter for MockSnipStore {
    fn ingest_public(&self, path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error> {
        let bytes = std::fs::read(path).map_err(SnipV2Error::CommandSpawn)?;
        let id_bytes = *blake3::hash(&bytes).as_bytes();
        self.inner.lock().unwrap().insert(id_bytes, bytes);
        Ok(SnipV2ObjectRef {
            merkle_root: SnipV2ObjectId::from_bytes(id_bytes),
            lifecycle: SnipV2Lifecycle::Active,
            plaintext_size_bytes: None,
        })
    }

    fn download_public(
        &self,
        root: &SnipV2ObjectId,
        output_path: &Path,
    ) -> Result<(), SnipV2Error> {
        let guard = self.inner.lock().unwrap();
        let bytes = guard
            .get(root.as_bytes())
            .ok_or_else(|| SnipV2Error::InputNotFound {
                path: PathBuf::from(format!("mock-store: no object at {root}")),
            })?;
        std::fs::write(output_path, bytes).map_err(SnipV2Error::CommandSpawn)?;
        Ok(())
    }
}
