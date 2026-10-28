//! Test helpers — in-memory SNIP adapter for integration tests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use omni_store::snip_v2::SnipV2Error;
use omni_store::snip_v2_publish::{
    SnipV2FileInfo, SnipV2ResumeOutcome, VISIBILITY_PUBLIC, public_chunk_count, public_object_root,
};
use omni_store::{SnipV2Adapter, SnipV2Publisher};
use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectId, SnipV2ObjectRef};

/// The account the mock publishes as.
#[allow(dead_code)]
pub const MOCK_ACCOUNT: &str = "MockContributorAccount1111111111";

/// In-memory SnipV2Adapter. `ingest_public` reads the file's bytes,
/// uses their canonical SNIP V2 root as the ID (BLAKE3 of the bytes for
/// anything up to one chunk), and stores. `download_public` looks up by
/// ID and writes to the requested path.
///
/// Issue #149: it also keeps the chain's record of every ingested root
/// (Active, owned by [`MOCK_ACCOUNT`]) and refuses to ingest a root that
/// has one, as the chain does, so the publication path cannot register a
/// root twice and pass. [`MockSnipStore::calls`] counts what was sent.
pub struct MockSnipStore {
    inner: Mutex<HashMap<[u8; 32], Vec<u8>>>,
    records: Mutex<HashMap<[u8; 32], SnipV2FileInfo>>,
    calls: Mutex<MockCalls>,
    faults: Mutex<MockFaults>,
}

/// Calls the mock received.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MockCalls {
    pub ingests: usize,
    pub resumes: usize,
    pub lookups: usize,
}

/// Faults the next calls hit.
#[derive(Debug, Default)]
struct MockFaults {
    /// The chain RPC is unreachable.
    rpc_down: bool,
    /// The next ingest registers the root Pending, then the chain RPC
    /// becomes unreachable and the ingest fails: issue #149's interrupted
    /// run.
    ingest_registers_then_rpc_lost: bool,
}

impl MockSnipStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            records: Mutex::new(HashMap::new()),
            calls: Mutex::new(MockCalls::default()),
            faults: Mutex::new(MockFaults::default()),
        }
    }

    /// What the mock received so far.
    #[allow(dead_code)]
    pub fn calls(&self) -> MockCalls {
        *self.calls.lock().unwrap()
    }

    /// The chain's record of `root`, if any.
    #[allow(dead_code)]
    pub fn record(&self, root: &SnipV2ObjectId) -> Option<SnipV2FileInfo> {
        self.records.lock().unwrap().get(root.as_bytes()).cloned()
    }

    /// Put a chain record for `bytes` (owner, lifecycle) without storing
    /// the bytes, then let `edit` change it.
    #[allow(dead_code)]
    pub fn put_record(
        &self,
        bytes: &[u8],
        owner: &str,
        lifecycle: u8,
        edit: impl FnOnce(&mut SnipV2FileInfo),
    ) -> SnipV2ObjectId {
        let root = public_object_root(bytes);
        let mut info = record_for(bytes, owner, lifecycle);
        edit(&mut info);
        self.records.lock().unwrap().insert(*root.as_bytes(), info);
        root
    }

    /// Make the chain RPC unreachable (or reachable again).
    #[allow(dead_code)]
    pub fn set_rpc_down(&self, down: bool) {
        self.faults.lock().unwrap().rpc_down = down;
    }

    /// The next ingest registers its root Pending, then the chain RPC
    /// becomes unreachable (until [`Self::set_rpc_down`]`(false)`) and the
    /// ingest fails.
    #[allow(dead_code)]
    pub fn lose_rpc_after_next_registration(&self) {
        self.faults.lock().unwrap().ingest_registers_then_rpc_lost = true;
    }

    /// `#[allow(dead_code)]` because some integration test crates
    /// (e.g. `handoff_integration`) compile the same `common`
    /// module but don't exercise this helper directly. Other
    /// test crates do.
    #[allow(dead_code)]
    pub fn insert_bytes(&self, bytes: &[u8]) -> SnipV2ObjectId {
        let id = *public_object_root(bytes).as_bytes();
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

fn record_for(bytes: &[u8], owner: &str, lifecycle: u8) -> SnipV2FileInfo {
    SnipV2FileInfo {
        merkle_root: public_object_root(bytes).to_hex(),
        owner: owner.to_string(),
        plaintext_size_bytes: bytes.len() as u64,
        stored_size_bytes: bytes.len() as u64,
        chunk_count: public_chunk_count(bytes.len() as u64),
        created_at: 10,
        activated_at_height: (lifecycle == 1).then_some(11),
        abandoned_at_height: (lifecycle == 2).then_some(99),
        visibility: VISIBILITY_PUBLIC,
        lifecycle,
    }
}

impl SnipV2Adapter for MockSnipStore {
    fn ingest_public(&self, path: &Path) -> Result<SnipV2ObjectRef, SnipV2Error> {
        self.calls.lock().unwrap().ingests += 1;
        let bytes = std::fs::read(path).map_err(SnipV2Error::CommandSpawn)?;
        let root = public_object_root(&bytes);
        let id_bytes = *root.as_bytes();
        {
            let mut records = self.records.lock().unwrap();
            if records.contains_key(&id_bytes) {
                // What the chain answers `RegisterFilePendingV2` for a root it
                // holds, and what `sum-node ingest-v2` 0.4.0-rc4 then does.
                return Err(SnipV2Error::OutcomeMissing {
                    command: "ingest-v2",
                    cause: "SNIP V2 ingest stdout is missing required `merkle_root:` line".into(),
                    diagnostics: "log: WARN V2 ingest FAILED before chain registration \
                                  source=RegisterFilePendingV2 failed: merkle_root already \
                                  registered as V2 file"
                        .into(),
                });
            }
            let mut faults = self.faults.lock().unwrap();
            if std::mem::take(&mut faults.ingest_registers_then_rpc_lost) {
                faults.rpc_down = true;
                records.insert(id_bytes, record_for(&bytes, MOCK_ACCOUNT, 0));
                self.inner.lock().unwrap().insert(id_bytes, bytes);
                return Err(SnipV2Error::OutcomeMissing {
                    command: "ingest-v2",
                    cause: "SNIP V2 ingest stdout is missing required `merkle_root:` line".into(),
                    diagnostics: "log: WARN RegisterFilePendingV2: error sending request".into(),
                });
            }
            records.insert(id_bytes, record_for(&bytes, MOCK_ACCOUNT, 1));
        }
        self.inner.lock().unwrap().insert(id_bytes, bytes);
        Ok(SnipV2ObjectRef {
            merkle_root: root,
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

impl SnipV2Publisher for MockSnipStore {
    fn file_info_v2(&self, root: &SnipV2ObjectId) -> Result<Option<SnipV2FileInfo>, SnipV2Error> {
        self.calls.lock().unwrap().lookups += 1;
        if self.faults.lock().unwrap().rpc_down {
            return Err(SnipV2Error::ChainLookup {
                url: "http://127.0.0.1:9".into(),
                root: root.to_hex(),
                detail: "storage_getFileInfoV2: Connection refused".into(),
            });
        }
        Ok(self.records.lock().unwrap().get(root.as_bytes()).cloned())
    }

    fn publisher_account(&self) -> Result<String, SnipV2Error> {
        Ok(MOCK_ACCOUNT.to_string())
    }

    fn resume_public(
        &self,
        root: &SnipV2ObjectId,
        path: &Path,
    ) -> Result<SnipV2ResumeOutcome, SnipV2Error> {
        self.calls.lock().unwrap().resumes += 1;
        let bytes = std::fs::read(path).map_err(SnipV2Error::CommandSpawn)?;
        if public_object_root(&bytes) != *root {
            return Ok(SnipV2ResumeOutcome::RootMismatch {
                diagnostics: String::new(),
            });
        }
        let mut records = self.records.lock().unwrap();
        let Some(info) = records.get_mut(root.as_bytes()) else {
            return Err(SnipV2Error::ResumeFailed {
                status: "exit status: 1".into(),
                diagnostics: "resume: storage_getFileInfoV2 returned no row".into(),
            });
        };
        match info.lifecycle {
            0 => {
                info.lifecycle = 1;
                info.activated_at_height = Some(20);
                self.inner.lock().unwrap().insert(*root.as_bytes(), bytes);
                Ok(SnipV2ResumeOutcome::Active)
            }
            1 => Ok(SnipV2ResumeOutcome::Active),
            _ => Ok(SnipV2ResumeOutcome::Abandoned {
                diagnostics: String::new(),
            }),
        }
    }
}
