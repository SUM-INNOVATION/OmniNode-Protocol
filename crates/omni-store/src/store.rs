//! On-disk shard storage.
//!
//! Layout: `<shard_dir>/<cid>.shard`
//!
//! Files are write-once and content-addressed, so there are no race
//! conditions to worry about — if two writers produce the same CID they
//! write identical bytes.

use std::fs;
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Result, StoreError};
use crate::mmap;

/// Filesystem-backed shard store keyed by CID.
pub struct ShardStore {
    shard_dir: PathBuf,
}

impl ShardStore {
    /// Open (or create) a shard store rooted at `shard_dir`.
    pub fn new(shard_dir: PathBuf) -> Result<Self> {
        fs::create_dir_all(&shard_dir)?;
        Ok(Self { shard_dir })
    }

    /// Path on disk for a given CID.
    pub fn shard_path(&self, cid: &str) -> PathBuf {
        self.shard_dir.join(format!("{cid}.shard"))
    }

    /// Check whether a shard exists locally.
    pub fn has(&self, cid: &str) -> bool {
        self.shard_path(cid).exists()
    }

    /// Write shard data to disk.
    pub fn put(&self, cid: &str, data: &[u8]) -> Result<()> {
        let path = self.shard_path(cid);
        fs::write(&path, data)?;
        Ok(())
    }

    /// Read entire shard into memory.
    /// Prefer [`Self::mmap`] for large shards.
    pub fn get(&self, cid: &str) -> Result<Vec<u8>> {
        let path = self.shard_path(cid);
        if !path.exists() {
            return Err(StoreError::NotFound(cid.to_string()));
        }
        Ok(fs::read(&path)?)
    }

    /// Memory-map a shard for zero-copy read access.
    pub fn mmap(&self, cid: &str) -> Result<Mmap> {
        let path = self.shard_path(cid);
        if !path.exists() {
            return Err(StoreError::NotFound(cid.to_string()));
        }
        mmap::mmap_file(&path)
    }

    /// Write shard data from an existing memory map to disk.
    pub fn put_from_mmap(&self, cid: &str, data: &Mmap) -> Result<()> {
        self.put(cid, data)
    }

    /// Root directory of this store.
    pub fn root(&self) -> &Path {
        &self.shard_dir
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_has_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();

        assert!(!store.has("baftest"));

        store.put("baftest", b"shard payload").unwrap();
        assert!(store.has("baftest"));

        let data = store.get("baftest").unwrap();
        assert_eq!(data, b"shard payload");
    }

    #[test]
    fn get_missing_returns_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();

        let err = store.get("nonexistent").unwrap_err();
        assert!(matches!(err, StoreError::NotFound(_)));
    }

    #[test]
    fn mmap_shard() {
        let dir = tempfile::tempdir().unwrap();
        let store = ShardStore::new(dir.path().join("shards")).unwrap();

        let payload = vec![0xABu8; 8192];
        store.put("bafmmap", &payload).unwrap();

        let mapped = store.mmap("bafmmap").unwrap();
        assert_eq!(&*mapped, &payload[..]);
    }
}
