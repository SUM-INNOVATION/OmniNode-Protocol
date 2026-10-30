//! Stage 12.0 — SNIP V2 byte-publish / byte-fetch helpers.
//!
//! Option A (chosen 2026-05-25 by the user): wrap the existing
//! `omni_store::SnipV2Adapter` path-based API with tempfile-backed
//! byte helpers. `omni-store` is **not** edited; no opaque-byte
//! method is added to the trait; no new SNIP wire / root format is
//! invented.
//!
//! Issue #149: every publication here is idempotent and retry-safe
//! ([`omni_store::snip_v2_publish::publish_public_idempotent`]). The root
//! is the canonical SNIP V2 root of the bytes, unchanged; what is sent
//! depends on the chain's record of it (none → ingest, Active → reuse,
//! the configured account's Pending → resume, anything else → refuse),
//! and a publication returns only once the record reads Active, matches
//! the bytes, and the object fetched back is those bytes. Publishing the
//! same bytes twice, from this process or another, therefore succeeds
//! instead of failing on the chain's duplicate-registration refusal.

use std::io::Write;
use std::path::Path;

use omni_store::snip_v2_publish::{publish_public_idempotent, PublicationAction};
use omni_store::SnipV2Adapter;
use omni_types::phase5::SnipV2ObjectId;

use crate::error::SnipError;

/// Publish `bytes` as a Public SNIP V2 object (idempotently, see the
/// module docs) and return its canonical root.
///
/// `label` is used for the tempfile name suffix and in error context;
/// it is NOT recorded on chain or in the returned root.
pub fn publish_bytes<A: SnipV2Adapter + ?Sized>(
    adapter: &A,
    bytes: &[u8],
    label: &str,
) -> Result<SnipV2ObjectId, SnipError> {
    publish_bytes_idempotent(adapter, bytes, label).map(|p| p.root)
}

/// Issue #149 — bytes published by [`publish_bytes_idempotent`] or
/// [`publish_file_idempotent`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedBytes {
    /// Canonical SNIP V2 root of the bytes, recorded Active on chain.
    pub root: SnipV2ObjectId,
    /// BLAKE3 of the bytes.
    pub hash: [u8; 32],
    /// How the root came to be Active: ingested now, already Active, or
    /// this account's Pending registration resumed.
    pub action: PublicationAction,
    /// Owner the chain records.
    pub owner: String,
}

/// Issue #149 — publish `bytes` as a Public SNIP V2 object, or confirm it
/// is already, without ever registering its root twice (module docs).
/// `label` names the tempfile only.
pub fn publish_bytes_idempotent<A: SnipV2Adapter + ?Sized>(
    adapter: &A,
    bytes: &[u8],
    label: &str,
) -> Result<PublishedBytes, SnipError> {
    let mut tmp = tempfile::Builder::new()
        .prefix("omni-contributor-publish-")
        .suffix(&format!("-{label}"))
        .tempfile()?;
    tmp.write_all(bytes)?;
    tmp.flush()?;
    publish_file_idempotent(adapter, tmp.path())
}

/// Issue #149 — [`publish_bytes_idempotent`] for bytes already in a file,
/// published from that file as it is (a run checkpoint's durable copy of
/// a response, for instance).
pub fn publish_file_idempotent<A: SnipV2Adapter + ?Sized>(
    adapter: &A,
    path: &Path,
) -> Result<PublishedBytes, SnipError> {
    let published = publish_public_idempotent(adapter, path)?;
    let bytes = std::fs::read(path)?;
    Ok(PublishedBytes {
        root: published.root,
        hash: blake3_32(&bytes),
        action: published.action,
        owner: published.owner,
    })
}

/// Call `adapter.download_public(root, tempfile_path)` and return the
/// downloaded bytes. The tempfile is removed on scope exit.
pub fn fetch_bytes<A: SnipV2Adapter + ?Sized>(
    adapter: &A,
    root: &SnipV2ObjectId,
) -> Result<Vec<u8>, SnipError> {
    let tmp = tempfile::Builder::new()
        .prefix("omni-contributor-fetch-")
        .tempfile()?;
    adapter.download_public(root, tmp.path())?;
    let bytes = std::fs::read(tmp.path())?;
    Ok(bytes)
}

/// Convenience: publish `bytes`, then BLAKE3-hash the same bytes
/// (no `omni-store` round-trip — the caller already has the bytes).
/// Returns `(SnipV2ObjectId, [u8; 32])` so a higher-level orchestrator
/// can populate `*_snip_root` and `*_hash` artifact fields in one
/// call.
pub fn publish_bytes_with_hash<A: SnipV2Adapter + ?Sized>(
    adapter: &A,
    bytes: &[u8],
    label: &str,
) -> Result<(SnipV2ObjectId, [u8; 32]), SnipError> {
    let root = publish_bytes(adapter, bytes, label)?;
    let hash = blake3_32(bytes);
    Ok((root, hash))
}

/// Convenience: fetch by root, then BLAKE3-hash and check the result
/// equals the caller's expected hash. Returns the fetched bytes on
/// success; returns `SnipError::Integrity` on hash mismatch.
pub fn fetch_bytes_with_integrity_check<A: SnipV2Adapter + ?Sized>(
    adapter: &A,
    root: &SnipV2ObjectId,
    expected_hash_hex: &str,
    label: &'static str,
) -> Result<Vec<u8>, SnipError> {
    let bytes = fetch_bytes(adapter, root)?;
    let actual = blake3_32(&bytes);
    let actual_hex = crate::canonical::hex_lower(&actual);
    if actual_hex != expected_hash_hex {
        return Err(SnipError::Integrity {
            label,
            expected: expected_hash_hex.to_string(),
            got: actual_hex,
        });
    }
    Ok(bytes)
}

fn blake3_32(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}
