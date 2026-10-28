//! Stage 12.0 — SNIP V2 byte-publish / byte-fetch helpers.
//!
//! Option A (chosen 2026-05-25 by the user): wrap the existing
//! `omni_store::SnipV2Adapter` path-based API with tempfile-backed
//! byte helpers. `omni-store` is **not** edited; no opaque-byte
//! method is added to the trait; no new SNIP wire / root format is
//! invented.
//!
//! All publish/fetch operations check the returned `SnipV2Lifecycle`
//! via the existing `omni_store::snip_v2::check_lifecycle` helper
//! (which rejects `Pending` / `Abandoned` unless explicitly allowed)
//! so behavior matches the rest of the `omni-store` consumer surface.

use std::io::Write;

use omni_store::snip_v2::check_lifecycle;
use omni_store::snip_v2_publish::{publish_public_idempotent, PublicationAction};
use omni_store::{SnipV2Adapter, SnipV2Publisher};
use omni_types::phase5::SnipV2ObjectId;

use crate::error::SnipError;

/// Write `bytes` to a tempfile, call `adapter.ingest_public(path)`,
/// and return the resulting SNIP V2 Merkle root.
///
/// `label` is used for the tempfile name suffix and in error context;
/// it is NOT recorded on chain or in the returned root.
pub fn publish_bytes<A: SnipV2Adapter>(
    adapter: &A,
    bytes: &[u8],
    label: &str,
) -> Result<SnipV2ObjectId, SnipError> {
    let mut tmp = tempfile::Builder::new()
        .prefix("omni-contributor-publish-")
        .suffix(&format!("-{label}"))
        .tempfile()?;
    tmp.write_all(bytes)?;
    tmp.flush()?;

    let obj_ref = adapter.ingest_public(tmp.path())?;
    check_lifecycle(&obj_ref.lifecycle, /* allow_non_active = */ false)?;
    Ok(obj_ref.merkle_root)
}

/// Issue #149 — bytes published by [`publish_bytes_idempotent`].
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
/// is already, without ever registering its root twice.
///
/// Decides from the chain's record of the bytes' canonical root
/// ([`omni_store::snip_v2_publish`]): no record → ingest; Active and
/// matching → reuse; Pending and owned by the configured account → resume;
/// anything else → refuse. Returns only once the chain records the root
/// Active with these bytes' size and chunk count. `label` names the
/// tempfile only.
pub fn publish_bytes_idempotent<A: SnipV2Publisher + ?Sized>(
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

    let published = publish_public_idempotent(adapter, tmp.path())?;
    Ok(PublishedBytes {
        root: published.root,
        hash: blake3_32(bytes),
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
pub fn publish_bytes_with_hash<A: SnipV2Adapter>(
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
pub fn fetch_bytes_with_integrity_check<A: SnipV2Adapter>(
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
