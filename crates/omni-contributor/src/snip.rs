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
use omni_store::SnipV2Adapter;
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
