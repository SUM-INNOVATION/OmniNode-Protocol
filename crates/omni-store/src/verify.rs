//! BLAKE3 integrity verification for shards.

use crate::content_id;
use crate::error::{Result, StoreError};

/// Verify that `data` matches the expected BLAKE3 hex hash.
pub fn verify_blake3(data: &[u8], expected_hex: &str) -> Result<()> {
    let actual = blake3::hash(data).to_hex().to_string();
    if actual != expected_hex {
        return Err(StoreError::IntegrityMismatch {
            expected: expected_hex.to_string(),
            actual,
        });
    }
    Ok(())
}

/// Verify that `data` hashes to the expected CIDv1 string.
pub fn verify_cid(data: &[u8], expected_cid: &str) -> Result<()> {
    let actual = content_id::cid_from_data(data);
    if actual != expected_cid {
        return Err(StoreError::IntegrityMismatch {
            expected: expected_cid.to_string(),
            actual,
        });
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_blake3_good() {
        let data = b"integrity test data";
        let hex = blake3::hash(data).to_hex().to_string();
        verify_blake3(data, &hex).unwrap();
    }

    #[test]
    fn verify_blake3_bad() {
        let data = b"integrity test data";
        let bad = "0".repeat(64);
        let err = verify_blake3(data, &bad).unwrap_err();
        assert!(matches!(err, StoreError::IntegrityMismatch { .. }));
    }

    #[test]
    fn verify_cid_good() {
        let data = b"cid verification";
        let cid = content_id::cid_from_data(data);
        verify_cid(data, &cid).unwrap();
    }

    #[test]
    fn verify_cid_bad() {
        let data = b"cid verification";
        let err = verify_cid(data, "bafkBADCID").unwrap_err();
        assert!(matches!(err, StoreError::IntegrityMismatch { .. }));
    }
}
