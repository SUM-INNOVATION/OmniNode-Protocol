//! BLAKE3 → CIDv1 content addressing.
//!
//! Every shard is identified by a CIDv1 string:
//! - Hash function: BLAKE3 (multicodec 0x1e)
//! - Codec:         raw    (multicodec 0x55)
//! - Encoding:      base32lower (multibase prefix 'b')

use cid::Cid;
use multihash::Multihash;

/// BLAKE3 multicodec identifier.
const BLAKE3_CODE: u64 = 0x1e;

/// "raw" codec — the data is unstructured bytes.
const RAW_CODEC: u64 = 0x55;

/// Hash `data` with BLAKE3 and return the CIDv1 string.
pub fn cid_from_data(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    cid_from_blake3_hash(&hash)
}

/// Convert an existing [`blake3::Hash`] into a CIDv1 string.
pub fn cid_from_blake3_hash(hash: &blake3::Hash) -> String {
    let mh = Multihash::<64>::wrap(BLAKE3_CODE, hash.as_bytes())
        .expect("blake3 32-byte digest always fits in 64-byte multihash");
    Cid::new_v1(RAW_CODEC, mh).to_string()
}

/// Return the raw BLAKE3 hash of `data` as a hex string (64 lowercase chars).
pub fn blake3_hex(data: &[u8]) -> String {
    blake3::hash(data).to_hex().to_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cid_is_deterministic() {
        let data = b"hello omni-store";
        let a = cid_from_data(data);
        let b = cid_from_data(data);
        assert_eq!(a, b);
        // CIDv1 base32lower starts with 'b'
        assert!(a.starts_with('b'), "CID should start with 'b': {a}");
    }

    #[test]
    fn different_data_different_cid() {
        let a = cid_from_data(b"aaa");
        let b = cid_from_data(b"bbb");
        assert_ne!(a, b);
    }

    #[test]
    fn blake3_hex_length() {
        let hex = blake3_hex(b"test");
        assert_eq!(hex.len(), 64);
    }

    #[test]
    fn cid_from_hash_matches_cid_from_data() {
        let data = b"round-trip check";
        let hash = blake3::hash(data);
        assert_eq!(cid_from_data(data), cid_from_blake3_hash(&hash));
    }
}
