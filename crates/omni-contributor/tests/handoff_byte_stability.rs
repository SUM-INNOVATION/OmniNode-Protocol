//! Stage 12.4 — byte-stability test for the `ActivationHandoff`
//! canonical signing body. The canonical body EXCLUDES
//! `tensor_chunk_bytes` and `sender_signature_hex`; the signature
//! binds `tensor_hash` instead.

use omni_contributor::{
    canonical::{canonical_activation_handoff_bytes, hex_lower},
    ActivationHandoff, TensorDtype, HANDOFF_SCHEMA_VERSION,
};

#[test]
fn activation_handoff_canonical_bytes_blake3_is_pinned() {
    let tensor_bytes = vec![0xAB; 32 * 64 * 2]; // 32 × 64 × f16 = 4096 B
    let tensor_hash = hex_lower(blake3::hash(&tensor_bytes).as_bytes());
    let h = ActivationHandoff {
        schema_version: HANDOFF_SCHEMA_VERSION,
        session_id: "11".repeat(32),
        from_assignment_id: "22".repeat(32),
        to_assignment_id: "33".repeat(32),
        from_contributor_pubkey_hex: "44".repeat(32),
        to_contributor_pubkey_hex: "55".repeat(32),
        dtype: TensorDtype::F16,
        shape: vec![32, 64],
        byte_len: tensor_bytes.len() as u64,
        tensor_hash,
        chunk_index: 0,
        chunk_count: 1,
        produced_at_utc: "2026-05-27T00:00:42Z".into(),
        tensor_chunk_bytes: tensor_bytes,
        sender_signature_hex: "ee".repeat(64),
    };
    let bytes = canonical_activation_handoff_bytes(&h).unwrap();
    let hash = hex_lower(blake3::hash(&bytes).as_bytes());
    assert_eq!(
        hash, "60c16b873e600c88586343ed3f4bf602b0178853774c135cda04ef38f2110989",
        "drift in canonical_activation_handoff_bytes — recompute and re-pin"
    );
}

#[test]
fn canonical_body_excludes_tensor_chunk_bytes() {
    // The signing body must NOT include tensor_chunk_bytes: two
    // handoffs with identical metadata but different chunk-bytes
    // payloads must produce identical canonical bytes (so the
    // sender's signature is independent of the bytes — which the
    // receiver re-checks separately via the BLAKE3 hash binding).
    fn build(extra_bytes: &[u8]) -> ActivationHandoff {
        let tensor_bytes: Vec<u8> = vec![0xAB; 16];
        let tensor_hash = hex_lower(blake3::hash(&tensor_bytes).as_bytes());
        ActivationHandoff {
            schema_version: HANDOFF_SCHEMA_VERSION,
            session_id: "11".repeat(32),
            from_assignment_id: "22".repeat(32),
            to_assignment_id: "33".repeat(32),
            from_contributor_pubkey_hex: "44".repeat(32),
            to_contributor_pubkey_hex: "55".repeat(32),
            dtype: TensorDtype::F16,
            shape: vec![8],
            byte_len: 16,
            tensor_hash,
            chunk_index: 0,
            chunk_count: 1,
            produced_at_utc: "2026-05-27T00:00:42Z".into(),
            // Different bytes between calls, but the canonical body
            // pins the hash — so the body bytes themselves must NOT
            // shift just because the payload bytes shift.
            tensor_chunk_bytes: extra_bytes.to_vec(),
            sender_signature_hex: "ee".repeat(64),
        }
    }
    // Both have the SAME tensor_hash even though tensor_chunk_bytes differ:
    // the canonical body is identical because it excludes the payload.
    let a = build(&[0xAB; 16]);
    let b = build(&[0xCD; 16]);
    let ca = canonical_activation_handoff_bytes(&a).unwrap();
    let cb = canonical_activation_handoff_bytes(&b).unwrap();
    assert_eq!(ca, cb, "canonical body must not depend on tensor_chunk_bytes");
}
