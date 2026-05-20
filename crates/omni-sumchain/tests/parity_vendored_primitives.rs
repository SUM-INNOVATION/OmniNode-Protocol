//! Stage 7b — local-vs-vendored parity tests.
//!
//! Stage 9a: file-level `#![cfg(feature = "submit")]`. Tests import
//! `sumchain_primitives::*`, which is only present with the vendored
//! chain crates pulled in by the `submit` feature.
#![cfg(feature = "submit")]

//!
//! Stage 6 defines `omni_zkml::InferenceAttestationDigest` and
//! (implicitly) `InferenceAttestationTxData` as the canonical OmniNode-
//! internal byte layout. Stage 7b's submit path converts these into the
//! vendored `sumchain_primitives::*` types at the outer-tx assembly
//! boundary. These tests prove the conversion is byte-preserving under
//! bincode 1.3 (the chain's canonical wire format).
//!
//! If any future bump of the chain rev introduces a divergence (field
//! reorder, field add, encoding change), these tests fail loudly and
//! the safe response is to bring Stage 6 / Stage 7b back in sync before
//! shipping.

use omni_zkml::InferenceAttestationDigest as LocalDigest;
use sumchain_primitives::inference_attestation::{
    InferenceAttestationDigest as ChainDigest,
    InferenceAttestationTxData as ChainTxData,
};
use sumchain_primitives::Address;

fn sample_local_digest() -> LocalDigest {
    LocalDigest {
        session_id: "omninode-stage7b-parity".into(),
        model_hash: [0x00; 32],
        manifest_root: [0x11; 32],
        response_hash: [0x22; 32],
        proof_root: [0x33; 32],
    }
}

fn sample_chain_digest(local: &LocalDigest) -> ChainDigest {
    ChainDigest {
        session_id: local.session_id.clone(),
        model_hash: local.model_hash,
        manifest_root: local.manifest_root,
        response_hash: local.response_hash,
        proof_root: local.proof_root,
    }
}

// ── #1: InferenceAttestationDigest bincode parity ────────────────────────────

/// Local Stage 6 digest and vendored chain digest with byte-identical
/// field values must produce byte-identical bincode-1.3 output. The
/// local side goes through Stage 6's public `canonical_digest_bytes`
/// helper (which uses `bincode 1.3` internally via the crate-local
/// `bincode1` alias); the chain side uses the workspace `bincode 1.3`
/// dev-dep directly.
#[test]
fn parity_inference_attestation_digest_bincode_bytes_match() {
    let local = sample_local_digest();
    let chain = sample_chain_digest(&local);

    let bytes_local =
        omni_zkml::canonical_digest_bytes(&local).expect("local canonical bytes");
    let bytes_chain = bincode1::serialize(&chain).expect("chain bincode 1.3");

    assert_eq!(
        bytes_local, bytes_chain,
        "InferenceAttestationDigest bincode 1.3 parity broken: local Stage 6 \
         emits {} bytes, vendored chain emits {} bytes. The two type \
         definitions must remain byte-compatible.",
        bytes_local.len(),
        bytes_chain.len()
    );
}

// ── #2: InferenceAttestationTxData bincode parity ────────────────────────────

/// Same as #1 for the wrapping type. `InferenceAttestationTxData` adds
/// a `[u8; 64]` `verifier_signature` field (serialised by both sides
/// via `serde_big_array::BigArray` — Stage 6 uses an equivalent
/// custom helper, chain uses BigArray directly). The wire emission is
/// identical: a fixed 64-element u8 tuple with no length prefix under
/// bincode 1.3 default config.
#[test]
fn parity_inference_attestation_tx_data_bincode_bytes_match() {
    use omni_zkml::InferenceAttestationTxData as LocalTxData;

    // Use a non-zero verifier_signature so a future bug that drops or
    // truncates the field is caught (a zero-filled sig would mask
    // some encoding errors).
    let mut sig = [0u8; 64];
    for (i, b) in sig.iter_mut().enumerate() {
        *b = i as u8;
    }

    let local_digest = sample_local_digest();
    let chain_digest = sample_chain_digest(&local_digest);

    let local_tx_data = LocalTxData {
        digest: local_digest,
        verifier_signature: sig,
    };
    let chain_tx_data = ChainTxData {
        digest: chain_digest,
        verifier_signature: sig,
    };

    let bytes_local = bincode1::serialize(&local_tx_data).expect("local bincode 1.3");
    let bytes_chain = bincode1::serialize(&chain_tx_data).expect("chain bincode 1.3");

    assert_eq!(
        bytes_local, bytes_chain,
        "InferenceAttestationTxData bincode 1.3 parity broken: local Stage 6 \
         and vendored chain emit different bytes. Inner signature wrapping \
         must remain byte-compatible."
    );
}

// ── #3: address derivation parity ───────────────────────────────────────────

/// Stage 6's `derive_chain_address_base58(pubkey)` and the vendored
/// `Address::from_public_key(pubkey).to_base58()` must produce the
/// identical string for every input pubkey. Stage 6's algorithm was
/// written against the chain spec; the parity test pins them together
/// against the canonical chain implementation.
#[test]
fn parity_address_from_pubkey_matches_omninode_derivation() {
    let pubkeys: &[[u8; 32]] = &[[0u8; 32], [42u8; 32], [0xFFu8; 32]];
    for pubkey in pubkeys {
        let addr_local = omni_zkml::derive_chain_address_base58(pubkey);
        let addr_chain = Address::from_public_key(pubkey).to_base58();
        assert_eq!(
            addr_local, addr_chain,
            "address derivation parity broken for pubkey [{}u8; 32]: \
             local Stage 6 = {}, vendored chain = {}",
            pubkey[0], addr_local, addr_chain
        );
    }
}
