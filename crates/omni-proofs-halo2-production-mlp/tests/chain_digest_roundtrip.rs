//! Stage 11d.2 — **R9 scaffold**: optional chain-digest +
//! signature roundtrip via the existing Stage 6/7
//! `InferenceAttestationDigest` primitives.
//!
//! Per the Stage 11d.2 plan §11 R9 precision (chain-team
//! correction): R9 builds a **real** `InferenceAttestationDigest`
//! with:
//!
//! - `session_id`     — synthetic sentinel (no chain session was
//!                      created in this test)
//! - `manifest_root`  — synthetic sentinel (no manifest tree was
//!                      built in this test)
//! - `model_hash`     — real, derived from `EXPECTED_PRODUCTION_SPEC_HASH`
//! - `response_hash`  — real, derived from
//!                      `BLAKE3(encode_canonical_output(committed_output))`
//! - `proof_root`     — real, derived from
//!                      `BLAKE3(committed_proof_bytes)`
//!
//! Then exercises the existing Stage 6 / 7 chain primitives:
//!
//! 1. [`canonical_digest_bytes`] → byte-stable canonical encoding.
//! 2. [`sign_chain_attestation_digest`] → 64-byte Ed25519 signature.
//! 3. Re-derive the public key from the same seed via
//!    [`signer_pubkey_bytes`] and verify the signature against the
//!    canonical `signing_input_bytes`.
//! 4. Bincode round-trip the digest (serialize → deserialize) and
//!    confirm canonical bytes are stable across the round-trip.
//!
//! **Not in CI by default.** This test is opt-in: it requires the
//! `r9-chain-digest-roundtrip` feature (which pulls in
//! `omni-sumchain`) AND it's marked `#[ignore]` so even
//! `cargo test --all-features` skips it unless `--ignored` is
//! passed. The actual R9 exercise/sign-off is a Stage 11d.3
//! deliverable; the scaffold exists in 11d.2 only to validate
//! the integration path is wired correctly.
//!
//! No chain wire / Stage 7b tx / RPC / chain-team-blocking
//! signing is performed; this is a local-only digest+signature
//! roundtrip against the existing primitives.

#![cfg(feature = "r9-chain-digest-roundtrip")]

use omni_proofs_halo2_production_mlp::{
    encoding::encode_canonical_output, EXPECTED_PRODUCTION_SPEC_HASH,
};
use omni_zkml::{
    canonical_digest_bytes, sign_chain_attestation_digest, signer_pubkey_bytes,
    signing_input_bytes, InferenceAttestationDigest, ProofArtifactBody,
};

const TEST_SEED: [u8; 32] = *b"OmniNode/Stage11d.2/r9-test-seed";

fn load_committed_artifact() -> ProofArtifactBody {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/halo2/proof_artifact.json");
    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("parse proof_artifact.json")
}

fn blake3_32(bytes: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(blake3::hash(bytes).as_bytes());
    out
}

fn build_synthetic_digest() -> InferenceAttestationDigest {
    let body = load_committed_artifact();
    let public_inputs = body
        .metadata
        .public_inputs
        .as_ref()
        .expect("metadata.public_inputs present");
    let output_arr = public_inputs["output"]
        .as_array()
        .expect("output is array");
    let mut output = [0i16; 8];
    for (i, v) in output_arr.iter().enumerate() {
        output[i] = v.as_i64().unwrap() as i16;
    }
    let proof_bytes = body.proof_bytes().expect("decode proof_bytes_hex");

    InferenceAttestationDigest {
        // Synthetic — Stage 11d.2 has no chain session.
        session_id: "stage11d2-r9-scaffold-synthetic-session".to_string(),
        // Real — production canonical spec hash.
        model_hash: EXPECTED_PRODUCTION_SPEC_HASH,
        // Synthetic — Stage 11d.2 has no manifest SNIP root.
        manifest_root: [0xAB; 32],
        // Real — BLAKE3 of LE-encoded committed output tensor.
        response_hash: blake3_32(&encode_canonical_output(&output)),
        // Real — BLAKE3 of committed proof bytes.
        proof_root: blake3_32(&proof_bytes),
    }
}

#[test]
#[ignore = "Stage 11d.2 R9 scaffold — opt-in via `--ignored` or the \
            `r9-chain-digest-roundtrip` feature; not in CI by default"]
fn r9_digest_bytes_are_deterministic() {
    let d = build_synthetic_digest();
    let a = canonical_digest_bytes(&d).unwrap();
    let b = canonical_digest_bytes(&d).unwrap();
    assert_eq!(a, b, "canonical_digest_bytes must be deterministic");
    assert!(!a.is_empty());
}

#[test]
#[ignore = "Stage 11d.2 R9 scaffold — opt-in via `--ignored`"]
fn r9_signature_verifies_with_matching_pubkey() {
    use libp2p_identity::ed25519;

    let digest = build_synthetic_digest();
    let sig = sign_chain_attestation_digest(&TEST_SEED, &digest).expect("sign");
    let pubkey_bytes = signer_pubkey_bytes(&TEST_SEED).expect("pubkey derive");
    let pubkey = ed25519::PublicKey::try_from_bytes(&pubkey_bytes).expect("PublicKey decode");
    let signing_input = signing_input_bytes(&digest).expect("signing_input_bytes");
    assert!(
        pubkey.verify(&signing_input, &sig),
        "Ed25519 signature must verify against the re-derived pubkey"
    );
}

#[test]
#[ignore = "Stage 11d.2 R9 scaffold — opt-in via `--ignored`"]
fn r9_signature_fails_against_unrelated_pubkey() {
    use libp2p_identity::ed25519;

    let digest = build_synthetic_digest();
    let sig = sign_chain_attestation_digest(&TEST_SEED, &digest).expect("sign");
    // Different seed → different pubkey → verify must fail.
    let other_seed = [0x42u8; 32];
    let other_pk_bytes = signer_pubkey_bytes(&other_seed).expect("pubkey derive");
    let other_pk =
        ed25519::PublicKey::try_from_bytes(&other_pk_bytes).expect("PublicKey decode");
    let signing_input = signing_input_bytes(&digest).expect("signing_input_bytes");
    assert!(
        !other_pk.verify(&signing_input, &sig),
        "unrelated pubkey must not verify the signature"
    );
}

#[test]
#[ignore = "Stage 11d.2 R9 scaffold — opt-in via `--ignored`"]
fn r9_digest_canonical_bytes_stable_across_bincode_roundtrip() {
    let d1 = build_synthetic_digest();
    // bincode 1.3 (matches canonical_digest_bytes encoding)
    let wire = canonical_digest_bytes(&d1).unwrap();
    let d2: InferenceAttestationDigest =
        bincode1::deserialize(&wire).expect("deserialize InferenceAttestationDigest");
    assert_eq!(d1, d2, "digest must round-trip through canonical bytes");
    let wire2 = canonical_digest_bytes(&d2).unwrap();
    assert_eq!(wire, wire2, "second-pass canonical bytes must equal first");
}

// bincode 1.3 is a dev-dependency aliased to `bincode1` to match the
// chain primitives' canonical-bytes encoding (which uses bincode 1.3).
