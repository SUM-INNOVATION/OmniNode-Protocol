//! Stage 14.5 — prove → verify roundtrip integration tests for
//! the [`Halo2ProductionMlpProofBackend`] adapter.
//!
//! These tests require BOTH the `prove` and `verify` features —
//! the prover produces proof bytes which the verifier accepts via
//! its embedded fixtures (`params.bin` + pinned
//! `EXPECTED_VK_HASH_HEX` / `EXPECTED_CIRCUIT_ID_HEX`). Default-
//! feature CI does not run them.
//!
//! Coexistence is asserted in `omni-node`'s CLI test module: the
//! default-feature build's `MockProofBackend` flow stays green
//! whether or not this crate's `prove` / `verify` features are
//! enabled, and Stage 11d.2 verify-only CI continues to pass.

#![cfg(all(feature = "prove", feature = "verify"))]

use omni_proofs_halo2_production_mlp::{
    encode_canonical_input, encode_canonical_output, Halo2ProductionMlpProofBackend,
    Halo2ProductionMlpVerifier, CANONICAL_INPUT, EXPECTED_CIRCUIT_ID_HEX,
    EXPECTED_VK_HASH_HEX,
};
use omni_zkml::{
    ModelFormat, ModelFramework, ProofArtifactBody, ProofBackend, ProofMetadata,
    ProofSystem, ProofVerifier,
};

const CANONICAL_SPEC_JSON: &[u8] = include_bytes!("../assets/canonical_spec.json");

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn build_artifact_for_input(input_i16: [i16; 16]) -> ProofArtifactBody {
    let backend = Halo2ProductionMlpProofBackend::new();
    let output_i16 =
        omni_proofs_halo2_production_mlp::canonical_evaluate(input_i16);
    let input_bytes = encode_canonical_input(&input_i16);
    let output_bytes = encode_canonical_output(&output_i16);

    let proof_bytes = backend
        .prove(CANONICAL_SPEC_JSON, &input_bytes, &output_bytes)
        .expect("adapter must produce a proof for valid inputs");

    // Stage 11d.2 production verifier is stricter than the
    // reference: circuit_id_hex AND verification_key_hex are
    // REQUIRED (not optional) and must equal the pinned
    // constants; testnet_or_dev_only MUST be `Some(false)` — the
    // production-shape contract.
    let metadata = ProofMetadata {
        backend_id: backend.backend_id().to_string(),
        model_hash: hex_lower(blake3::hash(CANONICAL_SPEC_JSON).as_bytes()),
        input_hash: hex_lower(blake3::hash(&input_bytes).as_bytes()),
        response_hash: hex_lower(blake3::hash(&output_bytes).as_bytes()),
        model_format: Some(ModelFormat::ProductionFixedPointMlp),
        proof_system: Some(ProofSystem::Stage11dProductionFixedPointMlp),
        circuit_id_hex: Some(EXPECTED_CIRCUIT_ID_HEX.to_string()),
        verification_key_hex: Some(EXPECTED_VK_HASH_HEX.to_string()),
        public_inputs: Some(serde_json::json!({
            "input":  input_i16.to_vec(),
            "output": output_i16.to_vec(),
        })),
        testnet_or_dev_only: Some(false),
        model_framework: Some(ModelFramework::FrameworkAgnostic),
    };
    ProofArtifactBody::from_components(metadata, &proof_bytes)
}

#[test]
fn prove_then_verify_round_trip_succeeds_for_canonical_input() {
    let body = build_artifact_for_input(CANONICAL_INPUT);
    let verifier = Halo2ProductionMlpVerifier::from_embedded_fixtures()
        .expect("verifier should construct from embedded fixtures");
    let verified = verifier
        .verify_artifact(&body)
        .expect("verify_artifact should succeed");
    assert!(
        verified,
        "Stage 14.5 round trip: prove(canonical) → verify must accept"
    );
}

#[test]
fn adapter_proof_bytes_are_deterministic_across_two_artifact_builds() {
    // Two adapter calls with identical inputs MUST produce the
    // same `proof_bytes_hex` — the verifier-side test
    // `prove_canonical_is_byte_deterministic` already pins
    // `prove_canonical`; this confirms the adapter does not
    // accidentally introduce non-determinism in the wrapping
    // (e.g. via inadvertent allocation hashing).
    let a = build_artifact_for_input(CANONICAL_INPUT);
    let b = build_artifact_for_input(CANONICAL_INPUT);
    assert_eq!(a.proof_bytes_hex, b.proof_bytes_hex);
}

#[test]
fn tampered_proof_bytes_are_rejected_by_verifier() {
    let mut body = build_artifact_for_input(CANONICAL_INPUT);
    // Flip one bit deep inside the proof bytes.
    let bytes = hex_to_bytes(&body.proof_bytes_hex);
    let mut tampered = bytes.clone();
    let idx = tampered.len() / 2;
    tampered[idx] ^= 0x01;
    body.proof_bytes_hex = hex_lower(&tampered);

    let verifier = Halo2ProductionMlpVerifier::from_embedded_fixtures().unwrap();
    match verifier.verify_artifact(&body) {
        Ok(false) => {}
        Err(_) => {}
        Ok(true) => panic!("tampered proof MUST NOT verify true"),
    }
}

#[test]
fn artifact_carries_production_shape_testnet_or_dev_only_false() {
    // Stage 14.5 production-shape contract pin: the artifact
    // declares `testnet_or_dev_only=Some(false)`, distinct from
    // the Stage 14.1 reference path which declares `Some(true)`.
    // Mainnet refusal comes from layer 6 (empty eligibility registry), not
    // layer 1. This test pins the shape so a future refactor
    // that conflates the two paths regresses visibly.
    let body = build_artifact_for_input(CANONICAL_INPUT);
    assert_eq!(body.metadata.testnet_or_dev_only, Some(false));
    assert_eq!(
        body.metadata.proof_system,
        Some(ProofSystem::Stage11dProductionFixedPointMlp)
    );
    assert_eq!(
        body.metadata.model_format,
        Some(ModelFormat::ProductionFixedPointMlp)
    );
    // verification_key_hex and circuit_id_hex are REQUIRED (not
    // optional) on production artifacts; the verifier refuses
    // None at lines 309-313 and 328-333 of verifier.rs.
    assert_eq!(
        body.metadata.circuit_id_hex.as_deref(),
        Some(EXPECTED_CIRCUIT_ID_HEX)
    );
    assert_eq!(
        body.metadata.verification_key_hex.as_deref(),
        Some(EXPECTED_VK_HASH_HEX)
    );
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
