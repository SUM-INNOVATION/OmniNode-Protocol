//! Stage 14.1 — prove → verify roundtrip integration tests for
//! the [`Halo2ReferenceProofBackend`] adapter.
//!
//! These tests require BOTH the `prove` and `verify` features —
//! the prover produces proof bytes which the verifier accepts via
//! its embedded fixtures. Default-feature CI does not run them.
//!
//! Coexistence is asserted in `omni-node`'s CLI test module: the
//! default-feature build's `MockProofBackend` flow stays green
//! whether or not this crate's `prove` / `verify` features are
//! enabled.

#![cfg(all(feature = "prove", feature = "verify"))]

use omni_proofs_halo2_reference::{
    canonical_evaluate, encode_canonical_input, encode_canonical_output,
    Halo2ReferenceProofBackend, Halo2ReferenceVerifier, CANONICAL_INPUT,
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

fn build_artifact_for_input(input_i16: [i16; 4]) -> ProofArtifactBody {
    let backend = Halo2ReferenceProofBackend::new();
    let output_i16 = canonical_evaluate(input_i16);
    let input_bytes = encode_canonical_input(&input_i16);
    let output_bytes = encode_canonical_output(&output_i16);

    let proof_bytes = backend
        .prove(CANONICAL_SPEC_JSON, &input_bytes, &output_bytes)
        .expect("adapter must produce a proof for valid inputs");

    let metadata = ProofMetadata {
        backend_id: backend.backend_id().to_string(),
        model_hash: hex_lower(blake3::hash(CANONICAL_SPEC_JSON).as_bytes()),
        input_hash: hex_lower(blake3::hash(&input_bytes).as_bytes()),
        response_hash: hex_lower(blake3::hash(&output_bytes).as_bytes()),
        model_format: Some(ModelFormat::Halo2ReferenceMlp),
        proof_system: Some(ProofSystem::Stage11bHalo2Reference),
        circuit_id_hex: backend
            .circuit_id()
            .as_ref()
            .map(|h| hex_lower(h)),
        verification_key_hex: None,
        public_inputs: Some(serde_json::json!({
            "input":  [input_i16[0],  input_i16[1],  input_i16[2],  input_i16[3]],
            "output": [output_i16[0], output_i16[1], output_i16[2], output_i16[3]],
        })),
        testnet_or_dev_only: Some(true),
        model_framework: Some(ModelFramework::FrameworkAgnostic),
    };
    ProofArtifactBody::from_components(metadata, &proof_bytes)
}

#[test]
fn prove_then_verify_round_trip_succeeds_for_canonical_input() {
    let body = build_artifact_for_input(CANONICAL_INPUT);
    let verifier = Halo2ReferenceVerifier::from_embedded_fixtures()
        .expect("verifier should construct from embedded fixtures");
    let verified = verifier
        .verify_artifact(&body)
        .expect("verify_artifact should succeed");
    assert!(
        verified,
        "Stage 14.1 round trip: prove(canonical) → verify must accept"
    );
}

#[test]
fn prove_then_verify_round_trip_succeeds_for_a_non_canonical_input() {
    // Exercises the seam beyond `CANONICAL_INPUT`: the verifier
    // re-runs `canonical_evaluate(input)` (its step 7.5
    // defense-in-depth check) so the prover and verifier must
    // agree on an arbitrary in-range input. A second `[i16; 4]`
    // value confirms the wiring isn't accidentally hardcoded.
    let input: [i16; 4] = [0, 1, -1, 2];
    let body = build_artifact_for_input(input);
    let verifier = Halo2ReferenceVerifier::from_embedded_fixtures().unwrap();
    let verified = verifier.verify_artifact(&body).unwrap();
    assert!(verified);
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

    let verifier = Halo2ReferenceVerifier::from_embedded_fixtures().unwrap();
    match verifier.verify_artifact(&body) {
        // Either path is acceptable: a tampered transcript may
        // surface as Ok(false) (constraint failure) or as a
        // VerifierInternal error (malformed transcript). Both
        // mean "the verifier did not accept the tamper."
        Ok(false) => {}
        Err(_) => {}
        Ok(true) => panic!("tampered proof MUST NOT verify true"),
    }
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
