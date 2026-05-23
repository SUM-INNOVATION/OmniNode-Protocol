//! Stage 11b.1.b — end-to-end prove + verify integration test.
//!
//! Runs entirely in-process: generates params via `prove_canonical`,
//! constructs an in-memory `Halo2ReferenceVerifier`, builds a full
//! `ProofArtifactBody`, and asserts `verify_artifact() == Ok(true)`
//! and that tampered bodies are rejected with the expected typed
//! errors.
//!
//! Gated by `--features prove` because the prover side is needed.

#![cfg(feature = "prove")]

use omni_proofs_halo2_reference::{
    canonical_evaluate, encoding::encode_tensor_4xi16_le, prover::prove_canonical,
    verifier::Halo2ReferenceVerifier, CANONICAL_INPUT, EXPECTED_SPEC_HASH,
};
use omni_zkml::{
    ModelFormat, ModelFramework, ProofArtifactBody, ProofMetadata, ProofSystem, ProofVerifier,
    ProofVerifierError,
};

fn spec_hash_hex() -> String {
    let mut s = String::with_capacity(64);
    for b in EXPECTED_SPEC_HASH {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn build_artifact(
    input_i16: [i16; 4],
    output_i16: [i16; 4],
    proof_bytes: &[u8],
) -> ProofArtifactBody {
    let spec_hex = spec_hash_hex();
    let input_hash = hex_blake3(&encode_tensor_4xi16_le(&input_i16));
    let output_hash = hex_blake3(&encode_tensor_4xi16_le(&output_i16));

    let mut metadata = ProofMetadata::new_stage11a(
        "halo2-reference-mlp-v1".to_string(),
        spec_hex,
        input_hash,
        output_hash,
    );
    metadata.proof_system = Some(ProofSystem::Stage11bHalo2Reference);
    metadata.model_format = Some(ModelFormat::Halo2ReferenceMlp);
    metadata.model_framework = Some(ModelFramework::FrameworkAgnostic);
    metadata.testnet_or_dev_only = Some(true);
    metadata.public_inputs = Some(serde_json::json!({
        "input":  input_i16.to_vec(),
        "output": output_i16.to_vec(),
    }));

    ProofArtifactBody::from_components(metadata, proof_bytes)
}

#[test]
fn canonical_proof_verifies_end_to_end() {
    let (params_bytes, proof_bytes, output_i16) = prove_canonical(CANONICAL_INPUT).unwrap();
    assert_eq!(output_i16, canonical_evaluate(CANONICAL_INPUT));

    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes).unwrap();
    let body = build_artifact(CANONICAL_INPUT, output_i16, &proof_bytes);

    let result = verifier.verify_artifact(&body).expect("verify_artifact");
    assert!(result, "valid canonical proof must verify");
}

#[test]
fn tampered_proof_bytes_are_rejected() {
    let (params_bytes, mut proof_bytes, output_i16) = prove_canonical(CANONICAL_INPUT).unwrap();
    proof_bytes[0] ^= 0x01; // flip a bit

    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes).unwrap();
    let body = build_artifact(CANONICAL_INPUT, output_i16, &proof_bytes);

    // A bit-flipped proof may either produce Ok(false) (constraints
    // unsatisfied) or VerifierInternal (transcript / IO failure
    // surfaced by halo2_proofs). Both are acceptable rejection
    // signals. The forbidden outcome is Ok(true).
    match verifier.verify_artifact(&body) {
        Ok(true) => panic!("tampered proof unexpectedly verified"),
        Ok(false) => {}
        Err(ProofVerifierError::VerifierInternal(_)) => {}
        Err(other) => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn wrong_output_hash_in_metadata_is_rejected() {
    let (params_bytes, proof_bytes, output_i16) = prove_canonical(CANONICAL_INPUT).unwrap();
    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes).unwrap();
    let mut body = build_artifact(CANONICAL_INPUT, output_i16, &proof_bytes);

    // Tamper with response_hash: now the body's BLAKE3(LE(output))
    // won't match metadata.response_hash → step 7 of the
    // verifier pipeline fails.
    body.metadata.response_hash = "0".repeat(64);

    let err = verifier.verify_artifact(&body).unwrap_err();
    assert!(matches!(err, ProofVerifierError::VerifierInternal(msg) if msg.contains("response_hash")));
}

#[test]
fn missing_testnet_or_dev_only_flag_is_rejected() {
    let (params_bytes, proof_bytes, output_i16) = prove_canonical(CANONICAL_INPUT).unwrap();
    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes).unwrap();
    let mut body = build_artifact(CANONICAL_INPUT, output_i16, &proof_bytes);
    body.metadata.testnet_or_dev_only = None;

    let err = verifier.verify_artifact(&body).unwrap_err();
    assert!(matches!(err, ProofVerifierError::VerifierInternal(msg) if msg.contains("testnet_or_dev_only")));
}

#[test]
fn wrong_proof_system_is_rejected() {
    let (params_bytes, proof_bytes, output_i16) = prove_canonical(CANONICAL_INPUT).unwrap();
    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes).unwrap();
    let mut body = build_artifact(CANONICAL_INPUT, output_i16, &proof_bytes);
    body.metadata.proof_system = Some(ProofSystem::Mock);

    let err = verifier.verify_artifact(&body).unwrap_err();
    assert!(matches!(err, ProofVerifierError::VerifierInternal(msg) if msg.contains("proof_system")));
}

#[test]
fn wrong_model_framework_is_rejected() {
    let (params_bytes, proof_bytes, output_i16) = prove_canonical(CANONICAL_INPUT).unwrap();
    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes).unwrap();
    let mut body = build_artifact(CANONICAL_INPUT, output_i16, &proof_bytes);
    body.metadata.model_framework = Some(ModelFramework::PyTorch);

    let err = verifier.verify_artifact(&body).unwrap_err();
    assert!(matches!(err, ProofVerifierError::VerifierInternal(msg) if msg.contains("model_framework")));
}

#[test]
fn verify_entry_point_returns_requires_artifact_dispatch() {
    let (params_bytes, _proof_bytes, _output) = prove_canonical(CANONICAL_INPUT).unwrap();
    let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes).unwrap();
    let pi = omni_zkml::PublicInputs {
        model_hash: [0u8; 32],
        input_hash: [0u8; 32],
        output_hash: [0u8; 32],
    };
    let err = verifier.verify(&[], &pi).unwrap_err();
    assert!(matches!(err, ProofVerifierError::RequiresArtifactDispatch(_)));
}
