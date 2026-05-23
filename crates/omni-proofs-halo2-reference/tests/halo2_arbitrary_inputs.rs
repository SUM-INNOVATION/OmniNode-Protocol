//! Stage 11c — arbitrary-input prove+verify integration test.
//!
//! For each entry in `tests/fixtures/corpus.json`, this test:
//!   1. Generates a fresh halo2 proof via `prove_canonical(input)`.
//!   2. Builds a full `ProofArtifactBody` with that proof.
//!   3. Verifies the body via `Halo2ReferenceVerifier::verify_artifact`.
//!
//! This is the **soundness assurance** for Stage 11c: it confirms
//! the Stage 11c gadget chain (dense → RHAZ → saturation → ReLU)
//! pins the canonical evaluator's output uniquely for every i16
//! input in the corpus (canonical, zero, four tie cases, and two
//! extreme i16 inputs).
//!
//! Gated by `--features prove` because the prove path pulls
//! `rand_chacha`. CI does NOT run this test on PR builds — the
//! prove side is developer-host only via the regen tool. CI runs
//! the verifier-only `halo2_proof_verifies` test against the
//! single committed canonical fixture.

#![cfg(feature = "prove")]

use omni_proofs_halo2_reference::{
    canonical_evaluate, encoding::encode_tensor_4xi16_le, prover::prove_canonical,
    verifier::Halo2ReferenceVerifier, EXPECTED_SPEC_HASH,
};
use omni_zkml::{
    ModelFormat, ModelFramework, ProofArtifactBody, ProofMetadata, ProofSystem, ProofVerifier,
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

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
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
fn every_corpus_entry_prove_and_verify_round_trip() {
    let corpus_bytes = std::fs::read(fixtures_dir().join("corpus.json"))
        .expect("read corpus.json");
    let corpus: serde_json::Value =
        serde_json::from_slice(&corpus_bytes).expect("parse corpus.json");
    let entries = corpus["entries"].as_array().expect("entries is array");
    assert_eq!(entries.len(), 8, "Stage 11c corpus must have 8 entries");

    for entry in entries {
        let label = entry["label"].as_str().unwrap_or("?");
        let input_arr = entry["input"].as_array().unwrap();
        let input: [i16; 4] = [
            input_arr[0].as_i64().unwrap() as i16,
            input_arr[1].as_i64().unwrap() as i16,
            input_arr[2].as_i64().unwrap() as i16,
            input_arr[3].as_i64().unwrap() as i16,
        ];

        // Cross-check: canonical_evaluate matches corpus.
        let expected_output = canonical_evaluate(input);

        // Prove.
        let (params_bytes, proof_bytes, prover_output) =
            prove_canonical(input).unwrap_or_else(|e| {
                panic!("corpus entry {label}: prove_canonical({input:?}) failed: {e:?}")
            });
        assert_eq!(
            prover_output, expected_output,
            "corpus entry {label}: prover output differs from canonical_evaluate"
        );

        // Verify.
        let verifier = Halo2ReferenceVerifier::from_params_bytes(&params_bytes)
            .expect("Halo2ReferenceVerifier::from_params_bytes");
        let body = build_artifact(input, expected_output, &proof_bytes);
        let ok = verifier.verify_artifact(&body).unwrap_or_else(|e| {
            panic!("corpus entry {label}: verify_artifact errored: {e:?}")
        });
        assert!(ok, "corpus entry {label}: verifier rejected its own honest proof");
    }
}
