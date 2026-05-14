//! Stage 7a — Stage 6 wire-parity smoke.
//!
//! Confirms that `omni-sumchain` can consume `omni-zkml`'s Stage 6
//! public surface and produce byte-identical output to the committed
//! chain-team deliverable fixture
//! (`crates/omni-zkml/tests/fixtures/chain_attestation_vectors.json`).
//!
//! Stage 7a does not yet use the Stage 6 surface for any in-crate
//! computation — Stage 7b will, when `tx::build_signed_transaction`
//! wraps the Stage 6 inner pipeline into the outer SignedTransaction.
//! This test is the cheapest possible cross-crate smoke that the Stage 6
//! deliverable is reachable and stable from `omni-sumchain`'s
//! compilation unit; Stage 7b expands the test surface to assert
//! specific outer-tx byte sequences against chain-provided parity
//! fixtures.

use std::path::PathBuf;

use omni_zkml::{compute_chain_attestation_vector, ChainAttestationVector};

#[test]
fn stage6_fixture_round_trips_via_omni_zkml_public_surface() {
    let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("omni-zkml")
        .join("tests")
        .join("fixtures")
        .join("chain_attestation_vectors.json");

    let bytes = std::fs::read(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "Stage 6 fixture must be present at {} — Stage 7a depends on it: {e}",
            fixture_path.display()
        )
    });
    let vectors: Vec<ChainAttestationVector> =
        serde_json::from_slice(&bytes).expect("fixture must be valid JSON");
    assert_eq!(
        vectors.len(),
        3,
        "Stage 6 fixture must contain exactly 3 vectors (chain-team deliverable)"
    );

    for v in &vectors {
        let computed = compute_chain_attestation_vector(
            &v.session_id,
            &v.model_hash,
            &v.manifest_root,
            &v.response_hash,
            &v.proof_root,
            &v.verifier_ed25519_seed,
        )
        .unwrap_or_else(|e| panic!("vector {} computation failed: {e}", v.session_id));

        assert_eq!(
            computed, *v,
            "vector {} drifted from committed Stage 6 fixture — omni-sumchain's \
             use of omni-zkml's Stage 6 surface no longer reproduces the \
             chain-team deliverable",
            v.session_id
        );
    }
}
