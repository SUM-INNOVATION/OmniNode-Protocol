//! Stage 11d.2 — verifier-only test against the embedded fixtures.
//!
//! Gated by `--features verify`. CI's verifier-only job runs this
//! test; it does NOT pull the prover surface (`prove`-only deps
//! like `rand_chacha`). Drift in `params.bin` or `proof.bin`
//! relative to the committed `proof_artifact.json` fails here.

#![cfg(feature = "verify")]

use omni_proofs_halo2_production_mlp::Halo2ProductionMlpVerifier;
use omni_zkml::{ProofArtifactBody, ProofVerifier};

fn load_committed_artifact() -> ProofArtifactBody {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/halo2/proof_artifact.json");
    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).expect("parse proof_artifact.json")
}

#[test]
fn embedded_params_verifies_committed_proof_artifact() {
    let verifier = Halo2ProductionMlpVerifier::from_embedded_fixtures()
        .expect("from_embedded_fixtures");
    let body = load_committed_artifact();
    let ok = verifier
        .verify_artifact(&body)
        .expect("verify_artifact must not error on committed fixture");
    assert!(ok, "committed proof must verify against embedded params.bin");
}

#[test]
fn committed_artifact_has_required_stage11d_metadata() {
    use omni_zkml::{ModelFormat, ModelFramework, ProofSystem};

    let body = load_committed_artifact();
    assert_eq!(
        body.metadata.proof_system,
        Some(ProofSystem::Stage11dProductionFixedPointMlp)
    );
    assert_eq!(
        body.metadata.model_format,
        Some(ModelFormat::ProductionFixedPointMlp)
    );
    assert_eq!(
        body.metadata.model_framework,
        Some(ModelFramework::FrameworkAgnostic)
    );
    // Stage 11d.2: production-shape artifact. Mainnet refusal at
    // layer 6 comes from the empty MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES
    // table, not from testnet_or_dev_only.
    assert_eq!(body.metadata.testnet_or_dev_only, Some(false));
}

#[test]
fn committed_artifact_input_output_match_canonical() {
    use omni_proofs_halo2_production_mlp::{canonical_evaluate, CANONICAL_INPUT, CANONICAL_OUTPUT};

    let body = load_committed_artifact();
    let pi = body
        .metadata
        .public_inputs
        .as_ref()
        .expect("metadata.public_inputs present");
    let (input, output) =
        Halo2ProductionMlpVerifier::decode_public_inputs_json(pi).unwrap();
    assert_eq!(input, CANONICAL_INPUT, "committed input must be canonical");
    assert_eq!(output, CANONICAL_OUTPUT, "committed output must be canonical");
    assert_eq!(canonical_evaluate(input), output);
}

#[test]
fn committed_artifact_carries_pinned_circuit_id_and_vk_hash() {
    use omni_proofs_halo2_production_mlp::{
        BACKEND_ID, EXPECTED_CIRCUIT_ID_HEX, EXPECTED_VK_HASH_HEX,
    };
    let body = load_committed_artifact();
    assert_eq!(body.metadata.backend_id, BACKEND_ID);
    assert_eq!(
        body.metadata.circuit_id_hex.as_deref(),
        Some(EXPECTED_CIRCUIT_ID_HEX),
        "committed artifact circuit_id_hex must equal pinned constant"
    );
    assert_eq!(
        body.metadata.verification_key_hex.as_deref(),
        Some(EXPECTED_VK_HASH_HEX),
        "committed artifact verification_key_hex (mainnet_vk_hash) must equal pinned constant"
    );
}

#[test]
fn verifier_construction_enforces_vk_identity_drift_detection() {
    // If `from_embedded_fixtures()` succeeds, the drift check inside
    // `from_params_bytes` passed for the committed params.bin —
    // i.e. the live VK's `(circuit_id_hex, vk_hash_hex)` pair equals
    // the pinned `EXPECTED_*` constants. A halo2_proofs version bump
    // or any unintended circuit edit fails this constructor.
    let v = Halo2ProductionMlpVerifier::from_embedded_fixtures();
    v.expect("VK identity must match pinned constants");
}

#[test]
fn committed_artifact_refused_on_mainnet() {
    // Stage 11d.1 invariant: every proof-system value (including this
    // one) is refused on mainnet through Stage 11d.2 because
    // MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES is empty. This test
    // pins that posture for the production proof class — any
    // accidental allowlist-table population would fail here.
    let body = load_committed_artifact();
    let outcome = omni_zkml::check_mainnet_eligible(&body.metadata);
    match outcome {
        Err(omni_zkml::MainnetRefusalReason::NotInMainnetAllowlist { .. }) => {}
        other => panic!(
            "Stage 11d.2 production artifact must be refused with NotInMainnetAllowlist, \
             got {other:?}"
        ),
    }
}
