//! Stage 11b.1.b — verifier-only test against the embedded fixtures.
//!
//! Gated by `--features verify`. CI's verifier-only job runs this
//! test; it does NOT pull the prover surface (`prove`-only deps
//! like `rand_chacha`). Drift in `params.bin` or `proof.bin`
//! relative to the committed `proof_artifact.json` fails here.

#![cfg(feature = "verify")]

use omni_proofs_halo2_reference::Halo2ReferenceVerifier;
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
    let verifier =
        Halo2ReferenceVerifier::from_embedded_fixtures().expect("from_embedded_fixtures");
    let body = load_committed_artifact();
    let ok = verifier
        .verify_artifact(&body)
        .expect("verify_artifact must not error on committed fixture");
    assert!(ok, "committed proof must verify against embedded params.bin");
}

#[test]
fn committed_artifact_has_required_stage11b_metadata() {
    use omni_zkml::{ModelFormat, ModelFramework, ProofSystem};

    let body = load_committed_artifact();
    assert_eq!(
        body.metadata.proof_system,
        Some(ProofSystem::Stage11bHalo2Reference)
    );
    assert_eq!(
        body.metadata.model_format,
        Some(ModelFormat::Halo2ReferenceMlp)
    );
    assert_eq!(
        body.metadata.model_framework,
        Some(ModelFramework::FrameworkAgnostic)
    );
    assert_eq!(body.metadata.testnet_or_dev_only, Some(true));
}
