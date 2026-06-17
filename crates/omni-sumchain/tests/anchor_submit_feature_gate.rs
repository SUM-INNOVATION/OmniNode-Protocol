//! Phase 5 Stage 13.2 — `submit_anchor` without `--features submit`
//! must refuse cleanly. Pinned because Stage 9c's gating posture
//! demands that the submit code path not exist at all in the
//! default build (no chain RPC reached, no panic, no surprise).

#![cfg(not(feature = "submit"))]

use omni_sumchain::{FakeJsonRpcTransport, JsonRpcTransport, SumChainClient};
use omni_zkml::{
    anchor_signer_pubkey_bytes, sign_anchor_digest, AnchoredArtifactKind, ChainClientError,
    EvidenceAnchorChainClient, IntegrityEvidenceAnchorDigest, IntegrityEvidenceAnchorTxData,
    INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
};

fn build_signed_tx(seed: [u8; 32]) -> IntegrityEvidenceAnchorTxData {
    let pubkey = anchor_signer_pubkey_bytes(&seed).unwrap();
    let digest = IntegrityEvidenceAnchorDigest {
        anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
        artifact_schema_version: 1,
        artifact_hash: [0x11; 32],
        signer_pubkey: pubkey,
        signed_at_utc_unix: 1_750_000_000,
    };
    let signature = sign_anchor_digest(&seed, &digest).unwrap();
    IntegrityEvidenceAnchorTxData {
        digest,
        submitter_signature: signature,
    }
}

#[test]
fn anchor_submit_without_submit_feature_refuses_without_reaching_chain() {
    let seed = [7u8; 32];
    let transport = FakeJsonRpcTransport::new();
    let client = SumChainClient::with_transport(seed, transport.clone());
    let tx_data = build_signed_tx(seed);
    let err = client.submit_anchor(&tx_data).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.contains("rebuild with --features submit"),
        "expected rebuild-with-submit message, got: {msg:?}"
    );
    // Critical: no chain RPC was made.
    assert!(
        transport.calls().is_empty(),
        "submit must not reach any RPC without --features submit; calls: {:?}",
        transport.calls()
    );
}
