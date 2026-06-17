//! Phase 5 Stage 13.0 — chain anchoring for Stage 12.25 signed
//! integrity-evidence-chain-report artifacts.
//!
//! See the per-file module docs for the full spec; the public
//! surface re-exported here is:
//!
//! - [`IntegrityEvidenceAnchorDigest`] / [`IntegrityEvidenceAnchorTxData`]
//!   — the chain wire payload (frozen for v1).
//! - [`AnchoredArtifactKind`] — closed-set kind enum (one
//!   variant for Stage 13.0).
//! - [`INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION`] /
//!   [`EVIDENCE_ANCHOR_DOMAIN`] — schema constants.
//! - [`EvidenceAnchorChainClient`] / [`StubEvidenceAnchorChainClient`]
//!   — chain client trait + stub (Stage 13.0 ships no real
//!   adapter).
//! - [`LocalEvidenceAnchorRegistry`] / [`AnchorRecord`] /
//!   [`LocalAnchorStatus`] — persistent stub registry.
//! - [`submit_evidence_anchor_workflow`] /
//!   [`query_evidence_anchor_workflow`] /
//!   [`verify_anchor_against_registry`] /
//!   [`verify_anchor_file_against_artifact_bytes`] — operator-
//!   facing workflow helpers.
//! - [`VerifiedWrapperMetadata`] — CLI ↔ library boundary
//!   carrying pre-verified Stage 12.25 wrapper metadata.
//! - [`evidence_anchor_reason_tag`] — closed-set reason-tag
//!   mapper for `event=...` lines.

pub mod client;
pub mod registry;
pub mod wire;
pub mod workflow;

pub use client::{
    AnchorStatus, AnchorSubmissionReceipt, EvidenceAnchorChainClient, StubEvidenceAnchorChainClient,
};
pub use registry::{
    AnchorRecord, LocalAnchorStatus, LocalEvidenceAnchorRegistry, local_status_from_chain,
};
pub use wire::{
    AnchoredArtifactKind, EVIDENCE_ANCHOR_DOMAIN, INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
    IntegrityEvidenceAnchorDigest, IntegrityEvidenceAnchorTxData, anchor_hex_lower,
    anchor_signer_pubkey_bytes, anchor_signing_input_bytes,
    bincode1_serialize_anchor_tx_data, canonical_anchor_bytes, parse_anchor_hex_32,
    sign_anchor_digest, verify_anchor_tx_data,
};
pub use workflow::{
    AnchorSelector, QueryAnchorOutcome, VerifiedWrapperMetadata,
    anchor_signing_input_for_digest, build_anchor_digest,
    query_evidence_anchor_workflow, reconcile_evidence_anchors_workflow,
    submit_evidence_anchor_workflow, verify_anchor_against_registry,
    verify_anchor_file_against_artifact_bytes,
};

use crate::error::EvidenceAnchorError;

/// Closed-set reason-tag mapper for
/// `event=integrity_evidence_anchor_..._failed reason=<tag>`
/// lines. Single source of truth for the stable tag strings.
pub fn evidence_anchor_reason_tag(err: &EvidenceAnchorError) -> &'static str {
    match err {
        EvidenceAnchorError::UnsupportedAnchorSchemaVersion { .. } => {
            "unsupported_anchor_schema_version"
        }
        EvidenceAnchorError::UnsupportedArtifactSchemaVersion { .. } => {
            "unsupported_artifact_schema_version"
        }
        EvidenceAnchorError::UnsupportedArtifactKind { .. } => "unsupported_artifact_kind",
        EvidenceAnchorError::WrapperSignatureInvalid => "wrapper_signature_invalid",
        EvidenceAnchorError::SubmitterPubkeyMismatch { .. } => "submitter_pubkey_mismatch",
        EvidenceAnchorError::SubmitterSignatureInvalid => "submitter_signature_invalid",
        EvidenceAnchorError::ArtifactHashMismatch { .. } => "artifact_hash_mismatch",
        EvidenceAnchorError::AnchoredSignerPubkeyMismatch { .. } => {
            "anchored_signer_pubkey_mismatch"
        }
        EvidenceAnchorError::AnchorNotFound { .. } => "anchor_not_found",
        EvidenceAnchorError::MalformedSeedFile { .. } => "malformed_seed_file",
        EvidenceAnchorError::MalformedJson { .. } => "malformed_json",
        EvidenceAnchorError::MalformedSignedAtUtc { .. } => "malformed_signed_at_utc",
        EvidenceAnchorError::CanonicalSerialization(_) => "canonical_serialization",
        EvidenceAnchorError::Signing(_) => "signing",
        EvidenceAnchorError::ChainClient(_) => "chain_client",
        EvidenceAnchorError::Io { .. } => "io",

        // ── Stage 13.2 chain-touching CLI preflight refusals ──
        EvidenceAnchorError::ChainIdMismatch { .. } => "chain_id_mismatch",
        EvidenceAnchorError::NotActivated { .. } => "not_activated",
        EvidenceAnchorError::MainnetPolicyUnresolved => "mainnet_policy_unresolved",
        EvidenceAnchorError::ChainRpc(_) => "chain_rpc",
        EvidenceAnchorError::ChainSubmitRefused(_) => "chain_submit_refused",
        EvidenceAnchorError::ChainResponseMalformed(_) => "chain_response_malformed",
    }
}

#[cfg(test)]
mod reason_tag_tests {
    use super::*;
    use crate::error::ChainClientError;

    #[test]
    fn every_variant_has_a_stable_tag() {
        // Exhaustive mapping check — adding a new variant
        // without updating the mapper is a compile error.
        let cases: &[(EvidenceAnchorError, &str)] = &[
            (
                EvidenceAnchorError::UnsupportedAnchorSchemaVersion {
                    got: 0,
                    expected: 1,
                },
                "unsupported_anchor_schema_version",
            ),
            (
                EvidenceAnchorError::UnsupportedArtifactSchemaVersion {
                    kind: "signed_integrity_evidence_chain_report",
                    got: 0,
                    expected: 1,
                },
                "unsupported_artifact_schema_version",
            ),
            (
                EvidenceAnchorError::UnsupportedArtifactKind {
                    kind: "weird".to_string(),
                },
                "unsupported_artifact_kind",
            ),
            (
                EvidenceAnchorError::WrapperSignatureInvalid,
                "wrapper_signature_invalid",
            ),
            (
                EvidenceAnchorError::SubmitterPubkeyMismatch {
                    derived_pubkey_hex: "a".repeat(64),
                    wrapper_pubkey_hex: "b".repeat(64),
                },
                "submitter_pubkey_mismatch",
            ),
            (
                EvidenceAnchorError::SubmitterSignatureInvalid,
                "submitter_signature_invalid",
            ),
            (
                EvidenceAnchorError::ArtifactHashMismatch {
                    recomputed_hex: "a".repeat(64),
                    anchored_hex: "b".repeat(64),
                },
                "artifact_hash_mismatch",
            ),
            (
                EvidenceAnchorError::AnchoredSignerPubkeyMismatch {
                    wrapper_pubkey_hex: "a".repeat(64),
                    anchored_pubkey_hex: "b".repeat(64),
                },
                "anchored_signer_pubkey_mismatch",
            ),
            (
                EvidenceAnchorError::AnchorNotFound {
                    selector: "tx_id=missing".to_string(),
                },
                "anchor_not_found",
            ),
            (
                EvidenceAnchorError::MalformedSeedFile {
                    path: std::path::PathBuf::from("/tmp/seed"),
                    reason: "wrong length".to_string(),
                },
                "malformed_seed_file",
            ),
            (
                EvidenceAnchorError::MalformedSignedAtUtc {
                    raw: "not-a-time".to_string(),
                    reason: "rfc3339 parse failed".to_string(),
                },
                "malformed_signed_at_utc",
            ),
            (
                EvidenceAnchorError::CanonicalSerialization("bincode-1 boom".into()),
                "canonical_serialization",
            ),
            (
                EvidenceAnchorError::Signing("decode boom".into()),
                "signing",
            ),
            (
                EvidenceAnchorError::ChainClient(ChainClientError::Other("rpc gone".into())),
                "chain_client",
            ),
            // ── Stage 13.2 additions ──
            (
                EvidenceAnchorError::ChainIdMismatch {
                    expected: 1,
                    actual: 42,
                },
                "chain_id_mismatch",
            ),
            (
                EvidenceAnchorError::NotActivated {
                    chain_id: 42,
                    activation_status: "dormant (no activation height set)".into(),
                },
                "not_activated",
            ),
            (
                EvidenceAnchorError::MainnetPolicyUnresolved,
                "mainnet_policy_unresolved",
            ),
            (
                EvidenceAnchorError::ChainRpc("HTTP transport failure: timed out".into()),
                "chain_rpc",
            ),
            (
                EvidenceAnchorError::ChainSubmitRefused("dedup conflict".into()),
                "chain_submit_refused",
            ),
            (
                EvidenceAnchorError::ChainResponseMalformed("missing tx_hash".into()),
                "chain_response_malformed",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(evidence_anchor_reason_tag(err), *expected, "{err:?}");
        }
    }
}
