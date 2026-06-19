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

pub mod archive;
pub mod cleanup;
pub mod client;
pub mod consistency;
pub mod export;
pub mod import;
pub mod operations;
pub mod registry;
pub mod wire;
pub mod workflow;

pub use client::{
    canonicalize_tx_hash, AnchorStatus, AnchorStatusReport, AnchorSubmissionReceipt,
    BatchStatusItem, EvidenceAnchorChainClient, StubEvidenceAnchorChainClient,
    TupleLookupResult, ANCHOR_STATUS_BATCH_MAX, FAILED_REASON_NULL_FALLBACK,
};
pub use cleanup::{
    apply_anchor_cleanup, plan_anchor_cleanup, restore_anchor_cleanup_quarantine,
    AnchorApplyOptions, AnchorCleanupAction, AnchorCleanupActionKind,
    AnchorCleanupActionOutcome, AnchorCleanupPlan, AnchorCleanupReport,
    AnchorPlanOptions, AnchorQuarantineEntry, AnchorQuarantineManifest,
    AnchorQuarantineRestoreOutcome, AnchorQuarantineRestoreReport,
    AnchorRestoreOptions, ANCHOR_CLEANUP_PLAN_SCHEMA_VERSION,
    ANCHOR_QUARANTINE_MANIFEST_SCHEMA_VERSION,
};
pub use export::{
    apply_anchor_export, plan_anchor_export, verify_anchor_export,
    AnchorExportEntry, AnchorExportEntryKind, AnchorExportManifest,
    AnchorExportOptions, AnchorExportPlan, AnchorExportReport,
    AnchorExportSelection, AnchorExportVerifyOptions,
    AnchorExportVerifyReport, ArtifactBytesInclusion,
    EVIDENCE_ANCHOR_EXPORT_MANIFEST_SCHEMA_VERSION,
    EXPORT_MANIFEST_FILENAME,
};
pub use archive::{
    apply_anchor_archive, plan_anchor_archive, restore_anchor_archive,
    AnchorArchiveAction, AnchorArchiveActionKind, AnchorArchiveActionOutcome,
    AnchorArchiveApplyOptions, AnchorArchiveEntry, AnchorArchiveManifest,
    AnchorArchivePlan, AnchorArchivePlanOptions, AnchorArchiveReport,
    AnchorArchiveRestoreOptions, AnchorArchiveRestoreOutcome,
    AnchorArchiveRestoreReport, AnchorArchiveSelection,
    ANCHOR_ARCHIVE_MANIFEST_SCHEMA_VERSION, ANCHOR_ARCHIVE_PLAN_SCHEMA_VERSION,
};
pub use consistency::{
    build_anchor_consistency_report, AnchorConsistencyFinding,
    AnchorConsistencyFindingKind, AnchorConsistencyOptions, AnchorConsistencyReport,
    AnchorConsistencySeverity, AnchorConsistencySummary,
    ANCHOR_CONSISTENCY_REPORT_SCHEMA_VERSION,
};
pub use import::{
    apply_anchor_export_import, plan_anchor_export_import,
    AnchorImportActionOutcome, AnchorImportOptions, AnchorImportPlan,
    AnchorImportReport, AnchorImportSelection, PlannedImportAction,
};
pub use operations::{
    check_evidence_anchor_registry_health, list_evidence_anchors_by_status,
    list_stale_submitted_or_included, EvidenceAnchorRegistryHealth,
    EvidenceAnchorRegistrySummary, StaleAnchorInfo,
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
    AnchorSelector, QueryAnchorOutcome, TupleLookupOutcome, VerifiedWrapperMetadata,
    anchor_signing_input_for_digest, build_anchor_digest,
    lookup_anchor_by_tuple_workflow, query_evidence_anchor_workflow,
    reconcile_evidence_anchors_workflow, submit_evidence_anchor_workflow,
    verify_anchor_against_registry, verify_anchor_file_against_artifact_bytes,
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

        // ── Stage 13.4 cleanup refusals ──
        // Three new tag strings + four reused from Stage 12.17/12.18.
        EvidenceAnchorError::CleanupDrift { .. } => "cleanup_drift",
        EvidenceAnchorError::CleanupPlanHashMismatch { .. } => "cleanup_plan_hash_mismatch",
        EvidenceAnchorError::CleanupGateRequired { .. } => "gate_required",
        EvidenceAnchorError::QuarantineBlake3Mismatch { .. } => "quarantine_blake3_mismatch",
        EvidenceAnchorError::RestoreTargetExists { .. } => "restore_target_exists",
        EvidenceAnchorError::CleanupInvalidPath { .. } => "cleanup_invalid_path",
        EvidenceAnchorError::CleanupPlanSchemaUnsupported { .. } => {
            "unsupported_cleanup_plan_schema_version"
        }

        // ── Stage 13.5 export-side refusals ──
        EvidenceAnchorError::ExportManifestSchemaUnsupported { .. } => {
            "unsupported_export_manifest_schema_version"
        }
        EvidenceAnchorError::ExportManifestHashMismatch { .. } => {
            "export_manifest_hash_mismatch"
        }
        EvidenceAnchorError::ExportInvalidPath { .. } => "export_invalid_path",
        EvidenceAnchorError::ExportBlake3Mismatch { .. } => "export_blake3_mismatch",
        EvidenceAnchorError::ExportEntryMetadataMismatch { .. } => {
            "export_entry_metadata_mismatch"
        }
        EvidenceAnchorError::ExportStrictModeArtifactBytesMissing { .. } => {
            "export_strict_mode_artifact_bytes_missing"
        }

        // ── Stage 13.6 import-side refusal ──
        EvidenceAnchorError::ImportTargetExists { .. } => "import_target_exists",

        // ── Stage 13.7 archive-side refusals ──
        EvidenceAnchorError::ArchivePlanSchemaUnsupported { .. } => {
            "unsupported_archive_plan_schema_version"
        }
        EvidenceAnchorError::ArchivePlanHashMismatch { .. } => "archive_plan_hash_mismatch",
        EvidenceAnchorError::ArchiveDrift { .. } => "archive_drift",
        EvidenceAnchorError::ArchiveInvalidPath { .. } => "archive_invalid_path",
        EvidenceAnchorError::ArchiveBlake3Mismatch { .. } => "archive_blake3_mismatch",
        EvidenceAnchorError::ArchiveTargetExists { .. } => "archive_target_exists",
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
            // ── Stage 13.4 cleanup variants ──
            (
                EvidenceAnchorError::CleanupDrift {
                    computed: "a".repeat(16),
                    expected: "b".repeat(16),
                },
                "cleanup_drift",
            ),
            (
                EvidenceAnchorError::CleanupPlanHashMismatch {
                    computed: "c".repeat(64),
                    expected: "d".repeat(64),
                },
                "cleanup_plan_hash_mismatch",
            ),
            (
                EvidenceAnchorError::CleanupGateRequired {
                    action_kind: "quarantine_stale_open_record",
                    gate_flag: "--allow-stale-quarantine",
                },
                "gate_required",
            ),
            (
                EvidenceAnchorError::QuarantineBlake3Mismatch {
                    source_relative: "11".repeat(32) + ".json",
                    computed: "e".repeat(64),
                    expected: "f".repeat(64),
                },
                "quarantine_blake3_mismatch",
            ),
            (
                EvidenceAnchorError::RestoreTargetExists {
                    target_path: std::path::PathBuf::from("/tmp/anchors/22.json"),
                },
                "restore_target_exists",
            ),
            (
                EvidenceAnchorError::CleanupInvalidPath {
                    action_kind: "remove_orphan_tmp_file",
                    source_relative: "../etc/passwd".to_string(),
                    reason: "parent traversal forbidden",
                },
                "cleanup_invalid_path",
            ),
            (
                EvidenceAnchorError::CleanupPlanSchemaUnsupported {
                    got: 2,
                    expected: 1,
                },
                "unsupported_cleanup_plan_schema_version",
            ),
            // ── Stage 13.5 export-side variants ──
            (
                EvidenceAnchorError::ExportManifestSchemaUnsupported {
                    got: 2,
                    expected: 1,
                },
                "unsupported_export_manifest_schema_version",
            ),
            (
                EvidenceAnchorError::ExportManifestHashMismatch {
                    computed: "1".repeat(64),
                    expected: "2".repeat(64),
                },
                "export_manifest_hash_mismatch",
            ),
            (
                EvidenceAnchorError::ExportInvalidPath {
                    entry_kind: "anchor_record",
                    relative_path: "../etc/passwd".to_string(),
                    reason: "parent traversal forbidden",
                },
                "export_invalid_path",
            ),
            (
                EvidenceAnchorError::ExportBlake3Mismatch {
                    relative_path: "anchors/aa.json".to_string(),
                    computed: "3".repeat(64),
                    expected: "4".repeat(64),
                },
                "export_blake3_mismatch",
            ),
            (
                EvidenceAnchorError::ExportEntryMetadataMismatch {
                    relative_path: "anchors/bb.json".to_string(),
                    field: "artifact_hash_hex",
                    computed: "aa".to_string(),
                    manifest: "bb".to_string(),
                },
                "export_entry_metadata_mismatch",
            ),
            (
                EvidenceAnchorError::ExportStrictModeArtifactBytesMissing {
                    anchor_record_relative_path: "anchors/cc.json".to_string(),
                    artifact_hash_hex: "cc".repeat(32),
                },
                "export_strict_mode_artifact_bytes_missing",
            ),
            // ── Stage 13.6 import-side variant ──
            (
                EvidenceAnchorError::ImportTargetExists {
                    field: "artifact_hash",
                    artifact_hash_hex: "dd".repeat(32),
                    tx_id: "anchor-1".to_string(),
                },
                "import_target_exists",
            ),
            // ── Stage 13.7 archive-side variants ──
            (
                EvidenceAnchorError::ArchivePlanSchemaUnsupported {
                    got: 2,
                    expected: 1,
                },
                "unsupported_archive_plan_schema_version",
            ),
            (
                EvidenceAnchorError::ArchivePlanHashMismatch {
                    computed: "1".repeat(64),
                    expected: "2".repeat(64),
                },
                "archive_plan_hash_mismatch",
            ),
            (
                EvidenceAnchorError::ArchiveDrift {
                    computed: "3".repeat(16),
                    expected: "4".repeat(16),
                },
                "archive_drift",
            ),
            (
                EvidenceAnchorError::ArchiveInvalidPath {
                    source_relative: "../etc/passwd".to_string(),
                    reason: "parent traversal forbidden",
                },
                "archive_invalid_path",
            ),
            (
                EvidenceAnchorError::ArchiveBlake3Mismatch {
                    archive_relative: "anchors/aa.json".to_string(),
                    computed: "5".repeat(64),
                    expected: "6".repeat(64),
                },
                "archive_blake3_mismatch",
            ),
            (
                EvidenceAnchorError::ArchiveTargetExists {
                    field: "tx_id",
                    artifact_hash_hex: "ee".repeat(32),
                    tx_id: "anchor-2".to_string(),
                },
                "archive_target_exists",
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(evidence_anchor_reason_tag(err), *expected, "{err:?}");
        }
    }
}
