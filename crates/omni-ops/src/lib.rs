//! O5 (issue #103) — operator observability report-generators extracted
//! from `omni-contributor`.
//!
//! Local-disk-only operator tooling: session status, repair/reassign
//! planning, restart/resume audit, state-integrity scan/diff,
//! archive/restore, and cleanup/quarantine. These modules generate
//! schema-versioned JSON reports and BLAKE3 report/plan hashes over
//! their own `serde_json` bytes; they do NOT touch chain wire, signing
//! domains (`canonical.rs`), proof systems, networking, or RPC.
//!
//! Dependency direction: `omni-node → omni-ops → omni-contributor`.
//! This crate depends on the RETAINED `omni-contributor` persistence
//! substrate (`ContributorStateStore`), session types, and session /
//! peer / supersession verification.

pub mod archive;
pub mod cleanup;
pub mod error;
pub mod integrity;
pub mod repair;
pub mod restore;
pub mod resume;
pub mod status;

pub use error::{
    ArchiveError, CleanupError, IntegrityDiffError, IntegrityError,
    QuarantineRestoreError, RepairError, RestoreError, StatusError,
};
pub use repair::{
    build_session_repair_plan, build_session_repair_plan_with_reason,
    check_invalid_partial_plan_eligible,
    check_reassign_eligible_allowing_invalid_partials,
    check_reassign_targets_active_missing, check_repair_eligible,
    repair_plan_hash_hex, source_status_hash_hex, RepairAction, RepairStrategy,
    SessionRepairPlan, REPAIR_PLAN_SCHEMA_VERSION,
};
pub use status::{
    build_session_status_report, AssignmentStatus, InvalidArtifactStatus,
    SessionOverallStatus, SessionStatusReport, SupersessionStatus,
    STATUS_SCHEMA_VERSION,
};
pub use resume::{
    compute_audit_health, load_verified_restart_snapshot, AuditCoherence,
    AuditHealth, RestartReport, RestartSnapshot,
};
pub use archive::{
    archive_session, ArchiveManifest, ArchiveMode, ArchiveOptions,
    ArchiveStatusRequirement, ArchivedFile, ARCHIVE_MANIFEST_SCHEMA_VERSION,
};
pub use restore::{
    restore_session_archive, verify_archive_manifest, RestoreOptions,
    RestoreReport, RestoreSource,
};
pub use integrity::{
    diff_presentation_view, diff_state_integrity_reports, scan_state_integrity,
    scan_state_integrity_with_audit_orphans, DiffCounts, DiffOptions,
    FindingKind, FindingSeverity, IntegrityFinding, RecommendedAction,
    ScanOptions, SessionIntegritySummary, StateIntegrityDiffReport,
    StateIntegrityReport, STATE_INTEGRITY_DIFF_SCHEMA_VERSION,
    STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
};
pub use cleanup::{
    apply_state_cleanup, cleanup_plan_hash_hex, plan_state_cleanup,
    restore_state_cleanup_quarantine, source_integrity_hash_hex,
    verify_quarantine_manifest, ApplyOptions as CleanupApplyOptions,
    CleanupAction, CleanupActionKind, CleanupActionOutcome, CleanupReport,
    PlanOptions as CleanupPlanOptions, QuarantineEntry, QuarantineManifest,
    QuarantineRestoreOptions, QuarantineRestoreOutcome,
    QuarantineRestoreReport, QuarantineRestoreSource, StateCleanupPlan,
    CLEANUP_PLAN_SCHEMA_VERSION, QUARANTINE_MANIFEST_SCHEMA_VERSION,
};
