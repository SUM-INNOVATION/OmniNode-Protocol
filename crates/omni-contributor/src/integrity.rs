//! Stage 12.16 — local state-dir integrity scan + repair
//! suggestions.
//!
//! Read-only consumer of the Stage 12.7 state-store, Stage 12.13
//! restart preload, Stage 12.14 archive manifest, and Stage
//! 12.15 restore-verify-only path. Walks the state-dir,
//! re-runs every Stage 12.3 / 12.11 verifier, surfaces stale /
//! missing seen markers, flags stray files inside documented
//! subtrees, optionally walks a parallel archive directory, and
//! emits a typed `StateIntegrityReport` with one
//! `IntegrityFinding` per structural anomaly. The scanner
//! **never writes** to the state-dir or archive directory.
//!
//! Findings carry closed-set `kind` + `severity` discriminators
//! and closed-set `recommended_action` strings so log scrapers
//! can branch deterministically. JSON is operator-scriptable;
//! per-line events are pattern-matchable.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::IntegrityError;
use crate::resume::{
    compute_audit_health, load_verified_restart_snapshot, AuditCoherence,
};
use crate::session::PartialContributorResult;
use crate::session_verify::{verify_aggregated_result_with_supersessions, verify_partial_result};
use crate::state::{ContributorStateStore, StateNamespace};
use crate::status::{build_session_status_report, SessionStatusReport};

/// Stage 12.16 — integrity report schema version. Tied to the
/// Stage 12.16 scanner output contract, NOT to any in-tree
/// envelope schema. Lives in `integrity.rs` (local-only;
/// scanner output is never SNIP-published or chain-bound).
pub const STATE_INTEGRITY_REPORT_SCHEMA_VERSION: u32 = 1;

// ── Closed-set discriminators ──────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum FindingSeverity {
    Ok,
    Warn,
    Error,
}

impl FindingSeverity {
    pub fn as_str(self) -> &'static str {
        match self {
            FindingSeverity::Ok => "ok",
            FindingSeverity::Warn => "warn",
            FindingSeverity::Error => "error",
        }
    }
}

/// Closed taxonomy of structural anomalies the Stage 12.16
/// scanner detects. Renaming any variant is a contract break;
/// adding new variants is forward-incompatible until the next
/// `STATE_INTEGRITY_REPORT_SCHEMA_VERSION` bump.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum FindingKind {
    /// `verified/sessions/<id>/session.json` failed
    /// `verify_execution_session`.
    InvalidSession,
    /// `verified/sessions/<id>/joins/<pubkey>.json` failed
    /// `verify_contributor_join`.
    InvalidJoin,
    /// `verified/sessions/<id>/assignments/<asn>.json` failed
    /// `verify_work_assignment`.
    InvalidAssignment,
    /// `verified/sessions/<id>/partials/<asn>.json` failed
    /// `verify_partial_result`.
    InvalidPartial,
    /// `verified/sessions/<id>/supersessions/<sup>.json` failed
    /// `verify_assignment_supersession` (or
    /// `verify_aggregated_result_with_supersessions` for the
    /// double-supersedes leg). Reuses Stage 12.11 reason tags.
    InvalidSupersession,
    /// `verified/sessions/<id>/aggregated.json` failed
    /// `verify_aggregated_result_with_supersessions`.
    InvalidAggregate,
    /// `seen/<ns>/<key>` exists but the corresponding
    /// `verified/...` body is missing. Operator either dropped
    /// the body manually or a Stage 12.7 prune cascade left
    /// markers behind.
    StaleSeenMarker,
    /// `verified/.../<body>` exists but the corresponding
    /// `seen/<ns>/<key>` marker is missing. The watcher would
    /// re-process the announcement on next restart — usually
    /// benign, sometimes indicates a Stage 12.13 restart
    /// preload rejected the marker write.
    MissingSeenMarker,
    /// A file under `verified/sessions/<id>/...` whose
    /// relative path does NOT match the Stage 12.14 archive
    /// whitelist. Operator-left `.tmp` / `.bak` / random file.
    StrayVerifiedFile,
    /// A file under `seen/<ns>/...` whose key shape does NOT
    /// match the Stage 12.14 archive whitelist (e.g. wrong
    /// hex length, missing `<sid>--` prefix for prefixed
    /// namespaces).
    StraySeenFile,
    /// Stage 12.13 audit projection emitted
    /// `OrphanReplacementAssignments` for this session — Phase
    /// B partial-apply suspect.
    OrphanReplacementAssignments,
    /// Stage 12.13 audit projection emitted
    /// `PartialApplySupersession` — out-of-order supersession
    /// arrival (best-effort accept) OR a Phase B2-corrupted
    /// supersession.
    PartialApplySupersession,
    /// Stage 12.13 audit projection: the session is
    /// `InvalidState` but every entry in `invalid_artifacts`
    /// is `InvalidPartial`. Operator can triage via
    /// `--reason invalid-partial`.
    ReassignTriagable,
    /// Stage 12.13 audit projection: the session is
    /// `InvalidState` AND at least one entry is NOT
    /// `InvalidPartial`. Reassign triage will refuse.
    NotReassignTriagable,
    /// `--include-archives` requested but a per-session
    /// archive directory's manifest is missing or
    /// schema/version/path-safety-malformed. Bubbles a
    /// stable `reason_tag` derived from the Stage 12.15
    /// `RestoreError`.
    ArchiveManifestMalformed,
    /// `--include-archives` BLAKE3 walk found a mismatch.
    /// One archive file's bytes don't match the manifest.
    ArchiveBlakeMismatch,
    /// `--include-archives`: a session_id has BOTH a
    /// state-dir presence AND an archive presence.
    /// Informational `Ok` finding — the archive is durable.
    ArchiveCoveredSession,
}

impl FindingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FindingKind::InvalidSession => "invalid_session",
            FindingKind::InvalidJoin => "invalid_join",
            FindingKind::InvalidAssignment => "invalid_assignment",
            FindingKind::InvalidPartial => "invalid_partial",
            FindingKind::InvalidSupersession => "invalid_supersession",
            FindingKind::InvalidAggregate => "invalid_aggregate",
            FindingKind::StaleSeenMarker => "stale_seen_marker",
            FindingKind::MissingSeenMarker => "missing_seen_marker",
            FindingKind::StrayVerifiedFile => "stray_verified_file",
            FindingKind::StraySeenFile => "stray_seen_file",
            FindingKind::OrphanReplacementAssignments => "orphan_replacement_assignments",
            FindingKind::PartialApplySupersession => "partial_apply_supersession",
            FindingKind::ReassignTriagable => "reassign_triagable",
            FindingKind::NotReassignTriagable => "not_reassign_triagable",
            FindingKind::ArchiveManifestMalformed => "archive_manifest_malformed",
            FindingKind::ArchiveBlakeMismatch => "archive_blake_mismatch",
            FindingKind::ArchiveCoveredSession => "archive_covered_session",
        }
    }
}

/// Closed/static set of operator-actionable suggestions. The
/// scanner emits these labels per finding; it never executes
/// them. Stage 12.16 v1 lifts the Stage 12.13 audit set + adds
/// archive-verify / restore-verify / stale-marker actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendedAction(pub Cow<'static, str>);

impl RecommendedAction {
    pub const NONE: Self = Self(Cow::Borrowed("none"));
    pub const RUN_SESSION_STATUS: Self = Self(Cow::Borrowed("run session-status"));
    pub const RUN_PLAN_REASSIGN_INVALID_PARTIAL: Self =
        Self(Cow::Borrowed("run plan-session-reassign --reason invalid-partial"));
    pub const RUN_PLAN_REASSIGN_MISSING_PARTIAL: Self =
        Self(Cow::Borrowed("run plan-session-reassign --reason missing-partial"));
    pub const RUN_ARCHIVE_AGGREGATED: Self =
        Self(Cow::Borrowed("run archive-session --require-status aggregated"));
    pub const RUN_ARCHIVE_EXPIRED: Self =
        Self(Cow::Borrowed("run archive-session --require-status expired-incomplete"));
    pub const RUN_ARCHIVE_VERIFY_ONLY: Self =
        Self(Cow::Borrowed("run archive-session --verify-only"));
    pub const RUN_RESTORE_VERIFY_ONLY: Self =
        Self(Cow::Borrowed("run restore-session-archive --verify-only"));
    pub const DELETE_STALE_SEEN_MARKER: Self =
        Self(Cow::Borrowed("delete stale seen marker"));
    pub const CLEAN_ORPHAN_REPLACEMENTS: Self =
        Self(Cow::Borrowed("clean state-dir orphan replacements before retry"));
    pub const OPERATOR_TRIAGE_REQUIRED: Self = Self(Cow::Borrowed("operator triage required"));

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl serde::Serialize for RecommendedAction {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> serde::Deserialize<'de> for RecommendedAction {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Self(Cow::Owned(s)))
    }
}

// ── Findings + report ─────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrityFinding {
    pub kind: FindingKind,
    pub severity: FindingSeverity,
    /// `None` when the finding is not session-scoped (e.g. an
    /// orphan top-level archive directory that doesn't match a
    /// state-dir session).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub session_id: Option<String>,
    /// Relative-to-state-dir (or relative-to-archive-dir for
    /// archive findings) path string, normalized to forward
    /// slashes for platform portability. `None` for findings
    /// not tied to a specific file (e.g. audit projections).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub path: Option<String>,
    /// Stable string. For verifier-failure findings, this is
    /// the Stage 12.12 `SessionVerifyOutcome::reason_tag`. For
    /// stray/stale markers, it's a closed-set tag (e.g.
    /// `"verified_body_missing"` / `"seen_marker_missing"`).
    /// For archive findings, the closed Stage 12.15
    /// `RestoreError` discriminator.
    pub reason_tag: String,
    pub recommended_action: RecommendedAction,
}

/// Per-session summary the scanner emits alongside its
/// findings. Informational only — `overall_status` is the v3
/// `SessionStatusReport.overall_status` Debug discriminator so
/// operators see the chain shape at a glance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionIntegritySummary {
    pub session_id: String,
    pub overall_status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateIntegrityReport {
    pub schema_version: u32,
    pub generated_at_utc: String,
    pub state_dir: String,
    pub state_version: u32,
    pub omni_contributor_version: String,

    pub sessions_scanned: u32,
    pub sessions_verified: u32,

    pub counts_ok: u32,
    pub counts_warn: u32,
    pub counts_error: u32,

    pub sessions: Vec<SessionIntegritySummary>,
    pub findings: Vec<IntegrityFinding>,
}

/// Stage 12.16 — read-only scan options.
#[derive(Debug, Clone, Default)]
pub struct ScanOptions<'a> {
    /// `None` = scan every session in the state-dir.
    /// `Some(hex)` = scope every session-scoped check to one
    /// session_id. Stray-file detection is also limited to
    /// that session's subtree.
    pub session_id_filter: Option<&'a str>,
    /// Optional sibling archive root. When set, every
    /// `<archive_dir>/<session_id>/manifest.json` is parsed
    /// via Stage 12.14 `verify_archive_manifest` and a full
    /// BLAKE3 walk runs via Stage 12.15
    /// `restore_session_archive(verify_only=true, dry_run=true)`
    /// — read-only end-to-end.
    pub archive_dir: Option<&'a Path>,
    /// RFC 3339 UTC. Goes into `generated_at_utc` and into
    /// `build_session_status_report`'s `now_utc` argument.
    pub now_utc: &'a str,
}

/// Run the scan. Returns a `StateIntegrityReport`. The store
/// itself is opened by the caller.
///
/// Stage 12.17 additive: the implementation also collects the
/// per-session orphan-assignment id list out of the audit
/// projection so the cleanup planner can plan
/// `QuarantineAndUnmarkOrphanAssignment` actions without
/// re-running the projection. The map is returned by the
/// sibling `scan_state_integrity_with_audit_orphans` entry; this
/// entry stays binary-compatible by discarding it.
pub fn scan_state_integrity(
    store: &ContributorStateStore,
    opts: &ScanOptions<'_>,
) -> Result<StateIntegrityReport, IntegrityError> {
    let (report, _orphans) = scan_state_integrity_with_audit_orphans(store, opts)?;
    Ok(report)
}

/// Stage 12.17 — same as [`scan_state_integrity`] but also
/// returns a `session_id → orphan_assignment_ids` map derived
/// from the Stage 12.13
/// `AuditCoherence::OrphanReplacementAssignments` projection.
/// The list is the structured side-channel the cleanup planner
/// consumes; it is **not** part of the public `IntegrityFinding`
/// surface, so the `STATE_INTEGRITY_REPORT_SCHEMA_VERSION = 1`
/// contract is preserved.
pub fn scan_state_integrity_with_audit_orphans(
    store: &ContributorStateStore,
    opts: &ScanOptions<'_>,
) -> Result<(StateIntegrityReport, HashMap<String, Vec<String>>), IntegrityError> {
    let mut findings: Vec<IntegrityFinding> = Vec::new();
    let mut session_summaries: Vec<SessionIntegritySummary> = Vec::new();
    let mut sessions_scanned: u32 = 0;
    let mut audit_orphans: HashMap<String, Vec<String>> = HashMap::new();

    // ── 1. Reuse Stage 12.13 restart preload for the
    //       sessions/joins/assignments/supersessions chain.
    let (snapshot, restart_report) = load_verified_restart_snapshot(store)?;
    let sessions_verified: u32 = restart_report.sessions_accepted;

    // Each Stage 12.13 rejection note maps to one finding. The
    // notes have a stable format
    // `kind=<...> session_id=<hex> id=<hex|-> reason_tag=<...>`;
    // parse them into typed findings so the scanner doesn't
    // expose Stage 12.13's debug string surface to its
    // consumers.
    for note in &restart_report.rejection_notes {
        if let Some(f) = parse_restart_rejection_note(note) {
            // Per Stage 12.16 decision: `--session-id` restricts
            // session-scoped findings to that one session even
            // when they come from the global rejection-note
            // stream. Drop rejection-note findings whose
            // session_id doesn't match the filter.
            if let Some(filter) = opts.session_id_filter {
                if f.session_id.as_deref() != Some(filter) {
                    continue;
                }
            }
            findings.push(f);
        }
    }

    // ── 2. Per accepted session: partials, aggregate, status,
    //       audit projection. Honors --session-id filter.
    let candidate_session_ids: Vec<String> = if let Some(sid) = opts.session_id_filter
    {
        if snapshot.sessions.contains_key(sid) {
            vec![sid.to_string()]
        } else {
            vec![]
        }
    } else {
        let mut ids: Vec<String> = snapshot.sessions.keys().cloned().collect();
        ids.sort();
        ids
    };

    for sid in &candidate_session_ids {
        sessions_scanned += 1;
        let session = snapshot.sessions.get(sid).expect("filtered above");
        let assignments = snapshot
            .assignments_by_session
            .get(sid)
            .cloned()
            .unwrap_or_default();
        let supersessions = snapshot
            .supersessions_by_session
            .get(sid)
            .cloned()
            .unwrap_or_default();
        let joins = snapshot.joins_by_session.get(sid).cloned().unwrap_or_default();
        let assignment_by_id: HashMap<String, _> = assignments
            .iter()
            .map(|a| (a.assignment_id.clone(), a.clone()))
            .collect();

        // ── Partial re-verify ──────────────────────────────
        let raw_partials = store.list_verified_partials_for(sid)?;
        let mut verified_partials: Vec<PartialContributorResult> = Vec::new();
        for p in raw_partials {
            if let Some(asn) = assignment_by_id.get(&p.assignment_id) {
                let outcome = verify_partial_result(asn, &p);
                if outcome.is_ok() {
                    verified_partials.push(p);
                } else {
                    findings.push(IntegrityFinding {
                        kind: FindingKind::InvalidPartial,
                        severity: FindingSeverity::Error,
                        session_id: Some(sid.clone()),
                        path: Some(format!(
                            "verified/sessions/{sid}/partials/{}.json",
                            p.assignment_id
                        )),
                        reason_tag: outcome.reason_tag().to_string(),
                        recommended_action:
                            RecommendedAction::RUN_PLAN_REASSIGN_INVALID_PARTIAL,
                    });
                }
            } else {
                // Orphan partial — assignment was rejected at
                // preload OR the body itself is malformed.
                findings.push(IntegrityFinding {
                    kind: FindingKind::InvalidPartial,
                    severity: FindingSeverity::Error,
                    session_id: Some(sid.clone()),
                    path: Some(format!(
                        "verified/sessions/{sid}/partials/{}.json",
                        p.assignment_id
                    )),
                    reason_tag: "orphan_no_assignment".to_string(),
                    recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
                });
            }
        }

        // ── Aggregate re-verify ────────────────────────────
        if let Some(agg) = store.read_verified_aggregate_for(sid)? {
            let outcome = verify_aggregated_result_with_supersessions(
                session,
                &joins,
                &assignments,
                &supersessions,
                &verified_partials,
                &agg,
            );
            if !outcome.is_ok() {
                findings.push(IntegrityFinding {
                    kind: FindingKind::InvalidAggregate,
                    severity: FindingSeverity::Error,
                    session_id: Some(sid.clone()),
                    path: Some(format!("verified/sessions/{sid}/aggregated.json")),
                    reason_tag: outcome.reason_tag().to_string(),
                    recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
                });
            }
        }

        // ── Per-session status report + audit projection ──
        let status: SessionStatusReport = build_session_status_report(
            store,
            sid,
            opts.now_utc,
            /* include_expired = */ false,
        )?;
        let audit = compute_audit_health(&status);
        session_summaries.push(SessionIntegritySummary {
            session_id: sid.clone(),
            overall_status: format!("{:?}", status.overall_status),
        });
        match audit.coherence {
            AuditCoherence::Coherent => {}
            AuditCoherence::OrphanReplacementAssignments { ref assignment_ids } => {
                findings.push(IntegrityFinding {
                    kind: FindingKind::OrphanReplacementAssignments,
                    severity: FindingSeverity::Error,
                    session_id: Some(sid.clone()),
                    path: None,
                    reason_tag: format!("orphan_count={}", assignment_ids.len()),
                    recommended_action: RecommendedAction::CLEAN_ORPHAN_REPLACEMENTS,
                });
                // Stage 12.17 — feed the side-channel the cleanup
                // planner reads to build per-orphan quarantine
                // actions without re-running compute_audit_health.
                audit_orphans.insert(sid.clone(), assignment_ids.clone());
            }
            AuditCoherence::PartialApplySupersession {
                ref supersession_id,
                unresolved_count,
            } => {
                findings.push(IntegrityFinding {
                    kind: FindingKind::PartialApplySupersession,
                    severity: FindingSeverity::Warn,
                    session_id: Some(sid.clone()),
                    path: None,
                    reason_tag: format!(
                        "supersession_id={supersession_id} unresolved={unresolved_count}"
                    ),
                    recommended_action: RecommendedAction::RUN_SESSION_STATUS,
                });
            }
            AuditCoherence::ReassignTriagable => {
                findings.push(IntegrityFinding {
                    kind: FindingKind::ReassignTriagable,
                    severity: FindingSeverity::Warn,
                    session_id: Some(sid.clone()),
                    path: None,
                    reason_tag: "invalid_state_with_invalid_partial_only".to_string(),
                    recommended_action:
                        RecommendedAction::RUN_PLAN_REASSIGN_INVALID_PARTIAL,
                });
            }
            AuditCoherence::NotReassignTriagable => {
                findings.push(IntegrityFinding {
                    kind: FindingKind::NotReassignTriagable,
                    severity: FindingSeverity::Error,
                    session_id: Some(sid.clone()),
                    path: None,
                    reason_tag: "invalid_state_not_reassignable".to_string(),
                    recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
                });
            }
        }
    }

    // ── 3. Seen-marker ↔ verified-body consistency. ─────────
    scan_seen_marker_consistency(store, opts.session_id_filter, &mut findings)?;

    // ── 4. Stray files inside documented `verified/` subtrees. ─
    scan_stray_files(store.root(), opts.session_id_filter, &mut findings)?;

    // ── 5. Stage 12.17 — stray files / shape-malformed keys
    //       under `seen/`. Catches files under unknown namespace
    //       dirs and files whose key shape isn't `<64-hex>` or
    //       `<64-hex>--<64-hex>` for prefixed namespaces.
    scan_stray_seen_files(store.root(), opts.session_id_filter, &mut findings)?;

    // ── 6. Optional --include-archives walker. ──────────────
    if let Some(archive_dir) = opts.archive_dir {
        scan_archive_dir(store, archive_dir, opts, &mut findings)?;
    }

    // ── 7. Sort + roll up counts. ───────────────────────────
    findings.sort_by(|a, b| {
        (
            a.session_id.as_deref().unwrap_or(""),
            a.kind.as_str(),
            a.path.as_deref().unwrap_or(""),
            a.reason_tag.as_str(),
        )
            .cmp(&(
                b.session_id.as_deref().unwrap_or(""),
                b.kind.as_str(),
                b.path.as_deref().unwrap_or(""),
                b.reason_tag.as_str(),
            ))
    });
    let counts_ok = findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Ok)
        .count() as u32;
    let counts_warn = findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Warn)
        .count() as u32;
    let counts_error = findings
        .iter()
        .filter(|f| f.severity == FindingSeverity::Error)
        .count() as u32;
    session_summaries.sort_by(|a, b| a.session_id.cmp(&b.session_id));

    let report = StateIntegrityReport {
        schema_version: STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
        generated_at_utc: opts.now_utc.to_string(),
        state_dir: store.root().to_string_lossy().replace('\\', "/"),
        state_version: crate::state::STATE_VERSION,
        omni_contributor_version: env!("CARGO_PKG_VERSION").to_string(),
        sessions_scanned,
        sessions_verified,
        counts_ok,
        counts_warn,
        counts_error,
        sessions: session_summaries,
        findings,
    };
    Ok((report, audit_orphans))
}

// ── Stage 12.13 rejection-note parser ─────────────────────────

fn parse_restart_rejection_note(note: &str) -> Option<IntegrityFinding> {
    // Stage 12.13 format:
    //   "kind=<...> session_id=<hex|-> id=<...> reason_tag=<...>"
    let mut kind: Option<&str> = None;
    let mut session_id: Option<String> = None;
    let mut id: Option<String> = None;
    let mut reason_tag: Option<String> = None;
    for token in note.split_whitespace() {
        if let Some(v) = token.strip_prefix("kind=") {
            kind = Some(v);
        } else if let Some(v) = token.strip_prefix("session_id=") {
            session_id = Some(v.to_string());
        } else if let Some(v) = token.strip_prefix("id=") {
            id = Some(v.to_string());
        } else if let Some(v) = token.strip_prefix("reason_tag=") {
            reason_tag = Some(v.to_string());
        }
    }
    let kind = kind?;
    let reason_tag = reason_tag.unwrap_or_else(|| "unspecified".to_string());
    let session_id_for_finding = session_id.clone();
    let (finding_kind, path) = match kind {
        "session" => (
            FindingKind::InvalidSession,
            session_id
                .as_ref()
                .map(|sid| format!("verified/sessions/{sid}/session.json")),
        ),
        "join" => {
            let path = match (&session_id, &id) {
                (Some(sid), Some(pubkey)) => {
                    Some(format!("verified/sessions/{sid}/joins/{pubkey}.json"))
                }
                _ => None,
            };
            (FindingKind::InvalidJoin, path)
        }
        "assignment" => {
            let path = match (&session_id, &id) {
                (Some(sid), Some(asn)) => Some(format!(
                    "verified/sessions/{sid}/assignments/{asn}.json"
                )),
                _ => None,
            };
            (FindingKind::InvalidAssignment, path)
        }
        "supersession" => {
            let path = match (&session_id, &id) {
                (Some(sid), Some(sup)) => Some(format!(
                    "verified/sessions/{sid}/supersessions/{sup}.json"
                )),
                _ => None,
            };
            (FindingKind::InvalidSupersession, path)
        }
        _ => return None,
    };
    Some(IntegrityFinding {
        kind: finding_kind,
        severity: FindingSeverity::Error,
        session_id: session_id_for_finding,
        path,
        reason_tag,
        recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
    })
}

// ── Seen-marker ↔ verified-body consistency walker ────────────

fn scan_seen_marker_consistency(
    store: &ContributorStateStore,
    session_filter: Option<&str>,
    findings: &mut Vec<IntegrityFinding>,
) -> Result<(), IntegrityError> {
    let root = store.root();

    // For each session subtree we know about, derive the
    // expected seen markers (one per body) and compare against
    // disk. Also walk seen/ namespaces to find markers without
    // verified bodies.
    let sessions_dir = root.join("verified").join("sessions");
    let mut session_ids: Vec<String> = if sessions_dir.is_dir() {
        std::fs::read_dir(&sessions_dir)
            .map_err(|e| IntegrityError::Io {
                path: sessions_dir.clone(),
                source: e,
            })?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| {
                e.path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(String::from)
            })
            .collect()
    } else {
        Vec::new()
    };
    if let Some(filter) = session_filter {
        session_ids.retain(|s| s == filter);
    }
    session_ids.sort();

    for sid in &session_ids {
        // Body → expected (ns, key) tuples.
        let body_pairs: Vec<(StateNamespace, String, String)> = {
            let mut out: Vec<(StateNamespace, String, String)> = Vec::new();
            let session_body = root
                .join("verified")
                .join("sessions")
                .join(sid)
                .join("session.json");
            if session_body.is_file() {
                out.push((
                    StateNamespace::Sessions,
                    sid.clone(),
                    format!("verified/sessions/{sid}/session.json"),
                ));
            }
            let agg_body = root
                .join("verified")
                .join("sessions")
                .join(sid)
                .join("aggregated.json");
            if agg_body.is_file() {
                out.push((
                    StateNamespace::Aggregates,
                    sid.clone(),
                    format!("verified/sessions/{sid}/aggregated.json"),
                ));
            }
            for (leaf, ns) in [
                ("joins", StateNamespace::Joins),
                ("assignments", StateNamespace::Assignments),
                ("partials", StateNamespace::Partials),
                ("peer-adverts", StateNamespace::PeerAdverts),
                ("supersessions", StateNamespace::AssignmentSupersessions),
            ] {
                let dir = root
                    .join("verified")
                    .join("sessions")
                    .join(sid)
                    .join(leaf);
                if !dir.is_dir() {
                    continue;
                }
                let entries = std::fs::read_dir(&dir).map_err(|e| {
                    IntegrityError::Io {
                        path: dir.clone(),
                        source: e,
                    }
                })?;
                for e in entries {
                    let e = e.map_err(|e| IntegrityError::Io {
                        path: dir.clone(),
                        source: e,
                    })?;
                    let p = e.path();
                    if !p.is_file() {
                        continue;
                    }
                    let stem = match p.file_stem().and_then(|s| s.to_str()) {
                        Some(s) => s.to_string(),
                        None => continue,
                    };
                    let key = format!("{sid}--{stem}");
                    let rel = format!(
                        "verified/sessions/{sid}/{leaf}/{}.json",
                        stem
                    );
                    out.push((ns, key, rel));
                }
            }
            out
        };

        for (ns, key, rel) in &body_pairs {
            let seen = store.is_seen(*ns, key)?;
            if !seen {
                findings.push(IntegrityFinding {
                    kind: FindingKind::MissingSeenMarker,
                    severity: FindingSeverity::Warn,
                    session_id: Some(sid.clone()),
                    path: Some(rel.clone()),
                    reason_tag: "seen_marker_missing".to_string(),
                    recommended_action: RecommendedAction::RUN_SESSION_STATUS,
                });
            }
        }
    }

    // Reverse direction: walk seen/<ns>/* and flag any marker
    // whose key implies a session_id that isn't in
    // session_ids OR whose corresponding body file doesn't
    // exist. Only check the namespaces Stage 12.14 archives.
    let session_set: HashSet<&String> = session_ids.iter().collect();
    let seen = root.join("seen");
    // Per-namespace metadata: (seen-side dir name, verified-side
    // leaf name, marker-key-is-prefixed-with-session-id). For
    // every namespace except `assignment-supersessions` the seen
    // dir and verified leaf coincide; supersessions is the one
    // historical mismatch (`seen/assignment-supersessions/` ↔
    // `verified/sessions/<sid>/supersessions/`).
    for (ns_dir, verified_leaf, prefixed) in [
        ("sessions", "sessions", false),
        ("aggregates", "aggregates", false),
        ("joins", "joins", true),
        ("assignments", "assignments", true),
        ("partials", "partials", true),
        ("peer-adverts", "peer-adverts", true),
        ("assignment-supersessions", "supersessions", true),
    ] {
        let dir = seen.join(ns_dir);
        if !dir.is_dir() {
            continue;
        }
        let entries = std::fs::read_dir(&dir).map_err(|e| IntegrityError::Io {
            path: dir.clone(),
            source: e,
        })?;
        for e in entries {
            let e = e.map_err(|e| IntegrityError::Io {
                path: dir.clone(),
                source: e,
            })?;
            let p = e.path();
            if !p.is_file() {
                continue;
            }
            let key = match p.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            // Stage 12.17 — shape-malformed keys (wrong hex
            // length, missing `<sid>--` split for prefixed
            // namespaces) are handled exclusively by
            // `scan_stray_seen_files`. Skip them here so the
            // same file doesn't accumulate both a
            // `StaleSeenMarker` and a `StraySeenFile` finding.
            let (sid_for_marker, body_rel) = if prefixed {
                let (sid, suffix) = match key.split_once("--") {
                    Some(parts) => parts,
                    None => continue,
                };
                if !is_64_hex(sid) || !is_64_hex(suffix) {
                    continue;
                }
                let body_rel = format!(
                    "verified/sessions/{sid}/{verified_leaf}/{suffix}.json"
                );
                (Some(sid.to_string()), Some(body_rel))
            } else {
                if !is_64_hex(&key) {
                    continue;
                }
                let body_rel = if verified_leaf == "sessions" {
                    Some(format!("verified/sessions/{key}/session.json"))
                } else if verified_leaf == "aggregates" {
                    Some(format!("verified/sessions/{key}/aggregated.json"))
                } else {
                    None
                };
                (Some(key.clone()), body_rel)
            };
            if let Some(filter) = session_filter {
                if sid_for_marker.as_deref() != Some(filter) {
                    continue;
                }
            }
            let session_present = sid_for_marker
                .as_ref()
                .map(|s| session_set.contains(s))
                .unwrap_or(false);
            let body_present = match &body_rel {
                Some(rel) => root.join(rel).is_file(),
                None => false,
            };
            if !session_present || !body_present {
                findings.push(IntegrityFinding {
                    kind: FindingKind::StaleSeenMarker,
                    severity: FindingSeverity::Warn,
                    session_id: sid_for_marker,
                    path: Some(format!("seen/{ns_dir}/{key}")),
                    reason_tag: "verified_body_missing".to_string(),
                    recommended_action: RecommendedAction::DELETE_STALE_SEEN_MARKER,
                });
            }
        }
    }
    Ok(())
}

// ── Stray-file walker ─────────────────────────────────────────

fn scan_stray_files(
    root: &Path,
    session_filter: Option<&str>,
    findings: &mut Vec<IntegrityFinding>,
) -> Result<(), IntegrityError> {
    // Walk verified/sessions/<id>/... — any file whose
    // (sid, segments) shape isn't recognized = StrayVerifiedFile.
    let sessions_dir = root.join("verified").join("sessions");
    if sessions_dir.is_dir() {
        let entries = std::fs::read_dir(&sessions_dir).map_err(|e| {
            IntegrityError::Io {
                path: sessions_dir.clone(),
                source: e,
            }
        })?;
        for e in entries {
            let e = e.map_err(|e| IntegrityError::Io {
                path: sessions_dir.clone(),
                source: e,
            })?;
            let session_path = e.path();
            if !session_path.is_dir() {
                // A FILE directly under verified/sessions/ — not
                // a recognized layout member.
                if let Some(name) = session_path
                    .file_name()
                    .and_then(|s| s.to_str())
                {
                    findings.push(IntegrityFinding {
                        kind: FindingKind::StrayVerifiedFile,
                        severity: FindingSeverity::Warn,
                        session_id: None,
                        path: Some(format!("verified/sessions/{name}")),
                        reason_tag: "unexpected_file_under_sessions_dir".to_string(),
                        recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
                    });
                }
                continue;
            }
            let sid = match session_path.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            if let Some(filter) = session_filter {
                if sid != filter {
                    continue;
                }
            }
            walk_session_subtree(root, &sid, &session_path, findings)?;
        }
    }
    Ok(())
}

fn walk_session_subtree(
    root: &Path,
    sid: &str,
    session_path: &Path,
    findings: &mut Vec<IntegrityFinding>,
) -> Result<(), IntegrityError> {
    let allowed_files: HashSet<&str> = ["session.json", "aggregated.json"]
        .into_iter()
        .collect();
    let allowed_dirs: HashSet<&str> = [
        "joins",
        "assignments",
        "partials",
        "peer-adverts",
        "supersessions",
    ]
    .into_iter()
    .collect();
    let entries = std::fs::read_dir(session_path).map_err(|e| IntegrityError::Io {
        path: session_path.to_path_buf(),
        source: e,
    })?;
    for e in entries {
        let e = e.map_err(|e| IntegrityError::Io {
            path: session_path.to_path_buf(),
            source: e,
        })?;
        let p = e.path();
        let name = match p.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let rel = match p.strip_prefix(root) {
            Ok(r) => r.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };
        if p.is_file() {
            if !allowed_files.contains(name.as_str()) {
                findings.push(IntegrityFinding {
                    kind: FindingKind::StrayVerifiedFile,
                    severity: FindingSeverity::Warn,
                    session_id: Some(sid.to_string()),
                    path: Some(rel),
                    reason_tag: "unrecognized_file_at_session_root".to_string(),
                    recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
                });
            }
        } else if p.is_dir() {
            if !allowed_dirs.contains(name.as_str()) {
                findings.push(IntegrityFinding {
                    kind: FindingKind::StrayVerifiedFile,
                    severity: FindingSeverity::Warn,
                    session_id: Some(sid.to_string()),
                    path: Some(rel),
                    reason_tag: "unrecognized_directory_at_session_root".to_string(),
                    recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
                });
                continue;
            }
            // Inside an allowed dir: each file should be
            // `<hex>.json`. Anything else = stray.
            let inner = std::fs::read_dir(&p).map_err(|e| IntegrityError::Io {
                path: p.clone(),
                source: e,
            })?;
            for ie in inner {
                let ie = ie.map_err(|e| IntegrityError::Io {
                    path: p.clone(),
                    source: e,
                })?;
                let ip = ie.path();
                let iname = match ip.file_name().and_then(|s| s.to_str()) {
                    Some(s) => s.to_string(),
                    None => continue,
                };
                let irel = match ip.strip_prefix(root) {
                    Ok(r) => r.to_string_lossy().replace('\\', "/"),
                    Err(_) => continue,
                };
                let json_stem_is_hex64 = iname
                    .strip_suffix(".json")
                    .map(|stem| stem.len() == 64
                        && stem.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
                    .unwrap_or(false);
                if !ip.is_file() || !json_stem_is_hex64 {
                    findings.push(IntegrityFinding {
                        kind: FindingKind::StrayVerifiedFile,
                        severity: FindingSeverity::Warn,
                        session_id: Some(sid.to_string()),
                        path: Some(irel),
                        reason_tag: "unrecognized_file_in_session_subtree".to_string(),
                        recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
                    });
                }
            }
        }
    }
    Ok(())
}

// ── Stage 12.17 — stray-seen walker + helpers ─────────────────

/// Lowercase-hex / length-64 predicate. Mirrors the
/// `state::is_blake3_hex` private helper but stays local so the
/// integrity module doesn't take a new state-module surface.
fn is_64_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Closed set of namespace dir names valid under `seen/`. Mirrors
/// the `(seen_dir, verified_leaf, prefixed)` tuple inlined in
/// [`scan_seen_marker_consistency`].
const SEEN_NAMESPACES: &[(&str, bool)] = &[
    ("sessions", false),
    ("aggregates", false),
    ("joins", true),
    ("assignments", true),
    ("partials", true),
    ("peer-adverts", true),
    ("assignment-supersessions", true),
];

/// Walk `seen/...` and emit `StraySeenFile` for:
///   1. Any file directly under `seen/` (no namespace dir).
///   2. Any file under `seen/<unknown-dir>/`.
///   3. Any file inside a known namespace dir whose key shape
///      isn't `<64-hex>` (unprefixed namespaces) or
///      `<64-hex>--<64-hex>` (prefixed namespaces).
///
/// Stage 12.16 v1 declared `StraySeenFile` in `FindingKind` but
/// never emitted it — Stage 12.17 wires the emission so the
/// cleanup planner has a closed-set finding to act on. The
/// `scan_seen_marker_consistency` reverse-walk was made
/// shape-strict in tandem so the same on-disk file never
/// accumulates both a `StaleSeenMarker` and a `StraySeenFile`
/// finding.
fn scan_stray_seen_files(
    root: &Path,
    session_filter: Option<&str>,
    findings: &mut Vec<IntegrityFinding>,
) -> Result<(), IntegrityError> {
    let seen_root = root.join("seen");
    if !seen_root.is_dir() {
        return Ok(());
    }

    let namespace_lookup: HashMap<&str, bool> = SEEN_NAMESPACES
        .iter()
        .map(|(name, prefixed)| (*name, *prefixed))
        .collect();

    let top_entries =
        std::fs::read_dir(&seen_root).map_err(|e| IntegrityError::Io {
            path: seen_root.clone(),
            source: e,
        })?;
    for top in top_entries {
        let top = top.map_err(|e| IntegrityError::Io {
            path: seen_root.clone(),
            source: e,
        })?;
        let top_path = top.path();
        let top_name = match top_path.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if top_path.is_file() {
            // (1) file directly under seen/ — never valid layout.
            if session_filter.is_some() {
                // A file at seen/<name> has no session_id; skip
                // when a filter is set — operators asking about
                // one session shouldn't see cross-cutting noise.
                continue;
            }
            findings.push(IntegrityFinding {
                kind: FindingKind::StraySeenFile,
                severity: FindingSeverity::Warn,
                session_id: None,
                path: Some(format!("seen/{top_name}")),
                reason_tag: "unexpected_file_under_seen_root".to_string(),
                recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
            });
            continue;
        }
        if !top_path.is_dir() {
            continue;
        }
        let prefixed = match namespace_lookup.get(top_name.as_str()) {
            Some(p) => *p,
            None => {
                // (2) unknown namespace dir — every file under it
                // is stray.
                walk_unknown_seen_namespace(
                    root,
                    &top_path,
                    &top_name,
                    session_filter,
                    findings,
                )?;
                continue;
            }
        };
        // (3) known namespace dir — flag shape-violating files.
        let inner = std::fs::read_dir(&top_path).map_err(|e| IntegrityError::Io {
            path: top_path.clone(),
            source: e,
        })?;
        for ie in inner {
            let ie = ie.map_err(|e| IntegrityError::Io {
                path: top_path.clone(),
                source: e,
            })?;
            let ip = ie.path();
            if !ip.is_file() {
                continue;
            }
            let key = match ip.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let (sid_for_filter, shape_ok) = if prefixed {
                match key.split_once("--") {
                    Some((sid, suffix)) => {
                        let ok = is_64_hex(sid) && is_64_hex(suffix);
                        (Some(sid.to_string()), ok)
                    }
                    None => (None, false),
                }
            } else {
                let ok = is_64_hex(&key);
                (Some(key.clone()), ok)
            };
            if shape_ok {
                continue;
            }
            if let Some(filter) = session_filter {
                if sid_for_filter.as_deref() != Some(filter) {
                    continue;
                }
            }
            findings.push(IntegrityFinding {
                kind: FindingKind::StraySeenFile,
                severity: FindingSeverity::Warn,
                session_id: sid_for_filter,
                path: Some(format!("seen/{top_name}/{key}")),
                reason_tag: "seen_key_shape_violation".to_string(),
                recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
            });
        }
    }
    Ok(())
}

fn walk_unknown_seen_namespace(
    root: &Path,
    ns_path: &Path,
    ns_name: &str,
    session_filter: Option<&str>,
    findings: &mut Vec<IntegrityFinding>,
) -> Result<(), IntegrityError> {
    if session_filter.is_some() {
        // Files under an unknown namespace dir have no
        // session_id; suppress under filter.
        return Ok(());
    }
    let _ = root;
    let entries = std::fs::read_dir(ns_path).map_err(|e| IntegrityError::Io {
        path: ns_path.to_path_buf(),
        source: e,
    })?;
    for e in entries {
        let e = e.map_err(|e| IntegrityError::Io {
            path: ns_path.to_path_buf(),
            source: e,
        })?;
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let key = match p.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        findings.push(IntegrityFinding {
            kind: FindingKind::StraySeenFile,
            severity: FindingSeverity::Warn,
            session_id: None,
            path: Some(format!("seen/{ns_name}/{key}")),
            reason_tag: "unknown_seen_namespace".to_string(),
            recommended_action: RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
        });
    }
    Ok(())
}

// ── Optional --include-archives walker ─────────────────────────

fn scan_archive_dir(
    store: &ContributorStateStore,
    archive_dir: &Path,
    opts: &ScanOptions<'_>,
    findings: &mut Vec<IntegrityFinding>,
) -> Result<(), IntegrityError> {
    use crate::restore::{restore_session_archive, RestoreOptions, RestoreSource};

    if !archive_dir.is_dir() {
        return Ok(());
    }
    let entries = std::fs::read_dir(archive_dir).map_err(|e| IntegrityError::Io {
        path: archive_dir.to_path_buf(),
        source: e,
    })?;
    let mut archive_session_ids: Vec<String> = Vec::new();
    for e in entries {
        let e = e.map_err(|e| IntegrityError::Io {
            path: archive_dir.to_path_buf(),
            source: e,
        })?;
        let p = e.path();
        if !p.is_dir() {
            continue;
        }
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if let Some(filter) = opts.session_id_filter {
                if name != filter {
                    continue;
                }
            }
            archive_session_ids.push(name.to_string());
        }
    }
    archive_session_ids.sort();

    let state_session_ids: HashSet<String> = store
        .list_verified_sessions()?
        .into_iter()
        .map(|(s, _)| s)
        .collect();

    for sid in &archive_session_ids {
        let source = RestoreSource::ArchiveRoot {
            archive_dir,
            session_id: sid,
        };
        let restore_opts = RestoreOptions {
            source,
            // verify_only=true + dry_run=true is the Stage 12.15
            // review-fix joint mode: verify-only wins, so the
            // BLAKE3 walk runs but the state-store write path
            // is bypassed end-to-end. Read-only by construction.
            dry_run: true,
            verify_only: true,
            overwrite_existing: false,
            include_results: false,
            now_utc: opts.now_utc,
        };
        match restore_session_archive(store, &restore_opts) {
            Ok(_) => {
                if state_session_ids.contains(sid) {
                    findings.push(IntegrityFinding {
                        kind: FindingKind::ArchiveCoveredSession,
                        severity: FindingSeverity::Ok,
                        session_id: Some(sid.clone()),
                        path: Some(format!("{}/{sid}", archive_dir.display())),
                        reason_tag: "archive_present_with_state".to_string(),
                        recommended_action: RecommendedAction::NONE,
                    });
                }
            }
            Err(e) => {
                let (kind, reason_tag, severity, action) =
                    classify_restore_error(&e);
                findings.push(IntegrityFinding {
                    kind,
                    severity,
                    session_id: Some(sid.clone()),
                    path: Some(format!("{}/{sid}", archive_dir.display())),
                    reason_tag,
                    recommended_action: action,
                });
            }
        }
    }
    Ok(())
}

fn classify_restore_error(
    e: &crate::error::RestoreError,
) -> (FindingKind, String, FindingSeverity, RecommendedAction) {
    use crate::error::RestoreError as R;
    match e {
        R::BlakeMismatch { .. } => (
            FindingKind::ArchiveBlakeMismatch,
            "blake3_mismatch".to_string(),
            FindingSeverity::Error,
            RecommendedAction::RUN_ARCHIVE_VERIFY_ONLY,
        ),
        R::ManifestFileMissing { .. } => (
            FindingKind::ArchiveManifestMalformed,
            "manifest_file_missing".to_string(),
            FindingSeverity::Error,
            RecommendedAction::RUN_RESTORE_VERIFY_ONLY,
        ),
        R::ManifestMissing { .. } => (
            FindingKind::ArchiveManifestMalformed,
            "manifest_missing".to_string(),
            FindingSeverity::Error,
            RecommendedAction::RUN_RESTORE_VERIFY_ONLY,
        ),
        R::MalformedManifest { .. } => (
            FindingKind::ArchiveManifestMalformed,
            "malformed_manifest".to_string(),
            FindingSeverity::Error,
            RecommendedAction::RUN_RESTORE_VERIFY_ONLY,
        ),
        R::UnsupportedManifestVersion { got, expected } => (
            FindingKind::ArchiveManifestMalformed,
            format!("unsupported_manifest_version got={got} expected={expected}"),
            FindingSeverity::Error,
            RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
        ),
        R::IncompatibleSourceStateVersion { archive, current } => (
            FindingKind::ArchiveManifestMalformed,
            format!("incompatible_source_state_version archive={archive} current={current}"),
            FindingSeverity::Error,
            RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
        ),
        R::SessionIdMismatch { .. } => (
            FindingKind::ArchiveManifestMalformed,
            "session_id_mismatch".to_string(),
            FindingSeverity::Error,
            RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
        ),
        R::UnsafeRelativePath { .. } => (
            FindingKind::ArchiveManifestMalformed,
            "unsafe_relative_path".to_string(),
            FindingSeverity::Error,
            RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
        ),
        R::DisallowedRelativePath { .. } => (
            FindingKind::ArchiveManifestMalformed,
            "disallowed_relative_path".to_string(),
            FindingSeverity::Error,
            RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
        ),
        R::ArchiveNotFound { .. } | R::DestinationExists { .. } | R::Io { .. } | R::State(_) => (
            FindingKind::ArchiveManifestMalformed,
            "archive_io_or_state_error".to_string(),
            FindingSeverity::Error,
            RecommendedAction::OPERATOR_TRIAGE_REQUIRED,
        ),
    }
}
