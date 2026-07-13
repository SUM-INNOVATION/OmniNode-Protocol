//! O5 (issue #103) — error types for the extracted operator observability
//! report-generator modules (status, repair, resume, integrity, archive,
//! restore, cleanup). Moved verbatim from `omni-contributor::error`;
//! `StateError` is imported from the retained `omni-contributor`
//! persistence substrate.

use omni_contributor::error::StateError;

/// Stage 12.9 — typed errors from the local session-status reporter.
///
/// `SessionStatusReport` is a local read-only snapshot — it never
/// leaves the operator's machine, is never signed, and is never
/// SNIP-published. Errors here describe why the reporter could not
/// load + re-verify enough state to produce the report at all. Bad
/// individual artifacts (forged joins, tampered partials) do NOT
/// produce errors — they produce report `notes` + `InvalidState`
/// overall status.
#[derive(Debug, thiserror::Error)]
pub enum StatusError {
    #[error("state-dir error: {0}")]
    State(#[from] StateError),

    #[error(
        "status reporter schema_version {got} not supported (expected {expected})"
    )]
    UnsupportedSchemaVersion { got: u32, expected: u32 },
}

/// Stage 12.10 — typed errors from the local pooled-session repair
/// planner / applier.
///
/// `SessionRepairPlan` is a local read-only operator hint — never
/// signed, never SNIP-published, never network-visible. Errors here
/// describe why the planner refused to emit actions, or why the
/// applier refused to publish them.
#[derive(Debug, thiserror::Error)]
pub enum RepairError {
    #[error(
        "repair planner schema_version {got} not supported (expected {expected})"
    )]
    UnsupportedSchemaVersion { got: u32, expected: u32 },

    /// The supplied `SessionStatusReport` describes a session that
    /// isn't on disk in the state-dir.
    #[error("no session: session_id={session_id} is not in the state-dir")]
    SessionNotPresent { session_id: String },

    /// Status is `CompletePartials`, `Aggregated`, or `NoAssignments`
    /// — the session does not need a reannounce-missing repair.
    #[error("nothing to repair: session status is {status:?}")]
    NothingToRepair { status: String },

    /// Status is `InvalidState`. Operators must clean tampered
    /// artifacts before repair planning. Stage 12.10 does NOT
    /// surface an `--allow-invalid-state` flag.
    #[error(
        "session has InvalidState; clean tampered artifacts before \
         repair (see Stage 12.9 status report notes)"
    )]
    InvalidState,

    /// Status is `ExpiredIncomplete`. Reannouncing past-expiry
    /// assignments is mostly noise; operators wanting to repair an
    /// expired session must extend it via `open-session` first.
    #[error(
        "session is expired (ExpiredIncomplete); reannouncing past \
         expiry is rejected"
    )]
    SessionExpired,

    /// Apply-time: the on-disk state's assignment-vs-partial shape
    /// has drifted from the plan's `source_status_hash` projection.
    /// Operator should re-plan from a fresh status report.
    #[error(
        "source status drift: plan was built against a session shape \
         that no longer matches the state-dir (a partial may have \
         arrived); re-plan from a fresh status report"
    )]
    SourceStatusDrift,

    /// Apply-time: the plan's `repair_plan_hash` does not match the
    /// recomputed hash of the loaded bytes. The plan was edited
    /// after creation.
    #[error(
        "repair_plan_hash drift: stored={stored} recomputed={recomputed} \
         — the plan may have been edited after creation"
    )]
    PlanHashDrift { stored: String, recomputed: String },

    /// Apply-time: a referenced assignment_id is not on disk in the
    /// state-dir's `verified/sessions/<id>/assignments/`.
    #[error(
        "assignment not in state-dir: session_id={session_id} \
         assignment_id={assignment_id}"
    )]
    AssignmentNotPresent {
        session_id: String,
        assignment_id: String,
    },

    /// Stage 12.12 apply-time refusal — `InvalidState` is not
    /// triagable via the `--reason invalid-partial` reassign path
    /// because of an additional non-`InvalidPartial` chain failure
    /// (e.g. invalid join, invalid assignment, invalid aggregate,
    /// invalid session, invalid supersession), OR because an
    /// `InvalidPartial` entry exists for an assignment NOT in the
    /// reassignment plan's superseded set.
    ///
    /// `kind` is one of: `"invalid_session"`, `"invalid_join"`,
    /// `"invalid_assignment"`, `"invalid_aggregate"`,
    /// `"invalid_supersession"`, `"invalid_partial_not_in_plan"`.
    /// `context` carries the relevant id (`assignment_id=...`,
    /// `contributor_pubkey_hex=...`, `supersession_id=...`) or is
    /// empty for `invalid_session` / `invalid_aggregate`. Both
    /// fields are stable across `schema_version: 3` for
    /// scripting.
    #[error("InvalidState not triagable via reassign: kind={kind}{context}")]
    InvalidStateNotTriagable {
        kind: &'static str,
        /// Optional id context. When non-empty, MUST start with a
        /// leading space (e.g. `" assignment_id=ff..."`) so the
        /// `Display` impl renders cleanly.
        context: String,
    },

    /// Stage 12.11 apply-time enforcement. A `ReassignAssignment`
    /// action targeted an assignment that, in the current rebuilt
    /// status, is NOT active-missing: it is either already
    /// completed (`partial_present == true`), already superseded
    /// by a prior verified `WorkAssignmentSupersession`, missing
    /// from the status report entirely, or has a `stage_index`
    /// that disagrees with the plan's `original_stage_index`.
    ///
    /// The plan is unsigned and local — a hand-edited plan can
    /// recompute its `repair_plan_hash` so the integrity check
    /// passes, so the applier must independently re-verify that
    /// every targeted assignment is still safe to retire.
    #[error(
        "reassign target not active-missing: session_id={session_id} \
         assignment_id={assignment_id} reason={reason}"
    )]
    ReassignTargetNotActiveMissing {
        session_id: String,
        assignment_id: String,
        /// `not_in_status` / `already_superseded` / `already_completed` /
        /// `stage_index_mismatch`.
        reason: &'static str,
    },
}

/// Stage 12.14 — local session archival errors. Every variant is
/// operator-actionable and carries enough context to triage
/// without re-running the command.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// The state-dir holds no `verified/sessions/<session_id>/`
    /// subtree. The session was never seen by this watcher, or has
    /// already been pruned/archived/cascaded out.
    #[error("session not present in state-dir: session_id={session_id}")]
    SessionNotPresent { session_id: String },

    /// The rebuilt `SessionStatusReport.overall_status` did not
    /// satisfy the `--require-status` policy. `got` and
    /// `requirement` are the stable Debug-stringified discriminators
    /// (closed sets) so scripts can pattern-match.
    #[error(
        "session overall_status {got} does not satisfy --require-status {requirement}"
    )]
    StatusRequirementUnmet {
        got: String,
        requirement: String,
    },

    /// The archive destination already holds a subtree named
    /// `<session_id>/`. Stage 12.14 refuses to overwrite — operators
    /// must move/rename/delete the existing archive directory before
    /// re-running.
    #[error("archive directory already contains session: {path}")]
    ArchiveAlreadyExists { path: std::path::PathBuf },

    /// BLAKE3 of the destination file did not match the BLAKE3
    /// computed at copy time. Fail-fast (no retry); operator must
    /// triage the FS / hardware before retrying.
    #[error(
        "blake3 mismatch on copied file: path={path} expected={expected} got={got}"
    )]
    BlakeMismatch {
        path: std::path::PathBuf,
        expected: String,
        got: String,
    },

    /// Generic FS error from a `fs::read` / `fs::write` /
    /// `fs::copy` call. The `path` field names the artifact that
    /// failed so operators can `ls` it.
    #[error("archive io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Bubbled-up `StatusError` from
    /// `build_session_status_report`. The archive entry point
    /// builds the status report once (to enforce
    /// `--require-status` + populate the manifest's
    /// `session_overall_status` field).
    #[error("status build for archive: {0}")]
    Status(#[from] StatusError),

    /// Bubbled-up `StateError`.
    #[error("state error: {0}")]
    State(#[from] StateError),
}

/// Stage 12.15 — local session archive restore errors. The
/// inverse of `ArchiveError`: each variant names exactly one
/// safety check the restore path enforces before any state-dir
/// write.
#[derive(Debug, thiserror::Error)]
pub enum RestoreError {
    /// The supplied archive directory does not exist on disk.
    #[error("archive directory not found: {path}")]
    ArchiveNotFound { path: std::path::PathBuf },

    /// The expected `manifest.json` is missing from the archive
    /// directory. Stage 12.14 writes the manifest LAST, so a
    /// missing manifest typically means the archive was created
    /// by a tool that crashed mid-copy.
    #[error("manifest.json missing at {path}")]
    ManifestMissing { path: std::path::PathBuf },

    /// `manifest.json` did not parse as a v1 `ArchiveManifest`.
    /// Bubbled `serde_json::Error` carries the column/line.
    #[error("malformed archive manifest at {path}: {source}")]
    MalformedManifest {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// `manifest.schema_version` is not `ARCHIVE_MANIFEST_SCHEMA_VERSION`.
    /// Stage 12.15 v1 accepts exactly version `1`.
    #[error(
        "unsupported archive manifest schema_version: got={got} expected={expected}"
    )]
    UnsupportedManifestVersion { got: u32, expected: u32 },

    /// The `manifest.session_id` field does not match the
    /// session_id derived from the archive directory name (or
    /// the operator-supplied `--session-id`). Defends against
    /// hand-renamed archive directories.
    #[error(
        "session_id mismatch: manifest.session_id={manifest_session_id} \
         dir.session_id={dir_session_id}"
    )]
    SessionIdMismatch {
        manifest_session_id: String,
        dir_session_id: String,
    },

    /// `manifest.source_state_version != STATE_VERSION`. Stage
    /// 12.15 v1 enforces strict equality — restoring across a
    /// state-dir version boundary would require a migration
    /// story that does not exist yet.
    #[error(
        "incompatible source state-dir version: archive={archive} current={current}"
    )]
    IncompatibleSourceStateVersion { archive: u32, current: u32 },

    /// `source_relative` contains `..`, is absolute, or uses a
    /// backslash. Bubbled from `StateError::UnsafeRelativePath`.
    #[error("unsafe relative path in manifest: {path}")]
    UnsafeRelativePath { path: String },

    /// `source_relative` does not match the Stage 12.14 archive
    /// whitelist. Bubbled from `StateError::DisallowedRelativePath`.
    #[error("disallowed relative path in manifest: {path}")]
    DisallowedRelativePath { path: String },

    /// A file named in the manifest does not exist under the
    /// archive directory. Restore is fail-fast — operator
    /// triages then re-runs.
    #[error("archive file missing for manifest entry: {archive_path}")]
    ManifestFileMissing { archive_path: std::path::PathBuf },

    /// BLAKE3 of the archive file did not match the manifest's
    /// `blake3_hex`. Fail-fast (no retry); operator triages
    /// the FS / archive integrity.
    #[error(
        "blake3 mismatch on archive file: path={path} expected={expected} got={got}"
    )]
    BlakeMismatch {
        path: std::path::PathBuf,
        expected: String,
        got: String,
    },

    /// Preflight check found a destination file that already
    /// exists in the state-dir AND `--overwrite-existing` was
    /// false. Stage 12.15 enforces all-or-nothing: any
    /// pre-existing destination refuses BEFORE any state-dir
    /// write happens. Operator re-runs with
    /// `--overwrite-existing` or cleans state-dir first.
    #[error("destination already exists: {path} (re-run with --overwrite-existing)")]
    DestinationExists { path: std::path::PathBuf },

    /// Generic FS error.
    #[error("restore io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Bubbled `StateError` — typically from
    /// `write_archived_bytes`.
    #[error("state error: {0}")]
    State(#[from] StateError),
}

/// Stage 12.16 — local state-dir integrity scan errors. These are
/// **scanner-aborting** only: per-artifact problems (a tampered
/// session, a stale seen marker, a corrupt archive file) are
/// captured as findings in the
/// [`crate::integrity::StateIntegrityReport`], not as variants
/// here. `IntegrityError` is reserved for failures that prevent
/// the scan from even running (e.g. the state-dir itself
/// can't be walked).
#[derive(Debug, thiserror::Error)]
pub enum IntegrityError {
    /// The Stage 12.7 `ContributorStateStore` returned an error
    /// before the scan could start. Wraps the upstream typed
    /// error (e.g. `StateError::UnsupportedVersion`,
    /// `StateError::Io`) so operators see the underlying cause.
    #[error("state error: {0}")]
    State(#[from] StateError),

    /// `build_session_status_report` failed mid-scan. The
    /// scanner runs status build per session to drive the
    /// Stage 12.13 audit projection; a build failure indicates
    /// the state-dir walked OK but a per-session re-verify
    /// chain hit an internal error.
    #[error("status build during integrity scan: {0}")]
    Status(#[from] StatusError),

    /// Generic FS error encountered during stray-file detection
    /// or the optional `--include-archives` directory walk.
    #[error("integrity scan io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Stage 12.17 — local state-dir cleanup planner / applier
/// errors. The cleanup flow is a closed-set composition of
/// Stage 12.16 findings + Stage 12.13 audit projection +
/// Stage 12.14-shaped quarantine writes; this enum surfaces
/// the failures specific to that flow.
///
/// Per-action failures (one action's apply tripped) are NOT
/// captured here — they propagate as the variant the failing
/// primitive returned (`State`, `Status`, `Integrity`, `Io`).
/// The applier records action-level outcomes in its returned
/// report.
#[derive(Debug, thiserror::Error)]
pub enum CleanupError {
    /// `scan_state_integrity` itself failed (state-dir won't
    /// walk, status build crashed, FS error during stray
    /// detection). Plan-time and apply-time entry both wrap
    /// this so callers see one consistent surface.
    #[error("integrity scan during cleanup: {0}")]
    Integrity(#[from] IntegrityError),

    /// Wraps a status build failure that bubbled up through
    /// audit re-projection at apply time.
    #[error("status build during cleanup: {0}")]
    Status(#[from] StatusError),

    /// State-store primitive (`remove_verified_relative`,
    /// `unmark_seen`, `write_archived_bytes`) refused.
    #[error("state error during cleanup: {0}")]
    State(#[from] StateError),

    /// Apply-time `source_integrity_hash` re-projection
    /// disagreed with the plan's recorded hash. The state-dir
    /// has changed between plan and apply; the operator must
    /// re-plan.
    #[error(
        "source integrity drift: plan expected {expected}, current state \
         hashes to {got}; re-run plan-state-cleanup"
    )]
    SourceIntegrityDrift { expected: String, got: String },

    /// The plan file's `cleanup_plan_hash` doesn't match the
    /// BLAKE3 of the canonical body. The plan was hand-edited
    /// or corrupted after write.
    #[error(
        "plan hash mismatch: stored {stored}, recomputed {recomputed}"
    )]
    PlanHashMismatch {
        stored: String,
        recomputed: String,
    },

    /// A gated action (`QuarantineAndUnmarkPartial` /
    /// `QuarantineAndUnmarkOrphanAssignment`) is present in the
    /// plan but the operator did not pass the corresponding
    /// `--allow-…` flag. Plan-time always emits gated actions
    /// when the finding warrants them; apply-time refuses
    /// unless the gate flag is present.
    #[error(
        "gated action {kind} requires {flag}; pass the flag explicitly to apply"
    )]
    GateRequired { kind: String, flag: String },

    /// The audit projection's orphan-assignment set for a
    /// session changed between plan and apply. Mirrors the
    /// `SourceIntegrityDrift` posture but at finer granularity:
    /// the integrity hash may still match if the change is
    /// confined to non-finding fields, so the apply re-checks
    /// `compute_audit_health` per gated session and refuses
    /// when the orphan id set diverges.
    #[error(
        "orphan-assignment audit drift for session {session_id}: plan listed \
         {plan_count} orphans, current projection lists {current_count}; \
         re-run plan-state-cleanup"
    )]
    OrphanAuditDrift {
        session_id: String,
        plan_count: u32,
        current_count: u32,
    },

    /// A quarantine destination already exists. The applier
    /// refuses to overwrite — operator must clear the
    /// quarantine subtree (or pass a fresh `--quarantine-dir`)
    /// before re-running.
    #[error("quarantine destination already exists: {path}")]
    QuarantineCollision { path: std::path::PathBuf },

    /// Malformed plan JSON (schema violation, unknown fields,
    /// missing required fields).
    #[error("malformed plan at {path}: {source}")]
    MalformedPlan {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// Plan-schema version is not supported by this binary.
    /// Stage 12.17 v1 accepts schema_version == 1 only.
    #[error(
        "unsupported cleanup plan schema_version: got {got}, this binary supports {expected}"
    )]
    UnsupportedPlanVersion { got: u32, expected: u32 },

    /// One of the plan's `path` / `seen_marker_path` strings
    /// violates the per-kind whitelist (`verified/sessions/...`
    /// for tier B and `WriteSeenMarker`; `seen/...` for tier A
    /// stray/remove actions; no `..`, no absolute, no backslash,
    /// no empty segments). A self-consistent
    /// `cleanup_plan_hash` does not vouch for path safety —
    /// hand-edited plans can produce malicious paths whose
    /// hash recomputes correctly. Apply-time refuses BEFORE
    /// any IO when this fires.
    #[error("unsafe path in cleanup plan ({reason}): {path}")]
    UnsafePlanPath {
        path: String,
        reason: &'static str,
    },

    /// Generic FS error encountered while reading the plan,
    /// writing quarantine bytes, or walking the state-dir.
    #[error("cleanup io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Stage 12.18 — local cleanup-quarantine restore errors.
/// Consumes the existing v1 `QuarantineManifest` written by
/// Stage 12.17's `apply_state_cleanup`; no manifest schema bump.
///
/// Per-entry path / BLAKE3 / destination-existence problems are
/// fail-fast: the all-or-nothing preflight runs BEFORE any
/// state-dir write, so a single bad entry refuses the whole
/// restore without partially mutating the state-dir.
#[derive(Debug, thiserror::Error)]
pub enum QuarantineRestoreError {
    /// The supplied quarantine plan directory does not exist
    /// on disk.
    #[error("quarantine plan directory not found: {path}")]
    QuarantineDirNotFound { path: std::path::PathBuf },

    /// The expected `quarantine-manifest.json` is missing
    /// from the supplied plan directory. Stage 12.17 writes
    /// the manifest LAST under the Phase A→B→C ordering, so a
    /// missing manifest typically means the Stage 12.17 apply
    /// crashed between Phase A and Phase B.
    #[error("quarantine-manifest.json missing at {path}")]
    ManifestMissing { path: std::path::PathBuf },

    /// `quarantine-manifest.json` did not parse as a v1
    /// `QuarantineManifest`.
    #[error("malformed quarantine manifest at {path}: {source}")]
    MalformedManifest {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// `manifest.schema_version` is not
    /// `QUARANTINE_MANIFEST_SCHEMA_VERSION`. Stage 12.18 v1
    /// accepts exactly version `1`.
    #[error(
        "unsupported quarantine manifest schema_version: got={got} expected={expected}"
    )]
    UnsupportedManifestVersion { got: u32, expected: u32 },

    /// `manifest.source_state_version != STATE_VERSION`. Same
    /// strict-equality posture as Stage 12.15 archive restore.
    #[error(
        "incompatible source state-dir version: manifest={manifest} current={current}"
    )]
    IncompatibleSourceStateVersion { manifest: u32, current: u32 },

    /// `manifest.plan_id` does not match the
    /// caller-supplied `plan_id` (via `--quarantine-dir +
    /// --plan-id`) OR the directory name supplied via
    /// `--quarantine-plan-dir`. Defends against hand-renamed
    /// quarantine directories.
    #[error(
        "plan_id mismatch: manifest.plan_id={manifest_plan_id} \
         supplied={supplied_plan_id}"
    )]
    PlanIdMismatch {
        manifest_plan_id: String,
        supplied_plan_id: String,
    },

    /// An entry's `source_relative` or `quarantine_relative`
    /// contains `..`, is absolute, uses a backslash, has an
    /// empty segment, or fails the closed-set
    /// `verified/sessions/<64hex>/...` whitelist. Hand-edited
    /// manifests with malicious paths get this BEFORE any IO.
    #[error("unsafe relative path in manifest ({reason}): {path}")]
    UnsafeRelativePath {
        path: String,
        reason: &'static str,
    },

    /// A file named in the manifest does not exist under the
    /// quarantine plan directory. Restore is fail-fast.
    #[error("quarantine file missing for manifest entry: {path}")]
    ManifestFileMissing { path: std::path::PathBuf },

    /// BLAKE3 of the quarantine file did not match the
    /// manifest's `blake3_hex`. Fail-fast (no retry); operator
    /// triages the FS / quarantine integrity.
    #[error(
        "blake3 mismatch on quarantine file: path={path} expected={expected} got={got}"
    )]
    BlakeMismatch {
        path: std::path::PathBuf,
        expected: String,
        got: String,
    },

    /// All-or-nothing preflight found a destination that
    /// already exists in the state-dir AND
    /// `--overwrite-existing` was false. Mirrors Stage 12.15.
    #[error(
        "destination already exists: {path} (re-run with --overwrite-existing)"
    )]
    DestinationExists { path: std::path::PathBuf },

    /// All-or-nothing preflight found a seen-marker
    /// destination that is occupied by a non-file (typically
    /// a directory) so `store.mark_seen` would have failed
    /// mid-restore. Refused BEFORE any body write so a marker
    /// problem cannot partially mutate the state-dir.
    /// `--overwrite-existing` does NOT cover this — marker
    /// preflight is unconditional whenever
    /// `restore_seen_markers == true`.
    #[error("seen marker path blocked at {path}: {reason}")]
    SeenMarkerPathBlocked {
        path: std::path::PathBuf,
        reason: &'static str,
    },

    /// A `source_finding_kind` in the manifest matched a
    /// closed-set tag whose restore requires an opt-in flag
    /// the operator didn't pass (e.g. orphan-assignment
    /// entries without `--allow-restore-orphan-assignments`).
    /// Refused BEFORE any FS interaction.
    #[error("gated restore requires {flag}: source_finding_kind={kind}")]
    GatedRestoreRequired {
        kind: &'static str,
        flag: &'static str,
    },

    /// An entry's `source_finding_kind` is not in the Stage
    /// 12.17 closed set. Hand-edited manifest or future-stage
    /// tag — refused so v1 doesn't silently skip new
    /// variants.
    #[error("unknown source_finding_kind in manifest: {kind}")]
    UnknownFindingKind { kind: String },

    /// Bubbled `StateError` — typically from
    /// `write_archived_bytes` or `mark_seen`.
    #[error("state error: {0}")]
    State(#[from] StateError),

    /// Generic FS error.
    #[error("quarantine restore io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}

/// Stage 12.19 — integrity-report diff errors. The differ is a
/// pure JSON-to-JSON comparison; it never opens a state-store
/// or writes state-dir bytes. These variants surface schema /
/// state-version refusals, optional state_dir-pinning refusal,
/// and the v1 "shouldn't happen" finding-metadata-drift guard.
#[derive(Debug, thiserror::Error)]
pub enum IntegrityDiffError {
    /// The baseline report's `schema_version` is not
    /// `STATE_INTEGRITY_REPORT_SCHEMA_VERSION`. Stage 12.19 v1
    /// accepts exactly v1.
    #[error(
        "unsupported baseline schema_version: got={got} expected={expected}"
    )]
    UnsupportedBaselineSchemaVersion { got: u32, expected: u32 },

    /// Same as above but for the `current` report.
    #[error(
        "unsupported current schema_version: got={got} expected={expected}"
    )]
    UnsupportedCurrentSchemaVersion { got: u32, expected: u32 },

    /// `baseline.state_version != current.state_version`.
    /// Cross-state-dir-version diff isn't meaningful without a
    /// migration story.
    #[error(
        "incompatible state-dir version: baseline={baseline} current={current}"
    )]
    IncompatibleStateVersion { baseline: u32, current: u32 },

    /// `baseline.state_dir != current.state_dir` AND
    /// `--require-state-dir-match` was set. CI baselines are
    /// commonly captured on a different host so this is OFF by
    /// default; operators who want host pinning opt in.
    #[error(
        "state_dir mismatch: baseline={baseline} current={current} \
         (re-run without --require-state-dir-match to ignore)"
    )]
    StateDirMismatch { baseline: String, current: String },

    /// Two findings share the same identity tuple
    /// `(kind, session_id, path, reason_tag)` but disagree on
    /// `severity` or `recommended_action`. v1 treats this as a
    /// structural inconsistency rather than silently collapsing
    /// it; the closed-set scanner deterministically maps each
    /// identity to a fixed (severity, action) pair, so a drift
    /// here means one of the reports was tampered with or
    /// produced by a non-Stage-12.16 tool.
    #[error(
        "finding metadata drift for identity={identity}: \
         severity baseline={baseline_severity} current={current_severity}; \
         action baseline={baseline_recommended_action} \
         current={current_recommended_action}"
    )]
    FindingMetadataDrift {
        identity: String,
        baseline_severity: String,
        current_severity: String,
        baseline_recommended_action: String,
        current_recommended_action: String,
    },

    /// Baseline JSON failed to parse as a v1
    /// `StateIntegrityReport`. Bubbled `serde_json::Error`
    /// carries column/line.
    #[error("malformed baseline at {path}: {source}")]
    MalformedBaseline {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// `current` JSON (typically supplied via
    /// `state-integrity-diff --current <path>`) failed to parse
    /// as a v1 `StateIntegrityReport`.
    #[error("malformed current at {path}: {source}")]
    MalformedCurrent {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    /// Generic FS error encountered while reading a report
    /// JSON.
    #[error("integrity diff io error at {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },
}
