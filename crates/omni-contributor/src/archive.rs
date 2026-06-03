//! Stage 12.14 — local operator session archival + state-dir
//! compaction.
//!
//! `archive_session` walks the `verified/sessions/<session_id>/...`
//! subtree + every session-keyed `seen/*` marker, copies each file
//! byte-for-byte to `<archive-dir>/<session_id>/...`, verifies
//! BLAKE3 of every copy, and writes a manifest LAST so a partial
//! copy is visible (no manifest) without corrupting the source.
//!
//! ## Scope
//!
//! - **Local only.** Archives live under operator-chosen
//!   `<archive-dir>` paths; nothing is published, no chain wire,
//!   no SNIP wire, no gossipsub topic. Archives are inert JSON.
//! - **No new envelope, no new canonical bytes, no
//!   `schema_version` bump on any in-tree envelope.** Verified
//!   files are copied byte-for-byte; their Stage 12.0–12.11
//!   canonical-byte schemas are preserved by definition.
//! - **No `STATE_VERSION` bump.** The archive manifest is a new
//!   local-only document in `<archive-dir>`, NOT in `<state-dir>`.
//!   Its version is the separate
//!   [`ARCHIVE_MANIFEST_SCHEMA_VERSION`] constant.
//!
//! ## Safety contract
//!
//! - `--dry-run` returns a fully-populated `ArchiveManifest`
//!   without touching the FS at all (no archive dir written, no
//!   source removed).
//! - `--copy` (default) copies + verifies; the source is
//!   untouched.
//! - `--move` ONLY runs the destructive cascade
//!   ([`ContributorStateStore::cascade_remove_session`]) AFTER
//!   every file's BLAKE3 verifies AND the manifest write
//!   succeeds. A partial-copy failure leaves the source intact.
//! - BLAKE3 mismatch on any copied file is **fail-fast**: no
//!   retry, no partial accept. Operator triages the FS and
//!   re-runs.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ArchiveError;
use crate::resume::{compute_audit_health, AuditCoherence};
use crate::session::ExecutionSession;
use crate::state::{ContributorStateStore, StateObjectKind};
use crate::status::{
    build_session_status_report, SessionOverallStatus, SessionStatusReport,
};

/// Stage 12.14 — archive manifest schema. Bumps independently of
/// `STATE_VERSION`; the manifest lives in `<archive-dir>`, NOT
/// `<state-dir>`.
pub const ARCHIVE_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// Copy vs Move. Move runs the destructive cascade ONLY after a
/// successful copy + BLAKE3 verify + manifest write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ArchiveMode {
    Copy,
    Move,
}

/// Closed set of status-policy gates the operator can enforce.
/// `Complete` is the safe default (`Aggregated` OR
/// `CompletePartials`). `Any` is the escape valve for InvalidState
/// triage scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ArchiveStatusRequirement {
    Any,
    Complete,
    Aggregated,
    CompletePartials,
    ExpiredIncomplete,
}

impl ArchiveStatusRequirement {
    pub fn satisfied_by(self, status: SessionOverallStatus) -> bool {
        match self {
            ArchiveStatusRequirement::Any => true,
            ArchiveStatusRequirement::Complete => matches!(
                status,
                SessionOverallStatus::Aggregated
                    | SessionOverallStatus::CompletePartials
            ),
            ArchiveStatusRequirement::Aggregated => {
                matches!(status, SessionOverallStatus::Aggregated)
            }
            ArchiveStatusRequirement::CompletePartials => {
                matches!(status, SessionOverallStatus::CompletePartials)
            }
            ArchiveStatusRequirement::ExpiredIncomplete => {
                matches!(status, SessionOverallStatus::ExpiredIncomplete)
            }
        }
    }

    /// Stable string for `--require-status=<x>` CLI parsing and
    /// for the manifest's `require_status` field. Closed set;
    /// renaming a variant must NOT silently change the wire tag.
    pub fn as_str(self) -> &'static str {
        match self {
            ArchiveStatusRequirement::Any => "any",
            ArchiveStatusRequirement::Complete => "complete",
            ArchiveStatusRequirement::Aggregated => "aggregated",
            ArchiveStatusRequirement::CompletePartials => "complete_partials",
            ArchiveStatusRequirement::ExpiredIncomplete => "expired_incomplete",
        }
    }
}

/// Per-file inventory entry in the manifest. `source_relative`
/// and `archive_relative` are both rooted-relative; they are the
/// only paths a reader needs to reconstruct the archive layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchivedFile {
    /// Relative path under the source state-dir
    /// (`<state-dir>/...`). UTF-8 lossy is forbidden; non-UTF-8
    /// state-dir paths are rejected by `archive_session` at
    /// enumeration time.
    pub source_relative: String,
    /// Relative path under
    /// `<archive-dir>/<session_id>/...`.
    pub archive_relative: String,
    /// BLAKE3 of the file bytes at archive time. After copy, the
    /// destination's BLAKE3 must match this string exactly
    /// (lowercase hex, no `0x` prefix); mismatch is a fail-fast
    /// `ArchiveError::BlakeMismatch`.
    pub blake3_hex: String,
    pub bytes: u64,
}

/// Archive operation manifest. Written LAST after every file
/// copies + verifies; a partial-copy failure leaves the
/// `<archive-dir>/<session_id>/` directory visibly missing this
/// manifest, so a future archive reader can refuse it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveManifest {
    pub schema_version: u32,
    pub session_id: String,
    pub generated_at_utc: String,
    /// `STATE_VERSION` of the source state-dir at archive time.
    /// Lets a future reader know which envelope schemas these
    /// files belong to (the per-envelope schema_versions are
    /// inside each file's body).
    pub source_state_version: u32,
    /// `CARGO_PKG_VERSION` of the `omni-contributor` crate at
    /// archive time.
    pub omni_contributor_version: String,
    /// `Debug`-stringified discriminator (closed set) of
    /// `SessionOverallStatus` at archive time — same shape as
    /// the `kind` tag in `ArchiveStatusRequirement::as_str`.
    pub session_overall_status: String,
    /// `Debug`-stringified discriminator (closed set) of
    /// `AuditCoherence` at archive time. Informational only;
    /// scripts can branch on this field to record why the
    /// archive was permitted.
    pub audit_coherence: String,
    pub mode: ArchiveMode,
    pub require_status: ArchiveStatusRequirement,
    pub include_results: bool,
    pub files: Vec<ArchivedFile>,
}

pub struct ArchiveOptions<'a> {
    pub session_id: &'a str,
    pub archive_dir: &'a Path,
    pub mode: ArchiveMode,
    pub require_status: ArchiveStatusRequirement,
    pub include_results: bool,
    pub now_utc: &'a str,
    pub dry_run: bool,
}

/// Run the archive operation. Returns the manifest. On `dry_run`,
/// the manifest is returned without touching the filesystem.
pub fn archive_session(
    store: &ContributorStateStore,
    opts: &ArchiveOptions<'_>,
) -> Result<ArchiveManifest, ArchiveError> {
    // ── 1. Verify the session exists locally + build status ────
    let session: Option<ExecutionSession> =
        store.read_verified_json(StateObjectKind::Session, opts.session_id)?;
    if session.is_none() {
        return Err(ArchiveError::SessionNotPresent {
            session_id: opts.session_id.to_string(),
        });
    }
    let session = session.unwrap();

    let status = build_session_status_report(
        store,
        opts.session_id,
        opts.now_utc,
        /* include_expired = */ true,
    )?;
    if !opts
        .require_status
        .satisfied_by(status.overall_status)
    {
        return Err(ArchiveError::StatusRequirementUnmet {
            got: format!("{:?}", status.overall_status),
            requirement: opts.require_status.as_str().to_string(),
        });
    }

    // ── 2. Enumerate source files ─────────────────────────────
    let mut files = enumerate_session_files(store, opts.session_id)?;
    if opts.include_results {
        if let Some(extra) =
            posted_result_link_file(store, &session.posted_id)?
        {
            files.push(extra);
        }
    }

    // ── 3. Compute BLAKE3 + bytes for every entry ─────────────
    let prepared = compute_file_metadata(store, files)?;

    // ── 4. Build the manifest body ────────────────────────────
    let audit = compute_audit_health(&status);
    let manifest = ArchiveManifest {
        schema_version: ARCHIVE_MANIFEST_SCHEMA_VERSION,
        session_id: opts.session_id.to_string(),
        generated_at_utc: opts.now_utc.to_string(),
        source_state_version: crate::state::STATE_VERSION,
        omni_contributor_version: env!("CARGO_PKG_VERSION").to_string(),
        session_overall_status: format!("{:?}", status.overall_status),
        audit_coherence: format_audit_coherence(&audit.coherence),
        mode: opts.mode,
        require_status: opts.require_status,
        include_results: opts.include_results,
        files: prepared,
    };

    // ── 5. Dry-run short-circuit ──────────────────────────────
    if opts.dry_run {
        return Ok(manifest);
    }

    // ── 6. Refuse pre-existing archive dir ────────────────────
    let dest_root = opts.archive_dir.join(opts.session_id);
    if dest_root.exists() {
        return Err(ArchiveError::ArchiveAlreadyExists { path: dest_root });
    }

    // ── 7. Copy every file + verify BLAKE3 ────────────────────
    for entry in &manifest.files {
        let src = store.root().join(&entry.source_relative);
        let dst = dest_root.join(&entry.archive_relative);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ArchiveError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        let bytes = std::fs::read(&src).map_err(|e| ArchiveError::Io {
            path: src.clone(),
            source: e,
        })?;
        std::fs::write(&dst, &bytes).map_err(|e| ArchiveError::Io {
            path: dst.clone(),
            source: e,
        })?;
        // Re-read + BLAKE3 the destination so a hardware glitch
        // between write and verify is caught.
        let verified = std::fs::read(&dst).map_err(|e| ArchiveError::Io {
            path: dst.clone(),
            source: e,
        })?;
        let got = blake3_hex(&verified);
        if got != entry.blake3_hex {
            return Err(ArchiveError::BlakeMismatch {
                path: dst,
                expected: entry.blake3_hex.clone(),
                got,
            });
        }
    }

    // ── 8. Write manifest LAST ────────────────────────────────
    let manifest_path = dest_root.join("manifest.json");
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).expect("serialize archive manifest");
    std::fs::write(&manifest_path, &manifest_bytes).map_err(|e| {
        ArchiveError::Io {
            path: manifest_path.clone(),
            source: e,
        }
    })?;

    // ── 9. Move mode: run cascade only after manifest is on disk ─
    if matches!(opts.mode, ArchiveMode::Move) {
        store.cascade_remove_session(opts.session_id)?;
    }

    Ok(manifest)
}

// ── Internal helpers ────────────────────────────────────────────

fn blake3_hex(bytes: &[u8]) -> String {
    let h = blake3::hash(bytes);
    let mut s = String::with_capacity(64);
    for b in h.as_bytes() {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn format_audit_coherence(c: &AuditCoherence) -> String {
    match c {
        AuditCoherence::Coherent => "Coherent".to_string(),
        AuditCoherence::PartialApplySupersession { .. } => {
            "PartialApplySupersession".to_string()
        }
        AuditCoherence::OrphanReplacementAssignments { .. } => {
            "OrphanReplacementAssignments".to_string()
        }
        AuditCoherence::NotReassignTriagable => "NotReassignTriagable".to_string(),
        AuditCoherence::ReassignTriagable => "ReassignTriagable".to_string(),
    }
}

/// Build the per-file path inventory for the session subtree
/// (`verified/sessions/<id>/...`) plus session-keyed
/// `seen/*` markers. `archive_relative` mirrors `source_relative`
/// 1-for-1 — the archive layout under `<archive-dir>/<session_id>/`
/// is bit-for-bit equivalent to the state-dir layout.
fn enumerate_session_files(
    store: &ContributorStateStore,
    session_id: &str,
) -> Result<Vec<RelPathPair>, ArchiveError> {
    let mut out: Vec<RelPathPair> = Vec::new();
    let session_subtree = store
        .root()
        .join("verified")
        .join("sessions")
        .join(session_id);
    if session_subtree.is_dir() {
        walk_dir_collect(&session_subtree, store.root(), &mut out)?;
    }

    // Session-keyed seen markers. The cascade definition in
    // `state.rs:cascade_remove_session` is the canonical list;
    // mirror it here.
    let seen = store.root().join("seen");
    let push_marker = |dir: &str, file: &str, out: &mut Vec<RelPathPair>| {
        let p = seen.join(dir).join(file);
        if p.is_file() {
            if let Ok(rel) = p.strip_prefix(store.root()) {
                out.push(RelPathPair::from_relative(rel.to_path_buf()));
            }
        }
    };
    push_marker("sessions", session_id, &mut out);
    push_marker("aggregates", session_id, &mut out);
    let prefix = format!("{session_id}--");
    let prefixed_dirs: HashSet<&str> = [
        "joins",
        "assignments",
        "partials",
        "peer-adverts",
        "assignment-supersessions",
    ]
    .into_iter()
    .collect();
    for d in &prefixed_dirs {
        let dir = seen.join(d);
        if !dir.is_dir() {
            continue;
        }
        let entries = std::fs::read_dir(&dir).map_err(|e| ArchiveError::Io {
            path: dir.clone(),
            source: e,
        })?;
        for e in entries {
            let e = e.map_err(|e| ArchiveError::Io {
                path: dir.clone(),
                source: e,
            })?;
            let p = e.path();
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name.starts_with(&prefix) {
                    if let Ok(rel) = p.strip_prefix(store.root()) {
                        out.push(RelPathPair::from_relative(rel.to_path_buf()));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.relative_string.cmp(&b.relative_string));
    Ok(out)
}

fn posted_result_link_file(
    store: &ContributorStateStore,
    posted_id: &str,
) -> Result<Option<RelPathPair>, ArchiveError> {
    let p = store
        .root()
        .join("results")
        .join("result-links")
        .join(format!("{posted_id}.link.json"));
    if p.is_file() {
        let rel = p.strip_prefix(store.root()).expect("rel");
        Ok(Some(RelPathPair::from_relative(rel.to_path_buf())))
    } else {
        Ok(None)
    }
}

/// Compute BLAKE3 + byte count for every enumerated file. Reads
/// each file once.
fn compute_file_metadata(
    store: &ContributorStateStore,
    files: Vec<RelPathPair>,
) -> Result<Vec<ArchivedFile>, ArchiveError> {
    let mut out = Vec::with_capacity(files.len());
    for pair in files {
        let src = store.root().join(&pair.path);
        let bytes = std::fs::read(&src).map_err(|e| ArchiveError::Io {
            path: src.clone(),
            source: e,
        })?;
        out.push(ArchivedFile {
            source_relative: pair.relative_string.clone(),
            archive_relative: pair.relative_string,
            blake3_hex: blake3_hex(&bytes),
            bytes: bytes.len() as u64,
        });
    }
    Ok(out)
}

struct RelPathPair {
    path: PathBuf,
    relative_string: String,
}

impl RelPathPair {
    fn from_relative(path: PathBuf) -> Self {
        let relative_string = path
            .to_string_lossy()
            // Normalize to forward slashes so the manifest is
            // platform-portable.
            .replace('\\', "/");
        Self {
            path,
            relative_string,
        }
    }
}

fn walk_dir_collect(
    dir: &Path,
    root: &Path,
    out: &mut Vec<RelPathPair>,
) -> Result<(), ArchiveError> {
    let entries = std::fs::read_dir(dir).map_err(|e| ArchiveError::Io {
        path: dir.to_path_buf(),
        source: e,
    })?;
    for e in entries {
        let e = e.map_err(|e| ArchiveError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?;
        let p = e.path();
        if p.is_dir() {
            walk_dir_collect(&p, root, out)?;
        } else if p.is_file() {
            if let Ok(rel) = p.strip_prefix(root) {
                out.push(RelPathPair::from_relative(rel.to_path_buf()));
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn _ensure_status_report_in_scope(_x: &SessionStatusReport) {}
