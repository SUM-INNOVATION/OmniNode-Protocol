//! Stage 12.7 — local contributor workflow state persistence.
//!
//! Lets a contributor watcher / runner survive `omni-node` restart
//! without losing already-seen announcements, verified session
//! trees, peer-advert caches, or computed contributor-results.
//! Complements Stage 12.6 (`--net-identity-file` keeps the mesh
//! PeerId stable); Stage 12.7 keeps the contributor workflow state
//! stable.
//!
//! ## Scope
//!
//! - **Local only.** Not a registry, not a chain authority, not
//!   visible to any other peer. Safe to delete.
//! - **No private key material.** Mesh identity lives in
//!   `--net-identity-file` (Stage 12.6); contributor seed lives in
//!   `--contributor-seed` (Stage 12.0). 12.7 only stores already-
//!   public envelopes (sessions, joins, assignments, partials,
//!   aggregates, peer advertisements) plus seen-marker files.
//! - **Inspectable JSON.** Operators can `cat` everything.
//! - **Atomic writes.** Tempfile-in-same-directory + `fs::rename`.
//! - **Auto-prune on open.** Removes expired sessions / peer
//!   advertisements (anything with an explicit `expires_at_utc`).
//!   `--no-prune-state-on-start` disables for forensic re-runs.
//! - **Versioned.** `meta/state_version.json` is written on first
//!   open at `state_version: 1`; future versions are refused with
//!   `StateError::UnsupportedVersion` so a 12.8+ migration is clean.
//!
//! ## Layout
//!
//! ```text
//! <state-dir>/
//!   meta/state_version.json
//!   seen/posted-jobs/<posted_id>
//!   seen/network-job-announcements/<posted_id>
//!   seen/network-result-announcements/<posted_id>--<snip_root>
//!   seen/sessions/<session_id>
//!   seen/joins/<session_id>--<contributor_pubkey>
//!   seen/assignments/<session_id>--<assignment_id>
//!   seen/partials/<session_id>--<assignment_id>
//!   seen/aggregates/<session_id>
//!   seen/peer-adverts/<session_id>--<contributor_pubkey>
//!   verified/sessions/<session_id>/session.json
//!   verified/sessions/<session_id>/joins/<pubkey>.json
//!   verified/sessions/<session_id>/assignments/<id>.json
//!   verified/sessions/<session_id>/partials/<id>.json
//!   verified/sessions/<session_id>/aggregated.json
//!   verified/sessions/<session_id>/peer-adverts/<pubkey>.json
//!   results/contributor-results/<job_id>.json
//!   results/contributor-results/<job_id>.rejected.json
//!   results/result-links/<posted_id>.link.json
//! ```
//!
//! The `verified/sessions/<id>/...` subtree under the state-dir
//! is the same shape as the existing Stage 12.3 `watch-sessions
//! --out-dir <X>` layout — i.e. a pre-12.7 `<X>/<id>/...` tree
//! becomes a valid state-dir subtree once it lives at
//! `<state-dir>/verified/sessions/<id>/...`. The reader walks
//! ONLY `<state-dir>/verified/sessions/...`; an old `<X>/<id>/...`
//! tree at the top level is NOT auto-discovered. To migrate, either
//! point new runs at a fresh `--contributor-state-dir` (the old
//! `--out-dir` keeps working in parallel and the watchers re-fetch
//! envelopes as they appear), or `mkdir -p <state>/verified/sessions`
//! and move the per-session subdirectories under that prefix. See
//! `docs/stage12-contributor-protocol.md`.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::StateError;
use crate::peer_advert::ContributorPeerAdvertisement;
use crate::session::{
    AggregatedContributorResult, ContributorJoin, ExecutionSession,
    PartialContributorResult, WorkAssignment,
};

/// Pinned v1. Future versions are refused on `open`.
pub const STATE_VERSION: u32 = 1;

/// Stored in `<state-dir>/meta/state_version.json` so a future
/// 12.8+ migration is clean.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateVersionMeta {
    pub state_version: u32,
}

/// Per-namespace `seen/` marker directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateNamespace {
    PostedJobs,
    NetworkJobAnnouncements,
    NetworkResultAnnouncements,
    Sessions,
    Joins,
    Assignments,
    Partials,
    Aggregates,
    PeerAdverts,
}

impl StateNamespace {
    fn dir_name(self) -> &'static str {
        match self {
            StateNamespace::PostedJobs => "posted-jobs",
            StateNamespace::NetworkJobAnnouncements => "network-job-announcements",
            StateNamespace::NetworkResultAnnouncements => "network-result-announcements",
            StateNamespace::Sessions => "sessions",
            StateNamespace::Joins => "joins",
            StateNamespace::Assignments => "assignments",
            StateNamespace::Partials => "partials",
            StateNamespace::Aggregates => "aggregates",
            StateNamespace::PeerAdverts => "peer-adverts",
        }
    }
}

/// What kind of verified envelope is being written / read. Carries
/// the keying material so the same call can address `<session_id>/
/// joins/<pubkey>.json` etc.
#[derive(Debug, Clone)]
pub enum StateObjectKind {
    Session,
    Join { session_id: String },
    Assignment { session_id: String },
    Partial { session_id: String },
    /// Keyed by `session_id` (one aggregated.json per session).
    Aggregate,
    PeerAdvert { session_id: String },
    /// `<state>/results/contributor-results/<job_id>.json`
    ContributorResult,
    /// `<state>/results/contributor-results/<job_id>.rejected.json`
    RejectedResult,
    /// `<state>/results/result-links/<posted_id>.link.json`
    PostedResultLink,
}

impl StateObjectKind {
    /// Resolve to `(absolute_dir, filename)`.
    fn resolve(&self, root: &Path, id: &str) -> (PathBuf, String) {
        match self {
            StateObjectKind::Session => (
                root.join("verified").join("sessions").join(id),
                "session.json".to_string(),
            ),
            StateObjectKind::Join { session_id } => (
                root.join("verified")
                    .join("sessions")
                    .join(session_id)
                    .join("joins"),
                format!("{id}.json"),
            ),
            StateObjectKind::Assignment { session_id } => (
                root.join("verified")
                    .join("sessions")
                    .join(session_id)
                    .join("assignments"),
                format!("{id}.json"),
            ),
            StateObjectKind::Partial { session_id } => (
                root.join("verified")
                    .join("sessions")
                    .join(session_id)
                    .join("partials"),
                format!("{id}.json"),
            ),
            StateObjectKind::Aggregate => (
                root.join("verified").join("sessions").join(id),
                "aggregated.json".to_string(),
            ),
            StateObjectKind::PeerAdvert { session_id } => (
                root.join("verified")
                    .join("sessions")
                    .join(session_id)
                    .join("peer-adverts"),
                format!("{id}.json"),
            ),
            StateObjectKind::ContributorResult => (
                root.join("results").join("contributor-results"),
                format!("{id}.json"),
            ),
            StateObjectKind::RejectedResult => (
                root.join("results").join("contributor-results"),
                format!("{id}.rejected.json"),
            ),
            StateObjectKind::PostedResultLink => (
                root.join("results").join("result-links"),
                format!("{id}.link.json"),
            ),
        }
    }
}

/// Summary returned by [`ContributorStateStore::open`] (when
/// pruning was active) and [`ContributorStateStore::prune_expired`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PruneReport {
    pub removed_sessions: u64,
    pub removed_peer_adverts: u64,
    pub kept: u64,
}

/// Local contributor state store. Cheap to clone — only holds a
/// path. Concurrent processes pointing at the same directory work
/// because every write goes through tempfile-rename.
#[derive(Debug, Clone)]
pub struct ContributorStateStore {
    root: PathBuf,
}

impl ContributorStateStore {
    /// Open or create the state directory. Writes
    /// `meta/state_version.json` on first open; refuses unknown
    /// future versions. When `auto_prune == true`, expired
    /// sessions and peer advertisements are removed (and their
    /// `seen/` markers cascaded) before the store is returned.
    pub fn open(
        root: impl AsRef<Path>,
        auto_prune: bool,
        now_utc: &str,
    ) -> Result<(Self, PruneReport), StateError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| StateError::Io {
            path: root.clone(),
            source: e,
        })?;
        let meta_dir = root.join("meta");
        fs::create_dir_all(&meta_dir).map_err(|e| StateError::Io {
            path: meta_dir.clone(),
            source: e,
        })?;
        let meta_path = meta_dir.join("state_version.json");
        if meta_path.is_file() {
            let bytes = fs::read(&meta_path).map_err(|e| StateError::Io {
                path: meta_path.clone(),
                source: e,
            })?;
            let meta: StateVersionMeta =
                serde_json::from_slice(&bytes).map_err(|e| StateError::Json {
                    path: meta_path.clone(),
                    source: e,
                })?;
            if meta.state_version != STATE_VERSION {
                return Err(StateError::UnsupportedVersion {
                    got: meta.state_version,
                    expected: STATE_VERSION,
                });
            }
        } else {
            let meta = StateVersionMeta {
                state_version: STATE_VERSION,
            };
            let json = serde_json::to_vec_pretty(&meta).expect("serialize meta");
            atomic_write(&meta_path, &json)?;
        }
        // Pre-create the top-level subdirs so `mark_seen` /
        // `write_verified_*` don't race on first-use.
        for sub in &["seen", "verified", "results"] {
            fs::create_dir_all(root.join(sub)).map_err(|e| StateError::Io {
                path: root.join(sub),
                source: e,
            })?;
        }
        let store = Self { root };
        let report = if auto_prune {
            store.prune_expired(now_utc)?
        } else {
            PruneReport::default()
        };
        Ok((store, report))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn seen_path(&self, ns: StateNamespace, id: &str) -> PathBuf {
        self.root.join("seen").join(ns.dir_name()).join(id)
    }

    /// Idempotent. Writes a zero-byte marker file (atomic via
    /// `OpenOptions::create_new`; a pre-existing marker is treated
    /// as already-marked, NOT as an error).
    pub fn mark_seen(&self, ns: StateNamespace, id: &str) -> Result<(), StateError> {
        let path = self.seen_path(ns, id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| StateError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
        match fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
        {
            Ok(_) => Ok(()),
            // EEXIST → already marked; idempotent success.
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(source) => Err(StateError::Io { path, source }),
        }
    }

    pub fn is_seen(&self, ns: StateNamespace, id: &str) -> Result<bool, StateError> {
        Ok(self.seen_path(ns, id).is_file())
    }

    /// Atomic JSON write: serialize → tempfile in same dir → rename.
    /// Returns the final path.
    pub fn write_verified_json<T: Serialize>(
        &self,
        kind: StateObjectKind,
        id: &str,
        value: &T,
    ) -> Result<PathBuf, StateError> {
        let (dir, filename) = kind.resolve(&self.root, id);
        fs::create_dir_all(&dir).map_err(|e| StateError::Io {
            path: dir.clone(),
            source: e,
        })?;
        let path = dir.join(&filename);
        let bytes = serde_json::to_vec_pretty(value).map_err(|e| StateError::Json {
            path: path.clone(),
            source: e,
        })?;
        atomic_write(&path, &bytes)?;
        Ok(path)
    }

    pub fn read_verified_json<T: serde::de::DeserializeOwned>(
        &self,
        kind: StateObjectKind,
        id: &str,
    ) -> Result<Option<T>, StateError> {
        let (dir, filename) = kind.resolve(&self.root, id);
        let path = dir.join(&filename);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = fs::read(&path).map_err(|e| StateError::Io {
            path: path.clone(),
            source: e,
        })?;
        let v = serde_json::from_slice(&bytes).map_err(|e| StateError::Json {
            path: path.clone(),
            source: e,
        })?;
        Ok(Some(v))
    }

    // ── Listing helpers (used by run-assignment resume) ─────────

    /// Walk `verified/sessions/*/session.json`. Files that don't
    /// parse are skipped with a stderr-style warning (writable via
    /// `tracing` later if anyone wires it up; for now we just drop
    /// them so a half-written file doesn't break the resume).
    pub fn list_verified_sessions(
        &self,
    ) -> Result<Vec<(String, ExecutionSession)>, StateError> {
        let sessions_dir = self.root.join("verified").join("sessions");
        if !sessions_dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(&sessions_dir).map_err(|e| StateError::Io {
            path: sessions_dir.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| StateError::Io {
                path: sessions_dir.clone(),
                source: e,
            })?;
            let p = entry.path();
            if !p.is_dir() {
                continue;
            }
            let session_id = match p.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            let session_path = p.join("session.json");
            if !session_path.is_file() {
                continue;
            }
            let bytes = match fs::read(&session_path) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let session: ExecutionSession = match serde_json::from_slice(&bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };
            out.push((session_id, session));
        }
        Ok(out)
    }

    pub fn list_verified_joins_for(
        &self,
        session_id: &str,
    ) -> Result<Vec<ContributorJoin>, StateError> {
        self.list_verified_under(session_id, "joins")
    }

    pub fn list_verified_assignments_for(
        &self,
        session_id: &str,
    ) -> Result<Vec<WorkAssignment>, StateError> {
        self.list_verified_under(session_id, "assignments")
    }

    pub fn list_verified_partials_for(
        &self,
        session_id: &str,
    ) -> Result<Vec<PartialContributorResult>, StateError> {
        self.list_verified_under(session_id, "partials")
    }

    pub fn list_verified_peer_adverts_for(
        &self,
        session_id: &str,
    ) -> Result<Vec<ContributorPeerAdvertisement>, StateError> {
        self.list_verified_under(session_id, "peer-adverts")
    }

    pub fn read_verified_aggregate_for(
        &self,
        session_id: &str,
    ) -> Result<Option<AggregatedContributorResult>, StateError> {
        self.read_verified_json(StateObjectKind::Aggregate, session_id)
    }

    fn list_verified_under<T: serde::de::DeserializeOwned>(
        &self,
        session_id: &str,
        leaf: &str,
    ) -> Result<Vec<T>, StateError> {
        let dir = self
            .root
            .join("verified")
            .join("sessions")
            .join(session_id)
            .join(leaf);
        if !dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in fs::read_dir(&dir).map_err(|e| StateError::Io {
            path: dir.clone(),
            source: e,
        })? {
            let entry = entry.map_err(|e| StateError::Io {
                path: dir.clone(),
                source: e,
            })?;
            let p = entry.path();
            if !p.is_file()
                || p.extension().and_then(|s| s.to_str()) != Some("json")
            {
                continue;
            }
            let bytes = match fs::read(&p) {
                Ok(b) => b,
                Err(_) => continue,
            };
            match serde_json::from_slice::<T>(&bytes) {
                Ok(v) => out.push(v),
                Err(_) => continue,
            }
        }
        Ok(out)
    }

    // ── Pruning ────────────────────────────────────────────────

    /// Remove expired sessions and peer advertisements (anything
    /// with an explicit `expires_at_utc` that is past `now_utc`).
    /// Cascades to the session's joins/assignments/partials/
    /// aggregated/peer-adverts AND every matching `seen/` marker.
    /// Entries without an `expires_at_utc` (posted-jobs without
    /// expiry, contributor-results, rejected results, posted
    /// result links) are never auto-pruned.
    pub fn prune_expired(&self, now_utc: &str) -> Result<PruneReport, StateError> {
        use chrono::DateTime;
        let now = match DateTime::parse_from_rfc3339(now_utc) {
            Ok(t) => t,
            Err(_) => return Ok(PruneReport::default()),
        };
        let mut report = PruneReport::default();
        let sessions_dir = self.root.join("verified").join("sessions");
        if !sessions_dir.is_dir() {
            return Ok(report);
        }
        // Collect first so we can mutate the tree without holding
        // an active read_dir iterator on the directory we're
        // pruning.
        let session_dirs: Vec<PathBuf> = fs::read_dir(&sessions_dir)
            .map_err(|e| StateError::Io {
                path: sessions_dir.clone(),
                source: e,
            })?
            .filter_map(|e| e.ok().map(|x| x.path()))
            .filter(|p| p.is_dir())
            .collect();
        for sdir in session_dirs {
            let session_id = match sdir.file_name().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            // Read session.json's expires_at_utc; on any failure,
            // KEEP the directory (conservative — don't prune what
            // we can't validate).
            let session_path = sdir.join("session.json");
            let session_expired = if session_path.is_file() {
                match fs::read(&session_path)
                    .ok()
                    .and_then(|b| serde_json::from_slice::<ExecutionSession>(&b).ok())
                {
                    Some(s) => {
                        match DateTime::parse_from_rfc3339(&s.expires_at_utc) {
                            Ok(exp) => now >= exp,
                            Err(_) => false,
                        }
                    }
                    None => false,
                }
            } else {
                false
            };

            // Peer advertisements are pruned individually so a
            // partially-stale session can still serve fresh
            // adverts. (But if the whole session is expired, the
            // cascade below drops all adverts anyway.)
            let adverts_dir = sdir.join("peer-adverts");
            if adverts_dir.is_dir() {
                let entries: Vec<PathBuf> = fs::read_dir(&adverts_dir)
                    .map_err(|e| StateError::Io {
                        path: adverts_dir.clone(),
                        source: e,
                    })?
                    .filter_map(|e| e.ok().map(|x| x.path()))
                    .collect();
                for p in entries {
                    if !p.is_file()
                        || p.extension().and_then(|s| s.to_str()) != Some("json")
                    {
                        continue;
                    }
                    let advert: Option<ContributorPeerAdvertisement> =
                        fs::read(&p)
                            .ok()
                            .and_then(|b| serde_json::from_slice(&b).ok());
                    let advert_expired = match advert {
                        Some(a) => {
                            match DateTime::parse_from_rfc3339(&a.expires_at_utc) {
                                Ok(exp) => now >= exp,
                                Err(_) => false,
                            }
                        }
                        None => false,
                    };
                    if advert_expired || session_expired {
                        let _ = fs::remove_file(&p);
                        report.removed_peer_adverts += 1;
                        if let Some(stem) =
                            p.file_stem().and_then(|s| s.to_str())
                        {
                            // Best-effort cascade of the matching
                            // `seen/peer-adverts` marker.
                            let marker_id = format!("{session_id}--{stem}");
                            let _ = fs::remove_file(
                                self.seen_path(StateNamespace::PeerAdverts, &marker_id),
                            );
                        }
                    } else {
                        report.kept += 1;
                    }
                }
            }
            if session_expired {
                cascade_remove_session(&self.root, &session_id)?;
                report.removed_sessions += 1;
            } else if session_path.is_file() {
                report.kept += 1;
            }
        }
        Ok(report)
    }
}

/// Tempfile-in-same-dir + rename. Returns `StateError::Io` on any
/// failure. The temp file is removed if the rename fails so
/// half-written files never appear at the final path.
fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), StateError> {
    let parent = path.parent().ok_or_else(|| StateError::Io {
        path: path.to_path_buf(),
        source: std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "atomic_write: target path has no parent directory",
        ),
    })?;
    fs::create_dir_all(parent).map_err(|e| StateError::Io {
        path: parent.to_path_buf(),
        source: e,
    })?;
    // Tempfile name: <final>.<pid>.<unique>.tmp — unique within the
    // dir so two concurrent writers don't collide.
    let tmp = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("state"),
        std::process::id(),
        rand_token(),
    ));
    {
        let mut f = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)
            .map_err(|e| StateError::Io {
                path: tmp.clone(),
                source: e,
            })?;
        if let Err(e) = f.write_all(bytes) {
            drop(f);
            let _ = fs::remove_file(&tmp);
            return Err(StateError::Io {
                path: tmp,
                source: e,
            });
        }
        if let Err(e) = f.flush() {
            drop(f);
            let _ = fs::remove_file(&tmp);
            return Err(StateError::Io {
                path: tmp,
                source: e,
            });
        }
        // Drop closes the file; no fsync (matches the temp-rename
        // pattern used by every other JSON writer in this crate).
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(StateError::Io {
            path: path.to_path_buf(),
            source: e,
        });
    }
    Ok(())
}

/// Cheap unique token for tempfile names. Uses
/// `SystemTime` nanos + a counter to avoid collisions across
/// rapid back-to-back writes within the same process.
fn rand_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{n:x}")
}

/// Remove a whole `verified/sessions/<session_id>/...` subtree and
/// every matching `seen/*` marker for that session_id.
fn cascade_remove_session(root: &Path, session_id: &str) -> Result<(), StateError> {
    let sdir = root.join("verified").join("sessions").join(session_id);
    if sdir.is_dir() {
        let _ = fs::remove_dir_all(&sdir);
    }
    // Markers under `seen/`: sessions/<id> + aggregates/<id> + any
    // joins/assignments/partials/peer-adverts whose key starts with
    // `<id>--`.
    let seen = root.join("seen");
    let _ = fs::remove_file(seen.join("sessions").join(session_id));
    let _ = fs::remove_file(seen.join("aggregates").join(session_id));
    let prefix = format!("{session_id}--");
    let mut seen_dirs_to_walk: HashSet<&str> = HashSet::new();
    seen_dirs_to_walk.insert("joins");
    seen_dirs_to_walk.insert("assignments");
    seen_dirs_to_walk.insert("partials");
    seen_dirs_to_walk.insert("peer-adverts");
    for d in seen_dirs_to_walk {
        let dir = seen.join(d);
        if !dir.is_dir() {
            continue;
        }
        let entries: Vec<PathBuf> = match fs::read_dir(&dir) {
            Ok(it) => it.filter_map(|e| e.ok().map(|x| x.path())).collect(),
            Err(_) => continue,
        };
        for p in entries {
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name.starts_with(&prefix) {
                    let _ = fs::remove_file(&p);
                }
            }
        }
    }
    Ok(())
}
