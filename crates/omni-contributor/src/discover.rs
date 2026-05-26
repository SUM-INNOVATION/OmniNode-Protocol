//! Stage 12.1 — `JobSource` trait + `FilesystemSource` impl.
//!
//! A `JobSource` produces `PostedJob` envelopes a contributor's
//! `watch-jobs` loop can pick up. Stage 12.1 ships exactly one impl:
//! `FilesystemSource`, which watches a directory for `*.json` files
//! that deserialize into `PostedJob`. SNIP-index polling is
//! deliberately NOT implemented in 12.1 — SNIP roots are immutable,
//! so polling one root cannot discover updates. Real mutable
//! discovery (libp2p gossip / local head pointer / off-chain
//! registry) is Stage 12.2+ work.
//!
//! `FilesystemSource` is intentionally simple:
//!   - Walks `dir` for files with extension `.json`.
//!   - Parses each as a `PostedJob`.
//!   - Validates schema + recomputed `posted_id`.
//!   - Returns parsed envelopes to the caller.
//!
//! Dedup across polls (so the watch loop doesn't re-pick a job
//! every tick) lives in the watch loop, NOT in the source —
//! `FilesystemSource::poll` always returns whatever the directory
//! currently holds. The watch loop carries an in-memory
//! `HashSet<posted_id>` of jobs it has already considered.

use std::path::{Path, PathBuf};

use crate::canonical::posted_id_hex;
use crate::error::DiscoverError;
use crate::posted::PostedJob;

/// The minimal abstraction a watch loop depends on. Implementations
/// in 12.1: `FilesystemSource` only. (Real network discovery is
/// Stage 12.2+.)
pub trait JobSource {
    /// Returns every well-formed `PostedJob` the source can produce
    /// at call time. Discovery is read-only: a successful `poll` does
    /// NOT mutate the underlying source (no leases, claims, or
    /// per-pickup state). Errors from individual entries surface as
    /// `Vec<Result<…>>` items so the watch loop can skip bad files
    /// and continue.
    fn poll(&mut self) -> Result<Vec<DiscoveredEntry>, DiscoverError>;
}

/// Per-entry result. A source-level failure (e.g. dir unreadable)
/// short-circuits via `Err` on `poll`; per-file failures (malformed
/// JSON, schema error, posted_id drift) come back as `Err(...)`
/// entries here so the caller can log + skip + continue.
#[derive(Debug)]
pub struct DiscoveredEntry {
    /// Origin path (filesystem source) or a human-readable
    /// identifier suitable for log lines.
    pub source_label: String,
    pub result: Result<PostedJob, DiscoverError>,
}

// ── FilesystemSource ──────────────────────────────────────────────────────

/// Watches a single directory for posted-job files. Stage 12.1
/// ships dedup-by-`posted_id` in the watch loop, so this source
/// can stay completely stateless — no mtime tracking, no
/// last-seen map.
pub struct FilesystemSource {
    pub dir: PathBuf,
}

impl FilesystemSource {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }
}

impl JobSource for FilesystemSource {
    fn poll(&mut self) -> Result<Vec<DiscoveredEntry>, DiscoverError> {
        let mut out = Vec::new();
        let entries =
            std::fs::read_dir(&self.dir).map_err(|e| DiscoverError::Io {
                path: self.dir.display().to_string(),
                source: e,
            })?;
        for entry in entries {
            let entry = entry.map_err(|e| DiscoverError::Io {
                path: self.dir.display().to_string(),
                source: e,
            })?;
            let path = entry.path();
            if path
                .extension()
                .and_then(|e| e.to_str())
                .map(|s| s != "json")
                .unwrap_or(true)
            {
                continue;
            }
            let result = load_and_validate(&path);
            out.push(DiscoveredEntry {
                source_label: path.display().to_string(),
                result,
            });
        }
        Ok(out)
    }
}

fn load_and_validate(path: &Path) -> Result<PostedJob, DiscoverError> {
    let bytes = std::fs::read(path).map_err(|e| DiscoverError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let posted: PostedJob =
        serde_json::from_slice(&bytes).map_err(|e| DiscoverError::Parse {
            path: path.display().to_string(),
            source: e,
        })?;
    posted.validate_schema().map_err(|e| DiscoverError::Schema {
        path: path.display().to_string(),
        source: e,
    })?;
    // Recompute posted_id from canonical bytes and refuse on drift.
    let recomputed =
        posted_id_hex(&posted).map_err(|e| DiscoverError::FilesystemSourceOther(e.to_string()))?;
    if recomputed != posted.posted_id {
        return Err(DiscoverError::PostedIdMismatch {
            path: path.display().to_string(),
            posted_id: posted.posted_id.clone(),
            recomputed,
        });
    }
    Ok(posted)
}
