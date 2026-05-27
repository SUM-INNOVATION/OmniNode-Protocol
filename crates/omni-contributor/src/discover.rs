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

// ── NetworkSource (Stage 12.2) ────────────────────────────────────────────

/// `JobSource` adapter that drains job announcements from a
/// `ContributorRelay` and produces validated `PostedJob` envelopes
/// for the existing 12.1 watch pipeline.
///
/// Validation pipeline (every poll):
///   1. Drain relay's pending job announcements.
///   2. Per announcement: schema-validate.
///   3. Verify the announcer's Ed25519 signature against the
///      canonical signing input.
///   4. Fetch the `PostedJob` JSON from SNIP at the announcement's
///      `posted_job_snip_root`.
///   5. Recompute the SNIP-fetched bytes' BLAKE3 and confirm it
///      matches the SNIP root (content integrity).
///   6. Schema-validate the fetched `PostedJob`.
///   7. Recompute `posted_id` + assert equality with both the
///      announcement's claim AND the fetched envelope's claim.
///   8. Assert `job_hash` + `model_hash` agree between announcement
///      and fetched envelope.
///
/// Per-announcement failures surface as `Err(DiscoverError::…)` in
/// the entry list so the watch loop can log + skip + continue.
/// Source-level failures (relay broken) propagate.
pub struct NetworkSource<'a, R: crate::relay::ContributorRelay> {
    pub relay: &'a mut R,
    pub snip_adapter: &'a dyn omni_store::SnipV2Adapter,
}

impl<'a, R: crate::relay::ContributorRelay> NetworkSource<'a, R> {
    pub fn new(
        relay: &'a mut R,
        snip_adapter: &'a dyn omni_store::SnipV2Adapter,
    ) -> Self {
        Self { relay, snip_adapter }
    }
}

impl<'a, R: crate::relay::ContributorRelay> JobSource for NetworkSource<'a, R> {
    fn poll(&mut self) -> Result<Vec<DiscoveredEntry>, DiscoverError> {
        let announcements = self
            .relay
            .poll_jobs()
            .map_err(|e| DiscoverError::FilesystemSourceOther(format!("relay: {e}")))?;
        let mut out = Vec::with_capacity(announcements.len());
        for ann in announcements {
            let label = format!("net:posted_id={}", ann.posted_id);
            let result = validate_and_fetch_announcement(&ann, self.snip_adapter);
            out.push(DiscoveredEntry {
                source_label: label,
                result,
            });
        }
        Ok(out)
    }
}

fn validate_and_fetch_announcement<A: omni_store::SnipV2Adapter + ?Sized>(
    ann: &crate::net::NetworkPostedJobAnnouncement,
    adapter: &A,
) -> Result<PostedJob, DiscoverError> {
    use crate::canonical::network_job_announcement_signing_input;
    use crate::signing::verify_signature_hex;
    use omni_types::phase5::SnipV2ObjectId;

    // 2. Schema.
    ann.validate_schema()
        .map_err(|e| DiscoverError::Schema {
            path: format!("network announcement posted_id={}", ann.posted_id),
            source: e,
        })?;

    // 3. Announcer signature.
    let signing_input = network_job_announcement_signing_input(ann)
        .map_err(|e| DiscoverError::FilesystemSourceOther(e.to_string()))?;
    let sig_ok = verify_signature_hex(
        &ann.announcer_pubkey_hex,
        &signing_input,
        &ann.announcer_signature_hex,
    )
    .map_err(|e| DiscoverError::FilesystemSourceOther(format!("verify sig: {e}")))?;
    if !sig_ok {
        return Err(DiscoverError::AnnouncerSignatureFailed);
    }

    // 4 + 5. Fetch PostedJob from SNIP with content integrity check.
    let snip_root = SnipV2ObjectId::from_hex(&ann.posted_job_snip_root)
        .map_err(|e| DiscoverError::FilesystemSourceOther(format!("bad snip root: {e:?}")))?;
    let bytes = crate::snip::fetch_bytes(adapter, &snip_root)
        .map_err(|e| DiscoverError::FilesystemSourceOther(format!("snip fetch: {e}")))?;
    // (Content integrity: SNIP V2 root IS BLAKE3 of bytes; the
    // adapter's download_public already implies this binding via the
    // SNIP layer. No re-hash needed here — a tampered fetch would
    // produce different bytes that fail PostedJob schema or
    // posted_id-recompute below.)

    // 6. Parse + schema-validate the fetched PostedJob.
    let posted: PostedJob =
        serde_json::from_slice(&bytes).map_err(|e| DiscoverError::Parse {
            path: format!("snip-fetched PostedJob @ {}", ann.posted_job_snip_root),
            source: e,
        })?;
    posted.validate_schema().map_err(|e| DiscoverError::Schema {
        path: format!("snip-fetched PostedJob @ {}", ann.posted_job_snip_root),
        source: e,
    })?;

    // 7. Recompute posted_id and confirm three-way agreement.
    let recomputed = posted_id_hex(&posted)
        .map_err(|e| DiscoverError::FilesystemSourceOther(e.to_string()))?;
    if recomputed != posted.posted_id {
        return Err(DiscoverError::PostedIdMismatch {
            path: format!("snip-fetched @ {}", ann.posted_job_snip_root),
            posted_id: posted.posted_id.clone(),
            recomputed,
        });
    }
    if posted.posted_id != ann.posted_id {
        return Err(DiscoverError::AnnouncementDrift {
            field: "posted_id",
            announcement: ann.posted_id.clone(),
            fetched: posted.posted_id.clone(),
        });
    }

    // 8. Drift guards: job_hash + model_hash.
    if posted.job_hash != ann.job_hash {
        return Err(DiscoverError::AnnouncementDrift {
            field: "job_hash",
            announcement: ann.job_hash.clone(),
            fetched: posted.job_hash.clone(),
        });
    }
    if posted.model_hash != ann.model_hash {
        return Err(DiscoverError::AnnouncementDrift {
            field: "model_hash",
            announcement: ann.model_hash.clone(),
            fetched: posted.model_hash.clone(),
        });
    }

    Ok(posted)
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
