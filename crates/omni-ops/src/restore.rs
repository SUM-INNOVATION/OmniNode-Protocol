//! Stage 12.15 — local session archive restore / import.
//!
//! The inverse of Stage 12.14's `archive_session`. Reads
//! `<archive-session-dir>/manifest.json`, validates every file
//! entry against the manifest's BLAKE3 + path whitelist, then
//! writes each file's bytes back into the operator's state-dir
//! via [`ContributorStateStore::write_archived_bytes`].
//!
//! No protocol surface. No chain wire. No SNIP wire. No mesh.
//! Restore is **byte-identical replay** of the bytes the archive
//! captured at archive time; every Stage 12.3 / 12.11 verifier
//! accepts the restored envelopes unchanged because their
//! signatures are over canonical bytes that survived the copy.
//!
//! ## Safety contract
//!
//! - `--dry-run`: parse manifest + validate schema_version +
//!   source_state_version + session_id + every entry's path
//!   safety/whitelist; **no archive file reads beyond the
//!   manifest, no destination writes**.
//! - `--verify-only`: everything `--dry-run` does, PLUS reads
//!   each archive file + recomputes its BLAKE3 and compares
//!   against the manifest. Still no destination writes.
//! - Real restore: everything `--verify-only` does, PLUS a
//!   preflight existence check across every destination path
//!   (all-or-nothing); if any destination exists and
//!   `--overwrite-existing` is false, refuses BEFORE writing
//!   anything. Then writes every byte verbatim via the
//!   state-store's `write_archived_bytes` atomic helper.
//! - BLAKE3 mismatch is **fail-fast** — no retry, no partial
//!   accept.
//! - State-dir version compat is strict equality:
//!   `manifest.source_state_version == STATE_VERSION`.

use std::path::{Path, PathBuf};

use crate::archive::{ArchiveManifest, ARCHIVE_MANIFEST_SCHEMA_VERSION};
use crate::error::RestoreError;
use omni_contributor::state::{ContributorStateStore, STATE_VERSION};

/// Stage 12.15 — where the archive lives on disk. Operators can
/// supply either the full session subdirectory or the archive
/// root plus a session id.
pub enum RestoreSource<'a> {
    /// `<path>/manifest.json` is read directly.
    SessionDir(&'a Path),
    /// `<archive_dir>/<session_id>/manifest.json` is read.
    ArchiveRoot {
        archive_dir: &'a Path,
        session_id: &'a str,
    },
}

impl<'a> RestoreSource<'a> {
    /// Resolve to the absolute session-directory path the
    /// archive should live at.
    pub fn session_dir(&self) -> PathBuf {
        match self {
            RestoreSource::SessionDir(p) => p.to_path_buf(),
            RestoreSource::ArchiveRoot {
                archive_dir,
                session_id,
            } => archive_dir.join(session_id),
        }
    }

    /// Resolve the expected session_id for `SessionIdMismatch`
    /// reporting. For `SessionDir` this is the directory's
    /// terminal component; for `ArchiveRoot` it's the
    /// caller-supplied id.
    pub fn expected_session_id(&self) -> Option<String> {
        match self {
            RestoreSource::SessionDir(p) => p
                .file_name()
                .and_then(|s| s.to_str())
                .map(String::from),
            RestoreSource::ArchiveRoot { session_id, .. } => {
                Some((*session_id).to_string())
            }
        }
    }
}

pub struct RestoreOptions<'a> {
    pub source: RestoreSource<'a>,
    /// Manifest-only validation. No archive file reads beyond
    /// the manifest, no destination writes.
    pub dry_run: bool,
    /// Full archive-file BLAKE3 verification, no destination
    /// writes. If `dry_run` AND `verify_only` are both true,
    /// `verify_only` wins (it's the strict superset).
    pub verify_only: bool,
    /// Default `false`. When `false`, restore refuses
    /// BEFORE writing if any destination file already
    /// exists.
    pub overwrite_existing: bool,
    /// Default `false`. Skip
    /// `results/result-links/<posted_id>.link.json` entries
    /// even if the archive contains them.
    pub include_results: bool,
    pub now_utc: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreReport {
    pub session_id: String,
    pub manifest_schema_version: u32,
    pub source_state_version: u32,
    pub files_restored: u32,
    pub files_skipped_results: u32,
    pub bytes_restored: u64,
    /// `"restore"` / `"dry_run"` / `"verify_only"`. Closed set.
    pub mode: &'static str,
    /// Informational copy of the archive-time
    /// `session_overall_status` so the CLI can echo it on the
    /// `event=restore_started` line.
    pub manifest_session_overall_status: String,
    /// Informational copy of archive-time `audit_coherence`.
    pub manifest_audit_coherence: String,
}

/// Parse + validate `manifest.json` only. Used by the CLI's
/// `event=restore_started` emission AND by external tooling
/// that wants to introspect an archive's metadata without
/// running a full restore.
pub fn verify_archive_manifest(
    source: &RestoreSource<'_>,
) -> Result<ArchiveManifest, RestoreError> {
    let session_dir = source.session_dir();
    if !session_dir.exists() {
        return Err(RestoreError::ArchiveNotFound { path: session_dir });
    }
    let manifest_path = session_dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Err(RestoreError::ManifestMissing { path: manifest_path });
    }
    let bytes = std::fs::read(&manifest_path).map_err(|e| RestoreError::Io {
        path: manifest_path.clone(),
        source: e,
    })?;
    let manifest: ArchiveManifest =
        serde_json::from_slice(&bytes).map_err(|e| RestoreError::MalformedManifest {
            path: manifest_path.clone(),
            source: e,
        })?;
    if manifest.schema_version != ARCHIVE_MANIFEST_SCHEMA_VERSION {
        return Err(RestoreError::UnsupportedManifestVersion {
            got: manifest.schema_version,
            expected: ARCHIVE_MANIFEST_SCHEMA_VERSION,
        });
    }
    // Stage 12.15 v1 — strict equality only.
    if manifest.source_state_version != STATE_VERSION {
        return Err(RestoreError::IncompatibleSourceStateVersion {
            archive: manifest.source_state_version,
            current: STATE_VERSION,
        });
    }
    // Session-id binding: defends against renamed archive
    // directories. When the source is `ArchiveRoot`, the
    // caller-supplied `session_id` is the authoritative
    // expectation; for `SessionDir`, the directory name is.
    if let Some(expected) = source.expected_session_id() {
        if manifest.session_id != expected {
            return Err(RestoreError::SessionIdMismatch {
                manifest_session_id: manifest.session_id.clone(),
                dir_session_id: expected,
            });
        }
    }
    Ok(manifest)
}

/// Run the restore. Returns a `RestoreReport`.
pub fn restore_session_archive(
    store: &ContributorStateStore,
    opts: &RestoreOptions<'_>,
) -> Result<RestoreReport, RestoreError> {
    let manifest = verify_archive_manifest(&opts.source)?;
    let session_dir = opts.source.session_dir();

    // Mode resolution: verify-only wins when both are set.
    let mode = if opts.verify_only {
        "verify_only"
    } else if opts.dry_run {
        "dry_run"
    } else {
        "restore"
    };

    // ── Phase A: per-entry path safety + whitelist check ────
    //
    // We do this in a tight loop BEFORE any archive file read
    // so a manifest with a single bad entry fails fast without
    // hashing megabytes of bytes first. The actual write happens
    // via `store.write_archived_bytes`, which independently
    // re-validates the path — but doing it here too lets us
    // bubble the typed error EARLY for clearer operator
    // diagnostics, and lets verify-only / dry-run refuse a
    // hostile manifest without ever touching the FS beyond
    // the manifest read.
    for entry in &manifest.files {
        if let Err(e) = check_relative_path(&entry.source_relative) {
            return Err(e);
        }
        if let Err(e) = check_relative_path(&entry.archive_relative) {
            return Err(e);
        }
    }

    // ── Phase B: optional file existence + BLAKE3 verify ────
    //
    // Stage 12.15 review fix — the `verify_only` flag is the
    // strict superset of `dry_run` (mode resolution above
    // already returns `"verify_only"` when both are passed), so
    // verification runs whenever `verify_only` is true, even if
    // `dry_run` is ALSO true. The previous gate `!opts.dry_run`
    // silently skipped verification when both flags were set,
    // producing a `verify_only` mode tag without actually
    // hashing any archive bytes. That's been corrected here:
    // verify when `verify_only || !dry_run`. Pure `dry_run` (no
    // verify_only) still skips this phase.
    //
    // For `verify_only` / real restore, every archive file must
    // exist AND its BLAKE3 must match. We also pre-load the
    // bytes so the real-restore phase can write without
    // re-reading.
    let mut prepared: Vec<(usize, Vec<u8>)> = Vec::new();
    if opts.verify_only || !opts.dry_run {
        for (idx, entry) in manifest.files.iter().enumerate() {
            if !opts.include_results
                && entry.source_relative.starts_with("results/result-links/")
            {
                continue;
            }
            let archive_path = session_dir.join(&entry.archive_relative);
            if !archive_path.is_file() {
                return Err(RestoreError::ManifestFileMissing { archive_path });
            }
            let bytes =
                std::fs::read(&archive_path).map_err(|e| RestoreError::Io {
                    path: archive_path.clone(),
                    source: e,
                })?;
            let got = blake3_hex(&bytes);
            if got != entry.blake3_hex {
                return Err(RestoreError::BlakeMismatch {
                    path: archive_path,
                    expected: entry.blake3_hex.clone(),
                    got,
                });
            }
            // Verify-only doesn't need to keep the bytes; freeing
            // immediately limits peak memory.
            if !opts.verify_only {
                prepared.push((idx, bytes));
            }
        }
    }

    // ── Phase C: destination preflight + write ──────────────
    //
    // Real restore only. Preflight: every destination must be
    // either non-existent OR overwrite_existing is true. The
    // check happens BEFORE any write so a single mid-loop
    // preflight failure can't leave the state-dir partially
    // populated. After preflight, we write each prepared file
    // via the state-store atomic helper, which independently
    // re-runs the path-safety + whitelist gate.
    let mut files_restored = 0u32;
    let mut bytes_restored = 0u64;
    let mut files_skipped_results = 0u32;

    if !matches!(mode, "dry_run" | "verify_only") {
        for entry in &manifest.files {
            if !opts.include_results
                && entry.source_relative.starts_with("results/result-links/")
            {
                continue;
            }
            if !opts.overwrite_existing {
                let dest = store.root().join(&entry.source_relative);
                if dest.exists() {
                    return Err(RestoreError::DestinationExists { path: dest });
                }
            }
        }
        for (idx, bytes) in prepared {
            let entry = &manifest.files[idx];
            store.write_archived_bytes(
                &entry.source_relative,
                &bytes,
                opts.overwrite_existing,
            )?;
            files_restored += 1;
            bytes_restored += bytes.len() as u64;
        }
    }

    // Count result-link skips uniformly (applies to all modes
    // for `files_skipped_results`).
    if !opts.include_results {
        for entry in &manifest.files {
            if entry.source_relative.starts_with("results/result-links/") {
                files_skipped_results += 1;
            }
        }
    }

    Ok(RestoreReport {
        session_id: manifest.session_id.clone(),
        manifest_schema_version: manifest.schema_version,
        source_state_version: manifest.source_state_version,
        files_restored,
        files_skipped_results,
        bytes_restored,
        mode,
        manifest_session_overall_status: manifest.session_overall_status.clone(),
        manifest_audit_coherence: manifest.audit_coherence.clone(),
    })
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

/// Pre-check a relative path AT MANIFEST PARSE TIME so verify-only
/// / dry-run can refuse a hostile manifest before any FS work.
/// The state-store's `write_archived_bytes` is the canonical
/// gate at write time; this function is the lightweight
/// pre-check that mirrors the same rules.
fn check_relative_path(rel: &str) -> Result<(), RestoreError> {
    if rel.is_empty() || rel.contains('\\') || Path::new(rel).is_absolute() {
        return Err(RestoreError::UnsafeRelativePath {
            path: rel.to_string(),
        });
    }
    if rel.split('/').any(|seg| seg == "..") {
        return Err(RestoreError::UnsafeRelativePath {
            path: rel.to_string(),
        });
    }
    // Whitelist match — same matrix as
    // `state::validate_archive_relative_path`. Duplicated here
    // so a malformed manifest fails fast in verify-only/dry-run
    // without touching the state-store. The state-store check
    // is the authoritative gate at write time.
    let parts: Vec<&str> = rel.split('/').collect();
    let ok = match parts.as_slice() {
        ["verified", "sessions", sid, rest @ ..] => {
            is_hex64(sid) && !rest.is_empty() && rest.iter().all(|s| !s.is_empty())
        }
        ["seen", "sessions", sid] => is_hex64(sid),
        ["seen", "aggregates", sid] => is_hex64(sid),
        ["seen", ns, key]
            if matches!(
                *ns,
                "joins"
                    | "assignments"
                    | "partials"
                    | "peer-adverts"
                    | "assignment-supersessions"
            ) =>
        {
            match key.split_once("--") {
                Some((sid, suffix)) => is_hex64(sid) && !suffix.is_empty(),
                None => false,
            }
        }
        ["results", "result-links", file] => file
            .strip_suffix(".link.json")
            .map(is_hex64)
            .unwrap_or(false),
        _ => false,
    };
    if !ok {
        return Err(RestoreError::DisallowedRelativePath {
            path: rel.to_string(),
        });
    }
    Ok(())
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
