//! Stage 12.22 — local-only integrity evidence bundle manifest.
//!
//! Ties together a chosen set of Stage 12.16–12.21 forensic
//! artifacts by `(artifact_kind, base-dir-relative path,
//! byte_len, blake3_hex)` under a single `base_dir`, so an
//! operator can capture a single tamper-evident forensic record
//! over a curated set of evidence artifacts.
//!
//! The bundle is a **byte manifest**. It does not embed the
//! artifact bytes themselves — it only fingerprints them. There
//! is no signature, no semantic JSON validation, no recursive
//! directory bundling, no automatic artifact discovery. The
//! `artifact_kind` tag is recorded for forensic context;
//! verifiers don't enforce policy on the kind itself.
//!
//! ## Path policy (single, locked)
//!
//! Bundles use **`relative_to_base_dir` only** in v1. Every
//! entry's `path` is recorded relative to the bundle's
//! `base_dir`, forward-slash normalized, UTF-8 only. The
//! verifier resolves entries as `<effective_base_dir>/<path>`,
//! where `effective_base_dir` is `--base-dir` if supplied at
//! verify time, else `bundle.base_dir`.
//!
//! ## Symlink handling
//!
//! Symlinks under `--base-dir` are **followed at hash time**
//! (via `std::fs::read`, which dereferences). The recorded
//! `path` is the operator-supplied relative form — NOT the
//! symlink target. Operators wanting target-pinned paths
//! should pass canonicalized paths themselves.
//!
//! ## Refusal model
//!
//! Builder is **fail-fast** — any error short-circuits the
//! build. Verifier is **collect-all** — every entry gets an
//! outcome and `BundleVerifyReport` carries the per-entry
//! results plus aggregate counts. Envelope-level failures
//! (`UnsupportedSchemaVersion`, `EffectiveBaseDirNotFound`,
//! bundle JSON read/parse) refuse the verify entirely without
//! a per-entry walk.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::EvidenceBundleError;

/// Stage 12.22 schema version. Bumping this is a
/// forward-incompatible change. Independent of every existing
/// schema constant; v1 bundles describe v1 artifacts only.
pub const INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Cheap defense against operator-typo footguns at build time.
/// Raise via constant if real-world bundles approach the cap.
pub const BUNDLE_MAX_ENTRIES: usize = 1024;

/// 256 MiB per-entry byte cap. Stage 12.14 archives have no
/// explicit cap because they're gated by session-overall-status;
/// the bundle has no such gate, so we ship a v1 cap to bound
/// builder memory.
pub const BUNDLE_ENTRY_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// Bundle-level `--label` UTF-8 byte cap.
pub const BUNDLE_LABEL_MAX: usize = 128;

/// Bundle-level `--notes` UTF-8 byte cap.
pub const BUNDLE_NOTES_MAX: usize = 1024;

/// Closed taxonomy of bundleable artifact kinds. Only kinds
/// that correspond to a file-emitting CLI surface today are
/// represented; manually-captured outputs that have no typed
/// file emitter (e.g. `RestoreReport`) live under `Other`
/// until a real artifact lands.
///
/// Verifiers don't enforce policy on the kind itself — the
/// integrity guarantee is bytes-and-presence only. The tag is
/// recorded for forensic context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BundleArtifactKind {
    /// Stage 12.16 `state-integrity --json-out`.
    StateIntegrityReport,
    /// Stage 12.20 `sign-state-integrity-baseline --out`.
    SignedStateIntegrityBaseline,
    /// Stage 12.19 `state-integrity-diff --json-out`.
    StateIntegrityDiffReport,
    /// Stage 12.21 `sign-state-integrity-diff --out`.
    SignedStateIntegrityDiff,
    /// Stage 12.17 `plan-state-cleanup --out`.
    StateCleanupPlan,
    /// Stage 12.17 `apply-state-cleanup --format json` (the
    /// operator redirects stdout to a file).
    CleanupReport,
    /// Stage 12.17 `<quarantine-dir>/<plan_id>/quarantine-manifest.json`.
    QuarantineManifest,
    /// Stage 12.18 `restore-state-cleanup-quarantine --format json`
    /// (operator redirect).
    QuarantineRestoreReport,
    /// Stage 12.14 `<archive>/<session_id>/archive-manifest.json`.
    ArchiveManifest,
    /// Operator-supplied auxiliary file: CI log, ad-hoc note,
    /// manually-captured restore output, etc.
    Other,
}

impl BundleArtifactKind {
    /// Stable kebab/snake-case wire tag.
    pub fn as_str(self) -> &'static str {
        match self {
            BundleArtifactKind::StateIntegrityReport => "state_integrity_report",
            BundleArtifactKind::SignedStateIntegrityBaseline => {
                "signed_state_integrity_baseline"
            }
            BundleArtifactKind::StateIntegrityDiffReport => {
                "state_integrity_diff_report"
            }
            BundleArtifactKind::SignedStateIntegrityDiff => {
                "signed_state_integrity_diff"
            }
            BundleArtifactKind::StateCleanupPlan => "state_cleanup_plan",
            BundleArtifactKind::CleanupReport => "cleanup_report",
            BundleArtifactKind::QuarantineManifest => "quarantine_manifest",
            BundleArtifactKind::QuarantineRestoreReport => {
                "quarantine_restore_report"
            }
            BundleArtifactKind::ArchiveManifest => "archive_manifest",
            BundleArtifactKind::Other => "other",
        }
    }

    /// Parse a wire tag back into the closed enum. Used by the
    /// CLI's `--include <kind=path>` parser. Unknown tags →
    /// `None`; caller surfaces as a CLI argument error.
    pub fn from_wire_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "state_integrity_report" => BundleArtifactKind::StateIntegrityReport,
            "signed_state_integrity_baseline" => {
                BundleArtifactKind::SignedStateIntegrityBaseline
            }
            "state_integrity_diff_report" => {
                BundleArtifactKind::StateIntegrityDiffReport
            }
            "signed_state_integrity_diff" => {
                BundleArtifactKind::SignedStateIntegrityDiff
            }
            "state_cleanup_plan" => BundleArtifactKind::StateCleanupPlan,
            "cleanup_report" => BundleArtifactKind::CleanupReport,
            "quarantine_manifest" => BundleArtifactKind::QuarantineManifest,
            "quarantine_restore_report" => {
                BundleArtifactKind::QuarantineRestoreReport
            }
            "archive_manifest" => BundleArtifactKind::ArchiveManifest,
            "other" => BundleArtifactKind::Other,
            _ => return None,
        })
    }
}

impl std::fmt::Display for BundleArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One entry inside an `IntegrityEvidenceBundle`. Identity is
/// `(artifact_kind, path)`; duplicate identities are refused
/// at build time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleEntry {
    pub artifact_kind: BundleArtifactKind,
    /// Forward-slash-normalized path RELATIVE to the bundle's
    /// `base_dir`. Recorded the same way every time regardless
    /// of how the operator typed it on the command line.
    pub path: String,
    pub bytes: u64,
    /// 64-char lowercase hex BLAKE3 of the file bytes at build
    /// time (`blake3::hash(bytes)`).
    pub blake3_hex: String,
}

/// Stage 12.22 wrapper. Persisted as pretty JSON; the
/// verifier CLI accepts it via `--bundle`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrityEvidenceBundle {
    pub schema_version: u32,
    pub generated_at_utc: String,
    pub omni_contributor_version: String,
    /// Operator-supplied bundle-level label, ≤
    /// `BUNDLE_LABEL_MAX` UTF-8 bytes. Operator-facing naming
    /// only; no semantic meaning to the verifier.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub label: Option<String>,
    /// Operator-supplied freeform notes, ≤ `BUNDLE_NOTES_MAX`
    /// UTF-8 bytes.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub notes: Option<String>,
    /// Forward-slash-normalized absolute path. Every
    /// `entries[].path` is resolved against this at verify
    /// time unless the verifier was given a `--base-dir`
    /// override.
    pub base_dir: String,
    pub entries: Vec<BundleEntry>,
}

/// Single-input descriptor for the builder.
#[derive(Debug)]
pub struct BundleBuilderInput<'a> {
    pub artifact_kind: BundleArtifactKind,
    /// Path as supplied by the operator. May be absolute OR
    /// relative; resolution rule is documented at the module
    /// level.
    pub path: &'a Path,
}

#[derive(Debug)]
pub struct BundleBuilderOptions<'a> {
    /// RFC 3339 UTC. Stamped into `generated_at_utc`.
    pub now_utc: &'a str,
    /// REQUIRED. Forward-slash-normalized absolute path
    /// recorded into the bundle and used to canonicalize
    /// every entry's relative `path`.
    pub base_dir: &'a Path,
    pub label: Option<&'a str>,
    pub notes: Option<&'a str>,
}

/// Per-entry verify outcome. Closed set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum BundleEntryOutcome {
    /// File re-read at `resolved_path` matched the bundle
    /// entry on `bytes` AND `blake3_hex`.
    Ok,
    /// File present but byte count differs from the bundle
    /// entry. Cheap pre-check: hash is NOT recomputed when
    /// size already differs.
    SizeMismatch { expected: u64, got: u64 },
    /// File present and right size, but BLAKE3 differs.
    HashMismatch { expected: String, got: String },
    /// File not present at the resolved path.
    NotFound,
    /// FS read error other than not-found (permissions, etc.).
    ReadError { detail: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleEntryVerifyOutcome {
    pub artifact_kind: BundleArtifactKind,
    /// The base-dir-relative form recorded in the bundle.
    pub path: String,
    /// Forward-slash-normalized absolute path the verifier
    /// actually opened (after resolving against the effective
    /// base_dir).
    pub resolved_path: String,
    pub outcome: BundleEntryOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleVerifyReport {
    pub bundle_schema_version: u32,
    pub bundle_generated_at_utc: String,
    /// Forward-slash-normalized absolute path actually used as
    /// the resolution root. Override if supplied, else
    /// `bundle.base_dir`.
    pub effective_base_dir: String,
    pub counts_ok: u32,
    pub counts_size_mismatch: u32,
    pub counts_hash_mismatch: u32,
    pub counts_not_found: u32,
    pub counts_read_error: u32,
    pub entries: Vec<BundleEntryVerifyOutcome>,
}

impl BundleVerifyReport {
    /// True iff every per-entry outcome is `Ok`. The CLI
    /// surfaces this as the exit-code policy: nonzero on any
    /// non-`Ok` outcome.
    pub fn all_ok(&self) -> bool {
        self.counts_size_mismatch == 0
            && self.counts_hash_mismatch == 0
            && self.counts_not_found == 0
            && self.counts_read_error == 0
    }
}

#[derive(Debug, Default)]
pub struct VerifyOptions<'a> {
    /// Optional override. Default: use `bundle.base_dir`
    /// verbatim. This is the portability lever — the operator
    /// copies the bundle + the artifact tree to a new host and
    /// points the verifier at the new root.
    pub base_dir_override: Option<&'a Path>,
}

// ── Path helpers ──────────────────────────────────────────────

fn path_to_utf8(p: &Path) -> Result<&str, EvidenceBundleError> {
    p.to_str().ok_or_else(|| EvidenceBundleError::NonUtf8Path {
        path: p.to_path_buf(),
    })
}

/// Forward-slash normalize a path (UTF-8) for JSON wire stability.
/// Refuses non-UTF-8 input.
fn normalize_to_forward_slash(p: &Path) -> Result<String, EvidenceBundleError> {
    Ok(path_to_utf8(p)?.replace('\\', "/"))
}

/// Strict relative-path validator. Refuses any string that
/// could escape the `relative_to_base_dir` contract once
/// joined with `<base_dir>/...`. Called on EVERY recorded
/// entry path before any FS IO at both build and verify time
/// — closes the v1 path-traversal hole where a hand-edited
/// bundle (or a relative `--include` containing `..`) could
/// point the verifier outside `base_dir`.
///
/// Refusal taxonomy (closed):
/// - `empty`           — zero-length string
/// - `absolute`        — leading `/` (Unix-absolute)
/// - `backslash`       — contains `\\` (Windows path separator
///                       or escape that pre-normalization would
///                       have flipped to `/`; the bundle wire
///                       format is forward-slash only)
/// - `dot_segment`     — any `/.` or leading `.` segment
/// - `dotdot_segment`  — any `..` segment (the primary
///                       traversal vector)
/// - `empty_segment`   — `//` or trailing `/` (empty interior
///                       segment)
fn validate_relative_entry_path(p: &str) -> Result<(), EvidenceBundleError> {
    if p.is_empty() {
        return Err(EvidenceBundleError::InvalidRelativePath {
            path: p.to_string(),
            reason: "empty",
        });
    }
    if p.starts_with('/') {
        return Err(EvidenceBundleError::InvalidRelativePath {
            path: p.to_string(),
            reason: "absolute",
        });
    }
    if p.contains('\\') {
        return Err(EvidenceBundleError::InvalidRelativePath {
            path: p.to_string(),
            reason: "backslash",
        });
    }
    for segment in p.split('/') {
        match segment {
            "" => {
                return Err(EvidenceBundleError::InvalidRelativePath {
                    path: p.to_string(),
                    reason: "empty_segment",
                });
            }
            "." => {
                return Err(EvidenceBundleError::InvalidRelativePath {
                    path: p.to_string(),
                    reason: "dot_segment",
                });
            }
            ".." => {
                return Err(EvidenceBundleError::InvalidRelativePath {
                    path: p.to_string(),
                    reason: "dotdot_segment",
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Resolve a builder input into (recorded_relative_path,
/// absolute_path_to_read).
///
/// - Absolute input: canonicalize, refuse if not under
///   canonical(base_dir), strip prefix, forward-slash
///   normalize, then validate the stripped form via
///   [`validate_relative_entry_path`] as a defense-in-depth
///   check (canonical paths shouldn't contain `..`/`.`/empty
///   segments, but the validator catches any future
///   canonicalize regression).
/// - Relative input: forward-slash normalize the operator's
///   string verbatim, validate it via
///   [`validate_relative_entry_path`] **before any FS IO** so
///   `--include other=../outside/file` refuses cleanly instead
///   of escaping `base_dir`. Then treat
///   `<canonical(base_dir)>/<input>` as the read path. Symlinks
///   under base_dir are followed by the subsequent
///   `std::fs::read` per the locked Stage 12.22 policy.
fn resolve_input_path(
    input: &Path,
    canonical_base_dir: &Path,
) -> Result<(String, PathBuf), EvidenceBundleError> {
    if input.is_absolute() {
        let canon_input =
            std::fs::canonicalize(input).map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => EvidenceBundleError::EntryNotFound {
                    path: input.to_path_buf(),
                },
                _ => EvidenceBundleError::Io {
                    path: input.to_path_buf(),
                    source: e,
                },
            })?;
        let stripped =
            canon_input
                .strip_prefix(canonical_base_dir)
                .map_err(|_| EvidenceBundleError::PathOutsideBaseDir {
                    path: canon_input.clone(),
                    base_dir: canonical_base_dir.to_path_buf(),
                })?;
        let recorded = normalize_to_forward_slash(stripped)?;
        validate_relative_entry_path(&recorded)?;
        Ok((recorded, canon_input))
    } else {
        // Validate the RAW operator string BEFORE forward-slash
        // normalization so a backslash in `a\\b.json` refuses
        // explicitly instead of being silently flipped to `/`
        // and slipping past the validator.
        let raw = path_to_utf8(input)?;
        validate_relative_entry_path(raw)?;
        let recorded = raw.to_string();
        let read_path = canonical_base_dir.join(input);
        Ok((recorded, read_path))
    }
}

// ── Builder ───────────────────────────────────────────────────

/// Build a v1 `IntegrityEvidenceBundle`. Fail-fast: any error
/// short-circuits the build. The verifier is the collect-all
/// path.
///
/// **Determinism**: given the same inputs + same `base_dir` +
/// same `now_utc` + identical file contents, the returned
/// bundle is byte-identical across runs (entries sorted by
/// `(artifact_kind.as_str(), path)`, pinned by integration
/// test).
pub fn build_integrity_evidence_bundle(
    inputs: &[BundleBuilderInput<'_>],
    opts: &BundleBuilderOptions<'_>,
) -> Result<IntegrityEvidenceBundle, EvidenceBundleError> {
    // ── Bundle-level limits ───────────────────────────────
    if inputs.is_empty() {
        return Err(EvidenceBundleError::EmptyBundle);
    }
    if inputs.len() > BUNDLE_MAX_ENTRIES {
        return Err(EvidenceBundleError::TooManyEntries {
            count: inputs.len(),
            max: BUNDLE_MAX_ENTRIES,
        });
    }
    if let Some(label) = opts.label {
        let len = label.len();
        if len > BUNDLE_LABEL_MAX {
            return Err(EvidenceBundleError::BundleLabelTooLong {
                len,
                max: BUNDLE_LABEL_MAX,
            });
        }
    }
    if let Some(notes) = opts.notes {
        let len = notes.len();
        if len > BUNDLE_NOTES_MAX {
            return Err(EvidenceBundleError::NotesTooLong {
                len,
                max: BUNDLE_NOTES_MAX,
            });
        }
    }

    // ── Canonicalize base_dir ────────────────────────────
    let canonical_base_dir =
        std::fs::canonicalize(opts.base_dir).map_err(|e| {
            EvidenceBundleError::BaseDirInvalid {
                path: opts.base_dir.to_path_buf(),
                detail: e.to_string(),
            }
        })?;
    if !canonical_base_dir.is_dir() {
        return Err(EvidenceBundleError::BaseDirInvalid {
            path: opts.base_dir.to_path_buf(),
            detail: "not a directory".to_string(),
        });
    }
    let base_dir_recorded = normalize_to_forward_slash(&canonical_base_dir)?;

    // ── Hash each entry ──────────────────────────────────
    let mut entries: Vec<BundleEntry> = Vec::with_capacity(inputs.len());
    let mut seen: HashSet<(BundleArtifactKind, String)> = HashSet::new();
    for input in inputs {
        let (recorded_path, read_path) =
            resolve_input_path(input.path, &canonical_base_dir)?;

        let identity = (input.artifact_kind, recorded_path.clone());
        if !seen.insert(identity) {
            return Err(EvidenceBundleError::DuplicateEntry {
                artifact_kind: input.artifact_kind.as_str().to_string(),
                path: recorded_path,
            });
        }

        // Cheap pre-check on file size before reading bytes.
        let meta = std::fs::metadata(&read_path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => EvidenceBundleError::EntryNotFound {
                path: read_path.clone(),
            },
            _ => EvidenceBundleError::Io {
                path: read_path.clone(),
                source: e,
            },
        })?;
        let bytes = meta.len();
        if bytes > BUNDLE_ENTRY_MAX_BYTES {
            return Err(EvidenceBundleError::EntryTooLarge {
                path: recorded_path,
                bytes,
                max: BUNDLE_ENTRY_MAX_BYTES,
            });
        }

        // Read + hash. `std::fs::read` follows symlinks, which
        // matches the locked Stage 12.22 policy.
        let body = std::fs::read(&read_path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => EvidenceBundleError::EntryNotFound {
                path: read_path.clone(),
            },
            _ => EvidenceBundleError::Io {
                path: read_path.clone(),
                source: e,
            },
        })?;
        let blake3_hex = blake3::hash(&body).to_hex().to_string();

        entries.push(BundleEntry {
            artifact_kind: input.artifact_kind,
            path: recorded_path,
            bytes,
            blake3_hex,
        });
    }

    // ── Stable sort for byte-stable JSON ─────────────────
    entries.sort_by(|a, b| {
        a.artifact_kind
            .as_str()
            .cmp(b.artifact_kind.as_str())
            .then_with(|| a.path.cmp(&b.path))
    });

    Ok(IntegrityEvidenceBundle {
        schema_version: INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        generated_at_utc: opts.now_utc.to_string(),
        omni_contributor_version: env!("CARGO_PKG_VERSION").to_string(),
        label: opts.label.map(|s| s.to_string()),
        notes: opts.notes.map(|s| s.to_string()),
        base_dir: base_dir_recorded,
        entries,
    })
}

// ── Verifier (collect-all) ────────────────────────────────────

/// Verify an `IntegrityEvidenceBundle` against the filesystem.
/// Collect-all semantics: every entry gets a
/// `BundleEntryOutcome`; the verifier never short-circuits per
/// entry. Envelope-level failures (`UnsupportedSchemaVersion`,
/// `EffectiveBaseDirNotFound`) refuse with a typed error and
/// no per-entry walk.
///
/// Caller maps `BundleVerifyReport::all_ok()` to the exit
/// code.
pub fn verify_integrity_evidence_bundle(
    bundle: &IntegrityEvidenceBundle,
    opts: &VerifyOptions<'_>,
) -> Result<BundleVerifyReport, EvidenceBundleError> {
    if bundle.schema_version != INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION {
        return Err(EvidenceBundleError::UnsupportedSchemaVersion {
            got: bundle.schema_version,
            expected: INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        });
    }

    // Envelope-level path validation. Walk every entry path
    // BEFORE any FS IO and refuse on the first invalid one.
    // Closes the v1 path-traversal hole: a hand-edited bundle
    // pointing at `../outside/file` refuses cleanly here
    // instead of escaping `effective_base_dir` at join time.
    for entry in &bundle.entries {
        validate_relative_entry_path(&entry.path)?;
    }

    // Resolve effective base_dir. Operator override beats the
    // recorded base_dir; both get canonicalized so a bad root
    // refuses cleanly with one typed error rather than N false
    // `NotFound` outcomes.
    let candidate_base = match opts.base_dir_override {
        Some(p) => p.to_path_buf(),
        None => PathBuf::from(&bundle.base_dir),
    };
    let canonical_base = std::fs::canonicalize(&candidate_base).map_err(|_| {
        EvidenceBundleError::EffectiveBaseDirNotFound {
            path: candidate_base.clone(),
        }
    })?;
    if !canonical_base.is_dir() {
        return Err(EvidenceBundleError::EffectiveBaseDirNotFound {
            path: candidate_base,
        });
    }
    let effective_base_dir = normalize_to_forward_slash(&canonical_base)?;

    let mut outcomes: Vec<BundleEntryVerifyOutcome> =
        Vec::with_capacity(bundle.entries.len());
    let mut counts_ok = 0u32;
    let mut counts_size_mismatch = 0u32;
    let mut counts_hash_mismatch = 0u32;
    let mut counts_not_found = 0u32;
    let mut counts_read_error = 0u32;

    for entry in &bundle.entries {
        let resolved = canonical_base.join(&entry.path);
        let resolved_str = normalize_to_forward_slash(&resolved)?;

        let outcome = verify_one_entry(entry, &resolved);
        match &outcome {
            BundleEntryOutcome::Ok => counts_ok += 1,
            BundleEntryOutcome::SizeMismatch { .. } => counts_size_mismatch += 1,
            BundleEntryOutcome::HashMismatch { .. } => counts_hash_mismatch += 1,
            BundleEntryOutcome::NotFound => counts_not_found += 1,
            BundleEntryOutcome::ReadError { .. } => counts_read_error += 1,
        }
        outcomes.push(BundleEntryVerifyOutcome {
            artifact_kind: entry.artifact_kind,
            path: entry.path.clone(),
            resolved_path: resolved_str,
            outcome,
        });
    }

    Ok(BundleVerifyReport {
        bundle_schema_version: bundle.schema_version,
        bundle_generated_at_utc: bundle.generated_at_utc.clone(),
        effective_base_dir,
        counts_ok,
        counts_size_mismatch,
        counts_hash_mismatch,
        counts_not_found,
        counts_read_error,
        entries: outcomes,
    })
}

fn verify_one_entry(entry: &BundleEntry, resolved: &Path) -> BundleEntryOutcome {
    let meta = match std::fs::metadata(resolved) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return BundleEntryOutcome::NotFound;
        }
        Err(e) => {
            return BundleEntryOutcome::ReadError {
                detail: e.to_string(),
            };
        }
    };
    let got_bytes = meta.len();
    if got_bytes != entry.bytes {
        return BundleEntryOutcome::SizeMismatch {
            expected: entry.bytes,
            got: got_bytes,
        };
    }
    let body = match std::fs::read(resolved) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Race: file vanished between metadata + read.
            return BundleEntryOutcome::NotFound;
        }
        Err(e) => {
            return BundleEntryOutcome::ReadError {
                detail: e.to_string(),
            };
        }
    };
    let got_hex = blake3::hash(&body).to_hex().to_string();
    if got_hex != entry.blake3_hex {
        return BundleEntryOutcome::HashMismatch {
            expected: entry.blake3_hex.clone(),
            got: got_hex,
        };
    }
    BundleEntryOutcome::Ok
}

// ── Atomic writer / FS reader ─────────────────────────────────

/// Atomic temp+rename write of a bundle JSON. Same posture as
/// Stage 12.17 `plan-state-cleanup --out` / Stage 12.20
/// `sign-state-integrity-baseline --out` / Stage 12.21
/// `sign-state-integrity-diff --out`.
pub fn write_integrity_evidence_bundle_atomic(
    bundle: &IntegrityEvidenceBundle,
    out: &Path,
) -> Result<PathBuf, EvidenceBundleError> {
    let bytes = serde_json::to_vec_pretty(bundle).map_err(|e| {
        EvidenceBundleError::MalformedJson {
            path: out.to_path_buf(),
            source: e,
        }
    })?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                EvidenceBundleError::Io {
                    path: parent.to_path_buf(),
                    source: e,
                }
            })?;
        }
    }
    let tmp = out.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| EvidenceBundleError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, out).map_err(|e| EvidenceBundleError::Io {
        path: out.to_path_buf(),
        source: e,
    })?;
    Ok(out.to_path_buf())
}

/// Read a bundle JSON from disk. Bubbles FS / JSON errors with
/// their `path` for clean operator messages.
pub fn read_integrity_evidence_bundle_from_path(
    path: &Path,
) -> Result<IntegrityEvidenceBundle, EvidenceBundleError> {
    let bytes = std::fs::read(path).map_err(|e| EvidenceBundleError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| EvidenceBundleError::MalformedJson {
        path: path.to_path_buf(),
        source: e,
    })
}
