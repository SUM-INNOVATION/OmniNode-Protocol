//! Stage 12.24 — local-only integrity-evidence chain
//! verification.
//!
//! Composes the Stage 12.20–12.23 verifiers into a single
//! read-only operation. An operator runs `verify-integrity-evidence-chain`
//! against a signed bundle (Stage 12.23) and the chain
//! verifier:
//!
//! 1. Verifies the outermost Stage 12.23
//!    `SignedIntegrityEvidenceBundle` signature against an
//!    operator-supplied trust anchor (required).
//! 2. Runs Stage 12.22 bundle byte verification on the
//!    embedded `IntegrityEvidenceBundle` (re-hashes every
//!    referenced artifact file).
//! 3. For each child entry of kind
//!    `signed_state_integrity_baseline` /
//!    `signed_state_integrity_diff`, optionally verifies the
//!    child's Ed25519 signature against per-kind trust
//!    anchors. Omitted anchors record `Skipped` outcomes —
//!    NOT silent passes.
//!
//! ## Trust model
//!
//! - **Outermost gate is short-circuit.** If the signed-bundle
//!   wrapper doesn't verify (missing file, schema mismatch,
//!   forged pubkey, bad signature), the chain stops here and
//!   no per-entry / per-child events fire. Refusal bubbles as
//!   [`ChainVerifyError::SignedBundle`].
//! - **Bundle byte verify runs after signed-bundle OK.**
//!   Stage 12.22 envelope-level refusals
//!   (`EffectiveBaseDirNotFound`, `InvalidRelativePath`,
//!   schema) bubble as [`ChainVerifyError::BundleByte`]; no
//!   child-signature events fire.
//! - **Child-signature verify is collect-all.** Per-entry
//!   bundle-byte failures DON'T stop child verifies — a
//!   `HashMismatch` on bytes + a separately-verifiable
//!   signature on those bytes are independent forensic facts.
//!   Operators should see both.
//! - **Omitted child anchor records `Skipped`.** When the
//!   operator does not supply `expected_baseline_signer_pubkey_hex`
//!   or `expected_diff_signer_pubkey_hex`, the corresponding
//!   children record `ChainStepOutcome::Skipped` — not a pass.
//!   Skipped outcomes DO NOT fail the exit-code check.
//! - **Single trust anchor per child kind in v1.** Multiple
//!   signers per kind out of scope.
//! - **No recursive chain walking.** A bundle entry that is
//!   itself a signed bundle (currently recorded under
//!   `Other`) is NOT re-verified by this chain. Stage 12.22
//!   byte verify still covers it.
//!
//! ## What this attests vs what it doesn't
//!
//! - **Attests**: bundle wrapper signature, every referenced
//!   artifact's bytes match the bundle's recorded BLAKE3,
//!   each `signed_state_integrity_baseline` /
//!   `signed_state_integrity_diff` child's Ed25519 signature
//!   against the operator-supplied per-kind anchor (when
//!   supplied).
//! - **Does NOT attest**: semantic validity of any raw
//!   unsigned artifact (plain `state_integrity_report`,
//!   `state_cleanup_plan`, etc.); recursive signature chains
//!   beyond one bundle level; multi-anchor verification.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ChainVerifyError;
use crate::integrity_evidence_bundle::{
    verify_integrity_evidence_bundle, BundleArtifactKind, BundleVerifyReport,
    IntegrityEvidenceBundle, VerifyOptions as BundleVerifyOptions,
};
use crate::signed_baseline::{
    read_signed_baseline_from_path, verify_signed_state_integrity_baseline,
    BaselineSignerRole,
};
use crate::signed_bundle::{
    read_signed_integrity_evidence_bundle_from_path,
    verify_signed_integrity_evidence_bundle,
};
use crate::signed_diff::{
    read_signed_integrity_diff_from_path, verify_signed_state_integrity_diff,
};

/// Stage 12.24 chain report schema version. Bumping this is a
/// forward-incompatible change. Independent of every existing
/// schema constant; v1 reports describe v1 chain artifacts
/// only.
pub const INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION: u32 = 1;

/// Per-step verification outcome. Closed set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum ChainStepOutcome {
    /// Step verified successfully against the operator-supplied
    /// trust anchor.
    Ok,
    /// Operator did NOT supply the corresponding expected
    /// pubkey flag; the child signature check was deliberately
    /// skipped. **NOT a pass** — surfaces in the report so
    /// operators see exactly what wasn't verified.
    Skipped,
    /// Step failed; `reason` is the closed-set tag from the
    /// underlying verifier's mapper, `detail` is the typed
    /// error's `Display`.
    Failed { reason: String, detail: String },
}

/// One signed-child entry's outcome. Only emitted for
/// `signed_state_integrity_baseline` / `signed_state_integrity_diff`
/// entries — other kinds are bundle-byte-only and don't appear
/// here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChainChildEntryOutcome {
    pub artifact_kind: BundleArtifactKind,
    /// Base-dir-relative path recorded in the bundle.
    pub path: String,
    /// Forward-slash-normalized absolute path the chain
    /// verifier actually opened.
    pub resolved_path: String,
    pub signature_outcome: ChainStepOutcome,
}

/// Top-level chain report. Persisted as pretty JSON when the
/// operator supplies `--json-out`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrityEvidenceChainReport {
    pub schema_version: u32,
    pub generated_at_utc: String,
    pub omni_contributor_version: String,
    /// Forward-slash-normalized signed-bundle path the operator
    /// pointed the chain verifier at.
    pub signed_bundle_path: String,
    /// Canonical effective base_dir used for BOTH Stage 12.22
    /// byte verification AND child-signature reads. Override
    /// if supplied at chain-verify time, else
    /// `bundle.base_dir`.
    pub effective_base_dir: String,
    /// Outermost gate: Stage 12.23 signed-bundle signature
    /// verification result. `Skipped` is NOT possible here
    /// (the bundle anchor flag is required).
    pub bundle_signature: ChainStepOutcome,
    /// Verified Stage 12.23 wrapper's `signer_role` —
    /// surfaced into the chain report so the CLI's
    /// `..._signed_bundle_ok` event can carry signer
    /// identity, and so operators reading the JSON / pretty
    /// renderers see who signed the bundle without opening
    /// the wrapper file. v1 deliberately does NOT embed the
    /// full wrapper.
    pub bundle_signer_role: BaselineSignerRole,
    /// Verified Stage 12.23 wrapper's `signer_pubkey_hex`
    /// (64-char lowercase hex). Recorded for the same
    /// reason as `bundle_signer_role` — keeps the chain
    /// report self-describing on the signer-identity front
    /// without embedding the full wrapper.
    pub bundle_signer_pubkey_hex: String,
    /// Stage 12.22 bundle byte verification — embedded in full
    /// so operators get every per-entry outcome without
    /// re-running.
    pub bundle_byte_verify: BundleVerifyReport,
    /// Per-signed-child signature outcomes. Excludes entries
    /// of non-signed kinds (Stage 12.22 byte verify already
    /// covers them).
    pub child_signatures: Vec<ChainChildEntryOutcome>,
    pub counts_child_ok: u32,
    pub counts_child_skipped: u32,
    pub counts_child_failed: u32,
}

impl IntegrityEvidenceChainReport {
    /// True iff the chain verifier should exit 0:
    /// - bundle signature verified Ok,
    /// - every bundle-byte entry was Ok, AND
    /// - no signed-child signature verification failed.
    ///
    /// `Skipped` child outcomes DON'T fail the helper — they
    /// represent deliberate operator choice (no anchor
    /// supplied).
    pub fn all_required_ok(&self) -> bool {
        matches!(self.bundle_signature, ChainStepOutcome::Ok)
            && self.bundle_byte_verify.all_ok()
            && self.counts_child_failed == 0
    }
}

#[derive(Debug)]
pub struct ChainVerifyOptions<'a> {
    /// RFC 3339 UTC. Stamped into `generated_at_utc`.
    pub now_utc: &'a str,
    /// Path to the Stage 12.23 `SignedIntegrityEvidenceBundle`
    /// JSON to chain-verify.
    pub signed_bundle_path: &'a Path,
    /// REQUIRED Ed25519 trust anchor for the outermost
    /// signed-bundle wrapper.
    pub expected_bundle_signer_pubkey_hex: &'a str,
    /// Stage 12.22 base-dir override. None → use the embedded
    /// bundle's recorded `base_dir`.
    pub base_dir_override: Option<&'a Path>,
    /// Optional Stage 12.20 trust anchor — gates verification
    /// of every `signed_state_integrity_baseline` child entry.
    /// `None` → each baseline child records `Skipped`.
    pub expected_baseline_signer_pubkey_hex: Option<&'a str>,
    /// Optional Stage 12.21 trust anchor — gates verification
    /// of every `signed_state_integrity_diff` child entry.
    /// `None` → each diff child records `Skipped`.
    pub expected_diff_signer_pubkey_hex: Option<&'a str>,
}

/// Run the full chain verification.
///
/// Returns the chain report on signed-bundle and bundle-byte
/// envelope success. Per-entry and per-child outcomes are
/// inside the report; the caller inspects
/// [`IntegrityEvidenceChainReport::all_required_ok`] for the
/// exit-code decision.
///
/// Envelope-level failures from Stage 12.23 / Stage 12.22 bubble
/// as [`ChainVerifyError::SignedBundle`] / [`ChainVerifyError::BundleByte`].
pub fn verify_integrity_evidence_chain(
    opts: &ChainVerifyOptions<'_>,
) -> Result<IntegrityEvidenceChainReport, ChainVerifyError> {
    // ── 1. Outermost gate: signed-bundle wrapper ──────────
    let wrapper = read_signed_integrity_evidence_bundle_from_path(
        opts.signed_bundle_path,
    )?;
    verify_signed_integrity_evidence_bundle(
        &wrapper,
        opts.expected_bundle_signer_pubkey_hex,
    )?;
    // If we got here the signature gate is OK; from this point
    // we collect-all over children rather than fail-fast.
    // Capture the wrapper's signer identity BEFORE consuming
    // the wrapper — these flow into the chain report's
    // `bundle_signer_role` / `bundle_signer_pubkey_hex` so
    // the CLI's signed-bundle-ok event can carry signer
    // metadata without re-opening the wrapper, and so JSON /
    // pretty renderers stay self-describing.
    let bundle_signature = ChainStepOutcome::Ok;
    let bundle_signer_role = wrapper.signer_role;
    let bundle_signer_pubkey_hex = wrapper.signer_pubkey_hex.clone();

    let bundle: IntegrityEvidenceBundle = wrapper.bundle;

    // ── 2. Stage 12.22 bundle byte verify ─────────────────
    let bundle_byte_verify = verify_integrity_evidence_bundle(
        &bundle,
        &BundleVerifyOptions {
            base_dir_override: opts.base_dir_override,
        },
    )?;

    let effective_base_dir = bundle_byte_verify.effective_base_dir.clone();
    let resolution_root = PathBuf::from(&effective_base_dir);
    let signed_bundle_path_recorded =
        forward_slash_path(opts.signed_bundle_path);

    // ── 3. Per-signed-child signature verify ──────────────
    let mut child_signatures: Vec<ChainChildEntryOutcome> = Vec::new();
    let mut counts_child_ok = 0u32;
    let mut counts_child_skipped = 0u32;
    let mut counts_child_failed = 0u32;
    for entry in &bundle.entries {
        let outcome = match entry.artifact_kind {
            BundleArtifactKind::SignedStateIntegrityBaseline => {
                verify_baseline_child(
                    &resolution_root,
                    &entry.path,
                    opts.expected_baseline_signer_pubkey_hex,
                )
            }
            BundleArtifactKind::SignedStateIntegrityDiff => {
                verify_diff_child(
                    &resolution_root,
                    &entry.path,
                    opts.expected_diff_signer_pubkey_hex,
                )
            }
            // Bundle-byte verify already covered these; chain
            // verifier doesn't second-guess them here.
            _ => continue,
        };
        match &outcome.signature_outcome {
            ChainStepOutcome::Ok => counts_child_ok += 1,
            ChainStepOutcome::Skipped => counts_child_skipped += 1,
            ChainStepOutcome::Failed { .. } => counts_child_failed += 1,
        }
        child_signatures.push(ChainChildEntryOutcome {
            artifact_kind: entry.artifact_kind,
            path: entry.path.clone(),
            ..outcome
        });
    }

    Ok(IntegrityEvidenceChainReport {
        schema_version: INTEGRITY_EVIDENCE_CHAIN_REPORT_SCHEMA_VERSION,
        generated_at_utc: opts.now_utc.to_string(),
        omni_contributor_version: env!("CARGO_PKG_VERSION").to_string(),
        signed_bundle_path: signed_bundle_path_recorded,
        effective_base_dir,
        bundle_signature,
        bundle_signer_role,
        bundle_signer_pubkey_hex,
        bundle_byte_verify,
        child_signatures,
        counts_child_ok,
        counts_child_skipped,
        counts_child_failed,
    })
}

// ── Child-verify helpers ──────────────────────────────────────

fn verify_baseline_child(
    base_dir: &Path,
    entry_path: &str,
    expected_pubkey: Option<&str>,
) -> ChainChildEntryOutcome {
    let resolved = base_dir.join(entry_path);
    let resolved_str = forward_slash_path(&resolved);

    // No anchor → record Skipped without touching the file.
    let Some(anchor) = expected_pubkey else {
        return ChainChildEntryOutcome {
            artifact_kind: BundleArtifactKind::SignedStateIntegrityBaseline,
            path: entry_path.to_string(),
            resolved_path: resolved_str,
            signature_outcome: ChainStepOutcome::Skipped,
        };
    };

    let signature_outcome = match read_signed_baseline_from_path(&resolved) {
        Ok(wrapper) => {
            match verify_signed_state_integrity_baseline(&wrapper, anchor) {
                Ok(()) => ChainStepOutcome::Ok,
                Err(e) => ChainStepOutcome::Failed {
                    reason: baseline_child_reason_tag(&e).to_string(),
                    detail: format!("{e}"),
                },
            }
        }
        Err(e) => ChainStepOutcome::Failed {
            reason: baseline_child_reason_tag(&e).to_string(),
            detail: format!("{e}"),
        },
    };
    ChainChildEntryOutcome {
        artifact_kind: BundleArtifactKind::SignedStateIntegrityBaseline,
        path: entry_path.to_string(),
        resolved_path: resolved_str,
        signature_outcome,
    }
}

fn verify_diff_child(
    base_dir: &Path,
    entry_path: &str,
    expected_pubkey: Option<&str>,
) -> ChainChildEntryOutcome {
    let resolved = base_dir.join(entry_path);
    let resolved_str = forward_slash_path(&resolved);

    let Some(anchor) = expected_pubkey else {
        return ChainChildEntryOutcome {
            artifact_kind: BundleArtifactKind::SignedStateIntegrityDiff,
            path: entry_path.to_string(),
            resolved_path: resolved_str,
            signature_outcome: ChainStepOutcome::Skipped,
        };
    };

    let signature_outcome = match read_signed_integrity_diff_from_path(&resolved) {
        Ok(wrapper) => {
            match verify_signed_state_integrity_diff(&wrapper, anchor) {
                Ok(()) => ChainStepOutcome::Ok,
                Err(e) => ChainStepOutcome::Failed {
                    reason: diff_child_reason_tag(&e).to_string(),
                    detail: format!("{e}"),
                },
            }
        }
        Err(e) => ChainStepOutcome::Failed {
            reason: diff_child_reason_tag(&e).to_string(),
            detail: format!("{e}"),
        },
    };
    ChainChildEntryOutcome {
        artifact_kind: BundleArtifactKind::SignedStateIntegrityDiff,
        path: entry_path.to_string(),
        resolved_path: resolved_str,
        signature_outcome,
    }
}

/// Closed-set reason tag mapper for
/// [`crate::error::SignedBaselineError`]. Pinned by regression.
pub fn baseline_child_reason_tag(
    e: &crate::error::SignedBaselineError,
) -> &'static str {
    use crate::error::SignedBaselineError as E;
    match e {
        E::UnsupportedSchemaVersion { .. } => "unsupported_schema_version",
        E::UnsupportedReportSchemaVersion { .. } => {
            "unsupported_report_schema_version"
        }
        E::SignerPubkeyMismatch { .. } => "signer_pubkey_mismatch",
        E::SignatureMismatch => "signature_mismatch",
        E::Signing(_) => "signing",
        E::Canonical(_) => "canonical",
        E::Io { .. } => "io",
        E::MalformedJson { .. } => "malformed_json",
    }
}

/// Closed-set reason tag mapper for
/// [`crate::error::SignedIntegrityDiffError`]. Pinned by
/// regression.
pub fn diff_child_reason_tag(
    e: &crate::error::SignedIntegrityDiffError,
) -> &'static str {
    use crate::error::SignedIntegrityDiffError as E;
    match e {
        E::UnsupportedSchemaVersion { .. } => "unsupported_schema_version",
        E::UnsupportedDiffSchemaVersion { .. } => {
            "unsupported_diff_schema_version"
        }
        E::SignerPubkeyMismatch { .. } => "signer_pubkey_mismatch",
        E::SignatureMismatch => "signature_mismatch",
        E::Signing(_) => "signing",
        E::Canonical(_) => "canonical",
        E::Io { .. } => "io",
        E::MalformedJson { .. } => "malformed_json",
    }
}

fn forward_slash_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

// ── Atomic writer for --json-out ──────────────────────────────

/// Atomic temp+rename write of the chain report JSON. Used by
/// the CLI's optional `--json-out` flag (best-effort posture —
/// failure here logs a warn event and doesn't change exit). Same
/// shape as Stage 12.17/12.20/12.21/12.22/12.23 atomic writers.
pub fn write_integrity_evidence_chain_report_atomic(
    report: &IntegrityEvidenceChainReport,
    out: &Path,
) -> Result<PathBuf, ChainVerifyError> {
    let bytes = serde_json::to_vec_pretty(report).map_err(|e| {
        ChainVerifyError::MalformedJson {
            path: out.to_path_buf(),
            source: e,
        }
    })?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ChainVerifyError::Io {
                    path: parent.to_path_buf(),
                    source: e,
                }
            })?;
        }
    }
    let tmp = out.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| ChainVerifyError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, out).map_err(|e| ChainVerifyError::Io {
        path: out.to_path_buf(),
        source: e,
    })?;
    Ok(out.to_path_buf())
}
