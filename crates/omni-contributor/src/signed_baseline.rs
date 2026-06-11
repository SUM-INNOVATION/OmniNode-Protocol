//! Stage 12.20 — local-only signed integrity-baseline wrapper.
//!
//! Wraps a v1 [`StateIntegrityReport`] with an Ed25519
//! signature over a closed canonical body so an operator can
//! prove which key produced a baseline and detect hand-edited
//! baseline JSON before the Stage 12.19 diff runs.
//!
//! The wrapper is purely local: no protocol envelope, no SNIP
//! wire, no chain interaction. The signing input mirrors the
//! Stage 12.0–12.18 envelope pattern:
//!
//! ```text
//! bytes = SIGNED_BASELINE_DOMAIN || bincode1::serialize(canonical_body)
//! signature = Ed25519::sign(seed, bytes)
//! ```
//!
//! `canonical_body` excludes `signature_hex` (signing over your
//! own signature is circular). It embeds the full typed
//! [`StateIntegrityReport`] so a captured wrapper is
//! self-contained — no out-of-band report file is needed at
//! verify time. Stage 12.16's v1 report schema is consumed
//! UNCHANGED; bincode-1 over the typed struct is byte-stable
//! across Rust toolchains because the report's field order is
//! pinned by the struct definition and bincode-1 is positional.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::canonical::{hex_lower, SIGNED_BASELINE_DOMAIN};
use crate::error::{SignedBaselineError, SigningError};
use crate::integrity::{
    StateIntegrityReport, STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
};
use crate::signing::verify_signature_hex;

/// Stage 12.20 schema version. Bumping this is a
/// forward-incompatible change. Independent of every
/// existing schema constant; v1 wrappers wrap v1 reports
/// only.
pub const SIGNED_BASELINE_SCHEMA_VERSION: u32 = 1;

/// Closed taxonomy of baseline signer roles. The role tag is
/// recorded for forensics; verifiers don't enforce policy on
/// the role itself — the trust anchor is the
/// `signer_pubkey_hex` the caller passes to
/// [`verify_signed_state_integrity_baseline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum BaselineSignerRole {
    /// Operator-local key dedicated to baseline signing.
    /// Recommended for new deployments; lets operators
    /// rotate baselines without touching protocol-role keys.
    Operator,
    Contributor,
    Dispatcher,
    Coordinator,
}

impl BaselineSignerRole {
    /// Stable kebab/snake-case wire tag.
    pub fn as_str(self) -> &'static str {
        match self {
            BaselineSignerRole::Operator => "operator",
            BaselineSignerRole::Contributor => "contributor",
            BaselineSignerRole::Dispatcher => "dispatcher",
            BaselineSignerRole::Coordinator => "coordinator",
        }
    }
}

/// Stage 12.20 wrapper. Persisted as pretty JSON; the diff
/// CLIs accept it via `--signed-baseline`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedStateIntegrityBaseline {
    pub schema_version: u32,
    pub signed_at_utc: String,
    /// 64-char lowercase-hex Ed25519 public key. Verifiers MUST
    /// compare against an operator-supplied trust anchor; this
    /// field alone is forensic context, not authorization.
    pub signer_pubkey_hex: String,
    pub signer_role: BaselineSignerRole,
    /// The v1 `StateIntegrityReport` this baseline wraps,
    /// embedded in full.
    pub report: StateIntegrityReport,
    /// Ed25519 signature over
    /// `SIGNED_BASELINE_DOMAIN || bincode1(canonical_body)`,
    /// where `canonical_body` is this struct minus
    /// `signature_hex`. 128-char lowercase hex.
    pub signature_hex: String,
}

/// Canonical-body projection used as the signing input. Field
/// declaration order is the bincode wire order and is frozen
/// for `SIGNED_BASELINE_SCHEMA_VERSION = 1`.
#[derive(Debug, Serialize)]
struct SignedBaselineCanonicalBody<'a> {
    schema_version: u32,
    signed_at_utc: &'a str,
    signer_pubkey_hex: &'a str,
    signer_role: BaselineSignerRole,
    report: &'a StateIntegrityReport,
}

/// Build the canonical signing input. Used by both
/// [`sign_state_integrity_baseline`] (to produce a signature)
/// and [`verify_signed_state_integrity_baseline`] (to
/// recompute the same bytes for crypto verification).
pub fn signed_baseline_signing_input(
    schema_version: u32,
    signed_at_utc: &str,
    signer_pubkey_hex: &str,
    signer_role: BaselineSignerRole,
    report: &StateIntegrityReport,
) -> Result<Vec<u8>, SignedBaselineError> {
    let body = SignedBaselineCanonicalBody {
        schema_version,
        signed_at_utc,
        signer_pubkey_hex,
        signer_role,
        report,
    };
    let body_bytes = bincode1::serialize(&body).map_err(|e| {
        SignedBaselineError::Canonical(crate::error::CanonicalError::from(e))
    })?;
    let mut out = Vec::with_capacity(SIGNED_BASELINE_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(SIGNED_BASELINE_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Sign a v1 `StateIntegrityReport`. Returns the assembled
/// wrapper ready to be serialized to JSON.
///
/// The caller supplies the signer pubkey + a closure that
/// computes the signature; the library stays signer-agnostic
/// so it doesn't take a generic dependency on the four
/// role-typed signer structs.
///
/// **Determinism**: given the same `report` + `signer_role` +
/// `signer_pubkey_hex` + `signed_at_utc` + signing closure
/// (i.e. same seed), the returned wrapper is byte-identical
/// across runs. This is pinned by a regression test.
pub fn sign_state_integrity_baseline<F>(
    report: StateIntegrityReport,
    signer_pubkey_hex: &str,
    signer_role: BaselineSignerRole,
    signed_at_utc: &str,
    sign_fn: F,
) -> Result<SignedStateIntegrityBaseline, SignedBaselineError>
where
    F: FnOnce(&[u8]) -> [u8; 64],
{
    if report.schema_version != STATE_INTEGRITY_REPORT_SCHEMA_VERSION {
        return Err(SignedBaselineError::UnsupportedReportSchemaVersion {
            got: report.schema_version,
            expected: STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
        });
    }
    let signing_input = signed_baseline_signing_input(
        SIGNED_BASELINE_SCHEMA_VERSION,
        signed_at_utc,
        signer_pubkey_hex,
        signer_role,
        &report,
    )?;
    let signature = sign_fn(&signing_input);
    let signature_hex = hex_lower(&signature);
    Ok(SignedStateIntegrityBaseline {
        schema_version: SIGNED_BASELINE_SCHEMA_VERSION,
        signed_at_utc: signed_at_utc.to_string(),
        signer_pubkey_hex: signer_pubkey_hex.to_string(),
        signer_role,
        report,
        signature_hex,
    })
}

/// Verify a signed baseline against an operator-supplied trust
/// anchor (`expected_signer_pubkey_hex`). Two-step refusal
/// order:
///
/// 1. Cheap pre-check: refuse with
///    [`SignedBaselineError::SignerPubkeyMismatch`] when the
///    wrapper's `signer_pubkey_hex` doesn't equal the trust
///    anchor. No crypto burn.
/// 2. Cryptographic verify: recompute the canonical body
///    bytes and check the Ed25519 signature against the
///    wrapper's pubkey. Refuse with
///    [`SignedBaselineError::SignatureMismatch`] on
///    `Ok(false)`; bubble decode failures as
///    [`SignedBaselineError::Signing`].
///
/// Schema enforcement runs before either check.
pub fn verify_signed_state_integrity_baseline(
    baseline: &SignedStateIntegrityBaseline,
    expected_signer_pubkey_hex: &str,
) -> Result<(), SignedBaselineError> {
    if baseline.schema_version != SIGNED_BASELINE_SCHEMA_VERSION {
        return Err(SignedBaselineError::UnsupportedSchemaVersion {
            got: baseline.schema_version,
            expected: SIGNED_BASELINE_SCHEMA_VERSION,
        });
    }
    if baseline.report.schema_version != STATE_INTEGRITY_REPORT_SCHEMA_VERSION {
        return Err(SignedBaselineError::UnsupportedReportSchemaVersion {
            got: baseline.report.schema_version,
            expected: STATE_INTEGRITY_REPORT_SCHEMA_VERSION,
        });
    }
    if baseline.signer_pubkey_hex != expected_signer_pubkey_hex {
        return Err(SignedBaselineError::SignerPubkeyMismatch {
            expected: expected_signer_pubkey_hex.to_string(),
            got: baseline.signer_pubkey_hex.clone(),
        });
    }
    let signing_input = signed_baseline_signing_input(
        baseline.schema_version,
        &baseline.signed_at_utc,
        &baseline.signer_pubkey_hex,
        baseline.signer_role,
        &baseline.report,
    )?;
    let ok = verify_signature_hex(
        &baseline.signer_pubkey_hex,
        &signing_input,
        &baseline.signature_hex,
    )
    .map_err(SignedBaselineError::Signing)?;
    if !ok {
        return Err(SignedBaselineError::SignatureMismatch);
    }
    Ok(())
}

/// Read a signed-baseline wrapper JSON from disk.
/// Convenience helper for the CLI consumer side; bubbles
/// FS / JSON errors with their `path` for clean operator
/// messages.
pub fn read_signed_baseline_from_path(
    path: &Path,
) -> Result<SignedStateIntegrityBaseline, SignedBaselineError> {
    let bytes = std::fs::read(path).map_err(|e| SignedBaselineError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| SignedBaselineError::MalformedJson {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Atomic temp+rename write of a signed-baseline wrapper.
/// Used by the CLI's `sign-state-integrity-baseline --out`
/// path; same posture as Stage 12.17's
/// `plan-state-cleanup --out`.
pub fn write_signed_baseline_atomic(
    baseline: &SignedStateIntegrityBaseline,
    out: &Path,
) -> Result<PathBuf, SignedBaselineError> {
    let bytes = serde_json::to_vec_pretty(baseline)
        .map_err(|e| SignedBaselineError::MalformedJson {
            path: out.to_path_buf(),
            source: e,
        })?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| SignedBaselineError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
    }
    let tmp = out.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| SignedBaselineError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, out).map_err(|e| SignedBaselineError::Io {
        path: out.to_path_buf(),
        source: e,
    })?;
    Ok(out.to_path_buf())
}

// Compile-time guard: SigningError must be `From` into
// SignedBaselineError so the `?` operator works cleanly in
// callers. Already declared via #[from] in error.rs; this
// const_assert keeps a future refactor honest.
const _: fn() = || {
    fn assert_from<T, U: From<T>>() {}
    assert_from::<SigningError, SignedBaselineError>();
};
