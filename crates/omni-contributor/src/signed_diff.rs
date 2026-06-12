//! Stage 12.21 — local-only signed integrity-diff wrapper.
//!
//! Wraps a v1 [`StateIntegrityDiffReport`] with an Ed25519
//! signature over a closed canonical body so an operator can
//! archive tamper-evident evidence of (a) which baseline was
//! compared, (b) which current report was compared, (c) the
//! exact diff that was produced, and (d) which key signed the
//! diff artifact.
//!
//! The wrapper is purely local: no protocol envelope, no SNIP
//! wire, no chain interaction. Mirrors the Stage 12.20
//! `SignedStateIntegrityBaseline` posture verbatim — same
//! signing input recipe, same two-step verification (cheap
//! pubkey-match pre-check, then crypto verify), same atomic
//! temp+rename writer.
//!
//! ```text
//! bytes = SIGNED_INTEGRITY_DIFF_DOMAIN || bincode1::serialize(canonical_body)
//! signature = Ed25519::sign(seed, bytes)
//! ```
//!
//! `canonical_body` excludes `signature_hex` (signing over your
//! own signature is circular). It embeds the full typed
//! [`StateIntegrityDiffReport`] so a captured wrapper is
//! self-contained — no out-of-band diff file is needed at
//! verify time. Stage 12.19's v1 diff schema is consumed
//! UNCHANGED; bincode-1 over the typed struct is byte-stable
//! across Rust toolchains because the diff's field order is
//! pinned by the struct definition and bincode-1 is positional.
//!
//! The signer-role enum is reused from
//! [`crate::signed_baseline::BaselineSignerRole`] per the
//! Stage 12.21 plan — the four variants are role names, not
//! artifact-type names, so the same taxonomy applies to
//! baselines and diffs alike.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::canonical::{hex_lower, SIGNED_INTEGRITY_DIFF_DOMAIN};
use crate::error::{SignedIntegrityDiffError, SigningError};
use crate::integrity::{
    StateIntegrityDiffReport, STATE_INTEGRITY_DIFF_SCHEMA_VERSION,
};
use crate::signed_baseline::BaselineSignerRole;
use crate::signing::verify_signature_hex;

/// Stage 12.21 schema version. Bumping this is a
/// forward-incompatible change. Independent of every existing
/// schema constant; v1 wrappers wrap v1 diffs only.
pub const SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION: u32 = 1;

/// Stage 12.21 wrapper. Persisted as pretty JSON; the verifier
/// CLI accepts it via `--signed-diff`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedStateIntegrityDiff {
    pub schema_version: u32,
    pub signed_at_utc: String,
    /// 64-char lowercase-hex Ed25519 public key. Verifiers MUST
    /// compare against an operator-supplied trust anchor; this
    /// field alone is forensic context, not authorization.
    pub signer_pubkey_hex: String,
    pub signer_role: BaselineSignerRole,
    /// The v1 `StateIntegrityDiffReport` this wrapper signs,
    /// embedded in full.
    pub diff: StateIntegrityDiffReport,
    /// Ed25519 signature over
    /// `SIGNED_INTEGRITY_DIFF_DOMAIN || bincode1(canonical_body)`,
    /// where `canonical_body` is this struct minus
    /// `signature_hex`. 128-char lowercase hex.
    pub signature_hex: String,
}

/// Canonical-body projection used as the signing input. Field
/// declaration order is the bincode wire order and is frozen
/// for `SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION = 1`.
#[derive(Debug, Serialize)]
struct SignedIntegrityDiffCanonicalBody<'a> {
    schema_version: u32,
    signed_at_utc: &'a str,
    signer_pubkey_hex: &'a str,
    signer_role: BaselineSignerRole,
    diff: &'a StateIntegrityDiffReport,
}

/// Build the canonical signing input. Used by both
/// [`sign_state_integrity_diff`] (to produce a signature) and
/// [`verify_signed_state_integrity_diff`] (to recompute the same
/// bytes for crypto verification).
pub fn signed_integrity_diff_signing_input(
    schema_version: u32,
    signed_at_utc: &str,
    signer_pubkey_hex: &str,
    signer_role: BaselineSignerRole,
    diff: &StateIntegrityDiffReport,
) -> Result<Vec<u8>, SignedIntegrityDiffError> {
    let body = SignedIntegrityDiffCanonicalBody {
        schema_version,
        signed_at_utc,
        signer_pubkey_hex,
        signer_role,
        diff,
    };
    let body_bytes = bincode1::serialize(&body).map_err(|e| {
        SignedIntegrityDiffError::Canonical(crate::error::CanonicalError::from(e))
    })?;
    let mut out =
        Vec::with_capacity(SIGNED_INTEGRITY_DIFF_DOMAIN.len() + body_bytes.len());
    out.extend_from_slice(SIGNED_INTEGRITY_DIFF_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Sign a v1 `StateIntegrityDiffReport`. Returns the assembled
/// wrapper ready to be serialized to JSON.
///
/// The caller supplies the signer pubkey + a closure that
/// computes the signature; the library stays signer-agnostic
/// so it doesn't take a generic dependency on the four
/// role-typed signer structs.
///
/// **Determinism**: given the same `diff` + `signer_role` +
/// `signer_pubkey_hex` + `signed_at_utc` + signing closure
/// (i.e. same seed), the returned wrapper is byte-identical
/// across runs. This is pinned by a regression test.
pub fn sign_state_integrity_diff<F>(
    diff: StateIntegrityDiffReport,
    signer_pubkey_hex: &str,
    signer_role: BaselineSignerRole,
    signed_at_utc: &str,
    sign_fn: F,
) -> Result<SignedStateIntegrityDiff, SignedIntegrityDiffError>
where
    F: FnOnce(&[u8]) -> [u8; 64],
{
    if diff.schema_version != STATE_INTEGRITY_DIFF_SCHEMA_VERSION {
        return Err(SignedIntegrityDiffError::UnsupportedDiffSchemaVersion {
            got: diff.schema_version,
            expected: STATE_INTEGRITY_DIFF_SCHEMA_VERSION,
        });
    }
    let signing_input = signed_integrity_diff_signing_input(
        SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION,
        signed_at_utc,
        signer_pubkey_hex,
        signer_role,
        &diff,
    )?;
    let signature = sign_fn(&signing_input);
    let signature_hex = hex_lower(&signature);
    Ok(SignedStateIntegrityDiff {
        schema_version: SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION,
        signed_at_utc: signed_at_utc.to_string(),
        signer_pubkey_hex: signer_pubkey_hex.to_string(),
        signer_role,
        diff,
        signature_hex,
    })
}

/// Verify a signed diff against an operator-supplied trust
/// anchor (`expected_signer_pubkey_hex`). Two-step refusal
/// order:
///
/// 1. Cheap pre-check: refuse with
///    [`SignedIntegrityDiffError::SignerPubkeyMismatch`] when the
///    wrapper's `signer_pubkey_hex` doesn't equal the trust
///    anchor. No crypto burn.
/// 2. Cryptographic verify: recompute the canonical body
///    bytes and check the Ed25519 signature against the
///    wrapper's pubkey. Refuse with
///    [`SignedIntegrityDiffError::SignatureMismatch`] on
///    `Ok(false)`; bubble decode failures as
///    [`SignedIntegrityDiffError::Signing`].
///
/// Schema enforcement runs before either check.
pub fn verify_signed_state_integrity_diff(
    wrapper: &SignedStateIntegrityDiff,
    expected_signer_pubkey_hex: &str,
) -> Result<(), SignedIntegrityDiffError> {
    if wrapper.schema_version != SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION {
        return Err(SignedIntegrityDiffError::UnsupportedSchemaVersion {
            got: wrapper.schema_version,
            expected: SIGNED_INTEGRITY_DIFF_SCHEMA_VERSION,
        });
    }
    if wrapper.diff.schema_version != STATE_INTEGRITY_DIFF_SCHEMA_VERSION {
        return Err(SignedIntegrityDiffError::UnsupportedDiffSchemaVersion {
            got: wrapper.diff.schema_version,
            expected: STATE_INTEGRITY_DIFF_SCHEMA_VERSION,
        });
    }
    if wrapper.signer_pubkey_hex != expected_signer_pubkey_hex {
        return Err(SignedIntegrityDiffError::SignerPubkeyMismatch {
            expected: expected_signer_pubkey_hex.to_string(),
            got: wrapper.signer_pubkey_hex.clone(),
        });
    }
    let signing_input = signed_integrity_diff_signing_input(
        wrapper.schema_version,
        &wrapper.signed_at_utc,
        &wrapper.signer_pubkey_hex,
        wrapper.signer_role,
        &wrapper.diff,
    )?;
    let ok = verify_signature_hex(
        &wrapper.signer_pubkey_hex,
        &signing_input,
        &wrapper.signature_hex,
    )
    .map_err(SignedIntegrityDiffError::Signing)?;
    if !ok {
        return Err(SignedIntegrityDiffError::SignatureMismatch);
    }
    Ok(())
}

/// Read a signed-diff wrapper JSON from disk. Convenience
/// helper for the CLI consumer side; bubbles FS / JSON errors
/// with their `path` for clean operator messages.
pub fn read_signed_integrity_diff_from_path(
    path: &Path,
) -> Result<SignedStateIntegrityDiff, SignedIntegrityDiffError> {
    let bytes = std::fs::read(path).map_err(|e| SignedIntegrityDiffError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        SignedIntegrityDiffError::MalformedJson {
            path: path.to_path_buf(),
            source: e,
        }
    })
}

/// Atomic temp+rename write of a signed-diff wrapper. Used by
/// the CLI's `sign-state-integrity-diff --out` path; same
/// posture as Stage 12.17's `plan-state-cleanup --out` and
/// Stage 12.20's `sign-state-integrity-baseline --out`.
pub fn write_signed_integrity_diff_atomic(
    wrapper: &SignedStateIntegrityDiff,
    out: &Path,
) -> Result<PathBuf, SignedIntegrityDiffError> {
    let bytes = serde_json::to_vec_pretty(wrapper).map_err(|e| {
        SignedIntegrityDiffError::MalformedJson {
            path: out.to_path_buf(),
            source: e,
        }
    })?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SignedIntegrityDiffError::Io {
                    path: parent.to_path_buf(),
                    source: e,
                }
            })?;
        }
    }
    let tmp = out.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| SignedIntegrityDiffError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, out).map_err(|e| SignedIntegrityDiffError::Io {
        path: out.to_path_buf(),
        source: e,
    })?;
    Ok(out.to_path_buf())
}

// Compile-time guard: SigningError must be `From` into
// SignedIntegrityDiffError so the `?` operator works cleanly
// in callers. Already declared via #[from] in error.rs; this
// const_assert keeps a future refactor honest.
const _: fn() = || {
    fn assert_from<T, U: From<T>>() {}
    assert_from::<SigningError, SignedIntegrityDiffError>();
};
