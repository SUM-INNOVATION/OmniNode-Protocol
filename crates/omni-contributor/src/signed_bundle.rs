//! Stage 12.23 — local-only signed integrity-evidence-bundle
//! wrapper.
//!
//! Wraps a v1 [`IntegrityEvidenceBundle`] with an Ed25519
//! signature over a closed canonical body so an operator can
//! attest provenance ("this bundle JSON came from a specific
//! key at a specific time") while keeping byte-fingerprint
//! verification (Stage 12.22
//! `verify_integrity_evidence_bundle`) as a separate step.
//!
//! The wrapper is purely local: no protocol envelope, no SNIP
//! wire, no chain interaction. Mirrors the Stage 12.20
//! [`crate::signed_baseline::SignedStateIntegrityBaseline`]
//! and Stage 12.21
//! [`crate::signed_diff::SignedStateIntegrityDiff`] posture
//! verbatim — same signing input recipe, same two-step
//! verification (cheap pubkey-match pre-check, then crypto
//! verify), same atomic temp+rename writer.
//!
//! ```text
//! bytes = SIGNED_INTEGRITY_EVIDENCE_BUNDLE_DOMAIN
//!       || bincode1::serialize(canonical_body)
//! signature = Ed25519::sign(seed, bytes)
//! ```
//!
//! `canonical_body` excludes `signature_hex` (signing over your
//! own signature is circular). It embeds the full typed
//! [`IntegrityEvidenceBundle`] so a captured wrapper is
//! self-contained — no out-of-band bundle file is needed at
//! verify time. Stage 12.22's v1 bundle schema is consumed
//! UNCHANGED; bincode-1 over the typed struct is byte-stable
//! across Rust toolchains because the bundle's field order is
//! pinned by the struct definition and bincode-1 is positional.
//!
//! The signer-role enum is reused from
//! [`crate::signed_baseline::BaselineSignerRole`] per the
//! Stage 12.21 precedent — the four variants are role names,
//! not artifact-type names, so the same taxonomy applies to
//! baselines, diffs, AND bundles alike.
//!
//! ## What this signature covers vs what it doesn't
//!
//! - **Covered**: every field of the wrapper EXCEPT
//!   `signature_hex` — `schema_version`, `signed_at_utc`,
//!   `signer_pubkey_hex`, `signer_role`, AND every byte of the
//!   embedded `IntegrityEvidenceBundle` (`bundle.schema_version`,
//!   `bundle.generated_at_utc`, `bundle.omni_contributor_version`,
//!   `bundle.label`, `bundle.notes`, `bundle.base_dir`, and
//!   every `BundleEntry` with its `artifact_kind` / `path` /
//!   `bytes` / `blake3_hex`). Because the bundle already records
//!   per-entry BLAKE3 hashes, the signature transitively pins
//!   the entire artifact tree without re-embedding any artifact
//!   bytes.
//! - **NOT covered** (deliberate v1 scope): the artifact files
//!   the bundle references. Re-hashing them is Stage 12.22's
//!   `verify_integrity_evidence_bundle` job — operators run
//!   that separately if they care about byte integrity. This
//!   verifier attests to bundle bytes only.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::canonical::{hex_lower, SIGNED_INTEGRITY_EVIDENCE_BUNDLE_DOMAIN};
use crate::error::{SignedIntegrityEvidenceBundleError, SigningError};
use crate::integrity_evidence_bundle::{
    IntegrityEvidenceBundle, INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
};
use crate::signed_baseline::BaselineSignerRole;
use crate::signing::verify_signature_hex;

/// Stage 12.23 schema version. Bumping this is a
/// forward-incompatible change. Independent of every existing
/// schema constant; v1 wrappers wrap v1 bundles only.
pub const SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION: u32 = 1;

/// Stage 12.23 wrapper. Persisted as pretty JSON; the verifier
/// CLI accepts it via `--signed-bundle`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedIntegrityEvidenceBundle {
    pub schema_version: u32,
    pub signed_at_utc: String,
    /// 64-char lowercase-hex Ed25519 public key. Verifiers MUST
    /// compare against an operator-supplied trust anchor; this
    /// field alone is forensic context, not authorization.
    pub signer_pubkey_hex: String,
    pub signer_role: BaselineSignerRole,
    /// The v1 `IntegrityEvidenceBundle` this wrapper signs,
    /// embedded in full so a captured wrapper is
    /// self-contained.
    pub bundle: IntegrityEvidenceBundle,
    /// Ed25519 signature over
    /// `SIGNED_INTEGRITY_EVIDENCE_BUNDLE_DOMAIN || bincode1(canonical_body)`,
    /// where `canonical_body` is this struct minus
    /// `signature_hex`. 128-char lowercase hex.
    pub signature_hex: String,
}

/// Canonical-body projection used as the signing input. Field
/// declaration order is the bincode wire order and is frozen
/// for `SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION = 1`.
#[derive(Debug, Serialize)]
struct SignedIntegrityEvidenceBundleCanonicalBody<'a> {
    schema_version: u32,
    signed_at_utc: &'a str,
    signer_pubkey_hex: &'a str,
    signer_role: BaselineSignerRole,
    bundle: &'a IntegrityEvidenceBundle,
}

/// Build the canonical signing input. Used by both
/// [`sign_integrity_evidence_bundle`] (to produce a signature)
/// and [`verify_signed_integrity_evidence_bundle`] (to
/// recompute the same bytes for crypto verification).
pub fn signed_integrity_evidence_bundle_signing_input(
    schema_version: u32,
    signed_at_utc: &str,
    signer_pubkey_hex: &str,
    signer_role: BaselineSignerRole,
    bundle: &IntegrityEvidenceBundle,
) -> Result<Vec<u8>, SignedIntegrityEvidenceBundleError> {
    let body = SignedIntegrityEvidenceBundleCanonicalBody {
        schema_version,
        signed_at_utc,
        signer_pubkey_hex,
        signer_role,
        bundle,
    };
    let body_bytes = bincode1::serialize(&body).map_err(|e| {
        SignedIntegrityEvidenceBundleError::Canonical(
            crate::error::CanonicalError::from(e),
        )
    })?;
    let mut out = Vec::with_capacity(
        SIGNED_INTEGRITY_EVIDENCE_BUNDLE_DOMAIN.len() + body_bytes.len(),
    );
    out.extend_from_slice(SIGNED_INTEGRITY_EVIDENCE_BUNDLE_DOMAIN);
    out.extend_from_slice(&body_bytes);
    Ok(out)
}

/// Sign a v1 `IntegrityEvidenceBundle`. Returns the assembled
/// wrapper ready to be serialized to JSON.
///
/// The caller supplies the signer pubkey + a closure that
/// computes the signature; the library stays signer-agnostic
/// so it doesn't take a generic dependency on the four
/// role-typed signer structs.
///
/// **Determinism**: given the same `bundle` + `signer_role` +
/// `signer_pubkey_hex` + `signed_at_utc` + signing closure
/// (i.e. same seed), the returned wrapper is byte-identical
/// across runs. This is pinned by a regression test.
///
/// **Pre-sign validation (locked v1 scope)**: only the embedded
/// `bundle.schema_version` is checked. The signer does NOT
/// re-run Stage 12.22 envelope-level path validation or
/// `(kind, path)` duplicate checks; an operator who hands the
/// signer a poisoned bundle gets a wrapper that the Stage 12.22
/// `verify_integrity_evidence_bundle` will refuse downstream.
/// Per locked v1 scope.
pub fn sign_integrity_evidence_bundle<F>(
    bundle: IntegrityEvidenceBundle,
    signer_pubkey_hex: &str,
    signer_role: BaselineSignerRole,
    signed_at_utc: &str,
    sign_fn: F,
) -> Result<SignedIntegrityEvidenceBundle, SignedIntegrityEvidenceBundleError>
where
    F: FnOnce(&[u8]) -> [u8; 64],
{
    if bundle.schema_version != INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION {
        return Err(
            SignedIntegrityEvidenceBundleError::UnsupportedBundleSchemaVersion {
                got: bundle.schema_version,
                expected: INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            },
        );
    }
    let signing_input = signed_integrity_evidence_bundle_signing_input(
        SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        signed_at_utc,
        signer_pubkey_hex,
        signer_role,
        &bundle,
    )?;
    let signature = sign_fn(&signing_input);
    let signature_hex = hex_lower(&signature);
    Ok(SignedIntegrityEvidenceBundle {
        schema_version: SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
        signed_at_utc: signed_at_utc.to_string(),
        signer_pubkey_hex: signer_pubkey_hex.to_string(),
        signer_role,
        bundle,
        signature_hex,
    })
}

/// Verify a signed wrapper against an operator-supplied trust
/// anchor (`expected_signer_pubkey_hex`). Two-step refusal
/// order:
///
/// 1. Cheap pre-check: refuse with
///    [`SignedIntegrityEvidenceBundleError::SignerPubkeyMismatch`]
///    when the wrapper's `signer_pubkey_hex` doesn't equal the
///    trust anchor. No crypto burn.
/// 2. Cryptographic verify: recompute the canonical body
///    bytes and check the Ed25519 signature against the
///    wrapper's pubkey. Refuse with
///    [`SignedIntegrityEvidenceBundleError::SignatureMismatch`]
///    on `Ok(false)`; bubble decode failures as
///    [`SignedIntegrityEvidenceBundleError::Signing`].
///
/// Schema enforcement runs before either check.
pub fn verify_signed_integrity_evidence_bundle(
    wrapper: &SignedIntegrityEvidenceBundle,
    expected_signer_pubkey_hex: &str,
) -> Result<(), SignedIntegrityEvidenceBundleError> {
    if wrapper.schema_version != SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION {
        return Err(
            SignedIntegrityEvidenceBundleError::UnsupportedSchemaVersion {
                got: wrapper.schema_version,
                expected: SIGNED_INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            },
        );
    }
    if wrapper.bundle.schema_version != INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION {
        return Err(
            SignedIntegrityEvidenceBundleError::UnsupportedBundleSchemaVersion {
                got: wrapper.bundle.schema_version,
                expected: INTEGRITY_EVIDENCE_BUNDLE_SCHEMA_VERSION,
            },
        );
    }
    if wrapper.signer_pubkey_hex != expected_signer_pubkey_hex {
        return Err(
            SignedIntegrityEvidenceBundleError::SignerPubkeyMismatch {
                expected: expected_signer_pubkey_hex.to_string(),
                got: wrapper.signer_pubkey_hex.clone(),
            },
        );
    }
    let signing_input = signed_integrity_evidence_bundle_signing_input(
        wrapper.schema_version,
        &wrapper.signed_at_utc,
        &wrapper.signer_pubkey_hex,
        wrapper.signer_role,
        &wrapper.bundle,
    )?;
    let ok = verify_signature_hex(
        &wrapper.signer_pubkey_hex,
        &signing_input,
        &wrapper.signature_hex,
    )
    .map_err(SignedIntegrityEvidenceBundleError::Signing)?;
    if !ok {
        return Err(SignedIntegrityEvidenceBundleError::SignatureMismatch);
    }
    Ok(())
}

/// Read a signed-bundle wrapper JSON from disk. Convenience
/// helper for the CLI consumer side; bubbles FS / JSON errors
/// with their `path` for clean operator messages.
pub fn read_signed_integrity_evidence_bundle_from_path(
    path: &Path,
) -> Result<SignedIntegrityEvidenceBundle, SignedIntegrityEvidenceBundleError> {
    let bytes =
        std::fs::read(path).map_err(|e| SignedIntegrityEvidenceBundleError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
    serde_json::from_slice(&bytes).map_err(|e| {
        SignedIntegrityEvidenceBundleError::MalformedJson {
            path: path.to_path_buf(),
            source: e,
        }
    })
}

/// Atomic temp+rename write of a signed-bundle wrapper. Used
/// by the CLI's `sign-integrity-evidence-bundle --out` path;
/// same posture as Stage 12.17/12.20/12.21 atomic writers.
pub fn write_signed_integrity_evidence_bundle_atomic(
    wrapper: &SignedIntegrityEvidenceBundle,
    out: &Path,
) -> Result<PathBuf, SignedIntegrityEvidenceBundleError> {
    let bytes = serde_json::to_vec_pretty(wrapper).map_err(|e| {
        SignedIntegrityEvidenceBundleError::MalformedJson {
            path: out.to_path_buf(),
            source: e,
        }
    })?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                SignedIntegrityEvidenceBundleError::Io {
                    path: parent.to_path_buf(),
                    source: e,
                }
            })?;
        }
    }
    let tmp = out.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes).map_err(|e| {
        SignedIntegrityEvidenceBundleError::Io {
            path: tmp.clone(),
            source: e,
        }
    })?;
    std::fs::rename(&tmp, out).map_err(|e| {
        SignedIntegrityEvidenceBundleError::Io {
            path: out.to_path_buf(),
            source: e,
        }
    })?;
    Ok(out.to_path_buf())
}

// Compile-time guard: SigningError must be `From` into
// SignedIntegrityEvidenceBundleError so the `?` operator works
// cleanly in callers. Already declared via #[from] in error.rs;
// this const_assert keeps a future refactor honest.
const _: fn() = || {
    fn assert_from<T, U: From<T>>() {}
    assert_from::<SigningError, SignedIntegrityEvidenceBundleError>();
};
