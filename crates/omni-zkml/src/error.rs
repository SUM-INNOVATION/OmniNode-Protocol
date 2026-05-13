//! Local error type for `omni-zkml` proof / response artifact flow.
//!
//! Kept local to this crate — not bridged into `omni_store::StoreError` or
//! `omni_types::OmniError`. `omni-zkml` is allowed to depend on `omni-store`
//! and surface its errors upward via the `SnipV2` variant, but the reverse
//! is never true.

use std::io;
use std::path::PathBuf;

use omni_store::SnipV2Error;

#[derive(Debug, thiserror::Error)]
pub enum ProofArtifactError {
    #[error("local response file not found: {}", path.display())]
    ResponseFileNotFound { path: PathBuf },

    #[error("local proof file not found: {}", path.display())]
    ProofFileNotFound { path: PathBuf },

    #[error("session_id is empty")]
    EmptySessionId,

    #[error("manifest has no top-level SNIP V2 ref; cannot build commitment")]
    ManifestLacksSnipRoot,

    #[error("response artifact has no BLAKE3 hash; cannot build commitment")]
    ResponseLacksHash,

    #[error("proof artifact has no SNIP V2 ref; cannot build commitment")]
    ProofLacksSnipRoot,

    #[error("SNIP V2 error: {0}")]
    SnipV2(#[from] SnipV2Error),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

pub type Result<T> = std::result::Result<T, ProofArtifactError>;

// ── Stage 4: signer + attestation envelope ────────────────────────────────────

/// Failure reported by a [`crate::attestation::Signer`] implementation.
///
/// `Clone` is derived so test fixtures can store a canned
/// `Result<String, SignerError>` and `.clone()` it on every call to
/// `sign(...)`. The single-variant `Failed(String)` preserves the
/// implementation's diagnostic message verbatim; a future stage may
/// upgrade to a structured `#[source]` when concrete signer
/// implementations introduce their own error types.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SignerError {
    #[error("signer failure: {0}")]
    Failed(String),
}

/// Failure produced by [`crate::attestation::build_attestation`] and the
/// related canonical-bytes / digest helpers. Intentionally **not** `Clone`
/// — Stage 4 callers consume the error directly; fixtures only need
/// `SignerError` to be cloneable.
#[derive(Debug, thiserror::Error)]
pub enum AttestationError {
    #[error("canonical-bytes serialization failed: {0}")]
    Serialization(String),

    #[error("commitment has empty session_id")]
    EmptySessionId,

    #[error("commitment has empty model_hash")]
    EmptyModelHash,

    #[error("commitment has empty response_hash")]
    EmptyResponseHash,

    #[error("signer returned empty verifier_address")]
    EmptyVerifierAddress,

    #[error("signer returned empty signature")]
    EmptySignature,

    #[error("signer failure: {0}")]
    Signer(#[from] SignerError),
}

/// Stage 4 result alias. Deliberately separate from [`Result`] (Stage 3, =
/// `std::result::Result<T, ProofArtifactError>`) so callers in
/// `attestation.rs` never route attestation errors through the Stage-3
/// error domain.
pub type AttestationResult<T> = std::result::Result<T, AttestationError>;
