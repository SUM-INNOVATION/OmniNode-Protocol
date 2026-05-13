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

// ── Stage 5: chain client + offline attestation registry ──────────────────────

/// Failure reported by a [`crate::chain::ChainClient`] implementation.
///
/// `Clone` is derived so test fixtures (and a future real chain adapter's
/// own error machinery) can store a canned outcome and reuse it. The
/// single-variant `Other(String)` shape preserves the implementation's
/// diagnostic message verbatim; a future real chain adapter may upgrade
/// to structured variants.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ChainClientError {
    #[error("chain client failure: {0}")]
    Other(String),
}

/// Failure produced by the [`crate::registry::AttestationRegistry`] and
/// its workflow free functions.
///
/// Intentionally **not** `Clone` (`io::Error` is not `Clone`).
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("registry serialization failure: {0}")]
    Serialization(String),

    #[error("registry I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("attestation record not found: {0}")]
    RecordNotFound(crate::registry::AttestationId),

    /// Returned by `insert` when an existing record under the same
    /// `(session_id, verifier_address)` carries a byte-different
    /// `InferenceAttestation`. The existing record on disk is **not**
    /// overwritten; the caller can `load(&id)` to inspect the stored
    /// value.
    #[error(
        "conflicting attestation under existing id {id}: session_id and \
         verifier_address match but the stored attestation differs from \
         the one being inserted"
    )]
    ConflictingAttestation { id: crate::registry::AttestationId },

    #[error("invalid status transition for {id}: cannot go from {from:?} to {to}")]
    InvalidStatusTransition {
        id: crate::registry::AttestationId,
        from: crate::registry::LocalAttestationStatus,
        to: &'static str,
    },

    #[error("chain client failure: {0}")]
    ChainClient(#[from] ChainClientError),
}

/// Stage 5 result alias. Distinct from [`Result`] (Stage 3) and
/// [`AttestationResult`] (Stage 4) so each domain's errors stay typed at
/// every call site.
pub type RegistryResult<T> = std::result::Result<T, RegistryError>;
