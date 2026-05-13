//! `omni-zkml` — Phase 5 zkML and proof attestation crate.
//!
//! Stage 3 laid down the **proof artifact substrate** (publish opaque
//! response/proof byte files to SNIP V2 Public and build an
//! [`omni_types::phase5::InferenceCommitment`]).
//!
//! Stage 4 added the **local verifier attestation envelope** (canonical
//! domain-separated bytes, BLAKE3 digest, [`attestation::Signer`] trait
//! seam, [`attestation::build_attestation`] producing an
//! [`omni_types::phase5::InferenceAttestation`]).
//!
//! Stage 5 adds the **chain client abstraction** and the **offline
//! attestation registry**:
//! - [`chain::ChainClient`] — synchronous trait future chain adapters
//!   implement. No real RPC, no tx encoding this stage.
//! - [`registry::AttestationRegistry`] — filesystem-backed JSON-per-record
//!   store, keyed by `(session_id, verifier_address)` to match the chain
//!   proposal's de-duplication rule. Atomic `.tmp` + rename writes;
//!   deterministic `list()` order.
//! - [`registry::submit_attestation_workflow`] /
//!   [`registry::query_attestation_workflow`] — composite operations that
//!   drive the chain client + registry together. RPC failures propagate
//!   as [`error::RegistryError::ChainClient`] and **leave records
//!   unchanged**; only an explicit chain `Failed { reason }` or
//!   `Dropped { reason }` transitions a record into those states.
//!
//! No real cryptographic signing scheme, no real chain implementation,
//! no proof verifier is wired in this crate yet. Those are Stage 6+.

pub mod artifact;
pub mod attestation;
pub mod chain;
pub mod error;
pub mod registry;

pub use artifact::{
    build_commitment, publish_proof_artifacts, ProofArtifact, ProofPublishReport, ResponseArtifact,
};
pub use attestation::{
    build_attestation, compute_canonical_bytes, compute_digest, CommitmentDigest,
    CommitmentPayload, Signer, DOMAIN_TAG,
};
pub use chain::{AttestationStatus, ChainClient, SubmissionReceipt};
pub use error::{
    AttestationError, AttestationResult, ChainClientError, ProofArtifactError, RegistryError,
    RegistryResult, Result, SignerError,
};
pub use registry::{
    compute_attestation_id, query_attestation_workflow, submit_attestation_workflow,
    AttestationId, AttestationRecord, AttestationRegistry, LocalAttestationStatus,
    ATTESTATION_ID_DOMAIN,
};
