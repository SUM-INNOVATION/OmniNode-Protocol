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
//!   unchanged**; only an explicit chain `Failed { reason }`
//!   transitions a record into a terminal local state.
//!
//! Stage 5.2 layers **client-local staleness / retry policy** on top
//! via [`staleness`]. SUM Chain v1 has no chain-side `Dropped`; this
//! module is the only writer that transitions `Submitted → Dropped`,
//! based on a caller-constructed [`staleness::StalenessPolicy`] and a
//! `current_block` height the caller fetches from the chain.
//! Stage 7a/7b ship the real SUM Chain `ChainClient` adapter
//! (`omni-sumchain`).
//!
//! Stage 5.3 stitches everything above into one operator-facing
//! surface via [`orchestration`]. Four hermetic workflow helpers
//! ([`orchestration::submit_attestation_workflow_with_block`],
//! [`orchestration::poll_attestations_workflow`],
//! [`orchestration::sweep_stale_attestations_workflow`],
//! [`orchestration::retry_dropped_attestations_workflow`]) compose
//! over a sibling [`orchestration::OrchestrationClient`] trait that
//! extends [`chain::ChainClient`] with a single
//! `get_latest_block_height` method. The chain protocol surface is
//! unchanged; `SumChainClient` implements the new trait additively.

pub mod artifact;
pub mod attestation;
pub mod chain;
pub mod chain_wire;
pub mod error;
pub mod orchestration;
pub mod proof;
pub mod registry;
pub mod staleness;

pub use artifact::{
    build_commitment, publish_proof_artifacts, ProofArtifact, ProofPublishReport, ResponseArtifact,
};
pub use attestation::{
    build_attestation, compute_canonical_bytes, compute_digest, CommitmentDigest,
    CommitmentPayload, Signer, DOMAIN_TAG,
};
pub use chain::{AttestationStatus, ChainClient, SubmissionReceipt};
pub use chain_wire::{
    canonical_digest_bytes, commitment_to_chain_digest, compute_chain_attestation_vector,
    derive_chain_address_base58, sign_chain_attestation_digest, signer_chain_address_base58,
    signer_pubkey_bytes, signer_pubkey_hex, signing_input_bytes, ChainAttestationVector,
    InferenceAttestationDigest, InferenceAttestationTxData, MAX_SESSION_ID_BYTES,
};
pub use error::{
    AttestationError, AttestationResult, ChainClientError, ChainWireError, ChainWireResult,
    ProofArtifactError, ProofBackendError, ProofPipelineError, ProofVerifierError, RegistryError,
    RegistryResult, Result, SignerError,
};
pub use proof::{
    produce_proof_artifact, MockProofBackend, MockProofVerifier, ProofArtifactBody, ProofBackend,
    ProofMetadata, ProofPipelineInputs, ProofPipelineOutputs, ProofVerifier, PublicInputs,
    MOCK_BACKEND_ID,
};
pub use registry::{
    compute_attestation_id, query_attestation_workflow, submit_attestation_workflow,
    AttestationId, AttestationRecord, AttestationRegistry, LocalAttestationStatus,
    ATTESTATION_ID_DOMAIN,
};
pub use orchestration::{
    poll_attestations_workflow, retry_dropped_attestations_workflow,
    submit_attestation_workflow_with_block, sweep_stale_attestations_workflow,
    OrchestrationClient,
};
pub use staleness::{
    is_record_stale, mark_stale_if_overdue, StalenessPolicy, StalenessPolicyError,
};
