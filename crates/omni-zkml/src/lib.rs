//! `omni-zkml` — Phase 5 zkML and proof attestation crate.
//!
//! Stage 3 laid down the **proof artifact substrate**: types that describe
//! local response/proof byte files, a publish flow that ingests them into
//! SNIP V2 Public via the existing `omni-store::SnipV2Adapter`, and a
//! builder that assembles an [`omni_types::phase5::InferenceCommitment`].
//!
//! Stage 4 adds the **local verifier attestation envelope**: canonical
//! domain-separated bytes for an `InferenceCommitment`, a 32-byte BLAKE3
//! [`attestation::CommitmentDigest`], a [`attestation::Signer`] trait
//! abstraction, and a builder that produces the existing
//! [`omni_types::phase5::InferenceAttestation`].
//!
//! No real proof generation, no verifier, no chain client, and no real
//! cryptographic signing scheme is present in this crate yet — proof
//! bytes are opaque blobs supplied by the caller, and the `Signer` trait
//! is satisfied by a local fake in unit tests. The actual zk machinery
//! and chain integration are the subject of Stage 5+.

pub mod artifact;
pub mod attestation;
pub mod error;

pub use artifact::{
    build_commitment, publish_proof_artifacts, ProofArtifact, ProofPublishReport, ResponseArtifact,
};
pub use attestation::{
    build_attestation, compute_canonical_bytes, compute_digest, CommitmentDigest,
    CommitmentPayload, Signer, DOMAIN_TAG,
};
pub use error::{
    AttestationError, AttestationResult, ProofArtifactError, Result, SignerError,
};
