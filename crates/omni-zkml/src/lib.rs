//! `omni-zkml` — Phase 5 zkML and proof attestation crate.
//!
//! Stage 3 lays down the **proof artifact substrate**: types that describe
//! local response/proof byte files, a publish flow that ingests them into
//! SNIP V2 Public via the existing `omni-store::SnipV2Adapter`, and a
//! builder that assembles an [`omni_types::phase5::InferenceCommitment`]
//! ready for a future stage to sign and submit.
//!
//! No real proof generation, no verifier, no signing, no chain client is
//! present in this crate yet — proof bytes are opaque blobs supplied by the
//! caller. The actual zk machinery is the subject of Stage 4+.

pub mod artifact;
pub mod error;

pub use artifact::{
    build_commitment, publish_proof_artifacts, ProofArtifact, ProofPublishReport, ResponseArtifact,
};
pub use error::{ProofArtifactError, Result};
