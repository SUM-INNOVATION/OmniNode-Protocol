//! Proof identity — the adopted B0 candidate-neutral
//! [`PartialComputeProofV1`] (**137 bytes**) and candidate-specific
//! [`ProductionProofEnvelopeV1`] (**235 bytes**), plus the pure cross-binding
//! checks.
//!
//! The bespoke OmniNode `R0ProofArtifactEnvelopeV1` / `Candidate` / `ProofRefKind`
//! and the `OMNINODE.R0.ENVELOPE.V1` domain are **deleted**.
//!
//! * [`PartialComputeProofV1`] is **candidate-neutral**: it carries only the four
//!   shared consensus hashes (`computation_statement_hash`, `b0_pre_spec_hash`,
//!   `r0_guest_set_hash`, `proof_artifact_digest`) — no `candidate`, program id,
//!   or verifier material.
//! * [`ProductionProofEnvelopeV1`] is where candidate identity lives: the
//!   [`Candidate`] discriminant, `guest_program_id`, `candidate_dep_lock_hash`,
//!   and `verifier_material_manifest_hash`, on top of the same four shared hashes.
//! * [`shared_binding_ok`] checks a partial and an envelope agree on all four
//!   shared hashes; [`allowlist_membership`] locates/validates the allowlist entry
//!   an envelope claims.

pub use sumchain_wire::b0::enums::Candidate;
pub use sumchain_wire::b0::partial_proof::PartialComputeProofV1;
pub use sumchain_wire::b0::proof_envelope::{
    allowlist_membership, shared_binding_ok, MembershipError, ProductionProofEnvelopeV1,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::b0::codec::DecodeError;
    use crate::fixture;

    #[test]
    fn partial_proof_is_137_bytes_and_candidate_neutral() {
        let j = fixture::journal().unwrap();
        let a = fixture::standard_allowlist(&j).unwrap();
        let partial = fixture::partial_proof(&j, &a).unwrap();
        let bytes = partial.encode();
        assert_eq!(bytes.len(), 137);
        assert_eq!(bytes.len(), PartialComputeProofV1::LEN);
        assert_eq!(
            PartialComputeProofV1::decode_exact(&bytes).unwrap(),
            partial
        );
        // Candidate-neutral: exactly four 32-byte hashes after the 9-byte header,
        // and no `candidate` field exists on the type at all.
        assert_eq!(bytes.len(), 7 + 2 + 4 * 32);
    }

    #[test]
    fn production_envelope_is_235_bytes_and_carries_candidate_identity() {
        let j = fixture::journal().unwrap();
        let a = fixture::standard_allowlist(&j).unwrap();
        let env = fixture::envelope(&j, &a, Candidate::Risc0).unwrap();
        let bytes = env.encode();
        assert_eq!(bytes.len(), 235);
        assert_eq!(bytes.len(), ProductionProofEnvelopeV1::LEN);
        assert_eq!(
            ProductionProofEnvelopeV1::decode_exact(&bytes).unwrap(),
            env
        );
        assert_eq!(env.candidate, Candidate::Risc0);
    }

    #[test]
    fn unknown_candidate_discriminant_rejected() {
        let j = fixture::journal().unwrap();
        let a = fixture::standard_allowlist(&j).unwrap();
        let mut bytes = fixture::envelope(&j, &a, Candidate::Sp1).unwrap().encode();
        // candidate u16 at offset 9..11 (7 magic + 2 schema version).
        bytes[9..11].copy_from_slice(&0u16.to_le_bytes());
        assert!(matches!(
            ProductionProofEnvelopeV1::decode_exact(&bytes),
            Err(DecodeError::BadEnum {
                name: "Candidate",
                ..
            })
        ));
    }

    #[test]
    fn trailing_bytes_rejected_on_both() {
        let j = fixture::journal().unwrap();
        let a = fixture::standard_allowlist(&j).unwrap();
        let mut p = fixture::partial_proof(&j, &a).unwrap().encode();
        p.push(0);
        assert!(matches!(
            PartialComputeProofV1::decode_exact(&p),
            Err(DecodeError::TrailingBytes { .. })
        ));
        let mut e = fixture::envelope(&j, &a, Candidate::Sp1).unwrap().encode();
        e.push(0);
        assert!(matches!(
            ProductionProofEnvelopeV1::decode_exact(&e),
            Err(DecodeError::TrailingBytes { .. })
        ));
    }

    #[test]
    fn shared_binding_detects_candidate_neutral_disagreement() {
        let j = fixture::journal().unwrap();
        let a = fixture::standard_allowlist(&j).unwrap();
        let env = fixture::envelope(&j, &a, Candidate::Sp1).unwrap();
        let mut partial = fixture::partial_proof(&j, &a).unwrap();
        assert!(shared_binding_ok(&env, &partial));
        partial.proof_artifact_digest[0] ^= 0xff;
        assert!(!shared_binding_ok(&env, &partial));
    }
}
