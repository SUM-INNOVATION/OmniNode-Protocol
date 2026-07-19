//! Allowlist / identity / journal binding **logic** over the adopted B0 types
//! (NOT proof verification).
//!
//! ## What this is — and is NOT
//!
//! This models the host-side decision logic a real R0 verifier would wrap around
//! a zkVM verify call. It does **not** verify any SP1 or RISC Zero proof — no SDK
//! is integrated and no receipt is cryptographically checked. The "cryptographic
//! verify" step is modeled by a caller-supplied [`CannedReceipt`] (a synthetic
//! journal + a boolean). This is deliberately a stand-in so the surrounding
//! allowlist / identity / journal-binding logic can be exercised; it must not be
//! read as candidate proof verification.
//!
//! The binding itself reuses the frozen B0 checks:
//! [`allowlist_membership`](crate::b0::proof_envelope::allowlist_membership)
//! (candidate presence, program id, verifier-material, guest-set, dep-lock,
//! spec-hash, reproducibility) and
//! [`shared_binding_ok`](crate::b0::proof_envelope::shared_binding_ok) (the four
//! shared consensus hashes between the candidate-neutral partial proof and the
//! candidate-specific envelope). On top of those, [`verify_synthetic`] binds the
//! envelope to the exact expected 996-byte synthetic journal.

use crate::b0::allowlist::GuestProgramAllowlistV1;
use crate::b0::partial_proof::PartialComputeProofV1;
use crate::b0::proof_envelope::{
    allowlist_membership, shared_binding_ok, MembershipError, ProductionProofEnvelopeV1,
};
use crate::b0::statement::R0ComputationStatementV2;
use crate::blake3_32;
use crate::statement::SyntheticJournal;

/// A **synthetic** journal standing in for what a zkVM verify step would output.
/// Supplied by the caller — nothing here produces or verifies a real receipt. A
/// real verifier (once toolchains are authorized, which they are NOT here) would
/// obtain this journal *from* a cryptographically verified receipt instead. It
/// can never be confused with a real verifier: it holds raw bytes plus a boolean,
/// exposes no proof-system state, and [`verify_synthetic`] treats
/// `crypto_valid == false` as a hard rejection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CannedReceipt {
    /// The public journal bytes the (synthetic) receipt commits to.
    pub journal: Vec<u8>,
    /// Stand-in for the cryptographic verify outcome. A real verifier replaces
    /// this boolean with an actual SP1/RISC Zero receipt check (not implemented
    /// in this crate).
    pub crypto_valid: bool,
}

impl CannedReceipt {
    /// A synthetic receipt whose journal is the expected journal's exact 996
    /// template bytes and whose `crypto_valid` flag is caller-chosen.
    pub fn synthetic(journal: &SyntheticJournal, crypto_valid: bool) -> Self {
        Self {
            journal: journal.bytes().to_vec(),
            crypto_valid,
        }
    }
}

/// Why synthetic (non-proof) verification failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyntheticVerifyError {
    /// The envelope failed B0 allowlist membership (candidate / program id /
    /// verifier-material / guest-set / dep-lock / spec-hash / reproducibility).
    Membership(MembershipError),
    /// The candidate-neutral partial proof and the envelope disagree on one of
    /// the four shared consensus hashes.
    SharedBindingMismatch,
    /// Envelope `computation_statement_hash` != the expected journal's.
    StatementHashMismatch,
    /// Envelope `b0_pre_spec_hash` != the expected journal's spec hash.
    SpecHashMismatch,
    /// The synthetic zkVM-verify stand-in rejected the receipt.
    ProofRejected,
    /// The verified journal was not exactly 996 bytes.
    JournalLengthMismatch { got: usize },
    /// The verified journal bytes did not equal the exact expected journal.
    JournalMismatch,
}

/// Full synthetic envelope verification against an expected journal.
///
/// This is NOT proof verification. It runs, in order:
/// 1. B0 [`allowlist_membership`] for `env` against `allowlist`;
/// 2. B0 [`shared_binding_ok`] between `env` and the candidate-neutral `partial`;
/// 3. `env.computation_statement_hash == expected_journal`'s hash;
/// 4. `env.b0_pre_spec_hash == expected_journal.spec_hash()`;
/// 5. the synthetic `receipt.crypto_valid` stand-in;
/// 6. the verified journal is exactly 996 bytes and equals the expected journal;
/// 7. defense in depth: `BLAKE3(journal) == env.computation_statement_hash`.
pub fn verify_synthetic(
    env: &ProductionProofEnvelopeV1,
    partial: &PartialComputeProofV1,
    allowlist: &GuestProgramAllowlistV1,
    expected_journal: &SyntheticJournal,
    receipt: &CannedReceipt,
) -> Result<(), SyntheticVerifyError> {
    allowlist_membership(env, allowlist).map_err(SyntheticVerifyError::Membership)?;

    if !shared_binding_ok(env, partial) {
        return Err(SyntheticVerifyError::SharedBindingMismatch);
    }
    if env.computation_statement_hash != expected_journal.computation_statement_hash() {
        return Err(SyntheticVerifyError::StatementHashMismatch);
    }
    if env.b0_pre_spec_hash != expected_journal.spec_hash() {
        return Err(SyntheticVerifyError::SpecHashMismatch);
    }
    // Synthetic stand-in for the zkVM cryptographic verify — NOT a real check.
    if !receipt.crypto_valid {
        return Err(SyntheticVerifyError::ProofRejected);
    }
    if receipt.journal.len() != R0ComputationStatementV2::LEN {
        return Err(SyntheticVerifyError::JournalLengthMismatch {
            got: receipt.journal.len(),
        });
    }
    if receipt.journal.as_slice() != expected_journal.bytes() {
        return Err(SyntheticVerifyError::JournalMismatch);
    }
    if blake3_32(&receipt.journal) != env.computation_statement_hash {
        return Err(SyntheticVerifyError::JournalMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::b0::enums::Candidate;
    use crate::fixture;

    fn setup() -> (
        SyntheticJournal,
        GuestProgramAllowlistV1,
        ProductionProofEnvelopeV1,
        PartialComputeProofV1,
    ) {
        let j = fixture::journal().unwrap();
        let a = fixture::standard_allowlist(&j).unwrap();
        let env = fixture::envelope(&j, &a, Candidate::Sp1).unwrap();
        let partial = fixture::partial_proof(&j, &a).unwrap();
        (j, a, env, partial)
    }

    #[test]
    fn honest_synthetic_path_verifies() {
        let (j, a, env, partial) = setup();
        let receipt = CannedReceipt::synthetic(&j, true);
        assert_eq!(verify_synthetic(&env, &partial, &a, &j, &receipt), Ok(()));
    }

    #[test]
    fn canned_receipt_cannot_be_confused_with_a_real_verifier() {
        // crypto_valid == false is a hard rejection: the synthetic receipt is a
        // stand-in, never a proof.
        let (j, a, env, partial) = setup();
        let receipt = CannedReceipt::synthetic(&j, false);
        assert_eq!(
            verify_synthetic(&env, &partial, &a, &j, &receipt),
            Err(SyntheticVerifyError::ProofRejected)
        );
    }

    #[test]
    fn wrong_journal_length_rejected() {
        let (j, a, env, partial) = setup();
        let mut receipt = CannedReceipt::synthetic(&j, true);
        receipt.journal.push(0); // 997 bytes
        assert!(matches!(
            verify_synthetic(&env, &partial, &a, &j, &receipt),
            Err(SyntheticVerifyError::JournalLengthMismatch { got: 997 })
        ));
    }
}
