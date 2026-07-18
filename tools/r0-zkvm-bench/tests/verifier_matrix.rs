//! Envelope / allowlist-membership / binding mismatch matrix over the adopted B0
//! proof types (candidate/program/verifier-material/guest-set/spec/reproducible).

mod common;

use common::golden;
use r0_zkvm_bench::envelope::Candidate;
use r0_zkvm_bench::envelope::{allowlist_membership, MembershipError};
use r0_zkvm_bench::fixture;
use r0_zkvm_bench::verifier::{verify_synthetic, CannedReceipt, SyntheticVerifyError};

#[test]
fn happy_path_both_candidates() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let partial = fixture::partial_proof(&j, &allow).unwrap();
    for candidate in [Candidate::Sp1, Candidate::Risc0] {
        let env = fixture::envelope(&j, &allow, candidate).unwrap();
        let receipt = CannedReceipt::synthetic(&j, true);
        assert_eq!(
            verify_synthetic(&env, &partial, &allow, &j, &receipt),
            Ok(())
        );
    }
}

#[test]
fn unknown_candidate_rejected() {
    let (_s, j) = golden();
    // Allowlist has ONLY Sp1; an envelope for Risc0 has no matching entry.
    let allow = fixture::allowlist(&j, &[Candidate::Sp1]).unwrap();
    let env = fixture::envelope(&j, &allow, Candidate::Risc0).unwrap();
    assert_eq!(
        allowlist_membership(&env, &allow),
        Err(MembershipError::NoSuchCandidate)
    );
}

#[test]
fn wrong_program_id_rejected() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let mut env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    env.guest_program_id[0] ^= 0xff;
    assert_eq!(
        allowlist_membership(&env, &allow),
        Err(MembershipError::ProgramIdMismatch)
    );
}

#[test]
fn wrong_dep_lock_rejected() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let mut env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    env.candidate_dep_lock_hash[0] ^= 0xff;
    assert_eq!(
        allowlist_membership(&env, &allow),
        Err(MembershipError::DepLockMismatch)
    );
}

#[test]
fn wrong_verifier_material_rejected() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let mut env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    env.verifier_material_manifest_hash[0] ^= 0xff;
    assert_eq!(
        allowlist_membership(&env, &allow),
        Err(MembershipError::VerifierMaterialMismatch)
    );
}

#[test]
fn wrong_guest_set_rejected() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let mut env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    env.r0_guest_set_hash[0] ^= 0xff;
    assert_eq!(
        allowlist_membership(&env, &allow),
        Err(MembershipError::GuestSetMismatch)
    );
}

#[test]
fn non_reproducible_entry_rejected() {
    let (_s, j) = golden();
    let mut allow = fixture::standard_allowlist(&j).unwrap();
    allow.entries[0].reproducible = false;
    // Build the envelope from the mutated allowlist so its guest-set hash matches
    // and only the reproducibility flag differs.
    let env = fixture::envelope(&j, &allow, allow.entries[0].candidate).unwrap();
    assert_eq!(
        allowlist_membership(&env, &allow),
        Err(MembershipError::NotReproducible)
    );
}

#[test]
fn shared_binding_mismatch_rejected() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    let mut partial = fixture::partial_proof(&j, &allow).unwrap();
    partial.proof_artifact_digest[0] ^= 0xff; // disagree on a shared hash
    let receipt = CannedReceipt::synthetic(&j, true);
    assert_eq!(
        verify_synthetic(&env, &partial, &allow, &j, &receipt),
        Err(SyntheticVerifyError::SharedBindingMismatch)
    );
}

#[test]
fn envelope_binding_a_different_journal_rejected() {
    // The attacker binds a DIFFERENT (self-consistent) journal, but the honest
    // verifier expects `golden`.
    let (_s, golden_j) = golden();
    let other_j = fixture::journal().unwrap();
    let allow = fixture::standard_allowlist(&other_j).unwrap();
    let env = fixture::envelope(&other_j, &allow, Candidate::Sp1).unwrap();
    let partial = fixture::partial_proof(&other_j, &allow).unwrap();
    let receipt = CannedReceipt::synthetic(&other_j, true);
    assert_eq!(
        verify_synthetic(&env, &partial, &allow, &golden_j, &receipt),
        Err(SyntheticVerifyError::StatementHashMismatch)
    );
}

#[test]
fn wrong_journal_bytes_rejected() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    let partial = fixture::partial_proof(&j, &allow).unwrap();
    // A 996-byte receipt journal that differs from the expected journal.
    let other = fixture::journal().unwrap();
    let receipt = CannedReceipt {
        journal: other.bytes().to_vec(),
        crypto_valid: true,
    };
    assert_eq!(
        verify_synthetic(&env, &partial, &allow, &j, &receipt),
        Err(SyntheticVerifyError::JournalMismatch)
    );
}

#[test]
fn crypto_invalid_synthetic_receipt_rejected() {
    let (_s, j) = golden();
    let allow = fixture::standard_allowlist(&j).unwrap();
    let env = fixture::envelope(&j, &allow, Candidate::Sp1).unwrap();
    let partial = fixture::partial_proof(&j, &allow).unwrap();
    let receipt = CannedReceipt::synthetic(&j, false);
    assert_eq!(
        verify_synthetic(&env, &partial, &allow, &j, &receipt),
        Err(SyntheticVerifyError::ProofRejected)
    );
}
