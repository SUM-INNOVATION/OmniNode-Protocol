//! **Synthetic, non-selection** B0 identities and binding sets.
//!
//! Everything here is a placeholder for exercising the adopted B0 types and their
//! binding logic. NONE of it is a real proof, a real allowlist, real verifier
//! material, or a measurement, and none of it is ever written as a protocol
//! artifact. Placeholder identities come exclusively through these dedicated
//! `fixture` constructors (requirement: not inferred from a zero spec hash), and
//! the produced structures are self-consistent so the honest binding path in
//! [`crate::verifier`] accepts them while any mutation is rejected.

use crate::b0::allowlist::{BuilderArch, GuestProgramAllowlistV1, GuestProgramEntryV1};
use crate::b0::codec::DecodeError;
use crate::b0::consts;
use crate::b0::enums::{Arch, Candidate, ObjectKind, UnitKind, VerifierMaterialRole};
use crate::b0::partial_proof::PartialComputeProofV1;
use crate::b0::proof_envelope::ProductionProofEnvelopeV1;
use crate::b0::verifier_material::{VerifierMaterialEntry, VerifierMaterialManifestV1};
use crate::blake3_32;
use crate::object::ObjectCommitmentV1;
use crate::statement::{R0ComputationStatementV2, SyntheticJournal};

/// Domain label prefix making every synthetic identity self-evidently a
/// non-selection placeholder (never a real protocol identity).
const FIXTURE_LABEL: &[u8] = b"SUMCHAIN/R0/FIXTURE/NON-SELECTION/";

/// A deterministic, clearly-synthetic 32-byte identity for `tag` (optionally
/// per-candidate).
fn synth(tag: &str, candidate: Option<Candidate>) -> [u8; 32] {
    let mut buf = FIXTURE_LABEL.to_vec();
    buf.extend_from_slice(tag.as_bytes());
    if let Some(c) = candidate {
        buf.extend_from_slice(&c.to_repr().to_le_bytes());
    }
    blake3_32(&buf)
}

/// Synthetic guest program id (SP1 vkey / RISC Zero image-id stand-in).
pub fn program_id(candidate: Candidate) -> [u8; 32] {
    synth("guest-program-id", Some(candidate))
}

/// Synthetic candidate dependency-lock hash.
pub fn dep_lock(candidate: Candidate) -> [u8; 32] {
    synth("candidate-dep-lock", Some(candidate))
}

/// Synthetic proof-artifact digest. There is NO real proof; this digest merely
/// binds the four-hash proof identity to the journal for round-trip / binding
/// tests.
pub fn proof_digest(journal: &SyntheticJournal) -> [u8; 32] {
    let mut buf = FIXTURE_LABEL.to_vec();
    buf.extend_from_slice(b"no-real-proof-artifact/");
    buf.extend_from_slice(&journal.computation_statement_hash());
    blake3_32(&buf)
}

/// A minimal, valid, all-empty-commitment statement journal for tests.
pub fn journal() -> Result<SyntheticJournal, DecodeError> {
    let statement = R0ComputationStatementV2 {
        b0_pre_spec_hash: [0u8; 32],
        job_id: blake3_32(b"fixture.job"),
        session_id: blake3_32(b"fixture.session"),
        unit_id: blake3_32(b"fixture.unit"),
        unit_kind: UnitKind::TransformerLayerGroup,
        unit_index: 0,
        generation_index: 0,
        model_id: blake3_32(b"fixture.model"),
        model_commitment: ObjectCommitmentV1::empty(ObjectKind::Model),
        tokenizer_id: blake3_32(b"fixture.tokenizer"),
        head_dim: consts::HEAD_DIM,
        ffn_dim: consts::FFN_DIM,
        layer_start: 0,
        layer_end: 1,
        vocab_size: consts::VOCAB_SIZE,
        d_model: consts::D_MODEL,
        n_heads: consts::N_HEADS,
        derived_input_commitment: ObjectCommitmentV1::empty(ObjectKind::DerivedInput),
        prior_residual_stream: ObjectCommitmentV1::empty(ObjectKind::PriorResidual),
        prior_kv_cache: ObjectCommitmentV1::empty(ObjectKind::PriorKv),
        token_prefix: ObjectCommitmentV1::empty(ObjectKind::TokenPrefix),
        input_manifest: ObjectCommitmentV1::empty(ObjectKind::InputManifest),
        sequence_length: 1,
        position: 0,
        output_manifest: ObjectCommitmentV1::empty(ObjectKind::OutputManifest),
        selected_token: u32::MAX,
        updated_token_seq_commitment: ObjectCommitmentV1::empty(ObjectKind::TokenSeq),
        eos_flag: 0,
        max_cycles: consts::MAX_CYCLES,
        max_d_model: consts::MAX_D_MODEL,
        max_seq_len: consts::MAX_SEQ_LEN,
        max_output_tokens: consts::MAX_OUTPUT_TOKENS,
        max_manifest_slots: consts::MAX_MANIFEST_SLOTS,
        max_state_bytes: consts::MAX_STATE_BYTES,
    };
    SyntheticJournal::from_statement_template(statement)
}

/// A synthetic single-entry verifier-material manifest for `candidate`.
pub fn verifier_material(candidate: Candidate) -> VerifierMaterialManifestV1 {
    VerifierMaterialManifestV1 {
        candidate,
        entries: vec![VerifierMaterialEntry {
            label: "GROTH16_VK_BYTES".to_string(),
            role: VerifierMaterialRole::Groth16Vk,
            byte_len: 0,
            hash: synth("verifier-material-groth16-vk", Some(candidate)),
        }],
    }
}

/// A single synthetic allowlist entry for `candidate`, consistent with the
/// envelope this module builds and with the given journal's spec hash.
fn entry(
    journal: &SyntheticJournal,
    candidate: Candidate,
) -> Result<GuestProgramEntryV1, DecodeError> {
    Ok(GuestProgramEntryV1 {
        candidate,
        b0_pre_spec_hash: journal.spec_hash(),
        guest_source_tree_hash: synth("guest-source-tree", Some(candidate)),
        candidate_dep_lock_hash: dep_lock(candidate),
        arches: vec![
            BuilderArch {
                arch: Arch::X86_64,
                builder_container_digest: synth("builder-container-x86_64", Some(candidate)),
            },
            BuilderArch {
                arch: Arch::Aarch64,
                builder_container_digest: synth("builder-container-aarch64", Some(candidate)),
            },
        ],
        guest_image_hash: synth("guest-image", Some(candidate)),
        program_id: program_id(candidate),
        verifier_material_manifest_hash: verifier_material(candidate).try_identity()?,
        build_command_hash: synth("build-command", Some(candidate)),
        reproducible: true,
    })
}

/// A synthetic allowlist over `candidates` (which must be given ascending and
/// unique, per the frozen ordering rule). Fails closed via `validate`.
pub fn allowlist(
    journal: &SyntheticJournal,
    candidates: &[Candidate],
) -> Result<GuestProgramAllowlistV1, DecodeError> {
    let mut entries = Vec::with_capacity(candidates.len());
    for &c in candidates {
        entries.push(entry(journal, c)?);
    }
    let a = GuestProgramAllowlistV1 { entries };
    a.validate()?;
    Ok(a)
}

/// The standard synthetic allowlist: SP1 (1) then RISC Zero (2), ascending.
pub fn standard_allowlist(
    journal: &SyntheticJournal,
) -> Result<GuestProgramAllowlistV1, DecodeError> {
    allowlist(journal, &[Candidate::Sp1, Candidate::Risc0])
}

/// A candidate-neutral partial-compute proof (4 hashes only) bound to `journal`
/// and `allowlist`.
pub fn partial_proof(
    journal: &SyntheticJournal,
    allowlist: &GuestProgramAllowlistV1,
) -> Result<PartialComputeProofV1, DecodeError> {
    Ok(PartialComputeProofV1 {
        computation_statement_hash: journal.computation_statement_hash(),
        b0_pre_spec_hash: journal.spec_hash(),
        r0_guest_set_hash: allowlist.try_guest_set_hash()?,
        proof_artifact_digest: proof_digest(journal),
    })
}

/// A candidate-specific production proof envelope bound to `journal`,
/// `allowlist`, and `candidate` — consistent with the matching allowlist entry.
pub fn envelope(
    journal: &SyntheticJournal,
    allowlist: &GuestProgramAllowlistV1,
    candidate: Candidate,
) -> Result<ProductionProofEnvelopeV1, DecodeError> {
    Ok(ProductionProofEnvelopeV1 {
        candidate,
        candidate_dep_lock_hash: dep_lock(candidate),
        guest_program_id: program_id(candidate),
        verifier_material_manifest_hash: verifier_material(candidate).try_identity()?,
        computation_statement_hash: journal.computation_statement_hash(),
        b0_pre_spec_hash: journal.spec_hash(),
        r0_guest_set_hash: allowlist.try_guest_set_hash()?,
        proof_artifact_digest: proof_digest(journal),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_journal_and_bindings_are_self_consistent() {
        let j = journal().unwrap();
        let a = standard_allowlist(&j).unwrap();
        let env = envelope(&j, &a, Candidate::Sp1).unwrap();
        let partial = partial_proof(&j, &a).unwrap();
        // envelope <-> partial agree on the four shared hashes.
        assert!(crate::b0::proof_envelope::shared_binding_ok(&env, &partial));
        // envelope is a member of the allowlist.
        assert!(crate::b0::proof_envelope::allowlist_membership(&env, &a).is_ok());
    }
}
