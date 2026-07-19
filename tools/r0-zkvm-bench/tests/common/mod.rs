//! Shared fixtures for the integration tests — all built on the adopted B0 wire
//! types and the non-selection synthetic journal.

#![allow(dead_code)]

use r0_zkvm_bench::blake3_32;
use r0_zkvm_bench::envelope::Candidate;
use r0_zkvm_bench::fixture;
use r0_zkvm_bench::object::{ObjectCommitmentV1, ObjectKind};
use r0_zkvm_bench::statement::SyntheticJournal;
use r0_zkvm_bench::verifier::{verify_synthetic, CannedReceipt};
use r0_zkvm_bench::workload::{
    KvCache, Model, ReferenceExecutor, StepContext, UnitInputs, D_MODEL,
};
// The raw statement type is NOT public in r0-zkvm-bench; tests that need to build
// or mutate one depend on `sumchain-wire` directly and return through the
// zero-template-only `SyntheticJournal::from_template_bytes` boundary.
use sumchain_wire::b0::statement::{template_bytes, R0ComputationStatementV2};

pub fn model() -> Model {
    Model::deterministic(1234, blake3_32(b"golden-model"))
}

/// A rich golden statement + its journal. It starts from a real transformer
/// layer-group unit (populating model / prior-state / input-manifest / output-
/// manifest commitments) and is then enriched with a live selected-token / eos /
/// token-sequence so the full adversarial matrix has a non-default value for
/// every bindable field.
pub fn golden() -> (R0ComputationStatementV2, SyntheticJournal) {
    let m = model();
    let mut ctx = StepContext::sample(2);
    ctx.unit_index = 1;
    ctx.layer_start = 3;
    ctx.layer_end = 4;
    let prior_residual = [10i16, -20, 30, -40, 50, -60, 70, -80];
    let prior_kv = KvCache {
        k: vec![[1i16; D_MODEL], [2i16; D_MODEL]],
        v: vec![[3i16; D_MODEL], [4i16; D_MODEL]],
    };
    let token_prefix = [7u32, 9];
    let inputs = UnitInputs {
        prior_residual: &prior_residual,
        prior_kv: &prior_kv,
        token_prefix: &token_prefix,
    };
    let (_out, lg_journal) = ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs).unwrap();

    // Enrich the output half so selected_token / eos / token_seq are live. The
    // raw statement comes from `sumchain-wire` directly (not from r0-zkvm-bench).
    let mut s = R0ComputationStatementV2::decode_exact(lg_journal.bytes()).unwrap();
    s.selected_token = 5;
    s.eos_flag = 1;
    s.updated_token_seq_commitment =
        ObjectCommitmentV1::commit(ObjectKind::TokenSeq, b"tok-0-tok-1-tok-5").unwrap();
    let journal = journal_of(&s);
    (s, journal)
}

/// Turn a (possibly mutated) raw statement into its journal template, routing
/// through the zero-template-only `SyntheticJournal` boundary: `template_bytes`
/// (from `sumchain-wire`) zeroes the spec hash + validates, then
/// `from_template_bytes` re-validates canonicality on ingestion.
pub fn journal_of(statement: &R0ComputationStatementV2) -> SyntheticJournal {
    let bytes = template_bytes(statement.clone()).unwrap();
    SyntheticJournal::from_template_bytes(&bytes).unwrap()
}

/// Model the attacker: they build a self-consistent SP1 binding for `mutated`,
/// but the honest verifier independently expects `golden`. Returns `true` iff
/// verification rejects (which it must for any mutation that changes the journal).
pub fn attack_rejected(golden: &SyntheticJournal, mutated: &SyntheticJournal) -> bool {
    let allow = fixture::standard_allowlist(mutated).unwrap();
    let env = fixture::envelope(mutated, &allow, Candidate::Sp1).unwrap();
    let partial = fixture::partial_proof(mutated, &allow).unwrap();
    let receipt = CannedReceipt::synthetic(mutated, true);
    verify_synthetic(&env, &partial, &allow, golden, &receipt).is_err()
}

/// The honest path for a journal: consistent binding, verifies against itself.
pub fn honest_verifies(journal: &SyntheticJournal) -> bool {
    let allow = fixture::standard_allowlist(journal).unwrap();
    let env = fixture::envelope(journal, &allow, Candidate::Sp1).unwrap();
    let partial = fixture::partial_proof(journal, &allow).unwrap();
    let receipt = CannedReceipt::synthetic(journal, true);
    verify_synthetic(&env, &partial, &allow, journal, &receipt).is_ok()
}
