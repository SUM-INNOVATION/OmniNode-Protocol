//! The full adversarial binding matrix over the adopted B0 996-byte statement.
//!
//! For every *bindable* statement field, mutating it must EITHER change the
//! journal's `computation_statement_hash` OR fail verification. We assert both:
//! the mutated journal hashes differently, and an attacker who binds the mutated
//! journal is rejected when the honest verifier expects the golden journal.

mod common;

use common::{attack_rejected, golden, honest_verifies, journal_of};
use r0_zkvm_bench::object::{ObjectCommitmentV1, ObjectKind};
use r0_zkvm_bench::statement::{SyntheticJournal, UnitKind};
// The raw statement type is NOT public in r0-zkvm-bench; mutate it via the
// `sumchain-wire` dependency directly, then return through `journal_of`.
use sumchain_wire::b0::statement::R0ComputationStatementV2;

/// Assert a mutation is bound: different journal hash AND rejected on verify.
fn assert_bound(name: &str, golden_j: &SyntheticJournal, mutated: R0ComputationStatementV2) {
    let mutated_j = journal_of(&mutated);
    assert_ne!(
        mutated_j.computation_statement_hash(),
        golden_j.computation_statement_hash(),
        "[{name}] mutation did not change the statement hash"
    );
    assert!(
        attack_rejected(golden_j, &mutated_j),
        "[{name}] mutated statement was NOT rejected on verify"
    );
}

fn mutate(
    base: &R0ComputationStatementV2,
    f: impl FnOnce(&mut R0ComputationStatementV2),
) -> R0ComputationStatementV2 {
    let mut s = base.clone();
    f(&mut s);
    s
}

#[test]
fn full_binding_matrix() {
    let (base, golden_j) = golden();

    let oc = |kind, data: &[u8]| ObjectCommitmentV1::commit(kind, data).unwrap();

    let cases: Vec<(&str, R0ComputationStatementV2)> = vec![
        // ── research-chain identity ────────────────────────────────────────
        ("job_id", mutate(&base, |s| s.job_id[0] ^= 0xff)),
        ("session_id", mutate(&base, |s| s.session_id[1] ^= 0xff)),
        ("unit_id", mutate(&base, |s| s.unit_id[2] ^= 0xff)),
        (
            "unit_kind",
            mutate(&base, |s| s.unit_kind = UnitKind::SelectToken),
        ),
        ("unit_index", mutate(&base, |s| s.unit_index += 1)),
        (
            "generation_index",
            mutate(&base, |s| s.generation_index += 1),
        ),
        // ── model / tokenizer ──────────────────────────────────────────────
        ("model_id", mutate(&base, |s| s.model_id[9] ^= 0xff)),
        (
            "model_commitment",
            mutate(&base, |s| {
                s.model_commitment = oc(ObjectKind::Model, b"other-model-bytes")
            }),
        ),
        ("tokenizer_id", mutate(&base, |s| s.tokenizer_id[0] ^= 0xff)),
        // ── numeric / dimension context ────────────────────────────────────
        ("head_dim", mutate(&base, |s| s.head_dim += 1)),
        ("ffn_dim", mutate(&base, |s| s.ffn_dim += 1)),
        ("layer_start", mutate(&base, |s| s.layer_start += 1)),
        ("layer_end", mutate(&base, |s| s.layer_end += 1)),
        ("vocab_size", mutate(&base, |s| s.vocab_size += 1)),
        ("d_model", mutate(&base, |s| s.d_model += 1)),
        ("n_heads", mutate(&base, |s| s.n_heads += 1)),
        // ── inputs ─────────────────────────────────────────────────────────
        (
            "derived_input_commitment",
            mutate(&base, |s| {
                s.derived_input_commitment = oc(ObjectKind::DerivedInput, b"x")
            }),
        ),
        (
            "prior_residual_stream",
            mutate(&base, |s| {
                s.prior_residual_stream = oc(ObjectKind::PriorResidual, b"x")
            }),
        ),
        (
            "prior_kv_cache",
            mutate(&base, |s| s.prior_kv_cache = oc(ObjectKind::PriorKv, b"x")),
        ),
        (
            "token_prefix",
            mutate(&base, |s| {
                s.token_prefix = oc(ObjectKind::TokenPrefix, b"x")
            }),
        ),
        (
            "input_manifest",
            mutate(&base, |s| {
                s.input_manifest = oc(ObjectKind::InputManifest, b"x")
            }),
        ),
        ("sequence_length", mutate(&base, |s| s.sequence_length += 1)),
        ("position", mutate(&base, |s| s.position += 1)),
        // ── outputs ────────────────────────────────────────────────────────
        (
            "output_manifest",
            mutate(&base, |s| {
                s.output_manifest = oc(ObjectKind::OutputManifest, b"x")
            }),
        ),
        ("selected_token", mutate(&base, |s| s.selected_token = 6)),
        (
            "updated_token_seq_commitment",
            mutate(&base, |s| {
                s.updated_token_seq_commitment = oc(ObjectKind::TokenSeq, b"other-seq")
            }),
        ),
        ("eos_flag", mutate(&base, |s| s.eos_flag = 0)),
        // ── bounds ─────────────────────────────────────────────────────────
        ("max_cycles", mutate(&base, |s| s.max_cycles += 1)),
        ("max_d_model", mutate(&base, |s| s.max_d_model += 1)),
        ("max_seq_len", mutate(&base, |s| s.max_seq_len += 1)),
        (
            "max_output_tokens",
            mutate(&base, |s| s.max_output_tokens += 1),
        ),
        (
            "max_manifest_slots",
            mutate(&base, |s| s.max_manifest_slots += 1),
        ),
        ("max_state_bytes", mutate(&base, |s| s.max_state_bytes += 1)),
    ];

    assert!(
        cases.len() >= 30,
        "expected a broad matrix, got {}",
        cases.len()
    );

    for (label, mutated) in cases {
        assert_bound(label, &golden_j, mutated);
    }
}

#[test]
fn spec_hash_is_structurally_zero_in_a_template() {
    // The statement is only ever a zero-spec-hash TEMPLATE: mutating
    // `b0_pre_spec_hash` cannot change the journal, because the template route
    // zeroes those bytes. This proves the template discipline (never a final
    // statement) rather than a binding of the spec hash.
    let (base, golden_j) = golden();
    let mutated = mutate(&base, |s| s.b0_pre_spec_hash = [0xAB; 32]);
    let mutated_j = journal_of(&mutated);
    assert_eq!(
        mutated_j.computation_statement_hash(),
        golden_j.computation_statement_hash(),
        "the template zeroes the spec hash, so mutating it must not change the journal"
    );
    assert_eq!(mutated_j.spec_hash(), [0u8; 32]);
}

#[test]
fn identity_statement_verifies() {
    // Sanity: an unmutated golden journal DOES verify (the matrix is meaningful
    // only if the honest path passes).
    let (_s, golden_j) = golden();
    assert!(honest_verifies(&golden_j));
}
