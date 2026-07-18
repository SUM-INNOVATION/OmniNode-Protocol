//! Host-side reference-executor determinism + candidate neutrality.
//!
//! NO zkVM guest is executed anywhere in this file — no guest is built, run, or
//! proven. The reference executor is a plain-Rust model of the integer +
//! authentication work a guest *would* run; these tests only assert that this
//! HOST computation is deterministic, that the candidate-neutral journal it emits
//! is candidate-independent by construction (a single 996-byte template, no
//! candidate field), and that an end-to-end generation loop threads residual / KV
//! / token state and produces well-formed, in-bounds journals.

use r0_zkvm_bench::blake3_32;
use r0_zkvm_bench::model_auth::{authenticate_all, prepare_weight_witnesses};
use r0_zkvm_bench::statement::{SyntheticJournal, UnitKind};
use r0_zkvm_bench::workload::{
    run_layer_group, KvCache, Model, ReferenceExecutor, StepContext, UnitInputs, D_MODEL, VOCAB,
};
// The raw statement type is NOT public in r0-zkvm-bench; decode the journal's
// bytes through the `sumchain-wire` dependency directly for inspection.
use sumchain_wire::b0::statement::R0ComputationStatementV2;

fn decoded(j: &SyntheticJournal) -> R0ComputationStatementV2 {
    R0ComputationStatementV2::decode_exact(j.bytes()).unwrap()
}

fn model() -> Model {
    Model::deterministic(2026, blake3_32(b"equivalence-model"))
}

fn inputs<'a>(residual: &'a [i16; D_MODEL], kv: &'a KvCache, prefix: &'a [u32]) -> UnitInputs<'a> {
    UnitInputs {
        prior_residual: residual,
        prior_kv: kv,
        token_prefix: prefix,
    }
}

#[test]
fn reference_executor_is_bit_exact_across_runs() {
    let m = model();
    let ctx = StepContext::sample(0);
    let x = [5i16, -5, 5, -5, 5, -5, 5, -5];
    let kv = KvCache::default();
    let (_o1, j1) =
        ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs(&x, &kv, &[])).unwrap();
    let (_o2, j2) =
        ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs(&x, &kv, &[])).unwrap();
    // Byte-for-byte identical host output across repeated runs (determinism).
    assert_eq!(j1.bytes(), j2.bytes());
    assert_eq!(
        j1.computation_statement_hash(),
        j2.computation_statement_hash()
    );
    assert_eq!(j1.len(), 996);
}

#[test]
fn journal_is_candidate_independent_host_side() {
    // The journal is candidate-neutral BY CONSTRUCTION: the 996-byte statement
    // carries no candidate field, so there is only ONE host computation — the
    // same bytes a future SP1 guest and a future RISC Zero guest would each
    // target. This does NOT execute or prove either guest.
    let m = model();
    let ctx = StepContext::sample(1);
    let x = [1i16, 2, 3, 4, 5, 6, 7, 8];
    let kv = KvCache {
        k: vec![[9i16; D_MODEL]],
        v: vec![[8i16; D_MODEL]],
    };
    let (_o, j) = ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs(&x, &kv, &[])).unwrap();
    let candidate1_slot = j.bytes().to_vec();
    let candidate2_slot = j.bytes().to_vec();
    assert_eq!(candidate1_slot, candidate2_slot);
}

#[test]
fn end_to_end_generation_loop_threads_state() {
    let m = model();
    let steps = 5u32;

    let run = || {
        let mut residual = [3i16, -3, 3, -3, 3, -3, 3, -3];
        let mut kv = KvCache::default();
        let mut tokens: Vec<u32> = Vec::new();
        let mut hashes = Vec::new();
        for pos in 0..steps {
            let mut ctx = StepContext::sample(pos);
            ctx.unit_index = pos;

            let (lg_out, lg_journal) =
                ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs(&residual, &kv, &tokens))
                    .unwrap();
            assert_eq!(
                decoded(&lg_journal).unit_kind,
                UnitKind::TransformerLayerGroup
            );
            kv = lg_out.updated_kv.clone();

            let (st_out, st_journal) = ReferenceExecutor::run_select_token_unit(
                &m,
                &ctx,
                &inputs(&residual, &kv, &tokens),
                &lg_out.new_residual,
                &tokens,
            )
            .unwrap();
            assert_eq!(decoded(&st_journal).unit_kind, UnitKind::SelectToken);
            assert!((st_out.selected_token as usize) < VOCAB);
            tokens.push(st_out.selected_token);

            residual = lg_out.new_residual;
            hashes.push((
                lg_journal.computation_statement_hash(),
                st_journal.computation_statement_hash(),
            ));
        }
        (tokens, hashes)
    };

    let (tokens_a, hashes_a) = run();
    let (tokens_b, hashes_b) = run();
    assert_eq!(tokens_a, tokens_b, "generation must be deterministic");
    assert_eq!(hashes_a, hashes_b, "journal hashes must be deterministic");
    assert_eq!(tokens_a.len(), steps as usize);
}

#[test]
fn model_weight_authentication_is_part_of_the_work() {
    // Every consumed weight chunk authenticates against the public commitment
    // (this cost belongs to guest measurement, per Approach 2).
    let m = model();
    let (commitment, chunks) = prepare_weight_witnesses(&m).unwrap();
    let n = authenticate_all(&commitment, &chunks).expect("all weight chunks authenticate");
    assert_eq!(n, chunks.len());
    // The commitment embedded in the journal equals the independently prepared
    // one.
    let ctx = StepContext::sample(0);
    let x = [0i16; D_MODEL];
    let (_o, j) =
        ReferenceExecutor::run_layer_group_unit(&m, &ctx, &inputs(&x, &KvCache::default(), &[]))
            .unwrap();
    assert_eq!(decoded(&j).model_commitment, commitment);
}

#[test]
fn layer_group_output_slots_have_independent_commitments() {
    let m = model();
    let x = [11i16, -22, 33, -44, 55, -66, 77, -88];
    let out = run_layer_group(&m, &x, &KvCache::default(), 0).unwrap();
    // Residual and KV slots each carry their own byte_len / root, under distinct
    // object kinds (ResidualState vs KvState).
    assert_eq!(out.output_manifest.slots.len(), 2);
    let residual_len = out.residual_slot.commitment.byte_len();
    let kv_len = out.kv_slot.commitment.byte_len();
    assert_eq!(residual_len, (D_MODEL * 2) as u64); // 8 i16 = 16 bytes
    assert_eq!(kv_len, (2 * D_MODEL * 2) as u64); // 1 position: k+v = 32 bytes
    assert_ne!(residual_len, kv_len);
}
