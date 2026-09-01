//! Tripwire binding for the committed R0 closure record (OmniNode #101).
//!
//! Binds `docs/r0/r0-closure.v1.json` at compile time and asserts it carries the ratified
//! B0-FINAL constants and the corrected (two-cell) eligibility matrix. This does NOT run
//! proofs or change any behavior — it fails loudly if the committed record drifts from the
//! official seal, guest set, spec hash, or golden-fixture digest, or if it ever asserts an
//! ARM performance measurement or a production claim.

use serde_json::Value;

const RECORD: &str = include_str!("../../../docs/r0/r0-closure.v1.json");

fn record() -> Value {
    serde_json::from_str(RECORD).expect("r0-closure.v1.json must be valid JSON")
}

#[test]
fn record_binds_the_ratified_authority() {
    let r = record();
    let a = &r["authority"];
    assert_eq!(r["kind"], "r0-closure-record/v1");
    assert_eq!(
        a["official_seal_blake3"],
        "60ace32cc2775fd38c3a4b9ea81f49686121cdd25a38db7a5ca5a0f4580bd600"
    );
    assert_eq!(
        a["b0_pre_spec_hash"],
        "e933e7325c2639a48d8e25f20746d0f8abc822dee9fcfa87c2e6cdec226cf2a2"
    );
    assert_eq!(
        a["r0_guest_set_hash"],
        "11d059c7dbc37b3d80f0a0c1fcaee96ad5e0ba1916ba08bdb0f37e1a7d76401a"
    );
    assert_eq!(
        a["package_id_blake3"],
        "80ab5ecfbe7a24d96d02dad78db2e4aee712ea0b29a071c19893b0d83dd0f11b"
    );
    assert_eq!(
        a["measurement_head_sum_chain"],
        "9cccaa5ee6e038fb9dcb45af44ecb3cbdc2f48c6"
    );
    // durable release tag points at the measurement head
    assert_eq!(
        r["durable_evidence"]["release_tag_commit"],
        a["measurement_head_sum_chain"]
    );
}

#[test]
fn eligibility_matrix_is_the_corrected_two_cell_model() {
    let r = record();
    let e = &r["eligibility_matrix"];
    let cells: Vec<&str> = e["measurement_cells_x86_64"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(cells, ["Sp1", "Risc0"]);
    assert_eq!(e["sp1_aarch64"], "identity_only_reconciliation");
    assert_eq!(e["risc0_aarch64"], "unsupported_refused");
    // ARM performance was NEVER measured; identity reconciliation WAS required + satisfied.
    assert_eq!(e["arm_performance_measured"], Value::Bool(false));
    assert_eq!(e["arm_identity_reconciliation_required"], Value::Bool(true));
    assert_eq!(
        e["arm_identity_reconciliation_satisfied"],
        Value::Bool(true)
    );
}

#[test]
fn golden_roots_use_ratified_types_and_committed_digest() {
    let r = record();
    let g = &r["acceptance_confirmation"]["golden_roots"];
    let types: Vec<&str> = g["ratified_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(
        types,
        ["ObjectCommitmentV1", "OutputManifestV1", "InputManifestV1"]
    );
    assert_eq!(
        g["encoding_golden_vectors_sha256"],
        "26a6338e3572384adfc4e0aa379f4501cb1c350a4195ce85f8056b2f378875c1"
    );
    assert_eq!(g["present"], Value::Bool(true));
    // the obsolete names are recorded only as "corrected_from", never as ratified types.
    let from: Vec<&str> = g["types_corrected_from"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(from, ["StateObjectV1", "UnitOutputManifestV1"]);
    assert!(!types.contains(&"StateObjectV1"));
    assert!(!types.contains(&"UnitOutputManifestV1"));
}

#[test]
fn acceptance_confirmed_and_non_production() {
    let r = record();
    assert_eq!(r["non_production"], Value::Bool(true));
    assert_eq!(r["no_production_or_integration_claim"], Value::Bool(true));
    let ac = &r["acceptance_confirmation"];
    for k in [
        "both_eligible_x86_cells_completed",
        "bounded_transformer_layer_group_unit_proven_and_verified",
        "select_token_unit_proven_and_verified",
        "malformed_and_wrong_statement_proofs_deterministically_refused",
        "raw_metrics_and_reproducibility_retrievable_and_hash_bound",
    ] {
        assert_eq!(ac[k], Value::Bool(true), "acceptance flag {k} must be true");
    }
}
