//! Stage 9c.1 — chain-produced signed-transaction fixture.
//!
//! Asserts byte-stable roundtrip of an actual on-chain `SignedTransaction`
//! supplied by the SUM Chain team from the 2026-05-19 mainnet smoke
//! (tx `0x3a9cbf85…c32a56`). Provenance, hash methodology, and the
//! chain-id-source disclosure are preserved verbatim inside the
//! fixture JSON's `provenance` / `notes` blocks.
//!
//! Unlike `tests/parity_vendored_primitives.rs` — which builds a
//! synthetic `TransactionV2` in-test and checks local-vs-chain byte
//! equivalence — this test consumes raw bytes the chain wrote to its
//! own TRANSACTIONS column family. Together they form a two-sided
//! gate: synthetic parity proves equivalence under deterministic
//! inputs; this fixture proves real on-chain bytes round-trip.
//!
//! NOTE: the decoded payload's digest fields (`model_hash`,
//! `manifest_root`, `response_hash`, `proof_root`) are smoke-test
//! placeholder bytes (0xaa / 0x11 / 0xbb / 0x22), not real
//! model/proof data — see the `digest_payload` note inside the
//! fixture. This test asserts structure and byte-roundtrip, not
//! digest semantics.
#![cfg(feature = "submit")]

use serde::Deserialize;
use sumchain_primitives::{SignedTransaction, TxInner, TxPayload};

#[derive(Deserialize)]
struct Fixture {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    source: String,
    tx_hash: String,
    #[allow(dead_code)]
    tx_hash_recomputed: String,
    #[allow(dead_code)]
    hash_match: bool,
    signed_tx_hex: String,
    chain_id: u64,
    session_id: String,
    verifier_address: String,
    #[allow(dead_code)]
    attestation_id: String,
    #[allow(dead_code)]
    included_at_block_height: u64,
    #[allow(dead_code)]
    tx_index_in_block: u64,
    #[allow(dead_code)]
    status: String,
    fee_paid: u128,
    #[allow(dead_code)]
    provenance: serde_json::Value,
    #[allow(dead_code)]
    notes: serde_json::Value,
}

const FIXTURE_JSON: &str =
    include_str!("fixtures/chain_produced_signed_tx.json");

fn load_fixture() -> Fixture {
    serde_json::from_str(FIXTURE_JSON)
        .expect("fixtures/chain_produced_signed_tx.json must parse as Fixture")
}

// ── shape: fixture string is bare lowercase ASCII hex ───────────────────────

#[test]
fn fixture_signed_tx_hex_is_bare_lowercase_ascii_hex() {
    let fx = load_fixture();

    // Bare hex (no `0x` / `0X` prefix).
    assert!(
        !fx.signed_tx_hex.starts_with("0x") && !fx.signed_tx_hex.starts_with("0X"),
        "fixture.signed_tx_hex must be stored as BARE hex (no 0x prefix); got prefix {:?}",
        &fx.signed_tx_hex.get(..2).unwrap_or("")
    );

    // Lowercase ASCII hex digits — pins lowercase for THIS fixture only.
    // Not a protocol-level guarantee about SignedTransaction::to_hex(); the
    // roundtrip assertion below is what actually verifies the crate's
    // emission against this specific byte sequence.
    for (i, c) in fx.signed_tx_hex.chars().enumerate() {
        assert!(
            c.is_ascii_hexdigit() && !c.is_ascii_uppercase(),
            "fixture.signed_tx_hex[{i}] = {c:?} is not a lowercase ASCII hex digit"
        );
    }
}

// ── roundtrip: bare → from_hex → to_hex → bare, byte-identical ──────────────

#[test]
fn fixture_bare_hex_roundtrips_byte_identical() {
    let fx = load_fixture();

    let signed = SignedTransaction::from_hex(&fx.signed_tx_hex)
        .expect("SignedTransaction::from_hex(bare) must succeed against the chain-produced fixture");

    let reemitted = signed.to_hex();
    assert_eq!(
        reemitted, fx.signed_tx_hex,
        "SignedTransaction::from_hex(bare).to_hex() must equal fixture.signed_tx_hex byte-identically"
    );
}

// ── roundtrip: 0x-prefixed input is accepted and re-emits BARE hex ──────────

#[test]
fn fixture_0x_prefixed_hex_is_accepted_and_reemits_bare() {
    let fx = load_fixture();
    let prefixed = format!("0x{}", fx.signed_tx_hex);

    let signed = SignedTransaction::from_hex(&prefixed).expect(
        "SignedTransaction::from_hex(0x-prefixed) must succeed (the crate strips the prefix internally)",
    );

    let reemitted = signed.to_hex();
    assert_eq!(
        reemitted, fx.signed_tx_hex,
        "0x-prefixed input must re-emit identically as the bare hex stored in the fixture"
    );
}

// ── decode: shape of the parsed transaction ─────────────────────────────────

#[test]
fn fixture_decodes_to_v2_inference_attestation_with_expected_fields() {
    let fx = load_fixture();
    let signed = SignedTransaction::from_hex(&fx.signed_tx_hex)
        .expect("fixture must decode");

    let TxInner::V2(v2) = &signed.inner else {
        panic!(
            "expected TxInner::V2 from the chain-produced fixture; got a non-V2 variant"
        );
    };

    // chain_id: 1 (matches both fixture and the in-body little-endian u64
    // disclosed in the fixture `notes.chain_id_source`).
    assert_eq!(v2.chain_id, fx.chain_id);
    assert_eq!(v2.chain_id, 1);

    // fee: 1000 (matches the fixture `fee_paid` and the chain team's
    // fee_paid observation).
    assert_eq!(v2.fee, fx.fee_paid);
    assert_eq!(v2.fee, 1000u128);

    // from: the verifier address recorded in the fixture.
    let from_b58 = v2.from.to_base58();
    assert_eq!(from_b58, fx.verifier_address);
    assert_eq!(from_b58, "2mvPk4h883B7DrcZvwy7yWKXyGYHuVzGP");

    // payload: InferenceAttestation with the smoke session_id.
    let TxPayload::InferenceAttestation(tx_data) = &v2.payload else {
        panic!(
            "expected TxPayload::InferenceAttestation; got a different payload arm"
        );
    };
    assert_eq!(tx_data.digest.session_id, fx.session_id);
    assert_eq!(tx_data.digest.session_id, "mainnet-smoke-2026-05-18-001");
}

// ── chain-team hash match: BLAKE3(bincode(signed_tx)) ───────────────────────
//
// `sumchain_primitives::SignedTransaction::hash()` is the public surface
// of the methodology the chain team described in the fixture's
// `notes.hash_methodology` block — it is exactly
// `Hash::hash(bincode::serialize(self))`, which is identical to how
// `TxStore::put` keys the TRANSACTIONS CF row. Asserting equality with
// the fixture's `tx_hash` therefore pins both:
//
//   1. The raw bytes in `signed_tx_hex` deserialise into a
//      `SignedTransaction` whose re-bincode'd form hashes to the
//      on-chain key.
//   2. The chain-team handoff (tx_hash_input == tx_hash_recomputed)
//      remains consistent with what `sumchain-primitives v0.1.0`
//      computes locally — no drift between the chain's internal
//      hashing and the published crate.
#[test]
fn fixture_signed_tx_hash_matches_chain_produced_tx_hash() {
    let fx = load_fixture();
    let signed = SignedTransaction::from_hex(&fx.signed_tx_hex)
        .expect("fixture must decode");

    // `Hash::to_hex()` returns a `0x`-prefixed lowercase hex string;
    // the chain team supplied `tx_hash` in the same `0x`-prefixed form,
    // so we compare directly.
    let computed_hex = signed.hash().to_hex();

    assert_eq!(
        computed_hex, fx.tx_hash,
        "SignedTransaction::hash().to_hex() = {computed_hex} drifted from the chain-produced \
         tx_hash {expected}. Either the public sumchain-primitives crate's hashing diverged \
         from the chain's TRANSACTIONS-CF key derivation, or the fixture bytes are not the \
         exact stored row. Both are load-bearing for Stage 9c.1; do NOT update EXPECTED \
         without notifying the chain team per the publishing agreement.",
        expected = fx.tx_hash
    );
}
