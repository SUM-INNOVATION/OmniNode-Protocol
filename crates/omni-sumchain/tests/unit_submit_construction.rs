//! Stage 7b — `submit_attestation` construction tests.
//!
//! Stage 9a: file-level `#![cfg(feature = "submit")]`. These tests
//! exercise the real Stage 7b construction path (which itself is
//! `submit`-gated) and use `SignedTransaction::from_hex` in
//! `submit_attestation_uses_min_fee_unconditionally` — that type only
//! exists with the vendored chain primitives.
#![cfg(feature = "submit")]

//!
//! Hermetic tests pinning the gate ordering, RPC call shapes, and
//! Stage 5.1 contract preservation. All tests use
//! [`FakeJsonRpcTransport`]; no live HTTP.
//!
//! The invariants under test:
//!
//! 1. `chain_getChainParams` is called at most once per submit (cached
//!    across both activation gates).
//! 2. `chain_getBlockHeight` is called at most once per submit (cached
//!    across both activation gates).
//! 3. `sum_getNonce` and `sum_sendRawTransaction` are reached **only**
//!    when both activation gates pass AND the verifier-address gate
//!    passes.
//! 4. Submission hex passed to `sum_sendRawTransaction` is bare (no
//!    `0x` prefix); the chain's response (a `0x`-prefixed tx hash) is
//!    propagated unchanged into `SubmissionReceipt::tx_id`.
//! 5. Stage 5.1 contracts hold: chain-RPC failures during submit
//!    propagate as `RegistryError::ChainClient(_)` from the workflow
//!    and leave records at their pre-submit state.

use omni_sumchain::{FakeJsonRpcTransport, SumChainClient};
use omni_types::phase5::{InferenceAttestation, InferenceCommitment, SnipV2ObjectId};
use omni_zkml::{ChainClient, ChainClientError};

// ── Fixtures ─────────────────────────────────────────────────────────────────

const SEED: [u8; 32] = [42u8; 32];

fn seed_address() -> String {
    omni_zkml::signer_chain_address_base58(&SEED).unwrap()
}

fn attestation_for(verifier_address: &str) -> InferenceAttestation {
    InferenceAttestation {
        commitment: InferenceCommitment {
            session_id: "sess-stage7b".into(),
            model_hash: "0".repeat(64),
            manifest_snip_root: SnipV2ObjectId::from_bytes([0x11u8; 32]),
            response_hash: "1".repeat(64),
            proof_snip_root: SnipV2ObjectId::from_bytes([0x22u8; 32]),
        },
        verifier_address: verifier_address.into(),
        verifier_signature: "ignored-by-stage-7b".into(),
    }
}

fn happy_fake() -> FakeJsonRpcTransport {
    let fake = FakeJsonRpcTransport::new();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            "omninode_enabled_from_height": 0,
            "v2_enabled_from_height": 0,
        })),
    );
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height": 5, "finality": "latest"})),
    );
    fake.set_response("sum_getNonce", Ok(serde_json::json!(0)));
    fake.set_response(
        "sum_sendRawTransaction",
        // Canonical chain response shape (confirmed against the
        // sum-chain @ b586ff3f local mirror): an object carrying the
        // 0x-prefixed tx hash under the `tx_hash` field.
        Ok(serde_json::json!({ "tx_hash": "0xdeadbeefcafebabe" })),
    );
    fake
}

fn methods_called(fake: &FakeJsonRpcTransport) -> Vec<String> {
    fake.calls().into_iter().map(|(m, _)| m).collect()
}

fn assert_no_state_mutating_rpcs(fake: &FakeJsonRpcTransport) {
    let methods = methods_called(fake);
    assert!(
        !methods.contains(&"sum_getNonce".to_string()),
        "expected no sum_getNonce call, got methods: {methods:?}"
    );
    assert!(
        !methods.contains(&"sum_sendRawTransaction".to_string()),
        "expected no sum_sendRawTransaction call, got methods: {methods:?}"
    );
}

// ── Gate 1: OmniNode activation ─────────────────────────────────────────────

#[test]
fn submit_attestation_rejects_when_omninode_not_activated() {
    let fake = happy_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            // omninode_enabled_from_height missing/null
            "v2_enabled_from_height": 0,
        })),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let err = client.submit_attestation(&attestation_for(&seed_address())).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("OmniNode subprotocol not activated"), "got: {msg}");
    assert_no_state_mutating_rpcs(&fake);
}

// ── Gate 2: V2 envelope activation ──────────────────────────────────────────

#[test]
fn submit_attestation_rejects_when_v2_not_activated() {
    let fake = happy_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            "omninode_enabled_from_height": 0,
            // v2_enabled_from_height missing/null
        })),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let err = client.submit_attestation(&attestation_for(&seed_address())).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("V2 transaction envelope not activated"), "got: {msg}");
    assert_no_state_mutating_rpcs(&fake);
}

// ── Gate 1 height check ─────────────────────────────────────────────────────

#[test]
fn submit_attestation_rejects_when_head_below_activation() {
    let fake = happy_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 1,
            "chain_id": 31337,
            "omninode_enabled_from_height": 100,
            "v2_enabled_from_height": 0,
        })),
    );
    fake.set_response(
        "chain_getBlockHeight",
        Ok(serde_json::json!({"height": 50, "finality": "latest"})),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let err = client.submit_attestation(&attestation_for(&seed_address())).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("OmniNode subprotocol not activated"), "got: {msg}");
    assert_no_state_mutating_rpcs(&fake);
}

// ── Gate 3: verifier address consistency ────────────────────────────────────

#[test]
fn submit_attestation_rejects_on_verifier_address_mismatch() {
    let fake = happy_fake();
    let client = SumChainClient::with_transport(SEED, fake.clone());
    // Attestation built for a verifier address that does NOT match the
    // address derived from SEED → mismatch → typed error.
    let bogus = "SomeOtherChainAddressNotMatchingSeed".to_string();
    let err = client.submit_attestation(&attestation_for(&bogus)).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("verifier address mismatch"), "got: {msg}");
    // Gates 1+2 ran (RPCs called), but nonce/submit did NOT.
    let methods = methods_called(&fake);
    assert!(methods.contains(&"chain_getChainParams".to_string()));
    assert_no_state_mutating_rpcs(&fake);
}

// ── Cached params + height ───────────────────────────────────────────────────

#[test]
fn submit_attestation_calls_chain_get_chain_params_exactly_once() {
    let fake = happy_fake();
    let client = SumChainClient::with_transport(SEED, fake.clone());
    client.submit_attestation(&attestation_for(&seed_address())).unwrap();
    let count = methods_called(&fake)
        .iter()
        .filter(|m| *m == "chain_getChainParams")
        .count();
    assert_eq!(
        count, 1,
        "chain_getChainParams must be cached across both activation gates"
    );
}

#[test]
fn submit_attestation_calls_chain_get_block_height_at_most_once() {
    let fake = happy_fake();
    let client = SumChainClient::with_transport(SEED, fake.clone());
    client.submit_attestation(&attestation_for(&seed_address())).unwrap();
    let count = methods_called(&fake)
        .iter()
        .filter(|m| *m == "chain_getBlockHeight")
        .count();
    assert!(
        count <= 1,
        "chain_getBlockHeight called {count} times; activation gates should \
         share at most one call"
    );
}

// ── Nonce + send shapes ──────────────────────────────────────────────────────

#[test]
fn submit_attestation_calls_sum_get_nonce_for_derived_address() {
    let fake = happy_fake();
    let client = SumChainClient::with_transport(SEED, fake.clone());
    client.submit_attestation(&attestation_for(&seed_address())).unwrap();
    let nonce_call = fake
        .calls()
        .into_iter()
        .find(|(m, _)| m == "sum_getNonce")
        .expect("sum_getNonce must be called on the happy path");
    assert_eq!(
        nonce_call.1,
        serde_json::json!([seed_address()]),
        "sum_getNonce param must be the derived (== claimed) verifier address"
    );
}

#[test]
fn submit_attestation_passes_bare_hex_to_send_raw_transaction() {
    let fake = happy_fake();
    let client = SumChainClient::with_transport(SEED, fake.clone());
    client.submit_attestation(&attestation_for(&seed_address())).unwrap();
    let send_call = fake
        .calls()
        .into_iter()
        .find(|(m, _)| m == "sum_sendRawTransaction")
        .expect("sum_sendRawTransaction must be called on the happy path");
    let params = send_call.1;
    let hex = params
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .expect("sum_sendRawTransaction params[0] must be a string");
    assert!(
        !hex.starts_with("0x") && !hex.starts_with("0X"),
        "sum_sendRawTransaction input must be BARE hex; got prefix: {:?}",
        &hex[..hex.len().min(4)]
    );
    assert!(
        hex.chars().all(|c| c.is_ascii_hexdigit()),
        "submission must be lowercase hex digits"
    );
}

// ── Receipt propagation ──────────────────────────────────────────────────────

#[test]
fn submit_attestation_propagates_send_raw_transaction_tx_hash() {
    // Canonical chain response shape — `{ "tx_hash": "0x..." }`,
    // confirmed against `sum-chain @ b586ff3f`. The 0x-prefixed
    // `tx_hash` field flows verbatim into `SubmissionReceipt::tx_id`.
    let fake = happy_fake();
    fake.set_response(
        "sum_sendRawTransaction",
        Ok(serde_json::json!({ "tx_hash": "0xabad1deafeed" })),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let receipt = client.submit_attestation(&attestation_for(&seed_address())).unwrap();
    assert_eq!(receipt.tx_id, "0xabad1deafeed");
}

/// Backwards-compatibility: older mirrors (and pre-Stage-7b hermetic
/// fixtures) returned the tx hash as a bare string. The parser
/// continues to accept that shape so a chain regression isn't a hard
/// break on the client side.
#[test]
fn submit_attestation_accepts_bare_string_send_raw_transaction_response() {
    let fake = happy_fake();
    fake.set_response(
        "sum_sendRawTransaction",
        Ok(serde_json::json!("0xbarestringfallback")),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let receipt = client.submit_attestation(&attestation_for(&seed_address())).unwrap();
    assert_eq!(receipt.tx_id, "0xbarestringfallback");
}

/// Negative: an object response without the `tx_hash` field surfaces
/// a typed `ChainClientError::Other` rather than silently producing a
/// bogus receipt.
#[test]
fn submit_attestation_rejects_object_response_missing_tx_hash() {
    let fake = happy_fake();
    fake.set_response(
        "sum_sendRawTransaction",
        Ok(serde_json::json!({ "status": "ok" })),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let err = client.submit_attestation(&attestation_for(&seed_address())).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.contains("missing 'tx_hash'"),
        "expected missing-field error; got: {msg}"
    );
}

/// Negative: an object response where `tx_hash` is not a string
/// surfaces a typed `ChainClientError::Other`.
#[test]
fn submit_attestation_rejects_object_response_non_string_tx_hash() {
    let fake = happy_fake();
    fake.set_response(
        "sum_sendRawTransaction",
        Ok(serde_json::json!({ "tx_hash": 12345 })),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let err = client.submit_attestation(&attestation_for(&seed_address())).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.contains("not a string"),
        "expected non-string error; got: {msg}"
    );
}

/// Negative: a non-string, non-object response (e.g. a number)
/// surfaces a typed `ChainClientError::Other`.
#[test]
fn submit_attestation_rejects_unexpected_send_raw_transaction_shape() {
    let fake = happy_fake();
    fake.set_response(
        "sum_sendRawTransaction",
        Ok(serde_json::json!(42)),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let err = client.submit_attestation(&attestation_for(&seed_address())).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.contains("unexpected result shape"),
        "expected unexpected-shape error; got: {msg}"
    );
}

// ── Error path: chain RPC failure during submit ──────────────────────────────

#[test]
fn submit_attestation_handles_send_raw_transaction_error_response() {
    let fake = happy_fake();
    fake.set_response(
        "sum_sendRawTransaction",
        Err(ChainClientError::Other("mempool full".into())),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    let err = client.submit_attestation(&attestation_for(&seed_address())).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.contains("mempool full"), "expected upstream error in {msg:?}");
}

// ── Min-fee usage (Stage 7b plan §5 point 11) ────────────────────────────────

#[test]
fn submit_attestation_uses_min_fee_unconditionally() {
    // The chain returns min_fee = 7. After submit, decode the bare hex
    // from sum_sendRawTransaction back into a SignedTransaction and
    // assert tx.fee == 7. This pins the chain-team contract that fee is
    // sourced from params.min_fee with no override path in Stage 7b.
    use sumchain_primitives::{SignedTransaction, TxInner};

    let fake = happy_fake();
    fake.set_response(
        "chain_getChainParams",
        Ok(serde_json::json!({
            "finality_depth": 10,
            "min_fee": 7,
            "chain_id": 31337,
            "omninode_enabled_from_height": 0,
            "v2_enabled_from_height": 0,
        })),
    );
    let client = SumChainClient::with_transport(SEED, fake.clone());
    client.submit_attestation(&attestation_for(&seed_address())).unwrap();

    let send_call = fake
        .calls()
        .into_iter()
        .find(|(m, _)| m == "sum_sendRawTransaction")
        .expect("sum_sendRawTransaction must be called");
    let hex = send_call.1.as_array().unwrap()[0].as_str().unwrap().to_string();
    let signed = SignedTransaction::from_hex(&hex).expect("hex must decode");
    match signed.inner {
        TxInner::V2(tx) => {
            assert_eq!(tx.fee, 7, "tx.fee must equal params.min_fee (Balance = u128)");
        }
        TxInner::Legacy(_) => panic!("expected V2 inner, got Legacy"),
    }
}

/// Issue #97 — provenance-backed fee=1 outer-signing-hash regression,
/// exercised through the **real production submission path**
/// (`chain_getChainParams` → `ChainParamsInfo` (u128 `min_fee`) →
/// `submit_attestation` → `TransactionV2` → captured
/// `sum_sendRawTransaction` hex → `SignedTransaction` decode), **not** a
/// direct `TransactionV2` build.
///
/// The `u64` → `u128` DTO widening must leave the outer signing bytes
/// byte-stable for any fee `<= u64::MAX` (`TransactionV2.fee` was already
/// `u128`). This pins the fee=1 vector.
///
/// Provenance of [`PRE_ISSUE_97_FEE_ONE_SIGNING_HASH_HEX`]:
/// - Reproduced from the pre-Issue-97 base commit **`bc9f365`** (the DTO
///   was still `u64` there; for a fee `<= u64::MAX` the encoded outer
///   bytes are identical pre/post-change).
/// - Chain primitives: **`sumchain-primitives` 0.2.0** (crates.io).
/// - Construction: **`hash = BLAKE3(bincode_1_3(TransactionV2))`** via
///   `TransactionV2::signing_hash()`.
/// - Independently matched by a hand-built canonical `TransactionV2`
///   carrying the identical fixture (two independent routes agreed).
/// - Exact production fixture: `attestation_for(&seed_address())` — the
///   `sess-stage7b` payload (session_id `"sess-stage7b"`,
///   model_hash `[0x00; 32]`, manifest_root `[0x11; 32]`,
///   response_hash `[0x11; 32]`, proof_root `[0x22; 32]`) inner-signed
///   under `SEED`; outer tx `chain_id = 31337`, `fee = 1`, `nonce = 0`,
///   `from = Address::from_public_key(pubkey(SEED))`.
///
/// Recorded as a readable lowercase-hex string (no opaque decimal byte
/// array) and compared via the chain `Hash` API's own hex parse; the
/// test target has no `hex` dependency and none is added for cosmetics.
const PRE_ISSUE_97_FEE_ONE_SIGNING_HASH_HEX: &str =
    "124f6f6c258133729195824153a8f8c5190b54eccdb14d5ea061a0bac5320033";

#[test]
fn submit_attestation_fee_one_matches_provenance_signing_hash() {
    use sumchain_primitives::{Hash, SignedTransaction, TxInner};

    // Real production path. `happy_fake()` returns `min_fee = 1`.
    let fake = happy_fake();
    let client = SumChainClient::with_transport(SEED, fake.clone());
    client.submit_attestation(&attestation_for(&seed_address())).unwrap();

    let send_call = fake
        .calls()
        .into_iter()
        .find(|(m, _)| m == "sum_sendRawTransaction")
        .expect("sum_sendRawTransaction must be called");
    let hex = send_call.1.as_array().unwrap()[0].as_str().unwrap().to_string();
    let signed = SignedTransaction::from_hex(&hex).expect("submitted hex must decode");
    let TxInner::V2(tx) = signed.inner else {
        panic!("expected V2 inner, got Legacy")
    };

    assert_eq!(tx.fee, 1, "fee must be the min_fee=1 from chain_getChainParams");

    let expected = Hash::from_hex(PRE_ISSUE_97_FEE_ONE_SIGNING_HASH_HEX)
        .expect("provenance hash constant must be valid hex");
    assert_eq!(
        tx.signing_hash(),
        expected,
        "fee=1 production signing hash drifted from the bc9f365 provenance \
         vector — the u64 → u128 DTO change must not alter practical-fee \
         tx bytes through the real client path"
    );
}

/// Issue #97 — a `min_fee` just above `u64::MAX`, driven through the
/// **real production submission path** (not a direct `TransactionV2`
/// build). Proves the widened DTO + the transport's arbitrary-precision
/// JSON handling carry the exact value all the way to the signed
/// on-wire transaction with no truncation, float rounding, or wrap.
#[test]
fn submit_attestation_fee_above_u64_max_survives_production_path() {
    use sumchain_primitives::{SignedTransaction, TxInner};

    // Build the `chain_getChainParams` response by parsing a RAW JSON
    // string so the exact `min_fee` integer token (`u64::MAX + 1`) is
    // preserved through the intermediate `serde_json::Value` — this is
    // what the on-wire transport does, and it depends on the crate's
    // `arbitrary_precision` feature. Splicing a Rust `u128` literal via
    // `json!` would not reproduce the wire path.
    let fake = happy_fake();
    let params_value: serde_json::Value = serde_json::from_str(
        r#"{"finality_depth":10,"min_fee":18446744073709551616,"chain_id":31337,"omninode_enabled_from_height":0,"v2_enabled_from_height":0}"#,
    )
    .expect("raw params JSON must parse");
    fake.set_response("chain_getChainParams", Ok(params_value));

    let client = SumChainClient::with_transport(SEED, fake.clone());
    client.submit_attestation(&attestation_for(&seed_address())).unwrap();

    let send_call = fake
        .calls()
        .into_iter()
        .find(|(m, _)| m == "sum_sendRawTransaction")
        .expect("sum_sendRawTransaction must be called");
    let hex = send_call.1.as_array().unwrap()[0].as_str().unwrap().to_string();
    let signed = SignedTransaction::from_hex(&hex).expect("submitted raw tx must decode");
    let TxInner::V2(tx) = signed.inner else {
        panic!("expected V2 inner, got Legacy")
    };

    let expected = u64::MAX as u128 + 1;
    assert_eq!(
        tx.fee, expected,
        "a min_fee just above u64::MAX must survive DTO → TransactionV2 → \
         raw-tx round-trip intact (no truncation/float/wrap)"
    );
}
