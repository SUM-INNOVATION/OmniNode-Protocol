//! Phase 5 Stage 13.2 — `SumChainClient`'s
//! `EvidenceAnchorChainClient` adapter integration tests.
//!
//! All hermetic via [`FakeJsonRpcTransport`]; **no live chain.**
//! Covers the adapter activation gate, the same-key submitter
//! gate, the submit RPC happy/refusal paths, and the status
//! query mapping (lowercase status strings; `included_at_height`
//! parsed-but-dropped per Q5; foreign tx_hash → `Unknown`).

#![cfg(feature = "submit")]

use omni_sumchain::{
    error_prefixes, FakeJsonRpcTransport, JsonRpcTransport, SumChainClient,
};
use omni_zkml::{
    anchor_signer_pubkey_bytes, sign_anchor_digest, AnchorStatus, AnchoredArtifactKind,
    ChainClientError, EvidenceAnchorChainClient, IntegrityEvidenceAnchorDigest,
    IntegrityEvidenceAnchorTxData, INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
};

// ── Fixtures ──────────────────────────────────────────────────────────────────

const SUBMIT_METHOD: &str = "sum_submitIntegrityEvidenceAnchor";
const STATUS_METHOD: &str = "sum_getIntegrityEvidenceAnchorStatus";
const PARAMS_METHOD: &str = "chain_getChainParams";
const HEIGHT_METHOD: &str = "chain_getBlockHeight";

fn fresh_client(seed: [u8; 32]) -> (FakeJsonRpcTransport, SumChainClient<FakeJsonRpcTransport>) {
    let transport = FakeJsonRpcTransport::new();
    let client = SumChainClient::with_transport(seed, transport.clone());
    (transport, client)
}

fn build_signed_tx(seed: [u8; 32]) -> IntegrityEvidenceAnchorTxData {
    let pubkey = anchor_signer_pubkey_bytes(&seed).unwrap();
    let digest = IntegrityEvidenceAnchorDigest {
        anchor_schema_version: INTEGRITY_EVIDENCE_ANCHOR_SCHEMA_VERSION,
        artifact_kind: AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
        artifact_schema_version: 1,
        artifact_hash: [0x11; 32],
        signer_pubkey: pubkey,
        signed_at_utc_unix: 1_750_000_000,
    };
    let signature = sign_anchor_digest(&seed, &digest).unwrap();
    IntegrityEvidenceAnchorTxData {
        digest,
        submitter_signature: signature,
    }
}

fn seed_active_chain_params(t: &FakeJsonRpcTransport) {
    // Anchor activation = Some(0); chain_id = 42 (non-mainnet).
    t.set_response(
        PARAMS_METHOD,
        Ok(serde_json::json!({
            "finality_depth": 1u64,
            "min_fee": 0u64,
            "chain_id": 42u64,
            "omninode_enabled_from_height": 0u64,
            "v2_enabled_from_height": 0u64,
            "integrity_evidence_anchor_enabled_from_height": 0u64,
        })),
    );
    t.set_response(
        HEIGHT_METHOD,
        Ok(serde_json::json!({ "height": 100u64, "finality": "latest" })),
    );
}

fn seed_dormant_chain_params(t: &FakeJsonRpcTransport) {
    // integrity_evidence_anchor_enabled_from_height = None.
    t.set_response(
        PARAMS_METHOD,
        Ok(serde_json::json!({
            "finality_depth": 1u64,
            "min_fee": 0u64,
            "chain_id": 42u64,
            "omninode_enabled_from_height": 0u64,
            "v2_enabled_from_height": 0u64,
            // integrity_evidence_anchor_enabled_from_height absent → None
        })),
    );
}

fn seed_scheduled_not_yet_reached_chain_params(t: &FakeJsonRpcTransport) {
    t.set_response(
        PARAMS_METHOD,
        Ok(serde_json::json!({
            "finality_depth": 1u64,
            "min_fee": 0u64,
            "chain_id": 42u64,
            "omninode_enabled_from_height": 0u64,
            "v2_enabled_from_height": 0u64,
            "integrity_evidence_anchor_enabled_from_height": 1_000_000u64,
        })),
    );
    t.set_response(
        HEIGHT_METHOD,
        Ok(serde_json::json!({ "height": 100u64, "finality": "latest" })),
    );
}

// ── Adapter activation gate ───────────────────────────────────────────────────

#[test]
fn anchor_submit_adapter_refuses_when_chain_dormant() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    seed_dormant_chain_params(&t);
    let tx_data = build_signed_tx(seed);
    let err = client.submit_anchor(&tx_data).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.starts_with(error_prefixes::ADAPTER_NOT_ACTIVATED),
        "expected ADAPTER_NOT_ACTIVATED prefix, got: {msg:?}"
    );
    // submit RPC must NOT be reached.
    let submitted = t
        .calls()
        .iter()
        .any(|(method, _)| method == SUBMIT_METHOD);
    assert!(!submitted, "submit RPC unexpectedly invoked");
}

#[test]
fn anchor_submit_adapter_refuses_when_scheduled_not_yet_reached() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    seed_scheduled_not_yet_reached_chain_params(&t);
    let tx_data = build_signed_tx(seed);
    let err = client.submit_anchor(&tx_data).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(msg.starts_with(error_prefixes::ADAPTER_NOT_ACTIVATED));
    let submitted = t
        .calls()
        .iter()
        .any(|(method, _)| method == SUBMIT_METHOD);
    assert!(!submitted, "submit RPC unexpectedly invoked");
}

// ── Adapter same-key submitter gate ──────────────────────────────────────────

#[test]
fn anchor_submit_adapter_refuses_when_seed_derives_different_pubkey() {
    let wrapper_seed = [7u8; 32];
    let configured_seed = [9u8; 32]; // mismatched — client built with this
    let (t, client) = fresh_client(configured_seed);
    seed_active_chain_params(&t);
    // tx_data declares the wrapper_seed's pubkey, NOT the
    // configured client seed's.
    let tx_data = build_signed_tx(wrapper_seed);
    let err = client.submit_anchor(&tx_data).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.starts_with(error_prefixes::ADAPTER_SAME_KEY_FAIL),
        "expected ADAPTER_SAME_KEY_FAIL prefix, got: {msg:?}"
    );
    let submitted = t
        .calls()
        .iter()
        .any(|(method, _)| method == SUBMIT_METHOD);
    assert!(!submitted, "submit RPC unexpectedly invoked");
}

// ── Submit happy path ────────────────────────────────────────────────────────

#[test]
fn anchor_submit_happy_path_posts_148_byte_payload_and_returns_tx_hash() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    seed_active_chain_params(&t);
    t.set_response(
        SUBMIT_METHOD,
        Ok(serde_json::json!({ "tx_hash": "0xdeadbeef" })),
    );
    let tx_data = build_signed_tx(seed);
    let receipt = client.submit_anchor(&tx_data).unwrap();
    assert_eq!(receipt.tx_id, "0xdeadbeef");
    assert_eq!(receipt.note, None);

    // Find the submit call and assert the param shape:
    // single positional 0x-prefixed hex string, 148 bytes
    // (= 2 + 148*2 = 298 hex chars including 0x).
    let submit_call = t
        .calls()
        .into_iter()
        .find(|(method, _)| method == SUBMIT_METHOD)
        .expect("submit RPC should have been called");
    let serde_json::Value::Array(params) = &submit_call.1 else {
        panic!("expected array params, got {:?}", submit_call.1);
    };
    assert_eq!(params.len(), 1, "expected single positional param");
    let serde_json::Value::String(hex) = &params[0] else {
        panic!("expected string param, got {:?}", params[0]);
    };
    assert!(hex.starts_with("0x"), "expected 0x-prefixed hex, got {hex:?}");
    assert_eq!(hex.len(), 2 + 148 * 2, "expected 148-byte payload");
}

#[test]
fn anchor_submit_response_lenient_parser_accepts_bare_string() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    seed_active_chain_params(&t);
    t.set_response(SUBMIT_METHOD, Ok(serde_json::json!("0xfeedf00d")));
    let receipt = client.submit_anchor(&build_signed_tx(seed)).unwrap();
    assert_eq!(receipt.tx_id, "0xfeedf00d");
}

#[test]
fn anchor_submit_response_refuses_malformed_shape() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    seed_active_chain_params(&t);
    // Neither object-with-tx_hash nor bare string.
    t.set_response(SUBMIT_METHOD, Ok(serde_json::json!({ "result": "0xfoo" })));
    let err = client.submit_anchor(&build_signed_tx(seed)).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.starts_with(error_prefixes::ADAPTER_MALFORMED_SUBMIT_RESP),
        "expected ADAPTER_MALFORMED_SUBMIT_RESP prefix, got: {msg:?}"
    );
}

#[test]
fn anchor_submit_propagates_chain_jsonrpc_error_verbatim() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    seed_active_chain_params(&t);
    t.set_response(
        SUBMIT_METHOD,
        Err(ChainClientError::Other(format!(
            "{prefix}{{\"code\":-32000,\"message\":\"insufficient_fee\"}}",
            prefix = error_prefixes::JSONRPC_ERROR,
        ))),
    );
    let err = client.submit_anchor(&build_signed_tx(seed)).unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.starts_with(error_prefixes::JSONRPC_ERROR),
        "expected JSONRPC_ERROR prefix, got: {msg:?}"
    );
    assert!(msg.contains("insufficient_fee"));
}

// ── Status query mapping ─────────────────────────────────────────────────────

fn status_response(status: &str) -> serde_json::Value {
    serde_json::json!({ "status": status })
}

#[test]
fn anchor_query_status_decodes_submitted() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(STATUS_METHOD, Ok(status_response("submitted")));
    assert!(matches!(
        client.query_anchor_status("0xabc").unwrap(),
        AnchorStatus::Submitted
    ));
}

#[test]
fn anchor_query_status_decodes_included() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(STATUS_METHOD, Ok(status_response("included")));
    assert!(matches!(
        client.query_anchor_status("0xabc").unwrap(),
        AnchorStatus::Included
    ));
}

#[test]
fn anchor_query_status_decodes_finalized() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(STATUS_METHOD, Ok(status_response("finalized")));
    assert!(matches!(
        client.query_anchor_status("0xabc").unwrap(),
        AnchorStatus::Finalized
    ));
}

#[test]
fn anchor_query_status_decodes_failed_with_reason() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(
        STATUS_METHOD,
        Ok(serde_json::json!({
            "status": "failed",
            "reason": "fee below min",
        })),
    );
    let status = client.query_anchor_status("0xabc").unwrap();
    if let AnchorStatus::Failed { reason } = status {
        assert_eq!(reason, "fee below min");
    } else {
        panic!("expected AnchorStatus::Failed, got {status:?}");
    }
}

#[test]
fn anchor_query_status_decodes_failed_with_default_reason_when_missing() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(STATUS_METHOD, Ok(status_response("failed")));
    let status = client.query_anchor_status("0xabc").unwrap();
    if let AnchorStatus::Failed { reason } = status {
        assert_eq!(reason, "no reason provided");
    } else {
        panic!("expected AnchorStatus::Failed, got {status:?}");
    }
}

#[test]
fn anchor_query_foreign_tx_hash_returns_unknown_per_chain_contract() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(STATUS_METHOD, Ok(status_response("unknown")));
    assert!(matches!(
        client.query_anchor_status("0xforeign").unwrap(),
        AnchorStatus::Unknown
    ));
}

#[test]
fn anchor_query_status_refuses_unrecognized_status_string() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(STATUS_METHOD, Ok(status_response("rogue_status")));
    let err = client.query_anchor_status("0xabc").unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.starts_with(error_prefixes::ADAPTER_UNRECOGNIZED_STATUS),
        "expected ADAPTER_UNRECOGNIZED_STATUS prefix, got: {msg:?}"
    );
}

#[test]
fn anchor_query_status_refuses_missing_status_field() {
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    // Object without `status` field.
    t.set_response(STATUS_METHOD, Ok(serde_json::json!({ "foo": "bar" })));
    let err = client.query_anchor_status("0xabc").unwrap_err();
    let ChainClientError::Other(msg) = err;
    assert!(
        msg.starts_with(error_prefixes::ADAPTER_MALFORMED_STATUS_RESP),
        "expected ADAPTER_MALFORMED_STATUS_RESP prefix, got: {msg:?}"
    );
}

#[test]
fn anchor_query_status_parses_optional_fields_default_none() {
    // Only `status` present — `included_at_height` and `reason`
    // both default to None. Successful query.
    let seed = [7u8; 32];
    let (t, client) = fresh_client(seed);
    t.set_response(STATUS_METHOD, Ok(status_response("submitted")));
    // No panic, no error — confirms the optional default-None
    // serde attributes work.
    let _ = client.query_anchor_status("0xabc").unwrap();
}
