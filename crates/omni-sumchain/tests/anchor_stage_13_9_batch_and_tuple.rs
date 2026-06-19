//! Phase 5 Stage 13.9 — wire-shape and behavior pins for the
//! batch-status and by-tuple anchor RPCs. Hermetic; all uses
//! `FakeJsonRpcTransport`.

use omni_sumchain::{
    classify_chain_client_error, ChainErrorCategory, FakeJsonRpcTransport, SumChainClient,
};
use omni_zkml::{
    canonicalize_tx_hash, AnchorStatus, AnchoredArtifactKind, ChainClientError,
    EvidenceAnchorChainClient,
};

fn make_client(tx: FakeJsonRpcTransport) -> SumChainClient<FakeJsonRpcTransport> {
    SumChainClient::with_transport([7u8; 32], tx)
}

// ── Wire-shape pins (Stage 13.9 REJECT-fix Findings 1 + 2) ────────────────────

#[test]
fn batch_params_wire_shape_is_single_element_positional_array() {
    // Pin: params serializes as [[tx_hash, tx_hash, ...]] — one
    // positional param that is the array of hashes. NOT
    // {"tx_hashes": [...]}.
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        Ok(serde_json::json!([])),
    );
    let client = make_client(tx.clone());
    let _ = client
        .query_anchor_status_batch(&[])
        .unwrap();
    let calls = tx.calls();
    // Empty input → no transport call (per anchor_tx.rs).
    assert!(calls.is_empty(), "empty input should short-circuit");

    // Now with non-empty input.
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        Ok(serde_json::json!([
            { "tx_hash": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "result": { "status": "finalized", "included_at_height": 1u64 }, "error": null },
            { "tx_hash": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "result": { "status": "submitted" }, "error": null },
        ])),
    );
    let client = make_client(tx.clone());
    let _ = client
        .query_anchor_status_batch(&["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(), "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string()])
        .unwrap();
    let calls = tx.calls();
    assert_eq!(calls.len(), 1);
    let (method, params) = &calls[0];
    assert_eq!(method, "sum_getIntegrityEvidenceAnchorStatusBatch");
    // The params value must be an array (positional), whose
    // first (and only) element is itself an array of strings.
    let params_array = params.as_array().expect("params is positional array");
    assert_eq!(
        params_array.len(),
        1,
        "exactly one positional param (the array of hashes)"
    );
    let inner = params_array[0]
        .as_array()
        .expect("inner element is the hash array");
    assert_eq!(inner.len(), 2);
    assert_eq!(inner[0].as_str(), Some("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    assert_eq!(inner[1].as_str(), Some("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"));
}

#[test]
fn by_tuple_params_wire_shape_is_five_element_positional_array() {
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorByTuple",
        Ok(serde_json::Value::Null),
    );
    let client = make_client(tx.clone());
    let artifact_hash = [0xabu8; 32];
    let signer_pubkey = [0xcdu8; 32];
    let _ = client
        .lookup_anchor_by_tuple(
            1,
            AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
            1,
            &artifact_hash,
            &signer_pubkey,
        )
        .unwrap();
    let calls = tx.calls();
    let (method, params) = &calls[0];
    assert_eq!(method, "sum_getIntegrityEvidenceAnchorByTuple");
    let arr = params.as_array().expect("params is positional array");
    assert_eq!(arr.len(), 5, "five positional params");
    assert_eq!(arr[0].as_u64(), Some(1)); // anchor_schema_version
    assert_eq!(arr[1].as_u64(), Some(0)); // artifact_kind tag for v1
    assert_eq!(arr[2].as_u64(), Some(1)); // artifact_schema_version
    let hash_hex = arr[3].as_str().expect("artifact_hash hex string");
    assert!(hash_hex.starts_with("0x"));
    assert_eq!(hash_hex.len(), 66);
    let pk_hex = arr[4].as_str().expect("signer_pubkey hex string");
    assert!(pk_hex.starts_with("0x"));
    assert_eq!(pk_hex.len(), 66);
}

#[test]
fn by_tuple_artifact_kind_v1_is_numeric_zero() {
    // Compile-time pin: the wire tag for v1
    // `SignedIntegrityEvidenceChainReport` is 0.
    assert_eq!(
        AnchoredArtifactKind::SignedIntegrityEvidenceChainReport.to_chain_tag_u32(),
        0
    );
}

// ── Batch happy paths ────────────────────────────────────────────────────────

#[test]
fn batch_status_parses_all_five_status_values_in_request_order() {
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        Ok(serde_json::json!([
            { "tx_hash": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "result": { "status": "submitted" }, "error": null },
            { "tx_hash": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "result": { "status": "included", "included_at_height": 100u64 }, "error": null },
            { "tx_hash": "0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", "result": { "status": "finalized", "included_at_height": 200u64 }, "error": null },
            { "tx_hash": "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "result": { "status": "failed", "code": 61u32, "reason": "duplicate 5-tuple" }, "error": null },
            { "tx_hash": "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", "result": { "status": "unknown" }, "error": null },
        ])),
    );
    let client = make_client(tx);
    let out = client
        .query_anchor_status_batch(&[
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
            "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string(),
            "0xcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".to_string(),
            "0xdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".to_string(),
            "0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_string(),
        ])
        .unwrap();
    assert_eq!(out.len(), 5);
    assert!(matches!(
        out[0].result.as_ref().unwrap().status,
        AnchorStatus::Submitted
    ));
    assert!(matches!(
        out[1].result.as_ref().unwrap().status,
        AnchorStatus::Included
    ));
    assert_eq!(out[1].result.as_ref().unwrap().included_at_height, Some(100));
    assert!(matches!(
        out[2].result.as_ref().unwrap().status,
        AnchorStatus::Finalized
    ));
    let failed = out[3].result.as_ref().unwrap();
    match &failed.status {
        AnchorStatus::Failed { reason } => assert_eq!(reason, "duplicate 5-tuple"),
        other => panic!("expected Failed, got {other:?}"),
    }
    assert_eq!(failed.code, Some(61));
    assert!(matches!(
        out[4].result.as_ref().unwrap().status,
        AnchorStatus::Unknown
    ));
}

#[test]
fn batch_per_item_malformed_hash_error_becomes_per_record_failure_not_whole_batch_abort() {
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        Ok(serde_json::json!([
            { "tx_hash": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "result": { "status": "finalized" }, "error": null },
            { "tx_hash": "bad", "result": null, "error": "Invalid hash: not 0x-prefixed" },
        ])),
    );
    let client = make_client(tx);
    let out = client
        .query_anchor_status_batch(&["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(), "bad".to_string()])
        .unwrap();
    assert_eq!(out.len(), 2);
    // First record: success.
    assert!(out[0].result.is_some());
    assert!(out[0].error.is_none());
    // Second record: per-item error preserved.
    assert!(out[1].result.is_none());
    assert_eq!(
        out[1].error.as_deref(),
        Some("Invalid hash: not 0x-prefixed")
    );
}

#[test]
fn batch_status_response_order_matches_request_order_with_canonical_compare() {
    // Cosmetic differences (0x prefix / case) must be tolerated
    // by canonicalize_tx_hash. Request has uppercase + no
    // prefix; response has lowercase + 0x prefix.
    let req_hash_uc = "AA".repeat(32);
    let resp_hash_lc = format!("0x{}", "a".repeat(64));
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        Ok(serde_json::json!([
            { "tx_hash": resp_hash_lc, "result": { "status": "finalized" }, "error": null },
        ])),
    );
    let client = make_client(tx);
    let out = client.query_anchor_status_batch(&[req_hash_uc]).unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].result.is_some());
}

#[test]
fn batch_response_length_mismatch_maps_to_chain_response_malformed() {
    let tx = FakeJsonRpcTransport::new();
    // Requested 2, chain returns 1.
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        Ok(serde_json::json!([
            { "tx_hash": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "result": { "status": "finalized" }, "error": null },
        ])),
    );
    let client = make_client(tx);
    let err = client
        .query_anchor_status_batch(&["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(), "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string()])
        .unwrap_err();
    assert_eq!(
        classify_chain_client_error(&err),
        ChainErrorCategory::Malformed
    );
}

#[test]
fn batch_malformed_whole_response_maps_to_chain_response_malformed() {
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorStatusBatch",
        Ok(serde_json::json!({"not": "an array"})),
    );
    let client = make_client(tx);
    let err = client
        .query_anchor_status_batch(&["0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string()])
        .unwrap_err();
    assert_eq!(
        classify_chain_client_error(&err),
        ChainErrorCategory::Malformed
    );
}

// ── By-tuple cases ───────────────────────────────────────────────────────────

#[test]
fn by_tuple_result_null_handled_as_not_found() {
    let tx = FakeJsonRpcTransport::new();
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorByTuple",
        Ok(serde_json::Value::Null),
    );
    let client = make_client(tx);
    let out = client
        .lookup_anchor_by_tuple(
            1,
            AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
            1,
            &[0u8; 32],
            &[0u8; 32],
        )
        .unwrap();
    assert!(out.is_none());
}

#[test]
fn by_tuple_canonical_tx_hash_and_included_at_height_parsed_and_surfaced() {
    let tx = FakeJsonRpcTransport::new();
    let canonical = format!("0x{}", "1".repeat(64));
    tx.set_response(
        "sum_getIntegrityEvidenceAnchorByTuple",
        Ok(serde_json::json!({
            "tx_hash": canonical,
            "included_at_height": 120u64,
        })),
    );
    let client = make_client(tx);
    let out = client
        .lookup_anchor_by_tuple(
            1,
            AnchoredArtifactKind::SignedIntegrityEvidenceChainReport,
            1,
            &[0u8; 32],
            &[0u8; 32],
        )
        .unwrap()
        .unwrap();
    assert_eq!(out.included_at_height, 120);
    assert!(out.tx_hash.starts_with("0x"));
}

// ── Stage 13.9 implementation-lock pins ──────────────────────────────────────

#[test]
fn canonicalize_tx_hash_strips_0x_lowercases_and_validates_length() {
    assert_eq!(
        canonicalize_tx_hash(&format!("0x{}", "AB".repeat(32))),
        Some("ab".repeat(32))
    );
    assert_eq!(
        canonicalize_tx_hash(&"ab".repeat(32)),
        Some("ab".repeat(32))
    );
    // Wrong length → None.
    assert_eq!(canonicalize_tx_hash("0xaa"), None);
    // Non-hex → None.
    let mut bad = "ab".repeat(31);
    bad.push_str("gg");
    assert_eq!(canonicalize_tx_hash(&bad), None);
}

#[test]
fn read_rpc_error_for_jsonrpc_does_not_say_chain_submit_refused() {
    // Pin that omni-sumchain's classifier categorizes a
    // JSON-RPC error envelope as JsonRpcError (NOT Malformed
    // or Transport). The read-path mapper at the CLI layer
    // re-routes JsonRpcError to chain_rpc; this test pins the
    // upstream classification stays JsonRpcError so the CLI
    // mapper's re-route is meaningful.
    let err = ChainClientError::Other("JSON-RPC error: -32000 Server error".to_string());
    assert_eq!(
        classify_chain_client_error(&err),
        ChainErrorCategory::JsonRpcError
    );
}
