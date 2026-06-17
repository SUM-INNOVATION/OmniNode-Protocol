//! Phase 5 Stage 13.2 — pinned `ChainClientError::Other(_)`
//! prefix → [`ChainErrorCategory`] classification regression.
//!
//! `omni-sumchain` owns these prefix strings; OmniNode's CLI
//! reason-tag mapper goes through
//! [`classify_chain_client_error`] and never reads the strings
//! directly. Bumping any of these prefixes is a coordinated
//! `omni-sumchain` + CLI change that surfaces this regression
//! test.

use omni_sumchain::{
    ChainErrorCategory, classify_chain_client_error, error_prefixes::*,
};
use omni_zkml::ChainClientError;

fn err(s: &str) -> ChainClientError {
    ChainClientError::Other(s.to_string())
}

// ── Prefix-value pins (literal strings) ──────────────────────────────────────

/// Pins the literal prefix VALUES. Bumping any of these
/// requires updating OmniNode's reason-tag mapper consumers
/// AND regenerating this regression.
#[test]
fn pinned_prefix_values_are_unchanged() {
    assert_eq!(TRANSPORT_HTTP, "HTTP transport failure: ");
    assert_eq!(TRANSPORT_BODY_READ, "failed to read response body: ");
    assert_eq!(TRANSPORT_BODY_SERIALIZE, "failed to serialise JSON-RPC body: ");
    assert_eq!(NON_JSON_RESPONSE, "non-JSON response: ");
    assert_eq!(
        MISSING_RESULT_FIELD,
        "JSON-RPC response missing required `result` field"
    );
    assert_eq!(JSONRPC_ERROR, "JSON-RPC error: ");

    // Stage 13.2 adapter-layer additions.
    assert_eq!(ADAPTER_NOT_ACTIVATED, "integrity_evidence_anchor not activated");
    assert_eq!(ADAPTER_SAME_KEY_FAIL, "same-key submitter check: ");
    assert_eq!(
        ADAPTER_MALFORMED_SUBMIT_RESP,
        "malformed sum_submitIntegrityEvidenceAnchor response: "
    );
    assert_eq!(
        ADAPTER_MALFORMED_STATUS_RESP,
        "malformed sum_getIntegrityEvidenceAnchorStatus response: "
    );
    assert_eq!(ADAPTER_UNRECOGNIZED_STATUS, "unrecognized anchor status: ");
}

// ── Transport prefixes → Transport ───────────────────────────────────────────

#[test]
fn transport_http_classifies_as_transport() {
    assert_eq!(
        classify_chain_client_error(&err(&format!("{TRANSPORT_HTTP}timed out"))),
        ChainErrorCategory::Transport
    );
}

#[test]
fn transport_body_read_classifies_as_transport() {
    assert_eq!(
        classify_chain_client_error(&err(&format!("{TRANSPORT_BODY_READ}eof"))),
        ChainErrorCategory::Transport
    );
}

#[test]
fn transport_body_serialize_classifies_as_transport() {
    assert_eq!(
        classify_chain_client_error(&err(&format!(
            "{TRANSPORT_BODY_SERIALIZE}invalid utf-8"
        ))),
        ChainErrorCategory::Transport
    );
}

#[test]
fn non_json_response_classifies_as_transport() {
    assert_eq!(
        classify_chain_client_error(&err(&format!(
            "{NON_JSON_RESPONSE}expected value at line 1 column 1"
        ))),
        ChainErrorCategory::Transport
    );
}

#[test]
fn missing_result_field_exact_match_classifies_as_transport() {
    // This prefix is the FULL string, no suffix expected — the
    // classifier matches exact equality, not starts_with.
    assert_eq!(
        classify_chain_client_error(&err(MISSING_RESULT_FIELD)),
        ChainErrorCategory::Transport
    );
}

// ── JSON-RPC error → JsonRpcError ────────────────────────────────────────────

#[test]
fn jsonrpc_error_object_classifies_as_jsonrpc_error() {
    assert_eq!(
        classify_chain_client_error(&err(&format!(
            "{JSONRPC_ERROR}{{\"code\":-32601,\"message\":\"method not found\"}}"
        ))),
        ChainErrorCategory::JsonRpcError
    );
}

// ── Adapter-layer prefixes → corresponding categories ────────────────────────

#[test]
fn adapter_not_activated_classifies_as_adapter_not_activated() {
    assert_eq!(
        classify_chain_client_error(&err(ADAPTER_NOT_ACTIVATED)),
        ChainErrorCategory::AdapterNotActivated
    );
}

#[test]
fn adapter_same_key_fail_classifies_as_adapter_same_key_fail() {
    assert_eq!(
        classify_chain_client_error(&err(&format!(
            "{ADAPTER_SAME_KEY_FAIL}seed derives a, digest declares b"
        ))),
        ChainErrorCategory::AdapterSameKeyFail
    );
}

#[test]
fn adapter_malformed_submit_response_classifies_as_malformed() {
    assert_eq!(
        classify_chain_client_error(&err(&format!(
            "{ADAPTER_MALFORMED_SUBMIT_RESP}null"
        ))),
        ChainErrorCategory::Malformed
    );
}

#[test]
fn adapter_malformed_status_response_classifies_as_malformed() {
    assert_eq!(
        classify_chain_client_error(&err(&format!(
            "{ADAPTER_MALFORMED_STATUS_RESP}missing field 'status'"
        ))),
        ChainErrorCategory::Malformed
    );
}

#[test]
fn adapter_unrecognized_status_classifies_as_malformed() {
    assert_eq!(
        classify_chain_client_error(&err(&format!(
            "{ADAPTER_UNRECOGNIZED_STATUS}\"foo\""
        ))),
        ChainErrorCategory::Malformed
    );
}

// ── Fallthrough → Unknown ────────────────────────────────────────────────────

#[test]
fn unknown_message_classifies_as_unknown() {
    assert_eq!(
        classify_chain_client_error(&err("totally unrelated error text")),
        ChainErrorCategory::Unknown
    );
    assert_eq!(
        classify_chain_client_error(&err("")),
        ChainErrorCategory::Unknown
    );
}
