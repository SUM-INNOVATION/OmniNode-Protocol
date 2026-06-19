//! Phase 5 Stage 13.2 — DTOs for the integrity-evidence-anchor
//! RPCs. **OmniNode-owned, no vendored chain types.** Stage 13.2
//! treats SUM Chain purely as an external JSON-RPC service.
//!
//! The Stage 13.0 wire payload itself
//! (`omni_zkml::IntegrityEvidenceAnchorTxData`) is the byte-stable
//! submit contract; this module only models the RPC envelopes
//! around it.

use serde::Deserialize;

/// Successful response from `sum_submitIntegrityEvidenceAnchor`.
///
/// Lenient parsing per Stage 13.2 Q2 contract assumption:
/// [`parse_submit_anchor_result`] accepts either this object form
/// (preferred — `{ "tx_hash": "0x..." }`) OR a bare hex string
/// (`"0x..."`) — mirrors Stage 7b's
/// `parse_send_raw_transaction_result` shape. Anything else
/// surfaces as `chain_response_malformed`.
#[cfg(feature = "submit")]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct SubmitAnchorResult {
    pub tx_hash: String,
}

/// Lenient parser for the submit-anchor response. Returns the
/// `0x`-prefixed chain tx hash on success; `None` if the response
/// matches neither the preferred object form nor the
/// backwards-compat bare-string form.
#[cfg(feature = "submit")]
pub(crate) fn parse_submit_anchor_result(value: &serde_json::Value) -> Option<String> {
    // Preferred: { "tx_hash": "0x..." }
    if let Ok(parsed) = serde_json::from_value::<SubmitAnchorResult>(value.clone()) {
        return Some(parsed.tx_hash);
    }
    // Backwards-compat: bare hex string.
    if let serde_json::Value::String(s) = value {
        return Some(s.clone());
    }
    None
}

/// Response from `sum_getIntegrityEvidenceAnchorStatus`.
///
/// Per Stage 13.2 Q4 / Q5 contract assumptions:
/// - `status` lowercase: `"submitted" | "included" | "finalized" | "failed" | "unknown"`.
/// - `included_at_height` parsed but **dropped** at the
///   adapter / workflow boundary (Stage 13.0 `AnchorStatus` enum
///   unchanged in Stage 13.2).
/// - `reason` populated only on `status == "failed"`; surfaced
///   verbatim through `AnchorStatus::Failed { reason }`.
///
/// A foreign / non-anchor `tx_hash` returns `status: "unknown"`
/// per chain contract; the client mirrors as
/// `AnchorStatus::Unknown` without local-record mutation
/// (Stage 5.1 observation-only).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct AnchorStatusResult {
    pub status: String,
    #[serde(default)]
    pub included_at_height: Option<u64>,
    /// Stage 13.9 — failure code parsed and exposed as
    /// `Option<u32>`. Stable values per chain contract:
    /// `60` not activated, `61` duplicate 5-tuple,
    /// `62` invalid submitter signature, `63` `tx.from !=
    /// address(signer_pubkey)`. Other failures may carry
    /// `code: null`. Parse the field; do not interpret unknown
    /// codes.
    #[serde(default)]
    pub code: Option<u32>,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Stage 13.9 — per-item entry of
/// `sum_getIntegrityEvidenceAnchorStatusBatch`. Echo-back
/// `tx_hash` preserves request ordering; `result` carries the
/// status when no per-item error; `error` carries the chain's
/// per-item error text (e.g. malformed hash). Exactly one of
/// `result` / `error` is non-null per item.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct BatchStatusItem {
    pub tx_hash: String,
    #[serde(default)]
    pub result: Option<AnchorStatusResult>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Stage 13.9 — response from
/// `sum_getIntegrityEvidenceAnchorByTuple`. Either present (chain
/// has an anchor matching the 5-tuple) or `null` (no match — the
/// outer JSON-RPC `result` field is `null`; this DTO is only
/// constructed for the present case).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct ByTupleResult {
    /// Canonical `0x`-prefixed 32-byte lowercase hex.
    pub tx_hash: String,
    pub included_at_height: u64,
}

#[cfg(all(test, feature = "submit"))]
mod submit_response_tests {
    use super::*;

    #[test]
    fn submit_response_parses_preferred_object_form() {
        let v = serde_json::json!({ "tx_hash": "0xabcd" });
        assert_eq!(parse_submit_anchor_result(&v).as_deref(), Some("0xabcd"));
    }

    #[test]
    fn submit_response_parses_bare_string_form() {
        let v = serde_json::json!("0xdeadbeef");
        assert_eq!(
            parse_submit_anchor_result(&v).as_deref(),
            Some("0xdeadbeef")
        );
    }

    #[test]
    fn submit_response_refuses_unknown_shape() {
        let v = serde_json::json!({ "result": "0xfoo" }); // wrong key
        assert!(parse_submit_anchor_result(&v).is_none());
        let v = serde_json::json!(42); // not a string, not an object
        assert!(parse_submit_anchor_result(&v).is_none());
        let v = serde_json::json!([]); // array
        assert!(parse_submit_anchor_result(&v).is_none());
    }

    #[test]
    fn status_response_parses_with_optional_fields_absent() {
        let v = serde_json::json!({ "status": "submitted" });
        let parsed: AnchorStatusResult = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.status, "submitted");
        assert_eq!(parsed.included_at_height, None);
        assert_eq!(parsed.reason, None);
    }

    #[test]
    fn status_response_parses_with_all_fields_populated() {
        let v = serde_json::json!({
            "status": "included",
            "included_at_height": 12345u64,
            "reason": null,
        });
        let parsed: AnchorStatusResult = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.status, "included");
        assert_eq!(parsed.included_at_height, Some(12345));
        assert_eq!(parsed.reason, None);
    }

    #[test]
    fn status_response_parses_failed_with_reason() {
        let v = serde_json::json!({
            "status": "failed",
            "reason": "fee below min",
        });
        let parsed: AnchorStatusResult = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.status, "failed");
        assert_eq!(parsed.reason.as_deref(), Some("fee below min"));
    }

    #[test]
    fn status_response_parses_code_field_when_present() {
        let v = serde_json::json!({
            "status": "failed",
            "code": 61u32,
            "reason": "duplicate 5-tuple",
        });
        let parsed: AnchorStatusResult = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.status, "failed");
        assert_eq!(parsed.code, Some(61));
        assert_eq!(parsed.reason.as_deref(), Some("duplicate 5-tuple"));
    }

    #[test]
    fn status_response_parses_code_absent_as_none() {
        let v = serde_json::json!({
            "status": "finalized",
            "included_at_height": 100u64,
        });
        let parsed: AnchorStatusResult = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.code, None);
    }

    #[test]
    fn batch_status_item_parses_success_shape() {
        let v = serde_json::json!({
            "tx_hash": "0xabcd",
            "result": { "status": "finalized", "included_at_height": 4807033u64 },
            "error": null,
        });
        let parsed: BatchStatusItem = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.tx_hash, "0xabcd");
        assert!(parsed.error.is_none());
        let result = parsed.result.unwrap();
        assert_eq!(result.status, "finalized");
        assert_eq!(result.included_at_height, Some(4807033));
    }

    #[test]
    fn batch_status_item_parses_error_shape() {
        let v = serde_json::json!({
            "tx_hash": "bad",
            "result": null,
            "error": "Invalid hash: not 0x-prefixed",
        });
        let parsed: BatchStatusItem = serde_json::from_value(v).unwrap();
        assert!(parsed.result.is_none());
        assert_eq!(parsed.error.as_deref(), Some("Invalid hash: not 0x-prefixed"));
    }

    #[test]
    fn by_tuple_result_parses_canonical_tx_hash_and_height() {
        let v = serde_json::json!({
            "tx_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "included_at_height": 120u64,
        });
        let parsed: ByTupleResult = serde_json::from_value(v).unwrap();
        assert_eq!(parsed.included_at_height, 120);
        assert!(parsed.tx_hash.starts_with("0x"));
    }
}
