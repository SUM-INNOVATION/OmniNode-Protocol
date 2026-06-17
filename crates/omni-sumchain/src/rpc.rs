//! Phase 5 Stage 7a — JSON-RPC transport for SUM Chain.
//!
//! Two implementations:
//! - [`UreqTransport`] — production, sync HTTP via the `ureq` crate.
//! - [`FakeJsonRpcTransport`] — test fixture with a clonable
//!   `Arc<Mutex<_>>` state so tests can pre-seed responses and inspect
//!   call logs after moving the fake into a [`crate::SumChainClient`].
//!
//! The trait deliberately does not require `Send + Sync` — Stage 7a's
//! `ChainClient` callers are single-threaded (the registry workflow is
//! sync and not threaded). Production `UreqTransport` happens to be
//! `Send + Sync` via `ureq::Agent`'s internal pooling, but the test
//! fake's `Arc<Mutex<_>>` is sufficient without bolting the bounds
//! onto every implementor.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use omni_zkml::ChainClientError;

// ── Stage 13.2 — Pinned error-prefix constants + typed classifier ────────────

/// Stage 13.2 — pinned `ChainClientError::Other(_)` message
/// prefixes produced by `UreqTransport::call` and (additively)
/// by [`crate::client::SumChainClient`]'s `EvidenceAnchorChainClient`
/// impl. These strings are stable for the lifetime of Stage 13.x;
/// bumping any of them is a coordinated wire-classification
/// change that requires updating consumers (OmniNode's CLI
/// reason-tag mapper) AND the
/// `tests/error_prefix_classification_is_stable.rs` regression.
///
/// Consumers MUST go through [`classify_chain_client_error`]
/// rather than inspecting these strings directly — the typed
/// [`ChainErrorCategory`] is the stable public API.
pub mod error_prefixes {
    pub const TRANSPORT_HTTP: &str = "HTTP transport failure: ";
    pub const TRANSPORT_BODY_READ: &str = "failed to read response body: ";
    pub const TRANSPORT_BODY_SERIALIZE: &str = "failed to serialise JSON-RPC body: ";
    pub const NON_JSON_RESPONSE: &str = "non-JSON response: ";
    pub const MISSING_RESULT_FIELD: &str =
        "JSON-RPC response missing required `result` field";
    pub const JSONRPC_ERROR: &str = "JSON-RPC error: ";

    // ── Stage 13.2 adapter-layer additions ───────────────────────────
    pub const ADAPTER_NOT_ACTIVATED: &str = "integrity_evidence_anchor not activated";
    pub const ADAPTER_SAME_KEY_FAIL: &str = "same-key submitter check: ";
    pub const ADAPTER_MALFORMED_SUBMIT_RESP: &str =
        "malformed sum_submitIntegrityEvidenceAnchor response: ";
    pub const ADAPTER_MALFORMED_STATUS_RESP: &str =
        "malformed sum_getIntegrityEvidenceAnchorStatus response: ";
    pub const ADAPTER_UNRECOGNIZED_STATUS: &str = "unrecognized anchor status: ";
}

/// Closed-set category for `ChainClientError::Other(_)` produced
/// by this crate. Consumers (OmniNode's reason-tag mapper) match
/// on this rather than reading prefix strings.
///
/// Adding a new variant is a coordinated `omni-sumchain` + CLI
/// patch — the CLI's exhaustive `match` surfaces the change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainErrorCategory {
    /// Transport-layer failure (HTTP / body read / JSON-RPC
    /// body serialize / non-JSON response / missing `result`).
    /// Surfaces to CLI as `reason=chain_rpc`.
    Transport,
    /// Chain returned a JSON-RPC error object. Surfaces to CLI
    /// as `reason=chain_submit_refused`.
    JsonRpcError,
    /// Success response shape parse failure OR unrecognized
    /// enum string (e.g. status="foo"). Surfaces to CLI as
    /// `reason=chain_response_malformed`.
    Malformed,
    /// Adapter activation gate refused pre-POST. Surfaces to
    /// CLI as `reason=not_activated`.
    AdapterNotActivated,
    /// Adapter same-key gate refused pre-POST. Surfaces to CLI
    /// as `reason=chain_rpc` (catch-all — upstream workflow
    /// should have caught this).
    AdapterSameKeyFail,
    /// Anything we don't recognize. CLI emits `reason=chain_rpc`
    /// for safety.
    Unknown,
}

/// Classify a `ChainClientError` produced by this crate into a
/// [`ChainErrorCategory`]. The single chokepoint for
/// prefix-string inspection in the workspace; consumers never
/// look at the raw string.
pub fn classify_chain_client_error(err: &ChainClientError) -> ChainErrorCategory {
    use error_prefixes::*;
    let msg = match err {
        ChainClientError::Other(s) => s.as_str(),
    };
    // Adapter-layer prefixes (Stage 13.2 — these are emitted by
    // omni-sumchain BEFORE any transport call, so they take
    // precedence over the transport prefixes in the match order.
    // None of the transport prefixes start with these strings,
    // so ordering is for clarity, not correctness).
    if msg.starts_with(ADAPTER_NOT_ACTIVATED) {
        return ChainErrorCategory::AdapterNotActivated;
    }
    if msg.starts_with(ADAPTER_SAME_KEY_FAIL) {
        return ChainErrorCategory::AdapterSameKeyFail;
    }
    if msg.starts_with(ADAPTER_MALFORMED_SUBMIT_RESP)
        || msg.starts_with(ADAPTER_MALFORMED_STATUS_RESP)
        || msg.starts_with(ADAPTER_UNRECOGNIZED_STATUS)
    {
        return ChainErrorCategory::Malformed;
    }
    // Transport / JSON-RPC envelope prefixes.
    if msg.starts_with(JSONRPC_ERROR) {
        return ChainErrorCategory::JsonRpcError;
    }
    if msg.starts_with(TRANSPORT_HTTP)
        || msg.starts_with(TRANSPORT_BODY_READ)
        || msg.starts_with(TRANSPORT_BODY_SERIALIZE)
        || msg.starts_with(NON_JSON_RESPONSE)
        || msg == MISSING_RESULT_FIELD
    {
        return ChainErrorCategory::Transport;
    }
    ChainErrorCategory::Unknown
}

// ── Transport trait ───────────────────────────────────────────────────────────

/// Minimal JSON-RPC 2.0 transport. Implementations send a request with
/// the given `method` and `params` and return the `result` field of the
/// response. JSON-RPC error responses must be mapped to
/// `ChainClientError::Other(_)` with a clear message.
pub trait JsonRpcTransport {
    fn call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> std::result::Result<serde_json::Value, ChainClientError>;
}

// ── Production: ureq ──────────────────────────────────────────────────────────

/// Production transport using `ureq` for sync HTTP. Single endpoint
/// per instance.
pub struct UreqTransport {
    agent: ureq::Agent,
    url: String,
}

impl UreqTransport {
    pub fn new(url: String) -> Self {
        Self::with_agent(ureq::Agent::new(), url)
    }

    pub fn with_agent(agent: ureq::Agent, url: String) -> Self {
        Self { agent, url }
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl JsonRpcTransport for UreqTransport {
    fn call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> std::result::Result<serde_json::Value, ChainClientError> {
        use error_prefixes::*;

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });
        // Hand-serialise / hand-parse to avoid pulling `ureq`'s optional
        // `json` feature into the workspace. `serde_json` is already a
        // direct dep of this crate.
        let body_str = serde_json::to_string(&body).map_err(|e| {
            ChainClientError::Other(format!("{TRANSPORT_BODY_SERIALIZE}{e}"))
        })?;

        let resp = self
            .agent
            .post(&self.url)
            .set("content-type", "application/json")
            .send_string(&body_str)
            .map_err(|e| ChainClientError::Other(format!("{TRANSPORT_HTTP}{e}")))?;

        let resp_str = resp.into_string().map_err(|e| {
            ChainClientError::Other(format!("{TRANSPORT_BODY_READ}{e}"))
        })?;

        let envelope: serde_json::Value = serde_json::from_str(&resp_str).map_err(|e| {
            ChainClientError::Other(format!("{NON_JSON_RESPONSE}{e}"))
        })?;

        if let Some(err) = envelope.get("error") {
            return Err(ChainClientError::Other(format!("{JSONRPC_ERROR}{err}")));
        }
        envelope
            .get("result")
            .cloned()
            .ok_or_else(|| ChainClientError::Other(MISSING_RESULT_FIELD.into()))
    }
}

// ── Test fake ────────────────────────────────────────────────────────────────

/// In-memory transport for hermetic tests. Holds an `Arc<Mutex<_>>` so
/// the test can keep a clone outside the [`crate::SumChainClient`] and
/// inspect the call log after running operations.
#[derive(Default, Clone)]
pub struct FakeJsonRpcTransport {
    state: Arc<Mutex<FakeState>>,
}

#[derive(Default)]
struct FakeState {
    calls: Vec<(String, serde_json::Value)>,
    responses:
        HashMap<String, std::result::Result<serde_json::Value, ChainClientError>>,
}

impl FakeJsonRpcTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-seed a canned response for `method`. The fake clones it on
    /// every call to the same method, so a single seeding suffices for
    /// repeated calls during one test.
    pub fn set_response(
        &self,
        method: &str,
        response: std::result::Result<serde_json::Value, ChainClientError>,
    ) {
        self.state
            .lock()
            .expect("FakeJsonRpcTransport state poisoned")
            .responses
            .insert(method.to_string(), response);
    }

    pub fn calls(&self) -> Vec<(String, serde_json::Value)> {
        self.state
            .lock()
            .expect("FakeJsonRpcTransport state poisoned")
            .calls
            .clone()
    }

    pub fn call_count(&self) -> usize {
        self.state
            .lock()
            .expect("FakeJsonRpcTransport state poisoned")
            .calls
            .len()
    }

    pub fn last_call(&self) -> Option<(String, serde_json::Value)> {
        self.state
            .lock()
            .expect("FakeJsonRpcTransport state poisoned")
            .calls
            .last()
            .cloned()
    }
}

impl JsonRpcTransport for FakeJsonRpcTransport {
    fn call(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> std::result::Result<serde_json::Value, ChainClientError> {
        let mut s = self
            .state
            .lock()
            .expect("FakeJsonRpcTransport state poisoned");
        s.calls.push((method.to_string(), params));
        match s.responses.get(method) {
            Some(r) => r.clone(),
            None => Err(ChainClientError::Other(format!(
                "fake: no response configured for method '{method}'"
            ))),
        }
    }
}
