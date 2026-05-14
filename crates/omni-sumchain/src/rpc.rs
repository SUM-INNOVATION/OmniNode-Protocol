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
            ChainClientError::Other(format!("failed to serialise JSON-RPC body: {e}"))
        })?;

        let resp = self
            .agent
            .post(&self.url)
            .set("content-type", "application/json")
            .send_string(&body_str)
            .map_err(|e| {
                ChainClientError::Other(format!("HTTP transport failure: {e}"))
            })?;

        let resp_str = resp.into_string().map_err(|e| {
            ChainClientError::Other(format!("failed to read response body: {e}"))
        })?;

        let envelope: serde_json::Value = serde_json::from_str(&resp_str).map_err(|e| {
            ChainClientError::Other(format!("non-JSON response: {e}"))
        })?;

        if let Some(err) = envelope.get("error") {
            return Err(ChainClientError::Other(format!("JSON-RPC error: {err}")));
        }
        envelope.get("result").cloned().ok_or_else(|| {
            ChainClientError::Other(
                "JSON-RPC response missing required `result` field".into(),
            )
        })
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
