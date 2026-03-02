//! Flat #[pyclass] wrapper for OmniNetEvent.
//!
//! Uses a `kind` string discriminator + optional fields, avoiding Python-side
//! enum complexity.

use pyo3::prelude::*;

use omni_net::OmniNetEvent;

#[pyclass(name = "NetEvent")]
#[derive(Clone)]
pub struct PyNetEvent {
    /// Event kind: "listening", "peer_discovered", "peer_expired",
    /// "peer_connected", "peer_disconnected", "message_received",
    /// "shard_requested", "shard_received", "shard_request_failed",
    /// "tensor_received", "tensor_response_received", "tensor_request_failed".
    #[pyo3(get)]
    pub kind: String,

    #[pyo3(get)]
    pub peer_id: Option<String>,

    #[pyo3(get)]
    pub address: Option<String>,

    #[pyo3(get)]
    pub addresses: Option<Vec<String>>,

    #[pyo3(get)]
    pub topic: Option<String>,

    #[pyo3(get)]
    pub data: Option<Vec<u8>>,

    #[pyo3(get)]
    pub cid: Option<String>,

    #[pyo3(get)]
    pub channel_id: Option<u64>,

    #[pyo3(get)]
    pub offset: Option<u64>,

    #[pyo3(get)]
    pub total_bytes: Option<u64>,

    #[pyo3(get)]
    pub error: Option<String>,

    // ── Phase 4: Tensor fields ──────────────────────────────────────────

    #[pyo3(get)]
    pub session_id: Option<String>,

    #[pyo3(get)]
    pub micro_batch_index: Option<u32>,

    #[pyo3(get)]
    pub from_stage: Option<u32>,

    #[pyo3(get)]
    pub to_stage: Option<u32>,

    #[pyo3(get)]
    pub seq_len: Option<u32>,

    #[pyo3(get)]
    pub hidden_dim: Option<u32>,

    #[pyo3(get)]
    pub dtype: Option<u8>,

    #[pyo3(get)]
    pub stage_index: Option<u32>,

    #[pyo3(get)]
    pub accepted: Option<bool>,
}

#[pymethods]
impl PyNetEvent {
    fn __repr__(&self) -> String {
        let mut parts = vec![format!("kind='{}'", self.kind)];
        if let Some(ref p) = self.peer_id {
            parts.push(format!("peer_id='{p}'"));
        }
        if let Some(ref t) = self.topic {
            parts.push(format!("topic='{t}'"));
        }
        if let Some(ref c) = self.cid {
            parts.push(format!("cid='{c}'"));
        }
        if let Some(ref s) = self.session_id {
            parts.push(format!("session_id='{s}'"));
        }
        if let Some(si) = self.stage_index {
            parts.push(format!("stage_index={si}"));
        }
        if let Some(a) = self.accepted {
            parts.push(format!("accepted={a}"));
        }
        if let Some(ref e) = self.error {
            parts.push(format!("error='{e}'"));
        }
        format!("NetEvent({})", parts.join(", "))
    }
}

impl From<OmniNetEvent> for PyNetEvent {
    fn from(ev: OmniNetEvent) -> Self {
        match ev {
            OmniNetEvent::Listening { addr } => Self {
                kind: "listening".into(),
                address: Some(addr.to_string()),
                peer_id: None,
                addresses: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                error: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::PeerDiscovered { peer_id, addrs } => Self {
                kind: "peer_discovered".into(),
                peer_id: Some(peer_id.to_string()),
                addresses: Some(addrs.iter().map(|a| a.to_string()).collect()),
                address: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                error: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::PeerExpired { peer_id } => Self {
                kind: "peer_expired".into(),
                peer_id: Some(peer_id.to_string()),
                address: None,
                addresses: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                error: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::PeerConnected { peer_id } => Self {
                kind: "peer_connected".into(),
                peer_id: Some(peer_id.to_string()),
                address: None,
                addresses: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                error: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::PeerDisconnected { peer_id } => Self {
                kind: "peer_disconnected".into(),
                peer_id: Some(peer_id.to_string()),
                address: None,
                addresses: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                error: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::MessageReceived { from, topic, data } => Self {
                kind: "message_received".into(),
                peer_id: Some(from.to_string()),
                topic: Some(topic),
                data: Some(data),
                address: None,
                addresses: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                error: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::ShardRequested {
                peer_id,
                request,
                channel_id,
            } => Self {
                kind: "shard_requested".into(),
                peer_id: Some(peer_id.to_string()),
                cid: Some(request.cid),
                channel_id: Some(channel_id),
                offset: request.offset,
                address: None,
                addresses: None,
                topic: None,
                data: None,
                total_bytes: None,
                error: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::ShardReceived { peer_id, response } => Self {
                kind: "shard_received".into(),
                peer_id: Some(peer_id.to_string()),
                cid: Some(response.cid),
                offset: Some(response.offset),
                total_bytes: Some(response.total_bytes),
                data: Some(response.data),
                error: response.error,
                address: None,
                addresses: None,
                topic: None,
                channel_id: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::ShardRequestFailed { peer_id, error } => Self {
                kind: "shard_request_failed".into(),
                peer_id: Some(peer_id.to_string()),
                error: Some(error),
                address: None,
                addresses: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },

            // ── Phase 4: Tensor events ──────────────────────────────────

            OmniNetEvent::TensorReceived {
                peer_id,
                request,
                channel_id,
            } => Self {
                kind: "tensor_received".into(),
                peer_id: Some(peer_id.to_string()),
                channel_id: Some(channel_id),
                session_id: Some(request.session_id),
                micro_batch_index: Some(request.micro_batch_index),
                from_stage: Some(request.from_stage),
                to_stage: Some(request.to_stage),
                seq_len: Some(request.seq_len),
                hidden_dim: Some(request.hidden_dim),
                dtype: Some(request.dtype),
                data: Some(request.data),
                address: None,
                addresses: None,
                topic: None,
                cid: None,
                offset: None,
                total_bytes: None,
                error: None,
                stage_index: None,
                accepted: None,
            },
            OmniNetEvent::TensorResponseReceived { peer_id, response } => Self {
                kind: "tensor_response_received".into(),
                peer_id: Some(peer_id.to_string()),
                session_id: Some(response.session_id),
                micro_batch_index: Some(response.micro_batch_index),
                stage_index: Some(response.stage_index),
                accepted: Some(response.accepted),
                error: response.error,
                address: None,
                addresses: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
            },
            OmniNetEvent::TensorRequestFailed { peer_id, error } => Self {
                kind: "tensor_request_failed".into(),
                peer_id: Some(peer_id.to_string()),
                error: Some(error),
                address: None,
                addresses: None,
                topic: None,
                data: None,
                cid: None,
                channel_id: None,
                offset: None,
                total_bytes: None,
                session_id: None,
                micro_batch_index: None,
                from_stage: None,
                to_stage: None,
                seq_len: None,
                hidden_dim: None,
                dtype: None,
                stage_index: None,
                accepted: None,
            },
        }
    }
}
