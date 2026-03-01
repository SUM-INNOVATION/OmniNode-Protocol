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
    /// "shard_requested", "shard_received", "shard_request_failed".
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
            },
        }
    }
}
