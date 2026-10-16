//! Flat #[pyclass] wrapper for OmniNetEvent.
//!
//! Uses a `kind` string discriminator + optional fields, avoiding Python-side
//! enum complexity.

use pyo3::prelude::*;

use omni_net::OmniNetEvent;

#[pyclass(name = "NetEvent", skip_from_py_object)]
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

            // ── WAN events ──────────────────────────────────────────────
            OmniNetEvent::NatStatusChanged { is_public, public_addr } => Self {
                kind: "nat_status_changed".into(),
                address: public_addr.map(|a| a.to_string()),
                accepted: Some(is_public),
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
            },
            OmniNetEvent::RelayReservation { relay_peer_id, relay_addr } => Self {
                kind: "relay_reservation".into(),
                peer_id: Some(relay_peer_id.to_string()),
                address: Some(relay_addr.to_string()),
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
            OmniNetEvent::HolePunchSucceeded { peer_id } => Self {
                kind: "hole_punch_succeeded".into(),
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
            OmniNetEvent::HolePunchFailed { peer_id, error } => Self {
                kind: "hole_punch_failed".into(),
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
#[cfg(test)]
mod tests {
    //! The flattening is the whole Python-visible contract of `NetEvent`: a
    //! consumer dispatches on `kind` and reads the fields that kind sets. These
    //! tests pin, for every `OmniNetEvent` variant, the kind string, exactly
    //! which fields are populated, and their values. They need no Python
    //! interpreter: building a `PyNetEvent` is plain Rust.

    use std::collections::BTreeSet;

    use libp2p::{Multiaddr, PeerId};
    use omni_net::{OmniNetEvent, ShardRequest, ShardResponse, TensorRequest, TensorResponse};

    use super::PyNetEvent;

    /// The names of the optional fields this event populates.
    fn populated(ev: &PyNetEvent) -> BTreeSet<&'static str> {
        let mut s = BTreeSet::new();
        let mut add = |name: &'static str, present: bool| {
            if present {
                s.insert(name);
            }
        };
        add("peer_id", ev.peer_id.is_some());
        add("address", ev.address.is_some());
        add("addresses", ev.addresses.is_some());
        add("topic", ev.topic.is_some());
        add("data", ev.data.is_some());
        add("cid", ev.cid.is_some());
        add("channel_id", ev.channel_id.is_some());
        add("offset", ev.offset.is_some());
        add("total_bytes", ev.total_bytes.is_some());
        add("error", ev.error.is_some());
        add("session_id", ev.session_id.is_some());
        add("micro_batch_index", ev.micro_batch_index.is_some());
        add("from_stage", ev.from_stage.is_some());
        add("to_stage", ev.to_stage.is_some());
        add("seq_len", ev.seq_len.is_some());
        add("hidden_dim", ev.hidden_dim.is_some());
        add("dtype", ev.dtype.is_some());
        add("stage_index", ev.stage_index.is_some());
        add("accepted", ev.accepted.is_some());
        s
    }

    fn set(names: &[&'static str]) -> BTreeSet<&'static str> {
        names.iter().copied().collect()
    }

    fn addr(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    /// One sample of every variant, with the kind and the fields it must
    /// populate. The exhaustive `match` makes a new `OmniNetEvent` variant a
    /// compile error here until its flattening is pinned too.
    fn samples(peer: PeerId) -> Vec<(OmniNetEvent, &'static str, BTreeSet<&'static str>)> {
        let all = vec![
            OmniNetEvent::Listening {
                addr: addr("/ip4/127.0.0.1/udp/4001/quic-v1"),
            },
            OmniNetEvent::PeerDiscovered {
                peer_id: peer,
                addrs: vec![addr("/ip4/10.0.0.1/tcp/1"), addr("/ip4/10.0.0.2/tcp/2")],
            },
            OmniNetEvent::PeerExpired { peer_id: peer },
            OmniNetEvent::PeerConnected { peer_id: peer },
            OmniNetEvent::PeerDisconnected { peer_id: peer },
            OmniNetEvent::MessageReceived {
                from: peer,
                topic: "omni/test/v1".into(),
                data: vec![1, 2, 3],
            },
            OmniNetEvent::ShardRequested {
                peer_id: peer,
                request: ShardRequest {
                    cid: "bafy-req".into(),
                    offset: Some(4096),
                    max_bytes: Some(1024),
                },
                channel_id: 7,
            },
            OmniNetEvent::ShardReceived {
                peer_id: peer,
                response: ShardResponse {
                    cid: "bafy-resp".into(),
                    offset: 8,
                    total_bytes: 64,
                    data: vec![9; 4],
                    error: None,
                },
            },
            OmniNetEvent::ShardRequestFailed {
                peer_id: peer,
                error: "timeout".into(),
            },
            OmniNetEvent::TensorReceived {
                peer_id: peer,
                request: TensorRequest {
                    session_id: "s-1".into(),
                    micro_batch_index: 2,
                    from_stage: 0,
                    to_stage: 1,
                    seq_len: 16,
                    hidden_dim: 32,
                    dtype: 1,
                    data: vec![0xAB; 8],
                },
                channel_id: 11,
            },
            OmniNetEvent::TensorResponseReceived {
                peer_id: peer,
                response: TensorResponse {
                    session_id: "s-1".into(),
                    micro_batch_index: 2,
                    stage_index: 1,
                    accepted: false,
                    error: Some("shape mismatch".into()),
                },
            },
            OmniNetEvent::TensorRequestFailed {
                peer_id: peer,
                error: "connection lost".into(),
            },
            OmniNetEvent::NatStatusChanged {
                is_public: true,
                public_addr: Some(addr("/ip4/203.0.113.7/udp/4001/quic-v1")),
            },
            OmniNetEvent::RelayReservation {
                relay_peer_id: peer,
                relay_addr: addr("/ip4/198.51.100.1/tcp/4001"),
            },
            OmniNetEvent::HolePunchSucceeded { peer_id: peer },
            OmniNetEvent::HolePunchFailed {
                peer_id: peer,
                error: "no route".into(),
            },
        ];
        all.into_iter()
            .map(|ev| {
                let (kind, fields): (&'static str, &[&'static str]) = match &ev {
                    OmniNetEvent::Listening { .. } => ("listening", &["address"]),
                    OmniNetEvent::PeerDiscovered { .. } => {
                        ("peer_discovered", &["peer_id", "addresses"])
                    }
                    OmniNetEvent::PeerExpired { .. } => ("peer_expired", &["peer_id"]),
                    OmniNetEvent::PeerConnected { .. } => ("peer_connected", &["peer_id"]),
                    OmniNetEvent::PeerDisconnected { .. } => ("peer_disconnected", &["peer_id"]),
                    OmniNetEvent::MessageReceived { .. } => {
                        ("message_received", &["peer_id", "topic", "data"])
                    }
                    OmniNetEvent::ShardRequested { .. } => (
                        "shard_requested",
                        &["peer_id", "cid", "channel_id", "offset"],
                    ),
                    OmniNetEvent::ShardReceived { .. } => (
                        "shard_received",
                        &["peer_id", "cid", "offset", "total_bytes", "data"],
                    ),
                    OmniNetEvent::ShardRequestFailed { .. } => {
                        ("shard_request_failed", &["peer_id", "error"])
                    }
                    OmniNetEvent::TensorReceived { .. } => (
                        "tensor_received",
                        &[
                            "peer_id",
                            "channel_id",
                            "session_id",
                            "micro_batch_index",
                            "from_stage",
                            "to_stage",
                            "seq_len",
                            "hidden_dim",
                            "dtype",
                            "data",
                        ],
                    ),
                    OmniNetEvent::TensorResponseReceived { .. } => (
                        "tensor_response_received",
                        &[
                            "peer_id",
                            "session_id",
                            "micro_batch_index",
                            "stage_index",
                            "accepted",
                            "error",
                        ],
                    ),
                    OmniNetEvent::TensorRequestFailed { .. } => {
                        ("tensor_request_failed", &["peer_id", "error"])
                    }
                    // `is_public` travels in `accepted`; the public address,
                    // when known, in `address`.
                    OmniNetEvent::NatStatusChanged { .. } => {
                        ("nat_status_changed", &["address", "accepted"])
                    }
                    OmniNetEvent::RelayReservation { .. } => {
                        ("relay_reservation", &["peer_id", "address"])
                    }
                    OmniNetEvent::HolePunchSucceeded { .. } => {
                        ("hole_punch_succeeded", &["peer_id"])
                    }
                    OmniNetEvent::HolePunchFailed { .. } => {
                        ("hole_punch_failed", &["peer_id", "error"])
                    }
                };
                (ev, kind, set(fields))
            })
            .collect()
    }

    #[test]
    fn every_variant_has_its_kind_and_exactly_its_fields() {
        let peer = PeerId::random();
        let samples = samples(peer);
        assert_eq!(samples.len(), 16);
        let mut kinds = BTreeSet::new();
        for (ev, kind, fields) in samples {
            let flat = PyNetEvent::from(ev);
            assert_eq!(flat.kind, kind);
            assert_eq!(populated(&flat), fields, "fields of {kind}");
            assert!(kinds.insert(kind), "kind {kind} used twice");
        }
    }

    #[test]
    fn values_are_carried_through_unchanged() {
        let peer = PeerId::random();
        let by_kind = |kind: &str| -> PyNetEvent {
            samples(peer)
                .into_iter()
                .find(|(_, k, _)| *k == kind)
                .map(|(ev, _, _)| PyNetEvent::from(ev))
                .unwrap()
        };

        let e = by_kind("listening");
        assert_eq!(
            e.address.as_deref(),
            Some("/ip4/127.0.0.1/udp/4001/quic-v1")
        );

        let e = by_kind("peer_discovered");
        assert_eq!(e.peer_id, Some(peer.to_string()));
        assert_eq!(
            e.addresses,
            Some(vec![
                "/ip4/10.0.0.1/tcp/1".to_string(),
                "/ip4/10.0.0.2/tcp/2".to_string()
            ])
        );

        let e = by_kind("message_received");
        assert_eq!(e.peer_id, Some(peer.to_string()));
        assert_eq!(e.topic.as_deref(), Some("omni/test/v1"));
        assert_eq!(e.data, Some(vec![1, 2, 3]));

        let e = by_kind("shard_requested");
        assert_eq!(
            (e.cid.as_deref(), e.channel_id, e.offset),
            (Some("bafy-req"), Some(7), Some(4096))
        );

        let e = by_kind("shard_received");
        assert_eq!(
            (e.cid.as_deref(), e.offset, e.total_bytes, e.data.as_deref()),
            (Some("bafy-resp"), Some(8), Some(64), Some(&[9u8; 4][..]))
        );
        assert_eq!(e.error, None);

        let e = by_kind("tensor_received");
        assert_eq!(
            (
                e.session_id.as_deref(),
                e.micro_batch_index,
                e.from_stage,
                e.to_stage,
                e.seq_len,
                e.hidden_dim,
                e.dtype,
                e.channel_id,
            ),
            (
                Some("s-1"),
                Some(2),
                Some(0),
                Some(1),
                Some(16),
                Some(32),
                Some(1),
                Some(11)
            )
        );
        assert_eq!(e.data, Some(vec![0xAB; 8]));

        let e = by_kind("tensor_response_received");
        assert_eq!(
            (e.stage_index, e.accepted, e.error.as_deref()),
            (Some(1), Some(false), Some("shape mismatch"))
        );

        let e = by_kind("nat_status_changed");
        assert_eq!(e.accepted, Some(true));
        assert_eq!(
            e.address.as_deref(),
            Some("/ip4/203.0.113.7/udp/4001/quic-v1")
        );
    }

    #[test]
    fn optional_inner_values_stay_absent_rather_than_defaulted() {
        let peer = PeerId::random();
        // A shard request without an offset, a shard response without an
        // error, and a NAT change without a public address keep those fields
        // `None`; Python sees `None`, never 0 or "".
        let e = PyNetEvent::from(OmniNetEvent::ShardRequested {
            peer_id: peer,
            request: ShardRequest {
                cid: "c".into(),
                offset: None,
                max_bytes: None,
            },
            channel_id: 1,
        });
        assert_eq!(e.offset, None);

        let e = PyNetEvent::from(OmniNetEvent::NatStatusChanged {
            is_public: false,
            public_addr: None,
        });
        assert_eq!((e.accepted, e.address), (Some(false), None));

        let e = PyNetEvent::from(OmniNetEvent::TensorResponseReceived {
            peer_id: peer,
            response: TensorResponse {
                session_id: "s".into(),
                micro_batch_index: 0,
                stage_index: 0,
                accepted: true,
                error: None,
            },
        });
        assert_eq!((e.accepted, e.error), (Some(true), None));
    }

    #[test]
    fn repr_names_the_kind_and_the_identifying_fields() {
        let peer = PeerId::random();
        let e = PyNetEvent::from(OmniNetEvent::TensorResponseReceived {
            peer_id: peer,
            response: TensorResponse {
                session_id: "s-9".into(),
                micro_batch_index: 0,
                stage_index: 3,
                accepted: false,
                error: Some("bad".into()),
            },
        });
        let r = e.__repr__();
        assert!(
            r.starts_with("NetEvent(kind='tensor_response_received'"),
            "{r}"
        );
        for part in [
            format!("peer_id='{peer}'"),
            "session_id='s-9'".to_string(),
            "stage_index=3".to_string(),
            "accepted=false".to_string(),
            "error='bad'".to_string(),
        ] {
            assert!(r.contains(&part), "{r} lacks {part}");
        }
    }
}
