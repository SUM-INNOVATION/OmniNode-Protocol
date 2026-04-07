use libp2p::{Multiaddr, PeerId};

use crate::codec::{ShardRequest, ShardResponse};
use crate::tensor_codec::{TensorRequest, TensorResponse};

/// Clean, domain-level events emitted by the OmniNode networking layer.
/// Never exposes raw libp2p internals to callers.
#[derive(Debug, Clone)]
pub enum OmniNetEvent {
    // ── Phase 1 ─────────────────────────────────────────────────────────

    /// The local node is now listening on a new address.
    Listening { addr: Multiaddr },

    /// A new peer was discovered via mDNS or Kademlia DHT.
    PeerDiscovered {
        peer_id: PeerId,
        addrs: Vec<Multiaddr>,
    },

    /// A previously discovered mDNS peer is no longer visible.
    PeerExpired { peer_id: PeerId },

    /// A transport-layer connection was established.
    PeerConnected { peer_id: PeerId },

    /// A transport-layer connection was closed.
    PeerDisconnected { peer_id: PeerId },

    /// A Gossipsub message was received.
    MessageReceived {
        /// PeerId that propagated the message to us.
        from: PeerId,
        /// Topic string, e.g. `"omni/test/v1"`.
        topic: String,
        /// Raw payload bytes.
        data: Vec<u8>,
    },

    // ── Phase 2: Shard transfer ─────────────────────────────────────────

    /// A remote peer requested a shard from us.
    /// The higher layer (omni-store) should call
    /// `OmniNet::respond_shard(channel_id, response)`.
    ShardRequested {
        peer_id: PeerId,
        request: ShardRequest,
        /// Internal ID mapped to the response channel stored in the swarm.
        channel_id: u64,
    },

    /// We received a shard chunk from a remote peer (response to our request).
    ShardReceived {
        peer_id: PeerId,
        response: ShardResponse,
    },

    /// An outbound shard request failed.
    ShardRequestFailed {
        peer_id: PeerId,
        error: String,
    },

    // ── Phase 4: Tensor transfer ────────────────────────────────────────

    /// A remote pipeline stage sent us a hidden-state activation tensor.
    /// The handler should call `OmniNet::respond_tensor(channel_id, response)`.
    TensorReceived {
        peer_id: PeerId,
        request: TensorRequest,
        /// Internal ID mapped to the response channel stored in the swarm.
        channel_id: u64,
    },

    /// We received an acknowledgment for a tensor we sent.
    TensorResponseReceived {
        peer_id: PeerId,
        response: TensorResponse,
    },

    /// An outbound tensor request failed.
    TensorRequestFailed {
        peer_id: PeerId,
        error: String,
    },

    // ── WAN: NAT traversal ──────────────────────────────────────────────

    /// AutoNAT determined whether this node is publicly reachable.
    NatStatusChanged {
        is_public: bool,
        /// The confirmed public address, if reachable.
        public_addr: Option<Multiaddr>,
    },

    /// A relay reservation was established — this firewalled node is now
    /// reachable via the relay peer's circuit address.
    RelayReservation {
        relay_peer_id: PeerId,
        relay_addr: Multiaddr,
    },

    /// DCUtR upgraded a relay circuit to a direct QUIC connection via
    /// UDP hole-punching.
    HolePunchSucceeded { peer_id: PeerId },

    /// DCUtR hole-punch failed — the relay circuit remains active but
    /// at degraded latency. Higher layers may want to warn the user or
    /// attempt a different relay peer.
    HolePunchFailed {
        peer_id: PeerId,
        error: String,
    },
}
