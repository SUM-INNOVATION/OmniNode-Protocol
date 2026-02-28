use libp2p::{Multiaddr, PeerId};

use crate::codec::{ShardRequest, ShardResponse};

/// Clean, domain-level events emitted by the OmniNode networking layer.
/// Never exposes raw libp2p internals to callers.
#[derive(Debug, Clone)]
pub enum OmniNetEvent {
    // ── Phase 1 ─────────────────────────────────────────────────────────

    /// The local node is now listening on a new address.
    Listening { addr: Multiaddr },

    /// A new peer was discovered via mDNS on the local network.
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
}
