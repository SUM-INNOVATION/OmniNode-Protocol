use libp2p::{Multiaddr, PeerId};

/// Clean, domain-level events emitted by the OmniNode networking layer.
/// Never exposes raw libp2p internals to callers.
#[derive(Debug, Clone)]
pub enum OmniNetEvent {
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
}
