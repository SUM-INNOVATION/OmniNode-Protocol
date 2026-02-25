use libp2p::{gossipsub, identify, mdns, swarm::NetworkBehaviour};

/// Composed [`NetworkBehaviour`] for the OmniNode local mesh (Phase 1).
///
/// The `#[derive(NetworkBehaviour)]` macro generates `LocalMeshBehaviourEvent`
/// with variants:
/// - `Mdns(mdns::Event)`
/// - `Gossipsub(gossipsub::Event)`
/// - `Identify(identify::Event)`
///
/// **Phase 1 scope (LAN only):** mDNS + Gossipsub + Identify.
///
/// **Deferred to Phase 1b (WAN):** `kademlia`, `autonat`, `relay`, `dcutr`,
/// `request_response`.
#[derive(NetworkBehaviour)]
pub struct LocalMeshBehaviour {
    pub mdns:      mdns::tokio::Behaviour,
    pub gossipsub: gossipsub::Behaviour,
    pub identify:  identify::Behaviour,
}
