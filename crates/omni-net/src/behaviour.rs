use libp2p::{gossipsub, identify, mdns, request_response, swarm::NetworkBehaviour};

use crate::codec::ShardCodec;

/// Composed [`NetworkBehaviour`] for the OmniNode local mesh.
///
/// The `#[derive(NetworkBehaviour)]` macro generates `LocalMeshBehaviourEvent`
/// with variants:
/// - `Mdns(mdns::Event)`
/// - `Gossipsub(gossipsub::Event)`
/// - `Identify(identify::Event)`
/// - `ShardXfer(request_response::Event<ShardRequest, ShardResponse>)`
///
/// **Phase 1 scope:** mDNS + Gossipsub + Identify.
/// **Phase 2 addition:** `request_response` for shard transfer.
#[derive(NetworkBehaviour)]
pub struct LocalMeshBehaviour {
    pub mdns:       mdns::tokio::Behaviour,
    pub gossipsub:  gossipsub::Behaviour,
    pub identify:   identify::Behaviour,
    pub shard_xfer: request_response::Behaviour<ShardCodec>,
}
