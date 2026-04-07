use libp2p::{
    autonat, dcutr, gossipsub, identify, kad, mdns, relay,
    request_response, swarm::NetworkBehaviour,
};

use crate::codec::ShardCodec;
use crate::tensor_codec::TensorCodec;

/// Composed [`NetworkBehaviour`] for the OmniNode mesh — LAN + WAN.
///
/// The `#[derive(NetworkBehaviour)]` macro generates `OmniNodeBehaviourEvent`
/// with variants matching each field name in PascalCase:
///
/// **Existing (Phase 1–4):**
/// - `Mdns(mdns::Event)`
/// - `Gossipsub(gossipsub::Event)`
/// - `Identify(identify::Event)`
/// - `ShardXfer(request_response::Event<ShardRequest, ShardResponse>)`
/// - `TensorXfer(request_response::Event<TensorRequest, TensorResponse>)`
///
/// **New (WAN):**
/// - `Kademlia(kad::Event)`
/// - `Autonat(autonat::Event)`
/// - `Relay(relay::Event)`
/// - `RelayClient(relay::client::Event)`
/// - `Dcutr(dcutr::Event)`
#[derive(NetworkBehaviour)]
pub struct OmniNodeBehaviour {
    // ── Existing: LAN mesh (Phase 1–4) ───────────────────────
    pub mdns:        mdns::tokio::Behaviour,
    pub gossipsub:   gossipsub::Behaviour,
    pub identify:    identify::Behaviour,
    pub shard_xfer:  request_response::Behaviour<ShardCodec>,
    pub tensor_xfer: request_response::Behaviour<TensorCodec>,

    // ── New: WAN discovery & NAT traversal ───────────────────
    pub kademlia:     kad::Behaviour<kad::store::MemoryStore>,
    pub autonat:      autonat::Behaviour,
    pub relay:        relay::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub dcutr:        dcutr::Behaviour,
}
