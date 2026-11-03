use libp2p::{
    autonat, dcutr, gossipsub, identify, kad, mdns, relay, request_response,
    swarm::{NetworkBehaviour, behaviour::toggle::Toggle},
};

use crate::codec::ShardCodec;
use crate::gate::{Gated, PolicyBehaviour};
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
///
/// **Issue #158:** `policy` judges inbound connections against the outbound
/// allow-list. It must stay the first field (fields are asked in declaration
/// order), and it is only built when an allow-list is set. `mdns`, `identify`
/// and `kademlia` are wrapped in [`Gated`], which filters the peer addresses
/// they offer for dials and publish to other behaviours; with no allow-list it
/// passes everything through. Both dereference to the inner behaviour.
#[derive(NetworkBehaviour)]
pub struct OmniNodeBehaviour {
    // ── Issue #158: inbound policy (first) ──────────────────
    pub policy:      Toggle<PolicyBehaviour>,

    // ── Existing: LAN mesh (Phase 1–4) ───────────────────────
    /// Off when [`omni_types::config::NetConfig::mdns`] is `false`.
    pub mdns:        Gated<Toggle<mdns::tokio::Behaviour>>,
    pub gossipsub:   gossipsub::Behaviour,
    pub identify:    Gated<identify::Behaviour>,
    pub shard_xfer:  request_response::Behaviour<ShardCodec>,
    pub tensor_xfer: request_response::Behaviour<TensorCodec>,

    // ── New: WAN discovery & NAT traversal ───────────────────
    pub kademlia:     Gated<kad::Behaviour<kad::store::MemoryStore>>,
    pub autonat:      autonat::Behaviour,
    pub relay:        relay::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub dcutr:        dcutr::Behaviour,
}
