use libp2p::{gossipsub, identify, mdns, request_response, swarm::NetworkBehaviour};

use crate::codec::ShardCodec;
use crate::tensor_codec::TensorCodec;

/// Composed [`NetworkBehaviour`] for the OmniNode local mesh.
///
/// The `#[derive(NetworkBehaviour)]` macro generates `LocalMeshBehaviourEvent`
/// with variants matching each field name in PascalCase:
/// - `Mdns(mdns::Event)`
/// - `Gossipsub(gossipsub::Event)`
/// - `Identify(identify::Event)`
/// - `ShardXfer(request_response::Event<ShardRequest, ShardResponse>)`
/// - `TensorXfer(request_response::Event<TensorRequest, TensorResponse>)`
#[derive(NetworkBehaviour)]
pub struct LocalMeshBehaviour {
    pub mdns:        mdns::tokio::Behaviour,
    pub gossipsub:   gossipsub::Behaviour,
    pub identify:    identify::Behaviour,
    pub shard_xfer:  request_response::Behaviour<ShardCodec>,
    pub tensor_xfer: request_response::Behaviour<TensorCodec>,
}
