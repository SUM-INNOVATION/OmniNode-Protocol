//! Gossipsub shard availability announcements.
//!
//! When a node ingests a model it publishes one [`ShardAnnouncement`] per shard
//! on the `omni/shard/v1` topic so that other nodes learn which peer holds
//! which CID.

use serde::{Deserialize, Serialize};

use omni_types::model::LayerRange;

/// Message published on `omni/shard/v1` after ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAnnouncement {
    /// CIDv1 identifying this shard.
    pub cid: String,
    /// Human-readable model name.
    pub model_name: String,
    /// Which transformer blocks this shard covers.
    pub layer_range: LayerRange,
    /// Size of the shard data in bytes.
    pub size_bytes: u64,
}

/// Encode a [`ShardAnnouncement`] to bincode bytes for Gossipsub publishing.
pub fn encode_announcement(ann: &ShardAnnouncement) -> Vec<u8> {
    bincode::serde::encode_to_vec(ann, bincode::config::standard())
        .expect("ShardAnnouncement serialization is infallible")
}

/// Decode a [`ShardAnnouncement`] from bincode bytes received via Gossipsub.
pub fn decode_announcement(data: &[u8]) -> Option<ShardAnnouncement> {
    bincode::serde::decode_from_slice(data, bincode::config::standard())
        .ok()
        .map(|(ann, _)| ann)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn announcement_round_trip() {
        let ann = ShardAnnouncement {
            cid: "bafkr4itest".into(),
            model_name: "llama-7b".into(),
            layer_range: LayerRange { start: 0, end: 3 },
            size_bytes: 500_000_000,
        };
        let bytes = encode_announcement(&ann);
        let decoded = decode_announcement(&bytes).unwrap();
        assert_eq!(decoded.cid, ann.cid);
        assert_eq!(decoded.model_name, ann.model_name);
        assert_eq!(decoded.size_bytes, ann.size_bytes);
    }
}
