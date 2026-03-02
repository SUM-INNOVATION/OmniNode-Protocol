//! PipelineMessage encode/decode helpers for gossipsub transport.
//!
//! Messages are serialized with bincode (same config as TensorCodec/ShardCodec)
//! and published to the `omni/pipeline/v1` gossipsub topic.

use omni_types::pipeline::PipelineMessage;

use crate::error::{PipelineError, Result};

/// Encode a [`PipelineMessage`] to bytes for gossipsub publishing.
pub fn encode_pipeline_message(msg: &PipelineMessage) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(msg, bincode::config::standard())
        .map_err(|e| PipelineError::Serialization(e.to_string()))
}

/// Decode a [`PipelineMessage`] from gossipsub payload bytes.
pub fn decode_pipeline_message(data: &[u8]) -> Result<PipelineMessage> {
    let (msg, _) =
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map_err(|e| PipelineError::Serialization(e.to_string()))?;
    Ok(msg)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use omni_types::pipeline::PipelineCapability;
    use omni_types::model::LayerRange;

    #[test]
    fn propose_round_trip() {
        let msg = PipelineMessage::Propose {
            session_id: "sess-1".into(),
            model_name: "llama-7b".into(),
            model_hash: "abc".into(),
            total_layers: 32,
            proposer_peer_id: "peer-a".into(),
        };
        let bytes = encode_pipeline_message(&msg).unwrap();
        let decoded = decode_pipeline_message(&bytes).unwrap();
        match decoded {
            PipelineMessage::Propose { total_layers, .. } => assert_eq!(total_layers, 32),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn capability_offer_round_trip() {
        let msg = PipelineMessage::CapabilityOffer {
            session_id: "sess-1".into(),
            capability: PipelineCapability {
                peer_id: "peer-b".into(),
                ram_bytes: 16_000_000_000,
                available_ram_bytes: 12_000_000_000,
                platform: "AppleSilicon".into(),
                local_shard_cids: vec!["cid1".into()],
                available_layers: vec![LayerRange { start: 0, end: 15 }],
                pipeline_ready: true,
            },
        };
        let bytes = encode_pipeline_message(&msg).unwrap();
        let decoded = decode_pipeline_message(&bytes).unwrap();
        match decoded {
            PipelineMessage::CapabilityOffer { capability, .. } => {
                assert!(capability.pipeline_ready);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn heartbeat_round_trip() {
        let msg = PipelineMessage::Heartbeat {
            session_id: "sess-1".into(),
            peer_id: "peer-a".into(),
            stage_index: 0,
            micro_batches_completed: 3,
            timestamp: "2025-01-01T00:00:00Z".into(),
        };
        let bytes = encode_pipeline_message(&msg).unwrap();
        let decoded = decode_pipeline_message(&bytes).unwrap();
        match decoded {
            PipelineMessage::Heartbeat { micro_batches_completed, .. } => {
                assert_eq!(micro_batches_completed, 3);
            }
            _ => panic!("wrong variant"),
        }
    }
}
