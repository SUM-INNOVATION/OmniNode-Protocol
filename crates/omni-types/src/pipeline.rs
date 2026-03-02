// Phase 4 — Pipeline-parallel inference types.
//
// Shared across omni-pipeline, omni-net, omni-bridge.

use serde::{Deserialize, Serialize};

use crate::model::LayerRange;

// ── Tensor Dtype ─────────────────────────────────────────────────────────────

/// Activation data type for hidden-state tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TensorDtype {
    F16  = 0,
    BF16 = 1,
    F32  = 2,
}

impl TensorDtype {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F16),
            1 => Some(Self::BF16),
            2 => Some(Self::F32),
            _ => None,
        }
    }

    /// Bytes per element for this dtype.
    pub fn element_bytes(&self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

// ── Pipeline Stage ───────────────────────────────────────────────────────────

/// A single stage in a pipeline-parallel execution plan.
/// Each stage maps to one node running a contiguous range of transformer layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub stage_index: u32,
    pub peer_id: String,
    pub layer_range: LayerRange,
    pub includes_embedding: bool,
    pub includes_output_head: bool,
}

// ── Pipeline Schedule ────────────────────────────────────────────────────────

/// Complete pipeline schedule broadcast by the coordinator to all participants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSchedule {
    pub session_id: String,
    pub model_name: String,
    pub model_hash: String,
    pub total_layers: u32,
    pub stages: Vec<PipelineStage>,
    pub num_micro_batches: u32,
    pub max_seq_len: u32,
    pub hidden_dim: u32,
    pub created_at: String,
}

// ── Hidden State Header ──────────────────────────────────────────────────────

/// Metadata header for a hidden-state activation tensor.
/// The actual data follows as raw bytes in the transport layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenStateHeader {
    pub session_id: String,
    pub micro_batch_index: u32,
    pub from_stage: u32,
    pub to_stage: u32,
    pub seq_len: u32,
    pub hidden_dim: u32,
    pub dtype: TensorDtype,
    pub data_bytes: u64,
}

// ── Pipeline Capability ──────────────────────────────────────────────────────

/// Extended node capability for pipeline planning.
/// Sent as a Gossipsub `CapabilityOffer` during session formation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCapability {
    pub peer_id: String,
    pub ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub platform: String,
    pub local_shard_cids: Vec<String>,
    pub available_layers: Vec<LayerRange>,
    pub pipeline_ready: bool,
}

// ── Pipeline Message ─────────────────────────────────────────────────────────

/// Gossipsub messages exchanged on `omni/pipeline/v1` during session lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineMessage {
    /// Coordinator proposes a new pipeline session.
    Propose {
        session_id: String,
        model_name: String,
        model_hash: String,
        total_layers: u32,
        proposer_peer_id: String,
    },

    /// A node offers its capability for the proposed session.
    CapabilityOffer {
        session_id: String,
        capability: PipelineCapability,
    },

    /// Coordinator broadcasts the finalized schedule.
    ScheduleAssigned {
        schedule: PipelineSchedule,
    },

    /// A node confirms it is ready to execute its assigned stage.
    StageReady {
        session_id: String,
        peer_id: String,
        stage_index: u32,
    },

    /// Coordinator signals all stages to begin inference.
    StartInference {
        session_id: String,
    },

    /// Periodic liveness beacon from each active stage.
    Heartbeat {
        session_id: String,
        peer_id: String,
        stage_index: u32,
        micro_batches_completed: u32,
        timestamp: String,
    },

    /// Final stage reports session completion.
    SessionComplete {
        session_id: String,
        total_tokens_generated: u64,
    },

    /// A stage failed — coordinator may abort the session.
    StageFailure {
        session_id: String,
        peer_id: String,
        stage_index: u32,
        error: String,
    },

    /// Coordinator aborts the session.
    SessionAborted {
        session_id: String,
        reason: String,
    },
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_dtype_round_trip() {
        for v in 0..=2u8 {
            let dt = TensorDtype::from_u8(v).unwrap();
            assert_eq!(dt as u8, v);
        }
        assert!(TensorDtype::from_u8(3).is_none());
    }

    #[test]
    fn tensor_dtype_element_bytes() {
        assert_eq!(TensorDtype::F16.element_bytes(), 2);
        assert_eq!(TensorDtype::BF16.element_bytes(), 2);
        assert_eq!(TensorDtype::F32.element_bytes(), 4);
    }

    #[test]
    fn pipeline_stage_serde() {
        let stage = PipelineStage {
            stage_index: 0,
            peer_id: "12D3KooWTest".into(),
            layer_range: LayerRange { start: 0, end: 10 },
            includes_embedding: true,
            includes_output_head: false,
        };
        let json = serde_json::to_string(&stage).unwrap();
        let round: PipelineStage = serde_json::from_str(&json).unwrap();
        assert_eq!(round.stage_index, 0);
        assert_eq!(round.layer_range.start, 0);
    }

    #[test]
    fn pipeline_schedule_serde() {
        let schedule = PipelineSchedule {
            session_id: "test-session".into(),
            model_name: "llama-7b".into(),
            model_hash: "a".repeat(64),
            total_layers: 32,
            stages: vec![],
            num_micro_batches: 4,
            max_seq_len: 2048,
            hidden_dim: 4096,
            created_at: "2025-01-01T00:00:00Z".into(),
        };
        let json = serde_json::to_string(&schedule).unwrap();
        let round: PipelineSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(round.total_layers, 32);
        assert_eq!(round.hidden_dim, 4096);
    }

    #[test]
    fn pipeline_message_serde() {
        let msg = PipelineMessage::Propose {
            session_id: "sess-1".into(),
            model_name: "llama-7b".into(),
            model_hash: "abc".into(),
            total_layers: 32,
            proposer_peer_id: "12D3KooWTest".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let round: PipelineMessage = serde_json::from_str(&json).unwrap();
        match round {
            PipelineMessage::Propose { total_layers, .. } => {
                assert_eq!(total_layers, 32);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn hidden_state_header_serde() {
        let hdr = HiddenStateHeader {
            session_id: "sess-1".into(),
            micro_batch_index: 0,
            from_stage: 0,
            to_stage: 1,
            seq_len: 512,
            hidden_dim: 4096,
            dtype: TensorDtype::F16,
            data_bytes: 512 * 4096 * 2,
        };
        let json = serde_json::to_string(&hdr).unwrap();
        let round: HiddenStateHeader = serde_json::from_str(&json).unwrap();
        assert_eq!(round.data_bytes, 512 * 4096 * 2);
        assert_eq!(round.dtype, TensorDtype::F16);
    }

    #[test]
    fn pipeline_capability_serde() {
        let cap = PipelineCapability {
            peer_id: "12D3KooWTest".into(),
            ram_bytes: 16_000_000_000,
            available_ram_bytes: 8_000_000_000,
            platform: "AppleSilicon".into(),
            local_shard_cids: vec!["bafkr4itest".into()],
            available_layers: vec![LayerRange { start: 0, end: 15 }],
            pipeline_ready: true,
        };
        let json = serde_json::to_string(&cap).unwrap();
        let round: PipelineCapability = serde_json::from_str(&json).unwrap();
        assert!(round.pipeline_ready);
        assert_eq!(round.available_layers.len(), 1);
    }
}
