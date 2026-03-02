//! Per-node stage executor for pipeline-parallel inference.
//!
//! The executor manages the local state for a single pipeline stage.
//! It builds TensorRequest/TensorResponse messages but does NOT perform
//! network I/O — the caller sends them via OmniNet.
//!
//! The actual forward pass (matrix multiply, attention, etc.) is performed
//! in Python via omni-bridge. The executor only handles the coordination:
//!   receive tensor → (caller runs Python forward) → send tensor to next stage.

use omni_net::tensor_codec::{TensorRequest, TensorResponse};
use omni_types::pipeline::PipelineSchedule;

use crate::error::{PipelineError, Result};
use crate::scheduler::MicroBatchSchedule;

/// Manages execution state for one stage on the local node.
pub struct StageExecutor {
    session_id: String,
    stage_index: u32,
    schedule: PipelineSchedule,
    micro_batch_schedule: MicroBatchSchedule,
    micro_batches_completed: u32,
}

impl StageExecutor {
    /// Create an executor for the given stage within a pipeline schedule.
    pub fn new(stage_index: u32, schedule: PipelineSchedule) -> Result<Self> {
        if stage_index as usize >= schedule.stages.len() {
            return Err(PipelineError::Execution(format!(
                "stage index {} out of bounds (total {})",
                stage_index,
                schedule.stages.len()
            )));
        }

        let micro_batch_schedule = MicroBatchSchedule::new(
            schedule.stages.len() as u32,
            schedule.num_micro_batches,
        );

        Ok(Self {
            session_id: schedule.session_id.clone(),
            stage_index,
            schedule,
            micro_batch_schedule,
            micro_batches_completed: 0,
        })
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn stage_index(&self) -> u32 {
        self.stage_index
    }

    pub fn schedule(&self) -> &PipelineSchedule {
        &self.schedule
    }

    pub fn micro_batch_schedule(&self) -> &MicroBatchSchedule {
        &self.micro_batch_schedule
    }

    pub fn micro_batches_completed(&self) -> u32 {
        self.micro_batches_completed
    }

    // ── Stage topology ───────────────────────────────────────────────────

    /// True if this is the first stage (receives token input, runs embedding).
    pub fn is_first_stage(&self) -> bool {
        self.stage_index == 0
    }

    /// True if this is the last stage (runs lm_head, produces logits).
    pub fn is_last_stage(&self) -> bool {
        self.stage_index == self.schedule.stages.len() as u32 - 1
    }

    /// Peer ID of the next stage in the pipeline, if any.
    pub fn next_stage_peer_id(&self) -> Option<&str> {
        let next = (self.stage_index + 1) as usize;
        self.schedule.stages.get(next).map(|s| s.peer_id.as_str())
    }

    /// Peer ID of the previous stage in the pipeline, if any.
    pub fn prev_stage_peer_id(&self) -> Option<&str> {
        if self.stage_index == 0 {
            return None;
        }
        let prev = (self.stage_index - 1) as usize;
        self.schedule.stages.get(prev).map(|s| s.peer_id.as_str())
    }

    // ── Message builders ─────────────────────────────────────────────────

    /// Build a [`TensorRequest`] to forward activations to the next stage.
    ///
    /// The `data` bytes are the output of the local forward pass.
    pub fn build_forward_request(
        &self,
        micro_batch_index: u32,
        data: Vec<u8>,
        seq_len: u32,
        hidden_dim: u32,
        dtype: u8,
    ) -> TensorRequest {
        TensorRequest {
            session_id: self.session_id.clone(),
            micro_batch_index,
            from_stage: self.stage_index,
            to_stage: self.stage_index + 1,
            seq_len,
            hidden_dim,
            dtype,
            data,
        }
    }

    /// Build an acknowledgment [`TensorResponse`] for a received tensor.
    pub fn build_ack(
        &self,
        micro_batch_index: u32,
        accepted: bool,
        error: Option<String>,
    ) -> TensorResponse {
        TensorResponse {
            session_id: self.session_id.clone(),
            micro_batch_index,
            stage_index: self.stage_index,
            accepted,
            error,
        }
    }

    // ── Progress tracking ────────────────────────────────────────────────

    /// Increment the completed micro-batch counter.
    pub fn mark_micro_batch_completed(&mut self) {
        self.micro_batches_completed += 1;
    }

    /// Check if all micro-batches have been processed.
    pub fn is_complete(&self) -> bool {
        self.micro_batches_completed >= self.schedule.num_micro_batches
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use omni_types::model::LayerRange;
    use omni_types::pipeline::PipelineStage;

    fn three_stage_schedule() -> PipelineSchedule {
        PipelineSchedule {
            session_id: "sess-1".into(),
            model_name: "llama-7b".into(),
            model_hash: "abc".into(),
            total_layers: 32,
            stages: vec![
                PipelineStage {
                    stage_index: 0,
                    peer_id: "peer-a".into(),
                    layer_range: LayerRange { start: 0, end: 10 },
                    includes_embedding: true,
                    includes_output_head: false,
                },
                PipelineStage {
                    stage_index: 1,
                    peer_id: "peer-b".into(),
                    layer_range: LayerRange { start: 11, end: 21 },
                    includes_embedding: false,
                    includes_output_head: false,
                },
                PipelineStage {
                    stage_index: 2,
                    peer_id: "peer-c".into(),
                    layer_range: LayerRange { start: 22, end: 31 },
                    includes_embedding: false,
                    includes_output_head: true,
                },
            ],
            num_micro_batches: 4,
            max_seq_len: 2048,
            hidden_dim: 4096,
            created_at: "2025-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn topology() {
        let exec = StageExecutor::new(1, three_stage_schedule()).unwrap();
        assert!(!exec.is_first_stage());
        assert!(!exec.is_last_stage());
        assert_eq!(exec.next_stage_peer_id(), Some("peer-c"));
        assert_eq!(exec.prev_stage_peer_id(), Some("peer-a"));
    }

    #[test]
    fn first_stage() {
        let exec = StageExecutor::new(0, three_stage_schedule()).unwrap();
        assert!(exec.is_first_stage());
        assert!(!exec.is_last_stage());
        assert_eq!(exec.next_stage_peer_id(), Some("peer-b"));
        assert!(exec.prev_stage_peer_id().is_none());
    }

    #[test]
    fn last_stage() {
        let exec = StageExecutor::new(2, three_stage_schedule()).unwrap();
        assert!(!exec.is_first_stage());
        assert!(exec.is_last_stage());
        assert!(exec.next_stage_peer_id().is_none());
        assert_eq!(exec.prev_stage_peer_id(), Some("peer-b"));
    }

    #[test]
    fn forward_request_fields() {
        let exec = StageExecutor::new(0, three_stage_schedule()).unwrap();
        let req = exec.build_forward_request(2, vec![0xAB; 100], 32, 64, 0);
        assert_eq!(req.session_id, "sess-1");
        assert_eq!(req.micro_batch_index, 2);
        assert_eq!(req.from_stage, 0);
        assert_eq!(req.to_stage, 1);
        assert_eq!(req.data.len(), 100);
    }

    #[test]
    fn ack_fields() {
        let exec = StageExecutor::new(1, three_stage_schedule()).unwrap();
        let ack = exec.build_ack(0, true, None);
        assert_eq!(ack.session_id, "sess-1");
        assert_eq!(ack.stage_index, 1);
        assert!(ack.accepted);
    }

    #[test]
    fn completion_tracking() {
        let mut exec = StageExecutor::new(0, three_stage_schedule()).unwrap();
        assert!(!exec.is_complete());
        for _ in 0..4 {
            exec.mark_micro_batch_completed();
        }
        assert!(exec.is_complete());
    }

    #[test]
    fn invalid_stage_index() {
        let result = StageExecutor::new(5, three_stage_schedule());
        assert!(result.is_err());
    }
}
