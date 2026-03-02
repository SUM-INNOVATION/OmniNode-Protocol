//! RAM-proportional layer-to-node assignment for pipeline parallelism.
//!
//! Algorithm:
//! 1. Filter nodes to those marked `pipeline_ready`.
//! 2. Sort by `available_ram_bytes` descending (biggest first).
//! 3. Cap node count to `total_layers` (each node needs ≥ 1 layer).
//! 4. Assign layers proportionally to RAM; enforce contiguity.
//! 5. First stage gets embedding, last stage gets output head.

use omni_types::model::LayerRange;
use omni_types::pipeline::{PipelineCapability, PipelineSchedule, PipelineStage};

use crate::error::{PipelineError, Result};

/// Plan a pipeline-parallel execution schedule from capability offers.
pub fn plan_stages(
    capabilities: &[PipelineCapability],
    total_layers: u32,
    model_name: &str,
    model_hash: &str,
    session_id: &str,
    num_micro_batches: Option<u32>,
    max_seq_len: u32,
    hidden_dim: u32,
) -> Result<PipelineSchedule> {
    // ── 1. Filter to pipeline-ready nodes ────────────────────────────────
    let mut ready: Vec<&PipelineCapability> = capabilities
        .iter()
        .filter(|c| c.pipeline_ready)
        .collect();

    if ready.is_empty() {
        return Err(PipelineError::Planning(
            "no pipeline-ready nodes available".into(),
        ));
    }

    // ── 2. Sort by RAM descending ────────────────────────────────────────
    ready.sort_by(|a, b| b.available_ram_bytes.cmp(&a.available_ram_bytes));

    // ── 3. Cap to total_layers ───────────────────────────────────────────
    let num_nodes = ready.len().min(total_layers as usize);
    let ready = &ready[..num_nodes];

    let total_ram: u64 = ready.iter().map(|c| c.available_ram_bytes).sum();
    if total_ram == 0 {
        return Err(PipelineError::Planning(
            "total available RAM is zero".into(),
        ));
    }

    // ── 4. Assign layers proportionally ──────────────────────────────────
    let mut stages = Vec::with_capacity(num_nodes);
    let mut layer_cursor: u32 = 0;

    for (i, cap) in ready.iter().enumerate() {
        let remaining_layers = total_layers - layer_cursor;
        let remaining_nodes = (num_nodes - i) as u32;

        let assigned = if remaining_nodes == 1 {
            remaining_layers
        } else {
            let proportion = cap.available_ram_bytes as f64 / total_ram as f64;
            let ideal = (proportion * total_layers as f64).round() as u32;
            // At least 1, at most leaving 1 per remaining peer
            let max_here = remaining_layers - (remaining_nodes - 1);
            ideal.max(1).min(max_here)
        };

        let start = layer_cursor;
        let end = start + assigned - 1;

        stages.push(PipelineStage {
            stage_index: i as u32,
            peer_id: cap.peer_id.clone(),
            layer_range: LayerRange { start, end },
            includes_embedding: start == 0,
            includes_output_head: end == total_layers - 1,
        });

        layer_cursor = end + 1;
    }

    // ── 5. Build schedule ────────────────────────────────────────────────
    let num_stages = stages.len() as u32;
    let effective_micro_batches = num_micro_batches.unwrap_or(2 * num_stages);

    Ok(PipelineSchedule {
        session_id: session_id.to_string(),
        model_name: model_name.to_string(),
        model_hash: model_hash.to_string(),
        total_layers,
        stages,
        num_micro_batches: effective_micro_batches,
        max_seq_len,
        hidden_dim,
        created_at: chrono::Utc::now().to_rfc3339(),
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(peer: &str, ram: u64) -> PipelineCapability {
        PipelineCapability {
            peer_id: peer.into(),
            ram_bytes: ram,
            available_ram_bytes: ram,
            platform: "test".into(),
            local_shard_cids: vec![],
            available_layers: vec![],
            pipeline_ready: true,
        }
    }

    #[test]
    fn two_equal_nodes() {
        let caps = vec![cap("a", 8_000_000_000), cap("b", 8_000_000_000)];
        let schedule = plan_stages(&caps, 32, "m", "h", "s", None, 2048, 4096).unwrap();
        assert_eq!(schedule.stages.len(), 2);
        assert_eq!(schedule.stages[0].layer_range, LayerRange { start: 0, end: 15 });
        assert_eq!(schedule.stages[1].layer_range, LayerRange { start: 16, end: 31 });
        assert!(schedule.stages[0].includes_embedding);
        assert!(schedule.stages[1].includes_output_head);
    }

    #[test]
    fn three_nodes_proportional() {
        // 16 GB, 8 GB, 8 GB → proportion 0.5, 0.25, 0.25 of 32 layers
        let caps = vec![
            cap("a", 16_000_000_000),
            cap("b", 8_000_000_000),
            cap("c", 8_000_000_000),
        ];
        let schedule = plan_stages(&caps, 32, "m", "h", "s", None, 2048, 4096).unwrap();
        assert_eq!(schedule.stages.len(), 3);
        // Total layers across all stages must equal 32
        let total: u32 = schedule.stages.iter().map(|s| s.layer_range.end - s.layer_range.start + 1).sum();
        assert_eq!(total, 32);
        // First stage should have the most layers (highest RAM)
        let first_layers = schedule.stages[0].layer_range.end - schedule.stages[0].layer_range.start + 1;
        assert!(first_layers >= 14); // ~16 layers for 50% RAM
    }

    #[test]
    fn single_node_gets_all() {
        let caps = vec![cap("a", 16_000_000_000)];
        let schedule = plan_stages(&caps, 32, "m", "h", "s", None, 2048, 4096).unwrap();
        assert_eq!(schedule.stages.len(), 1);
        assert_eq!(schedule.stages[0].layer_range, LayerRange { start: 0, end: 31 });
        assert!(schedule.stages[0].includes_embedding);
        assert!(schedule.stages[0].includes_output_head);
    }

    #[test]
    fn no_ready_nodes_fails() {
        let mut c = cap("a", 8_000_000_000);
        c.pipeline_ready = false;
        let result = plan_stages(&[c], 32, "m", "h", "s", None, 2048, 4096);
        assert!(result.is_err());
    }

    #[test]
    fn auto_micro_batches() {
        let caps = vec![cap("a", 8_000_000_000), cap("b", 8_000_000_000)];
        let schedule = plan_stages(&caps, 32, "m", "h", "s", None, 2048, 4096).unwrap();
        // Auto = 2 × num_stages = 4
        assert_eq!(schedule.num_micro_batches, 4);
    }

    #[test]
    fn explicit_micro_batches() {
        let caps = vec![cap("a", 8_000_000_000), cap("b", 8_000_000_000)];
        let schedule = plan_stages(&caps, 32, "m", "h", "s", Some(8), 2048, 4096).unwrap();
        assert_eq!(schedule.num_micro_batches, 8);
    }
}
