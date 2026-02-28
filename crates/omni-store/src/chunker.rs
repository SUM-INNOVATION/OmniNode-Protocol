//! Layer-wise tensor chunker for GGUF models.
//!
//! Classifies every tensor by its role (embedding, transformer block N,
//! output head, or global) using standard llama-family naming conventions
//! (`token_embd.*`, `blk.{N}.*`, `output.*`), then groups them into
//! [`ChunkPlan`]s governed by `layers_per_shard`.

use omni_types::model::LayerRange;

use crate::error::{Result, StoreError};
use crate::gguf::GgufFile;

// ── Tensor classification ─────────────────────────────────────────────────────

/// Role of a tensor within the model graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRole {
    /// Token embedding layer (`token_embd.*`).
    Embedding,
    /// Belongs to transformer block N (`blk.{N}.*`).
    Block(u32),
    /// Output projection / final norm (`output.*`, `output_norm.*`).
    OutputHead,
    /// Global tensors that don't belong to a specific block
    /// (e.g. `rope_freqs.*`).  Packed with the embedding shard.
    Global,
}

/// Classify a tensor by its GGUF name.
///
/// Covers all standard llama-family architectures (LLaMA, Mistral, Phi,
/// Qwen, etc.) which uniformly use the `blk.{N}.*` convention.
pub fn classify_tensor(name: &str) -> TensorRole {
    if name.starts_with("token_embd") {
        TensorRole::Embedding
    } else if name.starts_with("blk.") {
        // "blk.{N}.rest…" — extract N
        let after_prefix = &name[4..]; // skip "blk."
        if let Some(n_str) = after_prefix.split('.').next() {
            if let Ok(n) = n_str.parse::<u32>() {
                return TensorRole::Block(n);
            }
        }
        // Malformed blk name — treat as global
        TensorRole::Global
    } else if name.starts_with("output") {
        TensorRole::OutputHead
    } else {
        TensorRole::Global
    }
}

// ── Chunk plan ────────────────────────────────────────────────────────────────

/// Describes which tensors belong to a single shard, before data extraction.
#[derive(Debug, Clone)]
pub struct ChunkPlan {
    /// Zero-based shard index.
    pub shard_index: u32,
    /// Which transformer blocks this shard covers.
    pub layer_range: LayerRange,
    /// Whether the token embedding is included in this shard.
    pub includes_embedding: bool,
    /// Whether the output head is included in this shard.
    pub includes_output_head: bool,
    /// Indices into [`GgufFile::tensors`].
    pub tensor_indices: Vec<usize>,
}

/// Partition a parsed GGUF file's tensors into shards.
///
/// - Embedding + global tensors → first shard
/// - Output head tensors → last shard
/// - Transformer blocks are grouped by `layers_per_shard`
pub fn plan_chunks(gguf: &GgufFile, layers_per_shard: u32) -> Result<Vec<ChunkPlan>> {
    if layers_per_shard == 0 {
        return Err(StoreError::Other("layers_per_shard must be > 0".into()));
    }

    // ── Classify every tensor ────────────────────────────────────────────
    let mut embedding_indices: Vec<usize> = Vec::new();
    let mut output_indices: Vec<usize> = Vec::new();
    let mut global_indices: Vec<usize> = Vec::new();
    let mut block_map: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();

    for (i, tensor) in gguf.tensors.iter().enumerate() {
        match classify_tensor(&tensor.name) {
            TensorRole::Embedding => embedding_indices.push(i),
            TensorRole::Block(n) => block_map.entry(n).or_default().push(i),
            TensorRole::OutputHead => output_indices.push(i),
            TensorRole::Global => global_indices.push(i),
        }
    }

    let total_blocks = block_map.keys().last().map_or(0, |&max| max + 1);

    // ── Edge case: no transformer blocks ─────────────────────────────────
    if total_blocks == 0 {
        let mut all: Vec<usize> = Vec::new();
        all.extend(&embedding_indices);
        all.extend(&global_indices);
        all.extend(&output_indices);
        return Ok(vec![ChunkPlan {
            shard_index: 0,
            layer_range: LayerRange { start: 0, end: 0 },
            includes_embedding: !embedding_indices.is_empty(),
            includes_output_head: !output_indices.is_empty(),
            tensor_indices: all,
        }]);
    }

    // ── Build one plan per shard ─────────────────────────────────────────
    let n_shards = total_blocks.div_ceil(layers_per_shard);
    let mut plans: Vec<ChunkPlan> = Vec::with_capacity(n_shards as usize);

    for shard_idx in 0..n_shards {
        let start = shard_idx * layers_per_shard;
        let end = std::cmp::min(start + layers_per_shard - 1, total_blocks - 1);
        let is_first = shard_idx == 0;
        let is_last = shard_idx == n_shards - 1;

        let mut indices: Vec<usize> = Vec::new();

        // Embedding + global tensors go in the first shard
        if is_first {
            indices.extend(&embedding_indices);
            indices.extend(&global_indices);
        }

        // Block tensors for [start..=end]
        for blk in start..=end {
            if let Some(blk_indices) = block_map.get(&blk) {
                indices.extend(blk_indices);
            }
        }

        // Output head tensors go in the last shard
        if is_last {
            indices.extend(&output_indices);
        }

        plans.push(ChunkPlan {
            shard_index: shard_idx,
            layer_range: LayerRange { start, end },
            includes_embedding: is_first && !embedding_indices.is_empty(),
            includes_output_head: is_last && !output_indices.is_empty(),
            tensor_indices: indices,
        });
    }

    Ok(plans)
}

// ── Byte-range helper ─────────────────────────────────────────────────────────

/// Compute the contiguous byte range `[abs_start, abs_end)` of a shard's
/// tensor data within the full GGUF file.
///
/// The range spans from the earliest tensor's start to the byte immediately
/// after the latest tensor's data.  Works correctly as long as tensors in the
/// GGUF file are packed in order (standard for llama.cpp output).
pub fn shard_data_range(
    gguf: &GgufFile,
    plan: &ChunkPlan,
    file_len: u64,
) -> (u64, u64) {
    if plan.tensor_indices.is_empty() {
        let off = gguf.tensor_data_offset;
        return (off, off);
    }

    let min_offset = plan
        .tensor_indices
        .iter()
        .map(|&i| gguf.tensors[i].offset)
        .min()
        .unwrap();

    let max_offset = plan
        .tensor_indices
        .iter()
        .map(|&i| gguf.tensors[i].offset)
        .max()
        .unwrap();

    // End of the last tensor = the next tensor's offset (from any shard),
    // or the remainder of the tensor data section for the very last tensor.
    let data_section_len = file_len - gguf.tensor_data_offset;
    let next_after_max = gguf
        .tensors
        .iter()
        .map(|t| t.offset)
        .filter(|&o| o > max_offset)
        .min()
        .unwrap_or(data_section_len);

    let abs_start = gguf.tensor_data_offset + min_offset;
    let abs_end = gguf.tensor_data_offset + next_after_max;

    (abs_start, abs_end)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufHeader, TensorInfo};

    fn make_tensor(name: &str, offset: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            n_dimensions: 2,
            dimensions: vec![4096, 4096],
            ggml_type: 1, // F16
            offset,
        }
    }

    // ── Classification ───────────────────────────────────────────────────

    #[test]
    fn classify_embedding() {
        assert_eq!(classify_tensor("token_embd.weight"), TensorRole::Embedding);
    }

    #[test]
    fn classify_blocks() {
        assert_eq!(classify_tensor("blk.0.attn_q.weight"), TensorRole::Block(0));
        assert_eq!(classify_tensor("blk.15.ffn_gate.weight"), TensorRole::Block(15));
        assert_eq!(classify_tensor("blk.31.attn_norm.weight"), TensorRole::Block(31));
    }

    #[test]
    fn classify_output_head() {
        assert_eq!(classify_tensor("output.weight"), TensorRole::OutputHead);
        assert_eq!(classify_tensor("output_norm.weight"), TensorRole::OutputHead);
    }

    #[test]
    fn classify_global() {
        assert_eq!(classify_tensor("rope_freqs.weight"), TensorRole::Global);
        assert_eq!(classify_tensor("some_unknown_tensor"), TensorRole::Global);
    }

    // ── Chunk planning ───────────────────────────────────────────────────

    fn make_llama_gguf(n_blocks: u32) -> GgufFile {
        let mut tensors = vec![make_tensor("token_embd.weight", 0)];
        let mut offset = 1000u64;
        for blk in 0..n_blocks {
            for suffix in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_down", "ffn_up", "attn_norm", "ffn_norm"] {
                tensors.push(make_tensor(
                    &format!("blk.{blk}.{suffix}.weight"),
                    offset,
                ));
                offset += 1000;
            }
        }
        tensors.push(make_tensor("output_norm.weight", offset));
        offset += 1000;
        tensors.push(make_tensor("output.weight", offset));

        GgufFile {
            header: GgufHeader {
                version: 3,
                tensor_count: tensors.len() as u64,
                metadata_kv_count: 0,
            },
            metadata: vec![],
            tensors,
            tensor_data_offset: 512,
        }
    }

    #[test]
    fn plan_32_blocks_4_per_shard() {
        let gguf = make_llama_gguf(32);
        let plans = plan_chunks(&gguf, 4).unwrap();

        assert_eq!(plans.len(), 8); // 32 / 4

        // First shard: embedding + blocks 0-3
        assert!(plans[0].includes_embedding);
        assert!(!plans[0].includes_output_head);
        assert_eq!(plans[0].layer_range, LayerRange { start: 0, end: 3 });
        // 1 embedding + 4 blocks × 9 tensors each = 37
        assert_eq!(plans[0].tensor_indices.len(), 1 + 4 * 9);

        // Last shard: blocks 28-31 + output head
        assert!(!plans[7].includes_embedding);
        assert!(plans[7].includes_output_head);
        assert_eq!(plans[7].layer_range, LayerRange { start: 28, end: 31 });
        // 4 blocks × 9 + 2 output tensors = 38
        assert_eq!(plans[7].tensor_indices.len(), 4 * 9 + 2);

        // Middle shard (e.g. shard 3): blocks 12-15, no special tensors
        assert!(!plans[3].includes_embedding);
        assert!(!plans[3].includes_output_head);
        assert_eq!(plans[3].layer_range, LayerRange { start: 12, end: 15 });
        assert_eq!(plans[3].tensor_indices.len(), 4 * 9);
    }

    #[test]
    fn plan_uneven_division() {
        // 10 blocks with 3 per shard → 4 shards: [0-2], [3-5], [6-8], [9-9]
        let gguf = make_llama_gguf(10);
        let plans = plan_chunks(&gguf, 3).unwrap();

        assert_eq!(plans.len(), 4);
        assert_eq!(plans[0].layer_range, LayerRange { start: 0, end: 2 });
        assert_eq!(plans[1].layer_range, LayerRange { start: 3, end: 5 });
        assert_eq!(plans[2].layer_range, LayerRange { start: 6, end: 8 });
        assert_eq!(plans[3].layer_range, LayerRange { start: 9, end: 9 });

        // Last shard has only 1 block + output head
        assert!(plans[3].includes_output_head);
        assert_eq!(plans[3].tensor_indices.len(), 1 * 9 + 2);
    }

    #[test]
    fn plan_no_blocks() {
        let tensors = vec![
            make_tensor("token_embd.weight", 0),
            make_tensor("output.weight", 1000),
        ];
        let gguf = GgufFile {
            header: GgufHeader {
                version: 3,
                tensor_count: 2,
                metadata_kv_count: 0,
            },
            metadata: vec![],
            tensors,
            tensor_data_offset: 64,
        };

        let plans = plan_chunks(&gguf, 4).unwrap();
        assert_eq!(plans.len(), 1);
        assert!(plans[0].includes_embedding);
        assert!(plans[0].includes_output_head);
        assert_eq!(plans[0].tensor_indices.len(), 2);
    }

    #[test]
    fn plan_rejects_zero_layers_per_shard() {
        let gguf = make_llama_gguf(4);
        assert!(plan_chunks(&gguf, 0).is_err());
    }

    // ── Byte range ───────────────────────────────────────────────────────

    #[test]
    fn shard_data_range_basic() {
        let tensors = vec![
            make_tensor("token_embd.weight", 0),
            make_tensor("blk.0.attn_q.weight", 1000),
            make_tensor("blk.1.attn_q.weight", 2000),
            make_tensor("output.weight", 3000),
        ];
        let gguf = GgufFile {
            header: GgufHeader { version: 3, tensor_count: 4, metadata_kv_count: 0 },
            metadata: vec![],
            tensors,
            tensor_data_offset: 100,
        };
        let file_len = 5000u64;

        // Shard covering embedding + blk.0
        let plan = ChunkPlan {
            shard_index: 0,
            layer_range: LayerRange { start: 0, end: 0 },
            includes_embedding: true,
            includes_output_head: false,
            tensor_indices: vec![0, 1],
        };
        let (start, end) = shard_data_range(&gguf, &plan, file_len);
        assert_eq!(start, 100); // tensor_data_offset + 0
        assert_eq!(end, 2100);  // tensor_data_offset + 2000 (next tensor offset)

        // Shard covering blk.1 + output
        let plan2 = ChunkPlan {
            shard_index: 1,
            layer_range: LayerRange { start: 1, end: 1 },
            includes_embedding: false,
            includes_output_head: true,
            tensor_indices: vec![2, 3],
        };
        let (start2, end2) = shard_data_range(&gguf, &plan2, file_len);
        assert_eq!(start2, 2100); // tensor_data_offset + 2000
        assert_eq!(end2, 5000);   // file_len (no tensor after output)
    }
}
