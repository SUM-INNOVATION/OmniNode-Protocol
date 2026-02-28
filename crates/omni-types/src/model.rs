// Phase 2 — shared model types for GGUF sharding.
//
// Consumed by `omni-store`, `omni-pipeline`, and `omni-node`.

use serde::{Deserialize, Serialize};

// ── Layer Range ───────────────────────────────────────────────────────────────

/// Inclusive range of transformer block indices assigned to a shard.
///
/// `LayerRange { start: 0, end: 3 }` covers blocks 0, 1, 2, 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn len(&self) -> u32 {
        self.end.saturating_sub(self.start) + 1
    }

    pub fn is_empty(&self) -> bool {
        self.start > self.end
    }
}

// ── Shard Descriptor ──────────────────────────────────────────────────────────

/// Descriptor for a single shard (chunk) of a sharded model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardDescriptor {
    /// Zero-based index of this shard within the model.
    pub shard_index: u32,
    /// CIDv1 string (BLAKE3, raw codec).
    pub cid: String,
    /// Which transformer blocks this shard contains.
    pub layer_range: LayerRange,
    /// `true` if this shard includes the token embedding layer.
    pub includes_embedding: bool,
    /// `true` if this shard includes the output head (lm_head / output_norm).
    pub includes_output_head: bool,
    /// Size of the shard in bytes.
    pub size_bytes: u64,
    /// BLAKE3 hash of the shard data (hex-encoded, 64 chars).
    pub blake3_hash: String,
}

// ── Model Manifest ────────────────────────────────────────────────────────────

/// Complete manifest describing a sharded model.
///
/// Serialized as CBOR for on-disk storage; JSON for debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Human-readable model identifier (e.g. `"llama-7b-q4_k_m"`).
    pub model_name: String,
    /// BLAKE3 hash of the entire original GGUF file (hex).
    pub model_hash: String,
    /// Architecture identifier from GGUF metadata (e.g. `"llama"`).
    pub architecture: String,
    /// Total number of transformer blocks.
    pub total_layers: u32,
    /// Quantization type string (e.g. `"Q4_K_M"`).
    pub quantization: String,
    /// Total size of the original GGUF file in bytes.
    pub total_size_bytes: u64,
    /// GGUF format version parsed from the file header.
    pub gguf_version: u32,
    /// Ordered list of shard descriptors.
    pub shards: Vec<ShardDescriptor>,
}

// ── GGML Quantization Types ───────────────────────────────────────────────────

/// GGML tensor quantization types.
///
/// Numeric discriminants match the GGUF specification exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum GgmlType {
    F32    = 0,
    F16    = 1,
    Q4_0   = 2,
    Q4_1   = 3,
    Q5_0   = 6,
    Q5_1   = 7,
    Q8_0   = 8,
    Q8_1   = 9,
    Q2K    = 10,
    Q3K    = 11,
    Q4K    = 12,
    Q5K    = 13,
    Q6K    = 14,
    Q8K    = 15,
    IQ2XXS = 16,
    IQ2XS  = 17,
    IQ3XXS = 18,
    IQ1S   = 19,
    IQ4NL  = 20,
    IQ3S   = 21,
    IQ2S   = 22,
    IQ4XS  = 23,
    I8     = 24,
    I16    = 25,
    I32    = 26,
    I64    = 27,
    F64    = 28,
    IQ1M   = 29,
    BF16   = 30,
    TQ1_0  = 34,
    TQ2_0  = 35,
}

impl GgmlType {
    /// Convert a raw `u32` from a GGUF tensor info entry into a [`GgmlType`].
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0  => Some(Self::F32),
            1  => Some(Self::F16),
            2  => Some(Self::Q4_0),
            3  => Some(Self::Q4_1),
            6  => Some(Self::Q5_0),
            7  => Some(Self::Q5_1),
            8  => Some(Self::Q8_0),
            9  => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::IQ2XXS),
            17 => Some(Self::IQ2XS),
            18 => Some(Self::IQ3XXS),
            19 => Some(Self::IQ1S),
            20 => Some(Self::IQ4NL),
            21 => Some(Self::IQ3S),
            22 => Some(Self::IQ2S),
            23 => Some(Self::IQ4XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1M),
            30 => Some(Self::BF16),
            34 => Some(Self::TQ1_0),
            35 => Some(Self::TQ2_0),
            _  => None,
        }
    }

    /// Block size for this quantization type. Non-quantized types return 1.
    pub fn block_size(&self) -> u32 {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64
            | Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::TQ1_0 | Self::TQ2_0 => 256,
            _ => 32,
        }
    }
}

impl std::fmt::Display for GgmlType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_range_len() {
        let r = LayerRange { start: 0, end: 3 };
        assert_eq!(r.len(), 4);
        assert!(!r.is_empty());
    }

    #[test]
    fn ggml_type_round_trip() {
        for id in 0..=35u32 {
            if let Some(t) = GgmlType::from_u32(id) {
                assert_eq!(t as u32, id);
            }
        }
    }

    #[test]
    fn shard_descriptor_serde_round_trip() {
        let sd = ShardDescriptor {
            shard_index: 0,
            cid: "bafkr4itest".into(),
            layer_range: LayerRange { start: 0, end: 3 },
            includes_embedding: true,
            includes_output_head: false,
            size_bytes: 512_000_000,
            blake3_hash: "a".repeat(64),
        };
        let json = serde_json::to_string(&sd).unwrap();
        let round: ShardDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(sd, round);
    }

    #[test]
    fn model_manifest_serde_round_trip() {
        let m = ModelManifest {
            model_name: "test-model".into(),
            model_hash: "b".repeat(64),
            architecture: "llama".into(),
            total_layers: 32,
            quantization: "Q4_K_M".into(),
            total_size_bytes: 4_000_000_000,
            gguf_version: 3,
            shards: vec![],
        };
        let json = serde_json::to_string(&m).unwrap();
        let round: ModelManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(m.model_name, round.model_name);
        assert_eq!(m.total_layers, round.total_layers);
    }
}
