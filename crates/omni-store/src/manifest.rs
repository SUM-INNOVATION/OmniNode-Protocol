//! Model manifest construction, CBOR serialization, and JSON export.
//!
//! A [`ModelManifest`] is the source of truth for a sharded model: it lists
//! every shard's CID, layer range, size, and BLAKE3 hash.  It is written to
//! disk as CBOR alongside the shard files.

use std::path::Path;

use omni_types::model::{ModelManifest, ShardDescriptor};

use crate::chunker::{self, ChunkPlan};
use crate::content_id;
use crate::error::{Result, StoreError};
use crate::gguf::GgufFile;

// ── Build ─────────────────────────────────────────────────────────────────────

/// Build a complete [`ModelManifest`] from a parsed GGUF file and its chunk
/// plans.
///
/// For each [`ChunkPlan`], this function:
/// 1. Computes the shard's byte range within the file data
/// 2. Hashes the shard data (BLAKE3)
/// 3. Derives a CIDv1
/// 4. Assembles a [`ShardDescriptor`]
///
/// `file_data` must be the full GGUF file (typically mmap-ed).
pub fn build_manifest(
    gguf: &GgufFile,
    file_data: &[u8],
    plans: &[ChunkPlan],
) -> Result<ModelManifest> {
    let model_hash = content_id::blake3_hex(file_data);
    let model_name = gguf.model_name().unwrap_or("unknown").to_string();
    let architecture = gguf.architecture().unwrap_or("unknown").to_string();
    let total_layers = gguf.block_count().unwrap_or(0);
    let quantization = gguf
        .file_type()
        .map(|ft| format!("file_type_{ft}"))
        .unwrap_or_else(|| "unknown".to_string());
    let file_len = file_data.len() as u64;

    let mut shards = Vec::with_capacity(plans.len());
    for plan in plans {
        let (abs_start, abs_end) =
            chunker::shard_data_range(gguf, plan, file_len);
        let shard_bytes = &file_data[abs_start as usize..abs_end as usize];
        let blake3_hash = content_id::blake3_hex(shard_bytes);
        let cid = content_id::cid_from_data(shard_bytes);

        shards.push(ShardDescriptor {
            shard_index: plan.shard_index,
            cid,
            layer_range: plan.layer_range,
            includes_embedding: plan.includes_embedding,
            includes_output_head: plan.includes_output_head,
            size_bytes: shard_bytes.len() as u64,
            blake3_hash,
            // Phase 5: SNIP V2 reference is populated in a later stage; not
            // produced by `build_manifest` itself.
            snip_v2: None,
        });
    }

    Ok(ModelManifest {
        model_name,
        model_hash,
        architecture,
        total_layers,
        quantization,
        total_size_bytes: file_len,
        gguf_version: gguf.header.version,
        shards,
        // Phase 5: filled in by a later SNIP V2 publishing step, not here.
        snip_v2: None,
    })
}

// ── Serialization ─────────────────────────────────────────────────────────────

/// Serialize a manifest to CBOR and write it to `path`.
pub fn write_manifest(manifest: &ModelManifest, path: &Path) -> Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    ciborium::ser::into_writer(manifest, &mut buf)
        .map_err(|e| StoreError::Other(format!("CBOR serialization: {e}")))?;
    std::fs::write(path, &buf)?;
    Ok(())
}

/// Read a manifest from a CBOR file.
pub fn read_manifest(path: &Path) -> Result<ModelManifest> {
    let data = std::fs::read(path)?;
    ciborium::de::from_reader(&data[..])
        .map_err(|e| StoreError::Other(format!("CBOR deserialization: {e}")))
}

/// Pretty-print a manifest as JSON (useful for debugging / inspection).
pub fn manifest_to_json(manifest: &ModelManifest) -> Result<String> {
    serde_json::to_string_pretty(manifest)
        .map_err(|e| StoreError::Other(format!("JSON serialization: {e}")))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use omni_types::model::LayerRange;

    fn sample_manifest() -> ModelManifest {
        ModelManifest {
            model_name: "test-model".into(),
            model_hash: "a".repeat(64),
            architecture: "llama".into(),
            total_layers: 32,
            quantization: "Q4_K_M".into(),
            total_size_bytes: 4_000_000_000,
            gguf_version: 3,
            shards: vec![ShardDescriptor {
                shard_index: 0,
                cid: "bafktest".into(),
                layer_range: LayerRange { start: 0, end: 3 },
                includes_embedding: true,
                includes_output_head: false,
                size_bytes: 500_000_000,
                blake3_hash: "b".repeat(64),
                snip_v2: None,
            }],
            snip_v2: None,
        }
    }

    #[test]
    fn cbor_round_trip() {
        let manifest = sample_manifest();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.cbor");

        write_manifest(&manifest, &path).unwrap();
        let loaded = read_manifest(&path).unwrap();

        assert_eq!(loaded.model_name, "test-model");
        assert_eq!(loaded.architecture, "llama");
        assert_eq!(loaded.total_layers, 32);
        assert_eq!(loaded.shards.len(), 1);
        assert_eq!(loaded.shards[0].cid, "bafktest");
        assert!(loaded.shards[0].includes_embedding);
    }

    #[test]
    fn json_export() {
        let manifest = sample_manifest();
        let json = manifest_to_json(&manifest).unwrap();
        assert!(json.contains("\"model_name\": \"test-model\""));
        assert!(json.contains("\"architecture\": \"llama\""));
        assert!(json.contains("\"total_layers\": 32"));
    }

    #[test]
    fn build_manifest_from_synthetic_gguf() {
        use crate::chunker::plan_chunks;

        // Helper to make GGUF string
        fn ws(buf: &mut Vec<u8>, s: &str) {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }

        // ── Build a minimal GGUF binary with 4 blocks ───────────────────
        let mut raw = Vec::new();
        raw.extend_from_slice(b"GGUF");
        raw.extend_from_slice(&3u32.to_le_bytes());
        // 2 tensors: blk.0, blk.1 (simplified — 1 tensor per block)
        raw.extend_from_slice(&2u64.to_le_bytes());
        // 1 metadata KV
        raw.extend_from_slice(&1u64.to_le_bytes());

        // KV: general.architecture = "llama"
        ws(&mut raw, "general.architecture");
        raw.extend_from_slice(&8u32.to_le_bytes());
        ws(&mut raw, "llama");

        // Tensor 0: blk.0.attn_q.weight
        ws(&mut raw, "blk.0.attn_q.weight");
        raw.extend_from_slice(&1u32.to_le_bytes()); // 1D
        raw.extend_from_slice(&128u64.to_le_bytes());
        raw.extend_from_slice(&1u32.to_le_bytes()); // F16
        raw.extend_from_slice(&0u64.to_le_bytes());

        // Tensor 1: blk.1.attn_q.weight
        ws(&mut raw, "blk.1.attn_q.weight");
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&128u64.to_le_bytes());
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&64u64.to_le_bytes()); // offset 64

        // Pad to 32-byte alignment
        while raw.len() % 32 != 0 {
            raw.push(0);
        }

        // Tensor data: 128 bytes (64 per tensor)
        raw.extend_from_slice(&[0xAA; 64]);
        raw.extend_from_slice(&[0xBB; 64]);

        // ── Parse and build manifest ────────────────────────────────────
        let gguf = crate::gguf::parse_gguf(&raw).unwrap();
        let plans = plan_chunks(&gguf, 1).unwrap(); // 1 block per shard
        assert_eq!(plans.len(), 2);

        let manifest = build_manifest(&gguf, &raw, &plans).unwrap();
        assert_eq!(manifest.architecture, "llama");
        assert_eq!(manifest.gguf_version, 3);
        assert_eq!(manifest.shards.len(), 2);

        // Each shard should have a unique CID
        assert_ne!(manifest.shards[0].cid, manifest.shards[1].cid);
        // Hash lengths are correct
        assert_eq!(manifest.shards[0].blake3_hash.len(), 64);
        assert_eq!(manifest.shards[1].blake3_hash.len(), 64);
        // Sizes are non-zero
        assert!(manifest.shards[0].size_bytes > 0);
        assert!(manifest.shards[1].size_bytes > 0);
    }

    // ── Phase 5 backward-compatibility (CBOR) ────────────────────────────

    /// A CBOR manifest written by pre-Phase-5 code (no `snip_v2` keys
    /// anywhere) must round-trip cleanly through `read_manifest`, with
    /// both optional `snip_v2` fields resolved to `None`.
    #[test]
    fn cbor_old_manifest_decodes_into_new_struct() {
        use serde::Serialize;

        // Mirror of the pre-Phase-5 struct shape. Used only here, to
        // produce bytes representative of an old on-disk manifest.
        #[derive(Serialize)]
        struct LegacyShard {
            shard_index: u32,
            cid: String,
            layer_range: LayerRange,
            includes_embedding: bool,
            includes_output_head: bool,
            size_bytes: u64,
            blake3_hash: String,
        }
        #[derive(Serialize)]
        struct LegacyManifest {
            model_name: String,
            model_hash: String,
            architecture: String,
            total_layers: u32,
            quantization: String,
            total_size_bytes: u64,
            gguf_version: u32,
            shards: Vec<LegacyShard>,
        }

        let legacy = LegacyManifest {
            model_name: "tinyllama".into(),
            model_hash: "a".repeat(64),
            architecture: "llama".into(),
            total_layers: 22,
            quantization: "F16".into(),
            total_size_bytes: 2_200_000_000,
            gguf_version: 3,
            shards: vec![LegacyShard {
                shard_index: 0,
                cid: "bafkrlegacy".into(),
                layer_range: LayerRange { start: 0, end: 10 },
                includes_embedding: true,
                includes_output_head: false,
                size_bytes: 215_000_000,
                blake3_hash: "b".repeat(64),
            }],
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy_manifest.cbor");
        let mut buf: Vec<u8> = Vec::new();
        ciborium::ser::into_writer(&legacy, &mut buf).unwrap();
        std::fs::write(&path, &buf).unwrap();

        let loaded = read_manifest(&path).unwrap();
        assert!(loaded.snip_v2.is_none());
        assert_eq!(loaded.shards.len(), 1);
        assert!(loaded.shards[0].snip_v2.is_none());
        assert_eq!(loaded.model_name, "tinyllama");
        assert_eq!(loaded.architecture, "llama");
        assert_eq!(loaded.total_layers, 22);
        assert_eq!(loaded.shards[0].cid, "bafkrlegacy");
        assert_eq!(loaded.shards[0].size_bytes, 215_000_000);
    }

    /// A manifest produced today with `Some(SnipV2ObjectRef)` populated on
    /// both struct and shard must survive a CBOR round trip via the
    /// existing `write_manifest`/`read_manifest` path.
    #[test]
    fn cbor_round_trip_new_manifest_with_snip_field() {
        use omni_types::phase5::{SnipV2Lifecycle, SnipV2ObjectId, SnipV2ObjectRef};

        let hex =
            "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let id = SnipV2ObjectId::from_hex(hex).unwrap();

        let manifest = ModelManifest {
            model_name: "test-model".into(),
            model_hash: "a".repeat(64),
            architecture: "llama".into(),
            total_layers: 4,
            quantization: "F16".into(),
            total_size_bytes: 1_000,
            gguf_version: 3,
            shards: vec![ShardDescriptor {
                shard_index: 0,
                cid: "bafknew".into(),
                layer_range: LayerRange { start: 0, end: 3 },
                includes_embedding: true,
                includes_output_head: true,
                size_bytes: 1_000,
                blake3_hash: "b".repeat(64),
                snip_v2: Some(SnipV2ObjectRef {
                    merkle_root: id,
                    lifecycle: SnipV2Lifecycle::Active,
                    plaintext_size_bytes: Some(1_000),
                }),
            }],
            snip_v2: Some(SnipV2ObjectRef {
                merkle_root: id,
                lifecycle: SnipV2Lifecycle::Active,
                plaintext_size_bytes: None,
            }),
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest_with_snip.cbor");
        write_manifest(&manifest, &path).unwrap();
        let loaded = read_manifest(&path).unwrap();

        assert!(loaded.snip_v2.is_some());
        let top = loaded.snip_v2.as_ref().unwrap();
        assert_eq!(top.merkle_root, id);
        assert_eq!(top.lifecycle, SnipV2Lifecycle::Active);
        assert_eq!(top.plaintext_size_bytes, None);

        let shard_ref = loaded.shards[0].snip_v2.as_ref().unwrap();
        assert_eq!(shard_ref.merkle_root, id);
        assert_eq!(shard_ref.lifecycle, SnipV2Lifecycle::Active);
        assert_eq!(shard_ref.plaintext_size_bytes, Some(1_000));
    }
}
