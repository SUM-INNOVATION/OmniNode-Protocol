pub mod announce;
pub mod chunker;
pub mod content_id;
pub mod error;
pub mod fetch;
pub mod gguf;
pub mod manifest;
pub mod mmap;
pub mod serve;
pub mod store;
pub mod verify;

pub use announce::{ShardAnnouncement, decode_announcement, encode_announcement};
pub use content_id::cid_from_data;
pub use error::StoreError;
pub use fetch::{FetchManager, FetchOutcome};
pub use gguf::GgufFile;
pub use store::ShardStore;
pub use chunker::ChunkPlan;

use std::path::Path;

use omni_net::{OmniNet, TOPIC_SHARD};
use omni_types::config::StoreConfig;
use omni_types::model::ModelManifest;
use tracing::info;

use crate::error::Result;

/// Top-level API for the OmniNode shard storage layer.
pub struct OmniStore {
    pub config: StoreConfig,
    pub local: ShardStore,
    pub fetcher: FetchManager,
}

impl OmniStore {
    /// Open (or create) the shard store from the given config.
    pub fn new(config: StoreConfig) -> Result<Self> {
        let local = ShardStore::new(config.store_dir.clone())?;
        let fetcher = FetchManager::new(config.max_shard_msg_bytes);
        Ok(Self { config, local, fetcher })
    }

    /// Ingest a GGUF model file: parse, chunk, compute CIDs, store shards,
    /// and build a manifest.
    pub fn ingest_model(&self, path: &Path) -> Result<ModelManifest> {
        let mapped = mmap::mmap_file(path)?;
        let gguf = gguf::parse_gguf(&mapped)?;
        let plans = chunker::plan_chunks(&gguf, self.config.layers_per_shard)?;

        info!(
            path = %path.display(),
            shards = plans.len(),
            "ingesting model"
        );

        // Build manifest (computes CIDs and hashes).
        let man = manifest::build_manifest(&gguf, &mapped, &plans)?;

        // Write each shard to disk.
        let file_len = mapped.len() as u64;
        for (plan, desc) in plans.iter().zip(man.shards.iter()) {
            if self.local.has(&desc.cid) {
                info!(cid = %desc.cid, "shard already exists — skipping");
                continue;
            }
            let (abs_start, abs_end) = chunker::shard_data_range(&gguf, plan, file_len);
            let shard_bytes = &mapped[abs_start as usize..abs_end as usize];
            self.local.put(&desc.cid, shard_bytes)?;
            info!(
                cid = %desc.cid,
                shard = desc.shard_index,
                bytes = shard_bytes.len(),
                "shard written"
            );
        }

        // Write manifest to disk.
        let manifest_path = self.config.store_dir.join("manifest.cbor");
        manifest::write_manifest(&man, &manifest_path)?;
        info!(path = %manifest_path.display(), "manifest written");

        Ok(man)
    }

    /// Announce all shards in a manifest via Gossipsub.
    pub async fn announce_shards(
        &self,
        net: &OmniNet,
        manifest: &ModelManifest,
    ) -> Result<()> {
        for desc in &manifest.shards {
            let ann = ShardAnnouncement {
                cid: desc.cid.clone(),
                model_name: manifest.model_name.clone(),
                layer_range: desc.layer_range,
                size_bytes: desc.size_bytes,
            };
            let bytes = encode_announcement(&ann);
            net.publish(TOPIC_SHARD, bytes)
                .await
                .map_err(|e| StoreError::Other(e.to_string()))?;
            info!(cid = %desc.cid, "announced shard");
        }
        Ok(())
    }

    /// Check whether a shard exists locally.
    pub fn has_shard(&self, cid: &str) -> bool {
        self.local.has(cid)
    }

    /// Memory-map a local shard for zero-copy read access.
    pub fn mmap_shard(&self, cid: &str) -> Result<memmap2::Mmap> {
        self.local.mmap(cid)
    }
}
