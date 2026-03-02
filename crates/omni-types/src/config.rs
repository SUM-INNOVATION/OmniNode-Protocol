// Global configuration structs. Populated as each phase adds settings.

use std::path::PathBuf;

// ── Phase 1: Networking ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NetConfig {
    /// UDP port for QUIC listener. 0 = OS-assigned.
    pub listen_port: u16,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self { listen_port: 0 }
    }
}

// ── Phase 2: Storage ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Root directory for shard files and manifests.
    /// Defaults to `$HOME/.omninode/store/`.
    pub store_dir: PathBuf,

    /// Number of consecutive transformer blocks per shard.
    /// Default: 4 (yields ~8 shards for a 32-layer model).
    pub layers_per_shard: u32,

    /// Maximum bytes per request-response shard transfer message.
    /// Shards larger than this are fetched in multiple round-trips.
    /// Default: 64 MiB.
    pub max_shard_msg_bytes: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        let store_dir = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/tmp/omninode"))
            .join(".omninode")
            .join("store");

        Self {
            store_dir,
            layers_per_shard: 4,
            max_shard_msg_bytes: 64 * 1024 * 1024,
        }
    }
}

// ── Phase 4: Pipeline ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of micro-batches for GPipe scheduling.
    /// `None` = auto (`2 × num_stages`).
    pub num_micro_batches: Option<u32>,

    /// Maximum sequence length for hidden state tensors.
    pub max_seq_len: u32,

    /// Heartbeat interval in seconds.
    pub heartbeat_interval_secs: u64,

    /// Number of missed heartbeats before declaring a stage dead.
    pub heartbeat_timeout_factor: u32,

    /// Timeout for tensor transfer requests in seconds.
    pub tensor_timeout_secs: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_micro_batches: None,
            max_seq_len: 2048,
            heartbeat_interval_secs: 3,
            heartbeat_timeout_factor: 3,
            tensor_timeout_secs: 60,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_config_defaults() {
        let cfg = StoreConfig::default();
        assert_eq!(cfg.layers_per_shard, 4);
        assert_eq!(cfg.max_shard_msg_bytes, 64 * 1024 * 1024);
        assert!(cfg.store_dir.ends_with("store"));
    }

    #[test]
    fn pipeline_config_defaults() {
        let cfg = PipelineConfig::default();
        assert!(cfg.num_micro_batches.is_none());
        assert_eq!(cfg.max_seq_len, 2048);
        assert_eq!(cfg.heartbeat_interval_secs, 3);
        assert_eq!(cfg.heartbeat_timeout_factor, 3);
        assert_eq!(cfg.tensor_timeout_secs, 60);
    }
}
