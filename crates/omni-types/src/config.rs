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
}
