// Global configuration structs. Populated as each phase adds settings.

use std::path::PathBuf;

// ── Phase 1: Networking ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NetConfig {
    /// UDP port for QUIC listener. 0 = OS-assigned.
    pub listen_port: u16,

    /// Kademlia bootstrap peers for WAN discovery.
    /// Format: `/ip4/<IP>/udp/<PORT>/quic-v1/p2p/<PEER_ID>`
    /// Empty = LAN-only mode (mDNS discovery only).
    pub bootstrap_peers: Vec<String>,

    /// Whether this node volunteers as a Circuit Relay v2 server.
    /// Enable on nodes with open NATs (VPS, port-forwarded home server).
    pub relay_server: bool,

    /// Stage 12.6 — libp2p mesh identity policy.
    ///
    /// `Ephemeral` (default) preserves pre-12.6 behavior: a fresh
    /// Ed25519 keypair is generated on every `OmniNet::new`, so
    /// `local_peer_id()` changes across process restart. Stage 12.5
    /// peer advertisements published before a restart become
    /// useless.
    ///
    /// `KeypairProtobufBytes(_)` carries an existing libp2p
    /// keypair encoded via `libp2p_identity::Keypair::to_protobuf_encoding`.
    /// The swarm builder decodes once at construction; subsequent
    /// `OmniNet::new` calls with the same bytes yield the same
    /// `local_peer_id()`, so peer advertisements remain valid for
    /// their full ≤24h freshness window after a restart.
    ///
    /// CLI operators don't construct `NetIdentity` directly — they
    /// pass `--net-identity-file <path>` and the CLI calls the
    /// `omni-net::identity` helper (auto-creates the file at 0600
    /// on Unix; refuses to silently fall back to ephemeral when an
    /// existing file is malformed).
    pub identity: NetIdentity,
}

#[derive(Clone)]
pub enum NetIdentity {
    /// Pre-12.6 default. Fresh keypair every `OmniNet::new`.
    Ephemeral,
    /// libp2p protobuf-encoded keypair bytes. Decoded at
    /// swarm-build time via `omni_net::identity::decode_keypair_protobuf`.
    KeypairProtobufBytes(Vec<u8>),
}

/// Manual `Debug` impl so that printing a `NetConfig` (which
/// derives `Debug`) cannot leak the libp2p private keypair bytes.
/// The variant tag + byte length are still surfaced so operators
/// can confirm the variant in logs.
impl std::fmt::Debug for NetIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetIdentity::Ephemeral => write!(f, "Ephemeral"),
            NetIdentity::KeypairProtobufBytes(bytes) => f
                .debug_struct("KeypairProtobufBytes")
                .field("bytes", &"<redacted>")
                .field("len", &bytes.len())
                .finish(),
        }
    }
}

impl Default for NetIdentity {
    fn default() -> Self {
        NetIdentity::Ephemeral
    }
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            listen_port: 0,
            bootstrap_peers: vec![],
            relay_server: false,
            identity: NetIdentity::Ephemeral,
        }
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
