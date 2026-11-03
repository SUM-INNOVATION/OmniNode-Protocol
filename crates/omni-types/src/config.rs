// Global configuration structs. Populated as each phase adds settings.

use std::path::PathBuf;

// ── Phase 1: Networking ───────────────────────────────────────────────────────

/// Default ceiling on the total bytes accepted for one shard fetch.
///
/// 1 GiB — a conservative compatibility limit, not a measured one. It clears
/// the ~510 MB shard this repository documents by a wide margin while refusing
/// the multi-gigabyte reservation a hostile peer would otherwise obtain.
/// Operators may lower it.
pub const DEFAULT_MAX_SHARD_TOTAL_BYTES: u64 = 1024 * 1024 * 1024;

/// Default per-message ceiling for a shard range request.
pub const DEFAULT_MAX_SHARD_MSG_BYTES: usize = 64 * 1024 * 1024;

/// Default number of shard fetches a store runs at once (issue #132).
///
/// 4 bounds a store's peer-to-peer fetch staging at 4 ×
/// [`DEFAULT_MAX_SHARD_TOTAL_BYTES`] (4 GiB at the defaults). This constant is
/// the only place the value is written; [`StoreConfig::default`] and the
/// constructors that take no configuration all read it.
pub const DEFAULT_MAX_CONCURRENT_FETCHES: usize = 4;

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
    /// Whether this node runs mDNS LAN discovery. Default `true`.
    ///
    /// With it on, every node on the local network that answers mDNS is
    /// discovered, added as a gossipsub explicit peer and dialled. A node
    /// whose peers are reached only by explicit dial or bootstrap peers can
    /// turn it off; test harnesses do, so that concurrently running nodes on
    /// one host do not connect to each other. `false` constructs no mDNS
    /// behaviour at all: the node neither announces itself nor discovers
    /// peers by multicast. The CLI sets it with `--mdns on|off` (issue #158).
    pub mdns: bool,

    /// Outbound address allow-list (issue #158). Empty (the default) means no
    /// restriction, as before.
    ///
    /// When set, every address the node dials must match an entry, whichever
    /// part of the node asked for the dial (`--peer`, mDNS, Identify,
    /// Kademlia, AutoNAT, relay, DCUtR); anything else is refused before a
    /// packet is sent. Refused addresses are kept out of the peer stores (so
    /// they are not redialled), and an inbound connection is kept only from an
    /// allowed address (a relayed one only through an allowed relay). Each
    /// entry is an IP address or CIDR (`127.0.0.1`, `10.0.0.0/8`, `::1`) or a
    /// multiaddr prefix (`/ip4/127.0.0.1/udp/4001/quic-v1`, optionally ending
    /// in `/p2p/<peer-id>`). The CLI sets it with `--outbound-allow`.
    ///
    /// Parsed and enforced by `omni-net` (`omni_net::policy`), which refuses
    /// a malformed entry when the swarm is built, logs the effective policy
    /// and counts refusals by source. An application-level routing control for
    /// cooperative networks, not an operating-system security boundary: a
    /// modified client can ignore it.
    pub outbound_allow: Vec<String>,

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
            mdns: true,
            outbound_allow: Vec::new(),
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

    /// Hard ceiling on the TOTAL bytes accepted for one shard fetch,
    /// regardless of what a serving peer claims.
    ///
    /// Distinct from `max_shard_msg_bytes`, which bounds a single
    /// request-response message: a shard is fetched in several messages, so
    /// the per-message bound does not constrain the accumulated total.
    ///
    /// It bounds *one* fetch, and since issue #132 it bounds **disk**, not
    /// memory. A shard download streams each message into a staging file
    /// under the store root and hashes it as it arrives, so a fetch holds at
    /// most one message (`max_shard_msg_bytes`) in memory, and stages at most
    /// this many bytes on disk until the CID is verified and the shard is
    /// published. [`StoreConfig::max_concurrent_fetches`] bounds how many
    /// fetches run at once, so a store's peer-to-peer staging is at most
    /// `max_concurrent_fetches × max_shard_total_bytes`.
    ///
    /// Unlike SNIP — where a chunk is protocol-fixed at 1 MiB and the bound
    /// follows from the format — an Omni shard is a model slice whose size
    /// depends on the GGUF tensor layout and `layers_per_shard`, so no
    /// format-derived ceiling exists. Operators may lower this.
    pub max_shard_total_bytes: u64,

    /// How many shard fetches the store runs at once (issue #132). Must be
    /// at least 1. Default: [`DEFAULT_MAX_CONCURRENT_FETCHES`] (4).
    ///
    /// A fetch started while this many are in flight waits in a first-in,
    /// first-out queue and starts when a slot frees; it is not refused. A
    /// fetch that completes, fails, is cancelled or is dropped frees its
    /// slot. `1` runs fetches one after another.
    ///
    /// The bound applies to each of the store's two download paths:
    /// - peer-to-peer fetches (`FetchManager`), whose staging is therefore at
    ///   most `max_concurrent_fetches × max_shard_total_bytes`;
    /// - SNIP V2 shard downloads (`restore_from_snip`, and so `fetch_set`),
    ///   shared by every thread that uses the same store.
    ///
    /// The two paths do not share one budget, so a process that runs both on
    /// one store at the same time can have twice this many downloads staged.
    pub max_concurrent_fetches: usize,
}

/// The single validation entry point for [`StoreConfig`].
///
/// Both the Rust constructors and the Python `StoreConfig` binding call this,
/// so the two cannot drift apart and a Python caller that changes only one
/// field still gets the whole configuration checked.
///
/// Deliberately **not** requiring `max_shard_total_bytes >= max_shard_msg_bytes`:
/// the message limit sizes the window a fetcher *asks* for, and a small shard
/// may legitimately be shorter than that window. A total below the message
/// size is unusual, not invalid.
impl StoreConfig {
    pub fn validate(&self) -> Result<(), String> {
        validate_shard_limits(self.max_shard_msg_bytes, self.max_shard_total_bytes)?;
        validate_max_concurrent_fetches(self.max_concurrent_fetches)
    }
}

/// The concurrency rule on its own: at least one fetch must be able to run.
///
/// `FetchManager` and `ShardStore` validate through this, so neither can be
/// built with a bound a [`StoreConfig`] would have rejected.
pub fn validate_max_concurrent_fetches(max_concurrent_fetches: usize) -> Result<(), String> {
    if max_concurrent_fetches == 0 {
        return Err("max_concurrent_fetches must be greater than zero".to_string());
    }
    Ok(())
}

/// The limit rules themselves, callable without a whole [`StoreConfig`].
///
/// `FetchManager` validates through this, so a fetcher constructed directly
/// cannot hold limits a `StoreConfig` would have rejected.
pub fn validate_shard_limits(
    max_shard_msg_bytes: usize,
    max_shard_total_bytes: u64,
) -> Result<(), String> {
    if max_shard_msg_bytes == 0 {
        return Err("max_shard_msg_bytes must be greater than zero".to_string());
    }
    if max_shard_total_bytes == 0 {
        return Err("max_shard_total_bytes must be greater than zero".to_string());
    }
    Ok(())
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
            max_shard_msg_bytes: DEFAULT_MAX_SHARD_MSG_BYTES,
            max_shard_total_bytes: DEFAULT_MAX_SHARD_TOTAL_BYTES,
            max_concurrent_fetches: DEFAULT_MAX_CONCURRENT_FETCHES,
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

    /// Issue #158: the defaults keep mDNS on and set no outbound restriction.
    #[test]
    fn net_config_defaults_keep_mdns_on_and_set_no_allow_list() {
        let cfg = NetConfig::default();
        assert!(cfg.mdns, "mDNS stays on unless explicitly turned off");
        assert!(
            cfg.outbound_allow.is_empty(),
            "no outbound restriction by default"
        );
    }

    #[test]
    fn store_config_defaults() {
        let cfg = StoreConfig::default();
        assert_eq!(cfg.layers_per_shard, 4);
        assert_eq!(cfg.max_shard_msg_bytes, 64 * 1024 * 1024);
        assert!(cfg.store_dir.ends_with("store"));
    }

    /// Issue #132: the fetch-concurrency bound defaults to 4, from the one
    /// constant, and the default configuration validates.
    #[test]
    fn the_fetch_concurrency_default_is_four_and_validates() {
        assert_eq!(DEFAULT_MAX_CONCURRENT_FETCHES, 4);
        let cfg = StoreConfig::default();
        assert_eq!(cfg.max_concurrent_fetches, DEFAULT_MAX_CONCURRENT_FETCHES);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn a_zero_fetch_concurrency_is_refused_and_one_is_allowed() {
        let mut cfg = StoreConfig {
            max_concurrent_fetches: 0,
            ..StoreConfig::default()
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("max_concurrent_fetches"), "{err}");
        assert!(validate_max_concurrent_fetches(0).is_err());

        cfg.max_concurrent_fetches = 1;
        assert!(cfg.validate().is_ok());
        cfg.max_concurrent_fetches = 64;
        assert!(cfg.validate().is_ok());
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
