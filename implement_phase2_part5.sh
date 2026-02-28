#!/usr/bin/env bash
set -euo pipefail

# ── Step 8+9: Network Integration + CLI ──────────────────────────────────────

# 1. Update omni-store/Cargo.toml to add omni-net and bincode
cat > "crates/omni-store/Cargo.toml" << 'EOF'
[package]
name             = "omni-store"
version.workspace    = true
edition.workspace    = true
license.workspace    = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
omni-types = { workspace = true }
omni-net   = { workspace = true }
memmap2    = { workspace = true }
blake3     = { workspace = true }
cid        = { workspace = true }
multihash  = { workspace = true }
thiserror  = { workspace = true }
tracing    = { workspace = true }
serde      = { workspace = true }
serde_json = { workspace = true }
ciborium   = { workspace = true }
bincode    = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
EOF

# 2. Create announce.rs
cat > "crates/omni-store/src/announce.rs" << 'EOF'
//! Gossipsub shard availability announcements.
//!
//! When a node ingests a model it publishes one [`ShardAnnouncement`] per shard
//! on the `omni/shard/v1` topic so that other nodes learn which peer holds
//! which CID.

use serde::{Deserialize, Serialize};

use omni_types::model::LayerRange;

/// Message published on `omni/shard/v1` after ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAnnouncement {
    /// CIDv1 identifying this shard.
    pub cid: String,
    /// Human-readable model name.
    pub model_name: String,
    /// Which transformer blocks this shard covers.
    pub layer_range: LayerRange,
    /// Size of the shard data in bytes.
    pub size_bytes: u64,
}

/// Encode a [`ShardAnnouncement`] to bincode bytes for Gossipsub publishing.
pub fn encode_announcement(ann: &ShardAnnouncement) -> Vec<u8> {
    bincode::serde::encode_to_vec(ann, bincode::config::standard())
        .expect("ShardAnnouncement serialization is infallible")
}

/// Decode a [`ShardAnnouncement`] from bincode bytes received via Gossipsub.
pub fn decode_announcement(data: &[u8]) -> Option<ShardAnnouncement> {
    bincode::serde::decode_from_slice(data, bincode::config::standard())
        .ok()
        .map(|(ann, _)| ann)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn announcement_round_trip() {
        let ann = ShardAnnouncement {
            cid: "bafkr4itest".into(),
            model_name: "llama-7b".into(),
            layer_range: LayerRange { start: 0, end: 3 },
            size_bytes: 500_000_000,
        };
        let bytes = encode_announcement(&ann);
        let decoded = decode_announcement(&bytes).unwrap();
        assert_eq!(decoded.cid, ann.cid);
        assert_eq!(decoded.model_name, ann.model_name);
        assert_eq!(decoded.size_bytes, ann.size_bytes);
    }
}
EOF

# 3. Create fetch.rs
cat > "crates/omni-store/src/fetch.rs" << 'EOF'
//! Outbound shard fetch orchestration.
//!
//! [`FetchManager`] tracks in-progress fetches.  Each fetch proceeds as a
//! sequence of 64 MiB windowed request-response round-trips, reassembles the
//! chunks, verifies the CID, and stores the shard to disk.

use std::collections::HashMap;

use libp2p::PeerId;
use omni_net::{OmniNet, ShardResponse};
use tracing::{info, warn};

use crate::error::{Result, StoreError};
use crate::store::ShardStore;
use crate::verify;

/// Tracks in-flight fetches keyed by CID.
pub struct FetchManager {
    /// Maximum bytes per request-response chunk (default 64 MiB).
    chunk_size: u64,
    /// In-progress fetches: CID → state.
    active: HashMap<String, FetchState>,
}

/// State for a single in-progress shard fetch.
struct FetchState {
    peer_id: PeerId,
    /// Total shard size (learned from first response).
    total_bytes: Option<u64>,
    /// Byte offset of the next chunk to request.
    next_offset: u64,
    /// Accumulated data chunks.
    buffer: Vec<u8>,
}

/// Outcome of processing a received chunk.
pub enum FetchOutcome {
    /// More chunks are needed — the next request has been sent.
    InProgress,
    /// The entire shard has been received, verified, and stored.
    Complete { cid: String, size: u64 },
    /// The fetch failed.
    Failed { cid: String, error: String },
}

impl FetchManager {
    /// Create a new fetch manager with the given chunk size.
    pub fn new(max_shard_msg_bytes: usize) -> Self {
        Self {
            chunk_size: max_shard_msg_bytes as u64,
            active: HashMap::new(),
        }
    }

    /// Start fetching a shard from a remote peer.
    ///
    /// Sends the first chunk request.  Subsequent chunks are requested
    /// automatically when [`Self::on_chunk_received`] is called.
    pub async fn start_fetch(
        &mut self,
        net: &OmniNet,
        peer_id: PeerId,
        cid: String,
    ) -> Result<()> {
        if self.active.contains_key(&cid) {
            return Err(StoreError::Other(format!("fetch already in progress: {cid}")));
        }

        info!(%cid, %peer_id, "starting shard fetch");

        net.request_shard_chunk(peer_id, cid.clone(), Some(0), Some(self.chunk_size))
            .await
            .map_err(|e| StoreError::Other(e.to_string()))?;

        self.active.insert(cid, FetchState {
            peer_id,
            total_bytes: None,
            next_offset: 0,
            buffer: Vec::new(),
        });

        Ok(())
    }

    /// Process a received shard chunk.
    ///
    /// Returns a [`FetchOutcome`] indicating whether the fetch is complete,
    /// still in progress, or has failed.
    pub async fn on_chunk_received(
        &mut self,
        net: &OmniNet,
        store: &ShardStore,
        response: &ShardResponse,
    ) -> FetchOutcome {
        // Check for server-side error.
        if let Some(ref err) = response.error {
            self.active.remove(&response.cid);
            return FetchOutcome::Failed {
                cid: response.cid.clone(),
                error: err.clone(),
            };
        }

        let state = match self.active.get_mut(&response.cid) {
            Some(s) => s,
            None => {
                warn!(cid = %response.cid, "received chunk for unknown fetch — ignoring");
                return FetchOutcome::Failed {
                    cid: response.cid.clone(),
                    error: "no active fetch for this CID".into(),
                };
            }
        };

        // Record total size from first response.
        if state.total_bytes.is_none() {
            state.total_bytes = Some(response.total_bytes);
            state.buffer.reserve(response.total_bytes as usize);
        }

        // Append chunk data.
        state.buffer.extend_from_slice(&response.data);
        state.next_offset = response.offset + response.data.len() as u64;

        let total = state.total_bytes.unwrap();
        info!(
            cid = %response.cid,
            received = state.next_offset,
            total,
            "chunk received"
        );

        // Check if we need more chunks.
        if state.next_offset < total {
            let peer_id = state.peer_id;
            let cid = response.cid.clone();
            let offset = state.next_offset;
            let chunk_size = self.chunk_size;

            if let Err(e) = net.request_shard_chunk(
                peer_id,
                cid.clone(),
                Some(offset),
                Some(chunk_size),
            ).await {
                self.active.remove(&cid);
                return FetchOutcome::Failed {
                    cid,
                    error: format!("failed to request next chunk: {e}"),
                };
            }

            return FetchOutcome::InProgress;
        }

        // All chunks received — verify and store.
        let cid = response.cid.clone();
        let state = self.active.remove(&cid).unwrap();

        if let Err(e) = verify::verify_cid(&state.buffer, &cid) {
            return FetchOutcome::Failed {
                cid,
                error: format!("integrity check failed: {e}"),
            };
        }

        let size = state.buffer.len() as u64;
        if let Err(e) = store.put(&cid, &state.buffer) {
            return FetchOutcome::Failed {
                cid,
                error: format!("failed to write shard: {e}"),
            };
        }

        info!(%cid, size, "shard fetch complete and verified");
        FetchOutcome::Complete { cid, size }
    }

    /// Check if a fetch is currently active for the given CID.
    pub fn is_active(&self, cid: &str) -> bool {
        self.active.contains_key(cid)
    }

    /// Number of in-progress fetches.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }
}
EOF

# 4. Create serve.rs
cat > "crates/omni-store/src/serve.rs" << 'EOF'
//! Inbound shard request handler.
//!
//! When a remote peer requests a shard (or sub-chunk), this module reads the
//! shard from the local [`ShardStore`] via mmap, slices the requested byte
//! window, and sends the response through [`OmniNet::respond_shard`].

use omni_net::{OmniNet, ShardRequest, ShardResponse};
use tracing::{info, warn};

use crate::store::ShardStore;

/// Handle an inbound shard request.
///
/// - Reads the shard from `store` via mmap
/// - Slices `[offset .. offset + max_bytes]` (clamped to shard size)
/// - Sends the response via `net.respond_shard(channel_id, ...)`
pub async fn handle_request(
    net: &OmniNet,
    store: &ShardStore,
    request: &ShardRequest,
    channel_id: u64,
) {
    let cid = &request.cid;

    // Check if we have the shard locally.
    if !store.has(cid) {
        warn!(%cid, channel_id, "requested shard not found locally");
        let resp = ShardResponse {
            cid: cid.clone(),
            offset: 0,
            total_bytes: 0,
            data: Vec::new(),
            error: Some(format!("shard not found: {cid}")),
        };
        if let Err(e) = net.respond_shard(channel_id, resp).await {
            warn!(%cid, %e, "failed to send error response");
        }
        return;
    }

    // Memory-map the shard.
    let mapped = match store.mmap(cid) {
        Ok(m) => m,
        Err(e) => {
            warn!(%cid, %e, "failed to mmap shard");
            let resp = ShardResponse {
                cid: cid.clone(),
                offset: 0,
                total_bytes: 0,
                data: Vec::new(),
                error: Some(format!("mmap error: {e}")),
            };
            let _ = net.respond_shard(channel_id, resp).await;
            return;
        }
    };

    let total = mapped.len() as u64;
    let offset = request.offset.unwrap_or(0).min(total);
    let max_bytes = request.max_bytes.unwrap_or(total);
    let end = (offset + max_bytes).min(total);
    let chunk = &mapped[offset as usize..end as usize];

    info!(
        %cid,
        offset,
        chunk_len = chunk.len(),
        total,
        channel_id,
        "serving shard chunk"
    );

    let resp = ShardResponse {
        cid: cid.clone(),
        offset,
        total_bytes: total,
        data: chunk.to_vec(),
        error: None,
    };

    if let Err(e) = net.respond_shard(channel_id, resp).await {
        warn!(%cid, %e, "failed to send shard response");
    }
}
EOF

# 5. Overwrite omni-store/src/lib.rs
cat > "crates/omni-store/src/lib.rs" << 'EOF'
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
EOF

# 6. Overwrite omni-node/Cargo.toml
cat > "crates/omni-node/Cargo.toml" << 'EOF'
[package]
name             = "omni-node"
version.workspace    = true
edition.workspace    = true
license.workspace    = true
repository.workspace = true
rust-version.workspace = true

[[bin]]
name = "omni-node"
path = "src/main.rs"

[dependencies]
omni-types         = { workspace = true }
omni-net           = { workspace = true }
omni-store         = { workspace = true }
tokio              = { workspace = true }
clap               = { workspace = true }
tracing            = { workspace = true }
tracing-subscriber = { workspace = true }
anyhow             = { workspace = true }
serde_json         = { workspace = true }
EOF

# 7. Overwrite omni-node/src/main.rs
cat > "crates/omni-node/src/main.rs" << 'EOF'
//! OmniNode binary — Phase 1 + Phase 2 CLI.
//!
//! ```bash
//! # Listen + serve shards
//! RUST_LOG=info cargo run --bin omni-node -- listen
//!
//! # Ingest a GGUF model, store shards, announce on mesh
//! RUST_LOG=info cargo run --bin omni-node -- shard path/to/model.gguf
//!
//! # Fetch a shard by CID from a LAN peer
//! RUST_LOG=info cargo run --bin omni-node -- fetch <cid>
//!
//! # Send a test Gossipsub message
//! RUST_LOG=info cargo run --bin omni-node -- send "Hello from OmniNode"
//! ```

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use omni_net::{OmniNet, OmniNetEvent, TOPIC_SHARD, TOPIC_TEST};
use omni_store::{OmniStore, FetchOutcome, decode_announcement};
use omni_types::config::{NetConfig, StoreConfig};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "omni-node",
    version = env!("CARGO_PKG_VERSION"),
    about   = "OmniNode Protocol — decentralized AI inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Listen indefinitely, serving shard requests and printing events.
    Listen,

    /// Ingest a GGUF model: parse, chunk, store shards, announce on mesh.
    Shard {
        /// Path to the GGUF model file.
        path: PathBuf,
    },

    /// Fetch a shard by CID from a LAN peer.
    Fetch {
        /// CIDv1 string of the shard to fetch.
        cid: String,
    },

    /// Discover a peer on the LAN, publish a test message, then exit.
    Send {
        /// UTF-8 message to broadcast on `omni/test/v1`.
        message: String,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Listen           => run_listen().await,
        Command::Shard { path }   => run_shard(path).await,
        Command::Fetch { cid }    => run_fetch(cid).await,
        Command::Send { message } => run_send(message).await,
    }
}

// ── Listen mode ───────────────────────────────────────────────────────────────

async fn run_listen() -> Result<()> {
    let mut net   = OmniNet::new(NetConfig::default()).await?;
    let store = OmniStore::new(StoreConfig::default())?;

    info!("OmniNode listening — serving shards, press Ctrl-C to stop");

    loop {
        tokio::select! {
            Some(event) = net.next_event() => {
                match &event {
                    OmniNetEvent::ShardRequested { request, channel_id, .. } => {
                        omni_store::serve::handle_request(
                            &net, &store.local, request, *channel_id,
                        ).await;
                    }
                    OmniNetEvent::MessageReceived { topic, data, from } => {
                        if topic == TOPIC_SHARD {
                            if let Some(ann) = decode_announcement(data) {
                                info!(
                                    from = %from,
                                    cid = %ann.cid,
                                    model = %ann.model_name,
                                    size = ann.size_bytes,
                                    "shard announced"
                                );
                            }
                        }
                        print_event(&event);
                    }
                    _ => print_event(&event),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl-C — shutting down");
                net.shutdown().await?;
                break;
            }
        }
    }
    Ok(())
}

// ── Shard mode ───────────────────────────────────────────────────────────────

async fn run_shard(path: PathBuf) -> Result<()> {
    let mut net   = OmniNet::new(NetConfig::default()).await?;
    let store = OmniStore::new(StoreConfig::default())?;

    info!(path = %path.display(), "ingesting model");
    let manifest = store.ingest_model(&path)?;

    let json = serde_json::to_string_pretty(&manifest)?;
    info!("manifest:\n{json}");

    // Wait for a peer before announcing.
    info!("waiting for a peer on the LAN to announce shards...");
    let timeout = Duration::from_secs(30);
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        tokio::select! {
            Some(event) = net.next_event() => {
                print_event(&event);
                if let OmniNetEvent::PeerDiscovered { peer_id, .. } = &event {
                    info!(%peer_id, "peer found — announcing shards");
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    store.announce_shards(&net, &manifest).await?;
                    info!("all shards announced — listening for requests");

                    // Continue listening to serve shard requests.
                    serve_loop(&mut net, &store).await?;
                    return Ok(());
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                warn!("timed out waiting for peer — announcing anyway");
                store.announce_shards(&net, &manifest).await?;
                serve_loop(&mut net, &store).await?;
                return Ok(());
            }
        }
    }
}

/// After announcing, serve shard requests until Ctrl-C.
async fn serve_loop(net: &mut OmniNet, store: &OmniStore) -> Result<()> {
    loop {
        tokio::select! {
            Some(event) = net.next_event() => {
                match &event {
                    OmniNetEvent::ShardRequested { request, channel_id, .. } => {
                        omni_store::serve::handle_request(
                            net, &store.local, request, *channel_id,
                        ).await;
                    }
                    _ => print_event(&event),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl-C — shutting down");
                net.shutdown().await?;
                break;
            }
        }
    }
    Ok(())
}

// ── Fetch mode ───────────────────────────────────────────────────────────────

async fn run_fetch(cid: String) -> Result<()> {
    let mut net   = OmniNet::new(NetConfig::default()).await?;
    let mut store = OmniStore::new(StoreConfig::default())?;

    if store.has_shard(&cid) {
        info!(%cid, "shard already exists locally");
        return Ok(());
    }

    info!(%cid, "waiting for a peer that has the shard...");
    let timeout = Duration::from_secs(60);
    let deadline = tokio::time::Instant::now() + timeout;
    let mut target_peer = None;

    // Phase 1: discover a peer advertising this CID.
    loop {
        tokio::select! {
            Some(event) = net.next_event() => {
                match &event {
                    OmniNetEvent::MessageReceived { topic, data, from } if topic == TOPIC_SHARD => {
                        if let Some(ann) = decode_announcement(data) {
                            if ann.cid == cid {
                                info!(from = %from, "found peer with shard");
                                target_peer = Some(*from);
                                break;
                            }
                        }
                    }
                    OmniNetEvent::PeerDiscovered { peer_id, .. } => {
                        // If no announcement, try the first peer we see.
                        if target_peer.is_none() {
                            info!(%peer_id, "discovered peer — will request shard");
                            target_peer = Some(*peer_id);
                            break;
                        }
                    }
                    _ => print_event(&event),
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                anyhow::bail!("timed out waiting for a peer with shard {cid}");
            }
        }
    }

    let peer = target_peer.unwrap();
    store.fetcher.start_fetch(&net, peer, cid.clone())
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Phase 2: chunk-receive loop.
    loop {
        tokio::select! {
            Some(event) = net.next_event() => {
                match event {
                    OmniNetEvent::ShardReceived { response, .. } => {
                        match store.fetcher.on_chunk_received(&net, &store.local, &response).await {
                            FetchOutcome::InProgress => {}
                            FetchOutcome::Complete { cid, size } => {
                                info!(%cid, size, "shard fetched successfully");
                                net.shutdown().await?;
                                return Ok(());
                            }
                            FetchOutcome::Failed { cid, error } => {
                                anyhow::bail!("shard fetch failed for {cid}: {error}");
                            }
                        }
                    }
                    OmniNetEvent::ShardRequestFailed { error, .. } => {
                        anyhow::bail!("shard request failed: {error}");
                    }
                    _ => print_event(&event),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl-C — aborting fetch");
                net.shutdown().await?;
                return Ok(());
            }
        }
    }
}

// ── Send mode ─────────────────────────────────────────────────────────────────

async fn run_send(message: String) -> Result<()> {
    let mut node = OmniNet::new(NetConfig::default()).await?;

    const TIMEOUT: Duration = Duration::from_secs(30);
    info!("waiting for a peer on the LAN (timeout: {TIMEOUT:?})");

    let result =
        tokio::time::timeout(TIMEOUT, discover_and_send(&mut node, &message)).await;

    match result {
        Ok(inner)     => inner,
        Err(_elapsed) => {
            warn!("timed out — are both nodes on the same LAN?");
            Err(anyhow::anyhow!("no peer discovered within timeout"))
        }
    }
}

async fn discover_and_send(node: &mut OmniNet, message: &str) -> Result<()> {
    while let Some(event) = node.next_event().await {
        print_event(&event);

        if let OmniNetEvent::PeerDiscovered { peer_id, .. } = &event {
            info!(%peer_id, "peer found — publishing");
            tokio::time::sleep(Duration::from_millis(500)).await;
            node.publish(TOPIC_TEST, message.as_bytes().to_vec()).await?;
            info!("sent on '{TOPIC_TEST}' — exiting");
            tokio::time::sleep(Duration::from_millis(300)).await;
            node.shutdown().await?;
            return Ok(());
        }
    }
    Err(anyhow::anyhow!("event stream closed before a peer was discovered"))
}

// ── Event printer ─────────────────────────────────────────────────────────────

fn print_event(event: &OmniNetEvent) {
    match event {
        OmniNetEvent::Listening { addr } =>
            info!("LISTENING    {addr}"),
        OmniNetEvent::PeerDiscovered { peer_id, addrs } =>
            info!("DISCOVERED   {peer_id}  addrs={addrs:?}"),
        OmniNetEvent::PeerExpired { peer_id } =>
            info!("EXPIRED      {peer_id}"),
        OmniNetEvent::PeerConnected { peer_id } =>
            info!("CONNECTED    {peer_id}"),
        OmniNetEvent::PeerDisconnected { peer_id } =>
            info!("DISCONNECTED {peer_id}"),
        OmniNetEvent::MessageReceived { from, topic, data } => {
            let text = String::from_utf8_lossy(data);
            info!("MESSAGE      topic={topic}  from={from}  len={}", data.len());
        }
        OmniNetEvent::ShardRequested { peer_id, request, channel_id } =>
            info!("SHARD_REQ    peer={peer_id}  cid={}  ch={channel_id}", request.cid),
        OmniNetEvent::ShardReceived { peer_id, response } =>
            info!("SHARD_RECV   peer={peer_id}  cid={}  offset={}  bytes={}", response.cid, response.offset, response.data.len()),
        OmniNetEvent::ShardRequestFailed { peer_id, error } =>
            info!("SHARD_FAIL   peer={peer_id}  error={error}"),
    }
}
EOF

echo "=== Files written. Running cargo check --workspace ==="
cargo check --workspace 2>&1

