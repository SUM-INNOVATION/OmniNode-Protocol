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
        .await
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
