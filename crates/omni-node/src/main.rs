//! OmniNode binary — Phase 1 two-machine LAN test harness.
//!
//! ```bash
//! # Mac 1 — listen indefinitely, print every mesh event
//! RUST_LOG=info cargo run --bin omni-node -- listen
//!
//! # Mac 2 — discover a peer, publish a message on omni/test/v1, then exit
//! RUST_LOG=info cargo run --bin omni-node -- send "Hello from OmniNode"
//! ```

use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use omni_net::{OmniNet, OmniNetEvent, TOPIC_TEST};
use omni_types::config::NetConfig;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "omni-node",
    version = env!("CARGO_PKG_VERSION"),
    about   = "OmniNode Protocol — Phase 1 local mesh"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Listen indefinitely, printing all mesh events.
    Listen,

    /// Discover a peer on the LAN, publish a test message, then exit.
    Send {
        /// UTF-8 message to broadcast on `omni/test/v1`.
        message: String,
    },
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    // Default log level: INFO. Override with RUST_LOG=omni_net=debug etc.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli  = Cli::parse();
    let node = OmniNet::new(NetConfig::default()).await?;

    match cli.command {
        Command::Listen           => run_listener(node).await,
        Command::Send { message } => run_sender(node, message).await,
    }
}

// ── Listen mode ───────────────────────────────────────────────────────────────

async fn run_listener(mut node: OmniNet) -> Result<()> {
    info!("OmniNode listening — press Ctrl-C to stop");
    loop {
        tokio::select! {
            Some(event) = node.next_event() => print_event(&event),
            _ = tokio::signal::ctrl_c() => {
                info!("Ctrl-C — shutting down");
                node.shutdown().await?;
                break;
            }
        }
    }
    Ok(())
}

// ── Send mode ─────────────────────────────────────────────────────────────────

/// Wait for the first mDNS peer discovery, publish `message` on
/// `omni/test/v1`, then exit. Fails if no peer appears within 30 seconds.
async fn run_sender(mut node: OmniNet, message: String) -> Result<()> {
    const TIMEOUT: Duration = Duration::from_secs(30);
    info!("waiting for a peer on the LAN (timeout: {TIMEOUT:?})");

    let result =
        tokio::time::timeout(TIMEOUT, discover_and_send(&mut node, &message)).await;

    match result {
        Ok(inner)     => inner,
        Err(_elapsed) => {
            warn!("timed out after {}s — are both nodes on the same LAN?", TIMEOUT.as_secs());
            Err(anyhow::anyhow!("no peer discovered within timeout"))
        }
    }
}

async fn discover_and_send(node: &mut OmniNet, message: &str) -> Result<()> {
    while let Some(event) = node.next_event().await {
        print_event(&event);

        if let OmniNetEvent::PeerDiscovered { peer_id, .. } = &event {
            info!(%peer_id, "peer found — publishing");

            // 500 ms grace period: Gossipsub needs to complete the mesh
            // handshake before a publish will propagate to the new peer.
            tokio::time::sleep(Duration::from_millis(500)).await;

            node.publish(TOPIC_TEST, message.as_bytes().to_vec()).await?;
            info!("sent on '{TOPIC_TEST}' — exiting");

            // Brief pause so the message drains the QUIC send buffer.
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
            info!("MESSAGE      topic={topic}  from={from}  body=\"{text}\"");
        }
    }
}
