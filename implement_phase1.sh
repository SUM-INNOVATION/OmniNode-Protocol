#!/usr/bin/env bash
set -euo pipefail

echo "==> Phase 1: writing omni-net implementation files..."

# ─── Ensure futures dep is present in workspace Cargo.toml ──────────────────
if ! grep -q '^futures' Cargo.toml; then
    sed -i '' 's/^# ─── Utilities/futures = "0.3"\n\n# ─── Utilities/' Cargo.toml
    echo "  added futures to workspace Cargo.toml"
else
    echo "  futures already present in workspace Cargo.toml"
fi

# ─── Ensure futures dep is present in omni-net/Cargo.toml ───────────────────
if ! grep -q 'futures' crates/omni-net/Cargo.toml; then
    sed -i '' 's/^libp2p     = { workspace = true }/libp2p     = { workspace = true }\nfutures    = { workspace = true }/' crates/omni-net/Cargo.toml
    echo "  added futures to crates/omni-net/Cargo.toml"
else
    echo "  futures already present in crates/omni-net/Cargo.toml"
fi

# ─── events.rs ───────────────────────────────────────────────────────────────
cat > crates/omni-net/src/events.rs << 'RUST_EOF'
use libp2p::{Multiaddr, PeerId};

/// Clean, domain-level events emitted by the OmniNode networking layer.
/// Never exposes raw libp2p internals to callers.
#[derive(Debug, Clone)]
pub enum OmniNetEvent {
    /// The local node is now listening on a new address.
    Listening { addr: Multiaddr },

    /// A new peer was discovered via mDNS on the local network.
    PeerDiscovered {
        peer_id: PeerId,
        addrs: Vec<Multiaddr>,
    },

    /// A previously discovered mDNS peer is no longer visible.
    PeerExpired { peer_id: PeerId },

    /// A transport-layer connection was established.
    PeerConnected { peer_id: PeerId },

    /// A transport-layer connection was closed.
    PeerDisconnected { peer_id: PeerId },

    /// A Gossipsub message was received.
    MessageReceived {
        /// PeerId that propagated the message to us.
        from: PeerId,
        /// Topic string, e.g. `"omni/test/v1"`.
        topic: String,
        /// Raw payload bytes.
        data: Vec<u8>,
    },
}
RUST_EOF
echo "  wrote events.rs"

# ─── gossip.rs ───────────────────────────────────────────────────────────────
cat > crates/omni-net/src/gossip.rs << 'RUST_EOF'
use anyhow::Result;
use libp2p::gossipsub::{self, IdentTopic, MessageId};
use tracing::{debug, info};

// ── Topic constants ───────────────────────────────────────────────────────────

pub const TOPIC_TEST: &str       = "omni/test/v1";
pub const TOPIC_CAPABILITY: &str = "omni/capability/v1";
pub const TOPIC_SHARD: &str      = "omni/shard/v1";
pub const TOPIC_PIPELINE: &str   = "omni/pipeline/v1";
pub const TOPIC_PROOF: &str      = "omni/proof/v1";

// ── GossipManager ─────────────────────────────────────────────────────────────

/// Manages Gossipsub topic subscriptions and message publishing.
///
/// Holds pre-built [`IdentTopic`] handles — the topic hash is computed once at
/// construction rather than on every call to `publish`.
pub struct GossipManager {
    topic_test:       IdentTopic,
    topic_capability: IdentTopic,
    topic_shard:      IdentTopic,
    topic_pipeline:   IdentTopic,
    topic_proof:      IdentTopic,
}

impl GossipManager {
    pub fn new() -> Self {
        Self {
            topic_test:       IdentTopic::new(TOPIC_TEST),
            topic_capability: IdentTopic::new(TOPIC_CAPABILITY),
            topic_shard:      IdentTopic::new(TOPIC_SHARD),
            topic_pipeline:   IdentTopic::new(TOPIC_PIPELINE),
            topic_proof:      IdentTopic::new(TOPIC_PROOF),
        }
    }

    /// Subscribe this node to all OmniNode Gossipsub topics.
    pub fn subscribe_all(&self, gs: &mut gossipsub::Behaviour) -> Result<()> {
        for topic in self.all_topics() {
            if gs.subscribe(topic)? {
                debug!(topic = %topic, "subscribed to gossipsub topic");
            }
        }
        Ok(())
    }

    /// Publish raw bytes to a named topic.
    /// Returns the [`MessageId`] on success.
    pub fn publish(
        &self,
        gs: &mut gossipsub::Behaviour,
        topic_name: &str,
        data: impl Into<Vec<u8>>,
    ) -> Result<MessageId> {
        let topic = self.topic_by_name(topic_name);
        let id = gs
            .publish(topic.clone(), data.into())
            .map_err(|e| anyhow::anyhow!("publish on '{topic_name}': {e}"))?;
        info!(topic = topic_name, "message published");
        Ok(id)
    }

    fn all_topics(&self) -> [&IdentTopic; 5] {
        [
            &self.topic_test,
            &self.topic_capability,
            &self.topic_shard,
            &self.topic_pipeline,
            &self.topic_proof,
        ]
    }

    fn topic_by_name(&self, name: &str) -> &IdentTopic {
        match name {
            TOPIC_CAPABILITY => &self.topic_capability,
            TOPIC_SHARD      => &self.topic_shard,
            TOPIC_PIPELINE   => &self.topic_pipeline,
            TOPIC_PROOF      => &self.topic_proof,
            _                => &self.topic_test,
        }
    }
}

impl Default for GossipManager {
    fn default() -> Self {
        Self::new()
    }
}
RUST_EOF
echo "  wrote gossip.rs"

# ─── discovery.rs ────────────────────────────────────────────────────────────
cat > crates/omni-net/src/discovery.rs << 'RUST_EOF'
use std::collections::{HashMap, HashSet};

use libp2p::{gossipsub, mdns, PeerId};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::events::OmniNetEvent;

/// Handle a single mDNS event.
///
/// Wires newly-discovered peers into Gossipsub (via `add_explicit_peer`) and
/// emits the appropriate [`OmniNetEvent`] for each distinct PeerId.
/// Called directly from the swarm event loop to keep that loop thin.
pub fn handle_mdns_event(
    event: mdns::Event,
    gossipsub: &mut gossipsub::Behaviour,
    event_tx: &mpsc::Sender<OmniNetEvent>,
) {
    match event {
        mdns::Event::Discovered(list) => {
            // Group all addresses by PeerId — a peer may be reachable on
            // multiple addresses, but we emit one event per peer.
            let mut by_peer: HashMap<PeerId, Vec<_>> = HashMap::new();
            for (peer_id, addr) in list {
                gossipsub.add_explicit_peer(&peer_id);
                by_peer.entry(peer_id).or_default().push(addr);
            }
            for (peer_id, addrs) in by_peer {
                info!(%peer_id, addr_count = addrs.len(), "mDNS discovered peer");
                if let Err(e) =
                    event_tx.try_send(OmniNetEvent::PeerDiscovered { peer_id, addrs })
                {
                    warn!(%e, "event channel full — dropping PeerDiscovered");
                }
            }
        }

        mdns::Event::Expired(list) => {
            // Deduplicate: one peer may appear once per expired address.
            let mut expired: HashSet<PeerId> = HashSet::new();
            for (peer_id, _addr) in list {
                expired.insert(peer_id);
            }
            for peer_id in expired {
                gossipsub.remove_explicit_peer(&peer_id);
                debug!(%peer_id, "mDNS peer expired");
                if let Err(e) = event_tx.try_send(OmniNetEvent::PeerExpired { peer_id }) {
                    warn!(%e, "event channel full — dropping PeerExpired");
                }
            }
        }
    }
}
RUST_EOF
echo "  wrote discovery.rs"

# ─── behaviour.rs ────────────────────────────────────────────────────────────
cat > crates/omni-net/src/behaviour.rs << 'RUST_EOF'
use libp2p::{gossipsub, identify, mdns, swarm::NetworkBehaviour};

/// Composed [`NetworkBehaviour`] for the OmniNode local mesh (Phase 1).
///
/// The `#[derive(NetworkBehaviour)]` macro generates `LocalMeshBehaviourEvent`
/// with variants:
/// - `Mdns(mdns::Event)`
/// - `Gossipsub(gossipsub::Event)`
/// - `Identify(identify::Event)`
///
/// **Phase 1 scope (LAN only):** mDNS + Gossipsub + Identify.
///
/// **Deferred to Phase 1b (WAN):** `kademlia`, `autonat`, `relay`, `dcutr`,
/// `request_response`.
#[derive(NetworkBehaviour)]
pub struct LocalMeshBehaviour {
    pub mdns:      mdns::tokio::Behaviour,
    pub gossipsub: gossipsub::Behaviour,
    pub identify:  identify::Behaviour,
}
RUST_EOF
echo "  wrote behaviour.rs"

# ─── swarm.rs ────────────────────────────────────────────────────────────────
cat > crates/omni-net/src/swarm.rs << 'RUST_EOF'
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    gossipsub, identify, mdns,
    swarm::SwarmEvent,
    Multiaddr, SwarmBuilder,
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use omni_types::config::NetConfig;

use crate::{
    behaviour::{LocalMeshBehaviour, LocalMeshBehaviourEvent},
    discovery,
    events::OmniNetEvent,
    gossip::GossipManager,
};

// ── SwarmCommand ──────────────────────────────────────────────────────────────

/// Commands sent from the [`crate::OmniNet`] handle into the running swarm loop.
#[derive(Debug)]
pub enum SwarmCommand {
    /// Publish bytes to a named Gossipsub topic.
    Publish { topic: String, data: Vec<u8> },
    /// Exit the event loop cleanly.
    Shutdown,
}

// ── OmniSwarm ─────────────────────────────────────────────────────────────────

/// Owns the [`libp2p::Swarm`] and the [`GossipManager`].
/// Constructed by [`OmniSwarm::build`] and consumed by [`OmniSwarm::run`].
pub struct OmniSwarm {
    inner:  libp2p::Swarm<LocalMeshBehaviour>,
    gossip: GossipManager,
}

impl OmniSwarm {
    /// Construct and configure the Swarm via the libp2p 0.55 `SwarmBuilder` API.
    ///
    /// Transport:  QUIC (TLS 1.3 baked-in — no separate Noise step required)
    /// Behaviour:  mDNS + Gossipsub + Identify
    /// Listener:   `0.0.0.0:<config.listen_port>` (0 = OS-assigned)
    pub fn build(config: &NetConfig) -> Result<Self> {
        let gossip_cfg = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .history_length(10)
            .history_gossip(3)
            .build()
            .map_err(|msg| anyhow::anyhow!("gossipsub config error: {msg}"))?;

        let mut swarm = SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_quic()
            .with_behaviour(|key| {
                let mdns = mdns::tokio::Behaviour::new(
                    mdns::Config::default(),
                    key.public().to_peer_id(),
                )?;

                let gossipsub = gossipsub::Behaviour::new(
                    gossipsub::MessageAuthenticity::Signed(key.clone()),
                    gossip_cfg,
                )
                .map_err(|msg| anyhow::anyhow!("gossipsub init: {msg}"))?;

                let identify = identify::Behaviour::new(identify::Config::new(
                    "/omni-node/0.1.0".into(),
                    key.public(),
                ));

                Ok(LocalMeshBehaviour { mdns, gossipsub, identify })
            })?
            .with_swarm_config(|c| {
                // Keep idle QUIC connections alive between pipeline requests.
                c.with_idle_connection_timeout(Duration::from_secs(60))
            })
            .build();

        let listen_addr: Multiaddr =
            format!("/ip4/0.0.0.0/udp/{}/quic-v1", config.listen_port)
                .parse()
                .context("invalid QUIC listen multiaddr")?;

        swarm
            .listen_on(listen_addr)
            .context("failed to bind QUIC listener")?;

        Ok(Self {
            inner:  swarm,
            gossip: GossipManager::new(),
        })
    }

    /// Subscribe the node to all OmniNode Gossipsub topics.
    pub fn subscribe_all_topics(&mut self) -> Result<()> {
        self.gossip
            .subscribe_all(&mut self.inner.behaviour_mut().gossipsub)
    }

    /// Publish bytes to a named topic from within the event loop.
    pub fn publish(&mut self, topic: &str, data: Vec<u8>) -> Result<()> {
        self.gossip
            .publish(&mut self.inner.behaviour_mut().gossipsub, topic, data)
            .map(|_| ())
    }

    /// The core async event loop.
    ///
    /// Runs until a [`SwarmCommand::Shutdown`] is received or `cmd_rx` is
    /// dropped. Forwards all meaningful events to `event_tx`.
    pub async fn run(
        mut self,
        event_tx:   mpsc::Sender<OmniNetEvent>,
        mut cmd_rx: mpsc::Receiver<SwarmCommand>,
    ) -> Result<()> {
        loop {
            tokio::select! {
                // ── Swarm events ──────────────────────────────────────────────
                event = self.inner.select_next_some() => {
                    self.handle_swarm_event(event, &event_tx);
                }

                // ── Commands from OmniNet API ──────────────────────────────────
                cmd = cmd_rx.recv() => {
                    match cmd {
                        Some(SwarmCommand::Publish { topic, data }) => {
                            if let Err(e) = self.publish(&topic, data) {
                                warn!(%e, %topic, "gossipsub publish failed");
                            }
                        }
                        Some(SwarmCommand::Shutdown) | None => {
                            info!("swarm event loop shutting down");
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    // ── Private event dispatcher ──────────────────────────────────────────────

    fn handle_swarm_event(
        &mut self,
        event:    SwarmEvent<LocalMeshBehaviourEvent>,
        event_tx: &mpsc::Sender<OmniNetEvent>,
    ) {
        match event {
            // ── mDNS ──────────────────────────────────────────────────────────
            SwarmEvent::Behaviour(LocalMeshBehaviourEvent::Mdns(e)) => {
                discovery::handle_mdns_event(
                    e,
                    &mut self.inner.behaviour_mut().gossipsub,
                    event_tx,
                );
            }

            // ── Gossipsub: incoming message ────────────────────────────────────
            SwarmEvent::Behaviour(LocalMeshBehaviourEvent::Gossipsub(
                gossipsub::Event::Message {
                    propagation_source,
                    message,
                    ..
                },
            )) => {
                let topic = message.topic.to_string();
                let data  = message.data;
                info!(
                    from  = %propagation_source,
                    %topic,
                    bytes = data.len(),
                    "gossipsub message received"
                );
                if let Err(e) = event_tx.try_send(OmniNetEvent::MessageReceived {
                    from:  propagation_source,
                    topic,
                    data,
                }) {
                    warn!(%e, "event channel full — dropping MessageReceived");
                }
            }

            // Gossipsub mesh formation events (subscribe/unsubscribe) — debug only.
            SwarmEvent::Behaviour(LocalMeshBehaviourEvent::Gossipsub(e)) => {
                debug!(?e, "gossipsub mesh event");
            }

            // ── Identify ──────────────────────────────────────────────────────
            SwarmEvent::Behaviour(LocalMeshBehaviourEvent::Identify(e)) => {
                debug!(?e, "identify event");
            }

            // ── Transport ─────────────────────────────────────────────────────
            SwarmEvent::NewListenAddr { address, .. } => {
                info!(%address, "listening on address");
                if let Err(e) = event_tx.try_send(OmniNetEvent::Listening { addr: address }) {
                    warn!(%e, "event channel full — dropping Listening");
                }
            }

            SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                info!(%peer_id, "connection established");
                if let Err(e) = event_tx.try_send(OmniNetEvent::PeerConnected { peer_id }) {
                    warn!(%e, "event channel full — dropping PeerConnected");
                }
            }

            SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                debug!(%peer_id, ?cause, "connection closed");
                if let Err(e) = event_tx.try_send(OmniNetEvent::PeerDisconnected { peer_id }) {
                    warn!(%e, "event channel full — dropping PeerDisconnected");
                }
            }

            SwarmEvent::IncomingConnectionError { error, .. } => {
                warn!(%error, "incoming connection error");
            }

            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                warn!(?peer_id, %error, "outgoing connection error");
            }

            _ => {}
        }
    }
}
RUST_EOF
echo "  wrote swarm.rs"

# ─── lib.rs ──────────────────────────────────────────────────────────────────
cat > crates/omni-net/src/lib.rs << 'RUST_EOF'
// ── Module declarations ───────────────────────────────────────────────────────

pub mod behaviour;
pub mod capability;  // deferred — WAN capability advertisement protocol
pub mod codec;       // deferred — custom request-response codec
pub mod discovery;
pub mod events;
pub mod gossip;
pub mod nat;         // deferred — AutoNAT / DCUtR / relay
pub mod swarm;
pub mod transport;   // deferred — TCP/Noise fallback transport

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use events::OmniNetEvent;
pub use gossip::{
    TOPIC_CAPABILITY, TOPIC_PIPELINE, TOPIC_PROOF, TOPIC_SHARD, TOPIC_TEST,
};

// ── Imports ───────────────────────────────────────────────────────────────────

use anyhow::Result;
use tokio::sync::mpsc;

use omni_types::config::NetConfig;

use crate::swarm::{OmniSwarm, SwarmCommand};

/// Internal channel buffer. 256 slots absorbs short bursts without dropping
/// events under normal two-node LAN conditions.
const CHANNEL_CAPACITY: usize = 256;

// ── OmniNet ───────────────────────────────────────────────────────────────────

/// Top-level handle to the OmniNode P2P networking layer.
///
/// Owns two async channels that communicate with a background `tokio` task
/// running the [`swarm::OmniSwarm`] event loop:
///
/// - `cmd_tx`   — send commands (publish, shutdown) **into** the loop
/// - `event_rx` — receive [`OmniNetEvent`]s **from** the loop
///
/// # Example
/// ```rust,no_run
/// use omni_net::{OmniNet, OmniNetEvent, TOPIC_TEST};
/// use omni_types::config::NetConfig;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut node = OmniNet::new(NetConfig::default()).await?;
///     node.publish(TOPIC_TEST, b"hello".to_vec()).await?;
///     while let Some(ev) = node.next_event().await {
///         println!("{ev:?}");
///     }
///     Ok(())
/// }
/// ```
pub struct OmniNet {
    cmd_tx:   mpsc::Sender<SwarmCommand>,
    event_rx: mpsc::Receiver<OmniNetEvent>,
}

impl OmniNet {
    /// Build the swarm, subscribe to all topics, and spawn the event loop task.
    /// Returns immediately — the swarm runs concurrently in a `tokio` task.
    pub async fn new(config: NetConfig) -> Result<Self> {
        let mut omni_swarm = OmniSwarm::build(&config)?;
        omni_swarm.subscribe_all_topics()?;

        let (event_tx, event_rx) = mpsc::channel::<OmniNetEvent>(CHANNEL_CAPACITY);
        let (cmd_tx, cmd_rx)     = mpsc::channel::<SwarmCommand>(CHANNEL_CAPACITY);

        tokio::spawn(async move {
            if let Err(e) = omni_swarm.run(event_tx, cmd_rx).await {
                tracing::error!(%e, "swarm event loop exited with error");
            }
        });

        Ok(Self { cmd_tx, event_rx })
    }

    /// Publish `data` to the named Gossipsub topic.
    /// Sends the command to the background task and returns immediately.
    pub async fn publish(&self, topic: &str, data: Vec<u8>) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::Publish {
                topic: topic.to_string(),
                data,
            })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot publish"))
    }

    /// Receive the next event from the mesh.
    /// Returns `None` when the swarm task has stopped and the buffer is drained.
    pub async fn next_event(&mut self) -> Option<OmniNetEvent> {
        self.event_rx.recv().await
    }

    /// Signal the swarm loop to shut down gracefully.
    pub async fn shutdown(&self) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::Shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("swarm task already stopped"))
    }
}
RUST_EOF
echo "  wrote lib.rs"

# ─── main.rs ─────────────────────────────────────────────────────────────────
cat > crates/omni-node/src/main.rs << 'RUST_EOF'
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
RUST_EOF
echo "  wrote main.rs"

# ─── Verify ──────────────────────────────────────────────────────────────────
echo ""
echo "==> Running cargo check -p omni-net -p omni-node ..."
cargo check -p omni-net -p omni-node 2>&1

echo ""
echo "==> Done. If cargo check passed, run the two-Mac test:"
echo "    Mac 1:  RUST_LOG=info cargo run --bin omni-node -- listen"
echo "    Mac 2:  RUST_LOG=info cargo run --bin omni-node -- send \"Hello from OmniNode\""

