// ── Module declarations ───────────────────────────────────────────────────────

pub mod behaviour;
pub mod capability;  // deferred — WAN capability advertisement protocol
pub mod codec;
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
pub use codec::{ShardCodec, ShardRequest, ShardResponse, SHARD_XFER_PROTOCOL};

// ── Imports ───────────────────────────────────────────────────────────────────

use anyhow::Result;
use libp2p::PeerId;
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
/// - `cmd_tx`   — send commands (publish, shutdown, shard ops) **into** the loop
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

    // ── Phase 1: Gossipsub ──────────────────────────────────────────────

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

    // ── Phase 2: Shard transfer ─────────────────────────────────────────

    /// Request a shard chunk from a remote peer.
    ///
    /// - `peer_id`:   the peer to request from (learned via gossipsub announcement)
    /// - `cid`:       CIDv1 string identifying the shard
    /// - `offset`:    byte offset within the shard (`None` = from beginning)
    /// - `max_bytes`: max bytes to return (`None` = entire shard)
    ///
    /// The response arrives later as [`OmniNetEvent::ShardReceived`].
    pub async fn request_shard_chunk(
        &self,
        peer_id: PeerId,
        cid: String,
        offset: Option<u64>,
        max_bytes: Option<u64>,
    ) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::RequestShard {
                peer_id,
                request: ShardRequest { cid, offset, max_bytes },
            })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot request shard"))
    }

    /// Send a shard response on a pending response channel.
    ///
    /// `channel_id` is the ID received in [`OmniNetEvent::ShardRequested`].
    pub async fn respond_shard(
        &self,
        channel_id: u64,
        response: ShardResponse,
    ) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::SendShardResponse { channel_id, response })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot respond shard"))
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

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
