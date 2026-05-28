// ── Module declarations ───────────────────────────────────────────────────────

pub mod behaviour;
pub mod capability;  // deferred — WAN capability advertisement protocol
pub mod codec;
pub mod discovery;
pub mod events;
pub mod gossip;
pub mod nat;
pub mod swarm;
pub mod tensor_codec;
pub mod transport;   // deferred — TCP/Noise fallback transport

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use events::OmniNetEvent;
pub use gossip::{
    TOPIC_CAPABILITY, TOPIC_CONTRIBUTOR_JOB, TOPIC_CONTRIBUTOR_RESULT,
    TOPIC_CONTRIBUTOR_SESSION_AGGREGATED, TOPIC_CONTRIBUTOR_SESSION_ASSIGN,
    TOPIC_CONTRIBUTOR_SESSION_JOIN, TOPIC_CONTRIBUTOR_SESSION_OPEN,
    TOPIC_CONTRIBUTOR_SESSION_PARTIAL, TOPIC_PIPELINE, TOPIC_PROOF, TOPIC_SHARD,
    TOPIC_TEST, UnknownTopic,
};
pub use codec::{ShardCodec, ShardRequest, ShardResponse, SHARD_XFER_PROTOCOL};
pub use tensor_codec::{TensorCodec, TensorRequest, TensorResponse, TENSOR_XFER_PROTOCOL};
pub use nat::NatStatus;

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
/// - `cmd_tx`   — send commands (publish, shutdown, shard/tensor ops) **into** the loop
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
    cmd_tx:        mpsc::Sender<SwarmCommand>,
    event_rx:      mpsc::Receiver<OmniNetEvent>,
    /// Stage 12.5-pre — local libp2p `PeerId`. Captured from the
    /// built swarm BEFORE the run loop is spawned, so callers can
    /// read it via [`OmniNet::local_peer_id`] without re-entering
    /// the swarm task. Stable for the lifetime of this `OmniNet`;
    /// regenerated on every `OmniNet::new` because the swarm uses
    /// `SwarmBuilder::with_new_identity()`. Restart = new PeerId.
    local_peer_id: PeerId,
}

impl OmniNet {
    /// Build the swarm, subscribe to all topics, and spawn the event loop task.
    /// Returns immediately — the swarm runs concurrently in a `tokio` task.
    pub async fn new(config: NetConfig) -> Result<Self> {
        let mut omni_swarm = OmniSwarm::build(&config)?;
        omni_swarm.subscribe_all_topics()?;

        // Stage 12.5-pre — capture the local PeerId BEFORE moving
        // `omni_swarm` into the background task. Cheap copy; the
        // value is immutable for the lifetime of this `OmniNet`.
        let local_peer_id = omni_swarm.local_peer_id();

        let (event_tx, event_rx) = mpsc::channel::<OmniNetEvent>(CHANNEL_CAPACITY);
        let (cmd_tx, cmd_rx)     = mpsc::channel::<SwarmCommand>(CHANNEL_CAPACITY);

        tokio::spawn(async move {
            if let Err(e) = omni_swarm.run(event_tx, cmd_rx).await {
                tracing::error!(%e, "swarm event loop exited with error");
            }
        });

        Ok(Self {
            cmd_tx,
            event_rx,
            local_peer_id,
        })
    }

    /// Stage 12.5-pre — local libp2p [`PeerId`] for this node.
    /// Stable for the lifetime of this `OmniNet` instance.
    ///
    /// Used by Stage 12.5's `advertise-peer` subcommand to bind a
    /// signed `ContributorPeerAdvertisement` to the actual running
    /// network identity rather than an operator-supplied string.
    ///
    /// Note: `OmniNet::new` calls `SwarmBuilder::with_new_identity`,
    /// so the libp2p identity keypair (and therefore this PeerId)
    /// is regenerated on every node restart. Stage 12.5 peer
    /// advertisements are therefore session-scoped, short-lived
    /// routing hints — not permanent identity records. Persistent
    /// libp2p identity is a separate, deferred concern.
    pub fn local_peer_id(&self) -> PeerId {
        self.local_peer_id
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

    // ── Phase 4: Tensor transfer ────────────────────────────────────────

    /// Send a hidden-state activation tensor to a remote pipeline stage.
    ///
    /// The `request` contains both the metadata (session, micro-batch, stage
    /// indices, dimensions) and the raw activation bytes.
    ///
    /// The acknowledgment arrives later as [`OmniNetEvent::TensorResponseReceived`].
    pub async fn request_tensor(
        &self,
        peer_id: PeerId,
        request: TensorRequest,
    ) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::RequestTensor { peer_id, request })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot send tensor"))
    }

    /// Send an acknowledgment on a pending tensor response channel.
    ///
    /// `channel_id` is the ID received in [`OmniNetEvent::TensorReceived`].
    pub async fn respond_tensor(
        &self,
        channel_id: u64,
        response: TensorResponse,
    ) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::SendTensorResponse { channel_id, response })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot respond tensor"))
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    /// Receive the next event from the mesh.
    /// Returns `None` when the swarm task has stopped and the buffer is drained.
    pub async fn next_event(&mut self) -> Option<OmniNetEvent> {
        self.event_rx.recv().await
    }

    /// Stage 12.2-pre — non-blocking event drain.
    ///
    /// Returns the next event immediately if one is queued, or
    /// `None` if the queue is currently empty (including the case
    /// where the swarm task has stopped and the buffer is drained).
    /// Distinct from `next_event` which awaits.
    ///
    /// Provided so synchronous consumers (e.g. the Stage 12.2
    /// contributor watch loop) can poll the event stream from a
    /// non-async context without blocking. Does not change the
    /// behavior of `next_event`.
    pub fn try_next_event(&mut self) -> Option<OmniNetEvent> {
        self.event_rx.try_recv().ok()
    }

    /// Signal the swarm loop to shut down gracefully.
    pub async fn shutdown(&self) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::Shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("swarm task already stopped"))
    }

    /// Test-only constructor that builds an `OmniNet` from pre-built
    /// channels instead of standing up a full libp2p swarm. Used to
    /// unit-test the synchronous `try_next_event` accessor without
    /// requiring real networking. The local peer id is synthesized
    /// from a fresh random keypair; tests that read `local_peer_id`
    /// should use [`OmniNet::new`] against a real (port-0) swarm
    /// instead — see `local_peer_id_tests` below.
    #[cfg(test)]
    pub(crate) fn from_test_channels(
        cmd_tx: mpsc::Sender<SwarmCommand>,
        event_rx: mpsc::Receiver<OmniNetEvent>,
    ) -> Self {
        Self {
            cmd_tx,
            event_rx,
            local_peer_id: PeerId::random(),
        }
    }
}

// ── Stage 12.2-pre — try_next_event unit tests ────────────────────────────
//
// These exercise the synchronous accessor in isolation, without
// standing up a real swarm. The async `next_event` path is unchanged
// and continues to be exercised by existing integration usage.

#[cfg(test)]
mod try_next_event_tests {
    use super::*;
    use libp2p::PeerId;

    /// Construct an `OmniNet` whose `event_rx` is the receiver-side of
    /// the returned `event_tx`. These tests never call methods that
    /// use the cmd channel, so the cmd_rx is allowed to drop —
    /// `try_next_event` only touches `event_rx`.
    fn make_pair() -> (mpsc::Sender<OmniNetEvent>, OmniNet) {
        let (cmd_tx, _cmd_rx) = mpsc::channel::<SwarmCommand>(CHANNEL_CAPACITY);
        let (event_tx, event_rx) = mpsc::channel::<OmniNetEvent>(CHANNEL_CAPACITY);
        let net = OmniNet::from_test_channels(cmd_tx, event_rx);
        // Drop cmd_rx — irrelevant to try_next_event behavior.
        drop(_cmd_rx);
        (event_tx, net)
    }

    #[tokio::test]
    async fn try_next_event_returns_none_when_empty() {
        let (_event_tx, mut net) = make_pair();
        // Channel has zero pending events → try_next_event must be None.
        assert!(net.try_next_event().is_none());
    }

    #[tokio::test]
    async fn try_next_event_returns_some_after_event_pushed() {
        let (event_tx, mut net) = make_pair();
        // Inject a synthetic event; try_next_event must return it.
        let from = PeerId::random();
        event_tx
            .send(OmniNetEvent::PeerConnected { peer_id: from })
            .await
            .unwrap();
        let ev = net.try_next_event().expect("event should be available");
        match ev {
            OmniNetEvent::PeerConnected { peer_id } => assert_eq!(peer_id, from),
            other => panic!("unexpected event: {other:?}"),
        }
        // Drained — the next call must be None again.
        assert!(net.try_next_event().is_none());
    }

    #[tokio::test]
    async fn try_next_event_returns_none_when_sender_dropped_and_drained() {
        let (event_tx, mut net) = make_pair();
        drop(event_tx);
        // No events were ever pushed; receiver is now closed. Behavior
        // mirrors next_event's "swarm stopped" None semantics.
        assert!(net.try_next_event().is_none());
    }
}

// ── Stage 12.5-pre — local_peer_id accessor tests ─────────────────────────
//
// Stands up a real (port-0, no bootstrap) OmniNet to exercise the
// public `local_peer_id` accessor end-to-end. Asserts the value is
// stable across repeated calls on the same instance — the field is
// captured BEFORE the run loop is spawned, so no race exists between
// reads and swarm events.

#[cfg(test)]
mod local_peer_id_tests {
    use super::*;
    use omni_types::config::NetConfig;

    #[tokio::test]
    async fn local_peer_id_is_stable_across_repeated_calls() {
        let net = OmniNet::new(NetConfig::default())
            .await
            .expect("OmniNet::new with default config");
        let a = net.local_peer_id();
        let b = net.local_peer_id();
        let c = net.local_peer_id();
        assert_eq!(a, b);
        assert_eq!(b, c);
        // Clean shutdown so the swarm task doesn't outlive the test.
        let _ = net.shutdown().await;
    }

    #[tokio::test]
    async fn two_omni_net_instances_have_distinct_peer_ids() {
        // `SwarmBuilder::with_new_identity()` regenerates the
        // keypair on every `OmniNet::new`. Two independent
        // instances must therefore see distinct PeerIds — the
        // documented "restart = new PeerId" property Stage 12.5
        // peer advertisements rely on.
        let net_a = OmniNet::new(NetConfig::default())
            .await
            .expect("first OmniNet");
        let net_b = OmniNet::new(NetConfig::default())
            .await
            .expect("second OmniNet");
        assert_ne!(net_a.local_peer_id(), net_b.local_peer_id());
        let _ = net_a.shutdown().await;
        let _ = net_b.shutdown().await;
    }
}
