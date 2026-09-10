// ── Module declarations ───────────────────────────────────────────────────────

pub mod behaviour;
pub mod budget;
pub mod capability;  // deferred — WAN capability advertisement protocol
pub mod codec;
pub mod discovery;
pub mod events;
mod framing;
pub mod gossip;
pub mod identity;    // Stage 12.6 — persistent libp2p mesh identity
pub mod nat;
pub mod request;
pub mod router;
pub mod swarm;
pub mod tensor_codec;
#[cfg(test)]
pub(crate) mod test_alloc;
pub mod transport;   // deferred — TCP/Noise fallback transport

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use events::OmniNetEvent;
pub use gossip::{
    TOPIC_CAPABILITY, TOPIC_CONTRIBUTOR_JOB, TOPIC_CONTRIBUTOR_RESULT,
    TOPIC_CONTRIBUTOR_SESSION_AGGREGATED, TOPIC_CONTRIBUTOR_SESSION_ASSIGN,
    TOPIC_CONTRIBUTOR_SESSION_ASSIGNMENT_SUPERSESSION, TOPIC_CONTRIBUTOR_SESSION_JOIN,
    TOPIC_CONTRIBUTOR_SESSION_OPEN, TOPIC_CONTRIBUTOR_SESSION_PARTIAL,
    TOPIC_CONTRIBUTOR_SESSION_PEER_ADVERT, TOPIC_PIPELINE, TOPIC_PROOF, TOPIC_SHARD,
    TOPIC_TEST, UnknownTopic,
};
pub use identity::{
    decode_keypair_protobuf, load_or_create_keypair_file_bytes, IdentityError,
};
pub use codec::{ShardCodec, ShardRequest, ShardResponse, SHARD_XFER_PROTOCOL};
pub use tensor_codec::{TensorCodec, TensorRequest, TensorResponse, TENSOR_XFER_PROTOCOL};
pub use nat::NatStatus;
pub use request::{Pending, RequestError};
pub use router::{
    classify, EventClass, EventRouter, Interests, RouterCounts, RouterHandle,
    RouterStopped, Subscription, SUBSCRIBER_CAPACITY,
};
pub use budget::{
    weight as event_weight, ByteCounts, ByteLedger, Charge, EVENT_FLOOR_BYTES,
    SHADOW_GLOBAL_BYTES, SHADOW_PEER_BYTES,
};

// ── Imports ───────────────────────────────────────────────────────────────────

use anyhow::Result;
use libp2p::{Multiaddr, PeerId};
use tokio::sync::{mpsc, oneshot};

use omni_types::config::NetConfig;

use crate::swarm::{OmniSwarm, SwarmCommand};

/// Internal channel buffer. 256 slots absorbs short bursts without dropping
/// events under normal two-node LAN conditions.
const CHANNEL_CAPACITY: usize = 256;

// ── NetHandle ─────────────────────────────────────────────────────────────────

/// A cheap, cloneable handle to a running node.
///
/// This is the half of the old `OmniNet` that can safely be shared. It carries
/// the command sender (`mpsc::Sender` is `Clone`), the local peer id, and a
/// [`RouterHandle`] — which can *ask the router for* an event stream but can
/// never take one away from another consumer.
///
/// That asymmetry is the whole point. The event `mpsc::Receiver` is not
/// `Clone`, so consumers used to share it by wrapping the entire `OmniNet` in
/// `Arc<tokio::sync::Mutex<_>>` and taking turns — and taking turns on a
/// receiver means each consumer eats the events the others needed. Handing out
/// `NetHandle` clones instead means no consumer is holding the receiver at all:
/// it lives in the [`EventRouter`], and each consumer reads its own
/// [`Subscription`].
#[derive(Clone)]
pub struct NetHandle {
    cmd_tx:        mpsc::Sender<SwarmCommand>,
    router:        RouterHandle,
    local_peer_id: PeerId,
}

impl std::fmt::Debug for NetHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NetHandle")
            .field("local_peer_id", &self.local_peer_id)
            .field("router", &self.router)
            .finish()
    }
}

impl NetHandle {
    /// Register interest and receive this consumer's own event stream.
    ///
    /// Two consumers of one node must each call this. Sharing a single
    /// subscription between them reintroduces exactly the bug the router
    /// exists to remove.
    ///
    /// Fails with [`RouterStopped`] once the swarm and its router have exited,
    /// rather than returning a stream that would never yield.
    pub fn subscribe(&self, interests: Interests) -> Result<Subscription, RouterStopped> {
        self.router.subscribe(interests)
    }

    /// The router's counters — events seen, copies delivered, events nobody
    /// subscribed to, and deliveries that failed a subscriber who did.
    pub fn router_counts(&self) -> RouterCounts {
        self.router.counts()
    }

    /// The shadow byte accounting — what every event weighed, and what a
    /// budget at the shadow ceilings *would* have refused. Nothing was
    /// refused; see [`crate::budget`].
    pub fn byte_counts(&self) -> ByteCounts {
        self.router.byte_counts()
    }

    /// The router behind this handle.
    pub fn router(&self) -> &RouterHandle {
        &self.router
    }

    /// Local libp2p [`PeerId`]. See [`OmniNet::local_peer_id`].
    pub fn local_peer_id(&self) -> PeerId {
        self.local_peer_id
    }

    // ── Phase 1: Gossipsub ──────────────────────────────────────────────

    /// Publish `data` to the named Gossipsub topic.
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

    /// Request a shard chunk; the response arrives on the event stream as
    /// [`OmniNetEvent::ShardReceived`]. Prefer
    /// [`NetHandle::fetch_shard_chunk`], which delivers it to this caller
    /// alone.
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
                completion: None,
            })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot request shard"))
    }

    /// Send a shard response on a pending response channel.
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

    /// Send a tensor; the acknowledgment arrives on the event stream as
    /// [`OmniNetEvent::TensorResponseReceived`]. Prefer
    /// [`NetHandle::send_tensor`].
    pub async fn request_tensor(
        &self,
        peer_id: PeerId,
        request: TensorRequest,
    ) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::RequestTensor { peer_id, request, completion: None })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot send tensor"))
    }

    /// Send an acknowledgment on a pending tensor response channel.
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

    // ── Solicited requests: private completion ──────────────────────────

    /// Request a shard chunk and get a handle to *this* request's response.
    ///
    /// The response never reaches the router: it is delivered to the returned
    /// [`Pending`] and to nobody else. See [`OmniNet::fetch_shard_chunk`].
    pub async fn fetch_shard_chunk(
        &self,
        peer_id: PeerId,
        cid: String,
        offset: Option<u64>,
        max_bytes: Option<u64>,
    ) -> Pending<ShardResponse> {
        let (completion, rx) = oneshot::channel();
        let sent = self
            .cmd_tx
            .send(SwarmCommand::RequestShard {
                peer_id,
                request: ShardRequest {
                    cid,
                    offset,
                    max_bytes,
                },
                completion: Some(completion),
            })
            .await;
        match sent {
            Ok(()) => Pending::new(rx),
            // The swarm loop is gone, so the command — and with it the
            // completion channel we just handed over — was dropped. Answer
            // the caller now rather than let them await a closed channel.
            Err(_) => Pending::failed(RequestError::NotSent),
        }
    }

    /// Send a tensor and get a handle to *this* request's acknowledgment.
    pub async fn send_tensor(
        &self,
        peer_id: PeerId,
        request: TensorRequest,
    ) -> Pending<TensorResponse> {
        let (completion, rx) = oneshot::channel();
        let sent = self
            .cmd_tx
            .send(SwarmCommand::RequestTensor {
                peer_id,
                request,
                completion: Some(completion),
            })
            .await;
        match sent {
            Ok(()) => Pending::new(rx),
            Err(_) => Pending::failed(RequestError::NotSent),
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    /// Dial a peer at an explicit multiaddr, bypassing mDNS and the DHT.
    pub async fn dial(&self, addr: Multiaddr) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::Dial { addr })
            .await
            .map_err(|_| anyhow::anyhow!("swarm task has stopped — cannot dial"))
    }

    /// Signal the swarm loop to shut down gracefully.
    ///
    /// The loop completes every in-flight solicited request with
    /// [`RequestError::RouterGone`] and closes the event lane; the router then
    /// releases every subscription, so each consumer's `recv()` returns `None`
    /// instead of waiting on a stream that can no longer produce.
    pub async fn shutdown(&self) -> Result<()> {
        self.cmd_tx
            .send(SwarmCommand::Shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("swarm task already stopped"))
    }
}

// ── OmniNet ───────────────────────────────────────────────────────────────────

/// Top-level handle to the OmniNode P2P networking layer.
///
/// Owns a background `tokio` task running the [`swarm::OmniSwarm`] event loop,
/// and a second task running the [`EventRouter`]. Three channels connect them:
///
/// - `cmd_tx`   — commands (publish, shutdown, shard/tensor ops) **into** the loop
/// - the swarm's event lane — [`OmniNetEvent`]s **out of** the loop, owned end
///   to end by the router and reachable by nobody else
/// - one bounded channel per [`Subscription`], fed by the router
///
/// This value additionally carries a subscription to *everything*, which backs
/// [`OmniNet::next_event`]. That is the single-consumer shape: one CLI command
/// that owns its own mesh. **A node with more than one consumer must not use
/// it** — hand each consumer its own [`NetHandle::subscribe`] stream instead,
/// or they will each see only the events the other did not take.
///
/// # Example
/// ```rust,no_run
/// use omni_net::{Interests, OmniNet, OmniNetEvent, TOPIC_TEST};
/// use omni_types::config::NetConfig;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let node = OmniNet::new(NetConfig::default()).await?;
///     let net = node.handle();
///     let mut events = net.subscribe(Interests::none().topic(TOPIC_TEST))?;
///     net.publish(TOPIC_TEST, b"hello".to_vec()).await?;
///     while let Some(ev) = events.recv().await {
///         println!("{ev:?}");
///     }
///     Ok(())
/// }
/// ```
pub struct OmniNet {
    handle: NetHandle,
    /// This value's own all-interest subscription, backing
    /// [`OmniNet::next_event`]. Registered eagerly in [`OmniNet::new`] so no
    /// event that arrives before the first read is lost.
    events: Subscription,
}

impl OmniNet {
    /// Build the swarm, subscribe to all topics, and spawn the event loop and
    /// the router. Returns immediately — both run concurrently in `tokio`
    /// tasks.
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

        // The receiver is moved into the router here and is unreachable from
        // anywhere else for the rest of the process's life.
        let (router, router_handle) = EventRouter::new(event_rx);
        router.spawn();

        let handle = NetHandle {
            cmd_tx,
            router: router_handle,
            local_peer_id,
        };
        // Registered before returning, so events emitted between `new` and the
        // caller's first `next_event` are buffered rather than counted as
        // unsubscribed.
        let events = handle
            .subscribe(Interests::everything())
            .map_err(|e| anyhow::anyhow!("event router stopped during startup: {e}"))?;

        Ok(Self { handle, events })
    }

    /// A cheap, cloneable handle to this node.
    ///
    /// This is what a consumer should hold. It can publish, request, and
    /// subscribe, and it cannot take another consumer's events.
    pub fn handle(&self) -> NetHandle {
        self.handle.clone()
    }

    /// Register interest and receive a consumer's own event stream.
    pub fn subscribe(&self, interests: Interests) -> Result<Subscription, RouterStopped> {
        self.handle.subscribe(interests)
    }

    /// The router's counters. See [`NetHandle::router_counts`].
    pub fn router_counts(&self) -> RouterCounts {
        self.handle.router_counts()
    }

    /// The shadow byte accounting. See [`NetHandle::byte_counts`].
    pub fn byte_counts(&self) -> ByteCounts {
        self.handle.byte_counts()
    }

    /// Stage 12.5-pre — local libp2p [`PeerId`] for this node.
    /// Stable for the lifetime of this `OmniNet` instance.
    ///
    /// Used by Stage 12.5's `advertise-peer` subcommand to bind a
    /// signed `ContributorPeerAdvertisement` to the actual running
    /// network identity rather than an operator-supplied string.
    ///
    /// Persistence across restart depends on
    /// [`omni_types::config::NetConfig::identity`]:
    ///
    /// - `NetIdentity::Ephemeral` (default, pre-12.6 behavior):
    ///   `OmniNet::new` generates a fresh keypair via
    ///   `SwarmBuilder::with_new_identity`, so restart = new
    ///   PeerId. Stage 12.5 advertisements published before the
    ///   restart die.
    /// - `NetIdentity::KeypairProtobufBytes(_)` (Stage 12.6):
    ///   reuses a persistent libp2p identity. Two `OmniNet::new`
    ///   calls with the same bytes (e.g. across `omni-node`
    ///   restarts using the same `--net-identity-file`) yield the
    ///   same PeerId, so advertisements remain valid for their
    ///   full ≤24h freshness window.
    ///
    /// Stage 12.5 advertisements are still session-scoped and
    /// short-lived even under 12.6 persistence — this method does
    /// NOT turn them into permanent identity records.
    pub fn local_peer_id(&self) -> PeerId {
        self.handle.local_peer_id()
    }

    // ── Phase 1: Gossipsub ──────────────────────────────────────────────

    /// Publish `data` to the named Gossipsub topic.
    /// Sends the command to the background task and returns immediately.
    pub async fn publish(&self, topic: &str, data: Vec<u8>) -> Result<()> {
        self.handle.publish(topic, data).await
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
        self.handle
            .request_shard_chunk(peer_id, cid, offset, max_bytes)
            .await
    }

    /// Send a shard response on a pending response channel.
    ///
    /// `channel_id` is the ID received in [`OmniNetEvent::ShardRequested`].
    pub async fn respond_shard(
        &self,
        channel_id: u64,
        response: ShardResponse,
    ) -> Result<()> {
        self.handle.respond_shard(channel_id, response).await
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
        self.handle.request_tensor(peer_id, request).await
    }

    /// Send an acknowledgment on a pending tensor response channel.
    ///
    /// `channel_id` is the ID received in [`OmniNetEvent::TensorReceived`].
    pub async fn respond_tensor(
        &self,
        channel_id: u64,
        response: TensorResponse,
    ) -> Result<()> {
        self.handle.respond_tensor(channel_id, response).await
    }

    // ── Solicited requests: private completion ──────────────────────────

    /// Request a shard chunk and get a handle to *this* request's response.
    ///
    /// Unlike [`OmniNet::request_shard_chunk`], the response never appears on
    /// the shared event stream: it is delivered to the returned [`Pending`]
    /// and to nobody else. Two concurrent requests to the same peer therefore
    /// complete independently, and their responses may arrive in any order.
    ///
    /// The returned handle always resolves. Dropping it cancels the caller's
    /// interest and lets the swarm release the request's retained state.
    pub async fn fetch_shard_chunk(
        &self,
        peer_id: PeerId,
        cid: String,
        offset: Option<u64>,
        max_bytes: Option<u64>,
    ) -> Pending<ShardResponse> {
        self.handle
            .fetch_shard_chunk(peer_id, cid, offset, max_bytes)
            .await
    }

    /// Send a tensor and get a handle to *this* request's acknowledgment.
    ///
    /// Same delivery guarantee as [`OmniNet::fetch_shard_chunk`].
    pub async fn send_tensor(
        &self,
        peer_id: PeerId,
        request: TensorRequest,
    ) -> Pending<TensorResponse> {
        self.handle.send_tensor(peer_id, request).await
    }

    // ── Lifecycle ───────────────────────────────────────────────────────

    /// Dial a peer at an explicit multiaddr, bypassing mDNS and the DHT.
    ///
    /// Returns once the command reaches the swarm loop; the connection
    /// itself surfaces later as [`OmniNetEvent::PeerConnected`].
    pub async fn dial(&self, addr: Multiaddr) -> Result<()> {
        self.handle.dial(addr).await
    }

    /// Receive the next event on **this value's own** subscription.
    ///
    /// Returns `None` when the swarm task has stopped and the buffer is
    /// drained. It reads a private channel fed by the router, not the swarm's
    /// receiver, so it cannot consume another consumer's events — but it also
    /// only sees what arrived after this `OmniNet` was constructed. On a node
    /// with several consumers, give each one its own
    /// [`NetHandle::subscribe`] stream rather than routing them all through
    /// here.
    pub async fn next_event(&mut self) -> Option<OmniNetEvent> {
        self.events.recv().await
    }

    /// Stage 12.2-pre — non-blocking drain of this value's own subscription.
    ///
    /// Returns the next event immediately if one is queued, or
    /// `None` if the queue is currently empty (including the case
    /// where the swarm task has stopped and the buffer is drained).
    /// Distinct from `next_event` which awaits.
    ///
    /// Provided so synchronous consumers can poll from a non-async context
    /// without blocking. Same single-consumer caveat as
    /// [`OmniNet::next_event`].
    pub fn try_next_event(&mut self) -> Option<OmniNetEvent> {
        self.events.try_recv()
    }

    /// Signal the swarm loop to shut down gracefully.
    pub async fn shutdown(&self) -> Result<()> {
        self.handle.shutdown().await
    }

    /// Test-only constructor that builds an `OmniNet` from pre-built
    /// channels instead of standing up a full libp2p swarm. Used to
    /// unit-test the synchronous `try_next_event` accessor without
    /// requiring real networking. A router is spawned over `event_rx`
    /// exactly as in production, so the accessor exercises the real
    /// delivery path. The local peer id is synthesized from a fresh
    /// random keypair; tests that read `local_peer_id` should use
    /// [`OmniNet::new`] against a real (port-0) swarm instead — see
    /// `local_peer_id_tests` below.
    #[cfg(test)]
    pub(crate) fn from_test_channels(
        cmd_tx: mpsc::Sender<SwarmCommand>,
        event_rx: mpsc::Receiver<OmniNetEvent>,
    ) -> Self {
        let (router, router_handle) = EventRouter::new(event_rx);
        router.spawn();
        let handle = NetHandle {
            cmd_tx,
            router: router_handle,
            local_peer_id: PeerId::random(),
        };
        let events = handle
            .subscribe(Interests::everything())
            .expect("a freshly spawned router accepts subscriptions");
        Self { handle, events }
    }
}

// ── The router owns the swarm's event lane ────────────────────────────────
//
// One receiver, one owner. The whole point of this module's shape is that no
// consumer can reach the swarm's `mpsc::Receiver<OmniNetEvent>` — it is moved
// into the `EventRouter` in `OmniNet::new` and never surfaces again. If it
// did, two consumers could take turns on it, and taking turns on a receiver
// means each eats the other's events.
//
// The compiler enforces the move; these pin the rest.

#[cfg(test)]
mod router_owns_the_event_lane {
    use super::*;

    /// Resolved by the compiler against this very file.
    const LIB_SRC: &str = include_str!("lib.rs");

    /// Assembled at runtime so this module's own text cannot satisfy the scan.
    fn needle(head: &str, tail: &str) -> String {
        format!("{head}{tail}")
    }

    #[test]
    fn the_receiver_is_handed_to_the_router_and_nowhere_else() {
        // `EventRouter::new` takes the receiver by value. Anything else that
        // wanted it would have to take it by value too, and there is only one.
        let _: fn(mpsc::Receiver<OmniNetEvent>) -> (EventRouter, RouterHandle) =
            EventRouter::new;
    }

    #[test]
    fn nothing_in_this_module_drains_the_swarm_lane() {
        for (head, tail) in [
            ("event_rx.re", "cv()"),
            ("event_rx.try_re", "cv()"),
            ("event_rx.blocking_re", "cv()"),
        ] {
            let forbidden = needle(head, tail);
            assert!(
                !LIB_SRC.contains(&forbidden),
                "lib.rs contains `{forbidden}`: a consumer reading the swarm's \
                 receiver directly consumes events other consumers needed"
            );
        }
    }

    #[test]
    fn a_handle_can_ask_for_a_stream_but_cannot_take_one() {
        // `NetHandle` is `Clone`, which is only sound because none of what it
        // carries is a receiver: a command sender (`Clone`), a peer id, and a
        // router handle that hands out fresh channels.
        fn assert_clone<T: Clone>() {}
        assert_clone::<NetHandle>();
        assert_clone::<RouterHandle>();
        let _: fn(&NetHandle, Interests) -> Result<Subscription, RouterStopped> =
            NetHandle::subscribe;
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

    /// Poll `try_next_event` until the router has had its turn.
    async fn wait_for_event(net: &mut OmniNet) -> OmniNetEvent {
        for _ in 0..200 {
            if let Some(ev) = net.try_next_event() {
                return ev;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        panic!("event should have been routed to this subscription");
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
        // The event now takes one task hop: the swarm lane is drained by the
        // router, which fans the event out to this value's own subscription.
        // `try_next_event` is therefore eventually-consistent by a scheduler
        // turn — which is invisible to the polling loops that use it, but has
        // to be waited for here.
        let ev = wait_for_event(&mut net).await;
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
        // peer advertisements rely on (for the ephemeral path —
        // Stage 12.6 introduces a persistent path tested below).
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

    #[tokio::test]
    async fn persistent_identity_yields_stable_peer_id_across_instances() {
        // Stage 12.6 — the load-bearing property: given the same
        // libp2p keypair protobuf bytes, two `OmniNet::new` calls
        // produce the same `local_peer_id()`. This is what makes
        // Stage 12.5 peer advertisements survive `omni-node` restart.
        use libp2p::identity::Keypair;
        use omni_types::config::NetIdentity;
        let kp = Keypair::generate_ed25519();
        let bytes = kp.to_protobuf_encoding().expect("encode");
        let config_a = NetConfig {
            identity: NetIdentity::KeypairProtobufBytes(bytes.clone()),
            ..NetConfig::default()
        };
        let config_b = NetConfig {
            identity: NetIdentity::KeypairProtobufBytes(bytes),
            ..NetConfig::default()
        };
        let net_a = OmniNet::new(config_a).await.expect("first OmniNet");
        let net_b = OmniNet::new(config_b).await.expect("second OmniNet");
        assert_eq!(
            net_a.local_peer_id(),
            net_b.local_peer_id(),
            "same identity bytes must yield the same PeerId"
        );
        let _ = net_a.shutdown().await;
        let _ = net_b.shutdown().await;
    }

    #[tokio::test]
    async fn persistent_identity_two_different_files_have_different_peer_ids() {
        use libp2p::identity::Keypair;
        use omni_types::config::NetIdentity;
        let bytes_a = Keypair::generate_ed25519().to_protobuf_encoding().unwrap();
        let bytes_b = Keypair::generate_ed25519().to_protobuf_encoding().unwrap();
        assert_ne!(bytes_a, bytes_b, "two random keypairs must differ");
        let net_a = OmniNet::new(NetConfig {
            identity: NetIdentity::KeypairProtobufBytes(bytes_a),
            ..NetConfig::default()
        })
        .await
        .expect("first OmniNet");
        let net_b = OmniNet::new(NetConfig {
            identity: NetIdentity::KeypairProtobufBytes(bytes_b),
            ..NetConfig::default()
        })
        .await
        .expect("second OmniNet");
        assert_ne!(net_a.local_peer_id(), net_b.local_peer_id());
        let _ = net_a.shutdown().await;
        let _ = net_b.shutdown().await;
    }

    #[tokio::test]
    async fn omninet_new_rejects_malformed_identity_bytes() {
        // Garbage bytes must fail at `OmniNet::new` time rather
        // than silently falling back to a fresh identity. (Can't
        // use `expect_err` because `OmniNet` doesn't derive `Debug`;
        // match on the Result directly.)
        use omni_types::config::NetIdentity;
        let cfg = NetConfig {
            identity: NetIdentity::KeypairProtobufBytes(
                b"definitely not a libp2p Keypair protobuf".to_vec(),
            ),
            ..NetConfig::default()
        };
        match OmniNet::new(cfg).await {
            Ok(_) => panic!("OmniNet::new must fail on malformed identity bytes"),
            Err(e) => {
                let s = format!("{e:?}");
                assert!(
                    s.contains("malformed keypair protobuf bytes"),
                    "expected the swarm-builder branch's typed error, got: {s}"
                );
            }
        }
    }
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOC: test_alloc::Failing = test_alloc::Failing;
