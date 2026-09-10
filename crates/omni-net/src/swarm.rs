use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    dcutr, gossipsub, identify, mdns,
    request_response::{
        self, InboundRequestId, OutboundRequestId, ProtocolSupport, ResponseChannel,
    },
    swarm::SwarmEvent,
    Multiaddr, PeerId, SwarmBuilder,
};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use omni_types::config::NetConfig;

use crate::{
    behaviour::{OmniNodeBehaviour, OmniNodeBehaviourEvent},
    codec::{ShardRequest, ShardResponse, SHARD_XFER_PROTOCOL},
    tensor_codec::{TensorRequest, TensorResponse, TENSOR_XFER_PROTOCOL},
    discovery,
    events::OmniNetEvent,
    gossip::GossipManager,
    nat::{self, NatStatus},
    request::{Completion, PendingRequests, RequestError},
};

// ── SwarmCommand ──────────────────────────────────────────────────────────────

/// Commands sent from the [`crate::OmniNet`] handle into the running swarm loop.
#[derive(Debug)]
pub enum SwarmCommand {
    /// Publish bytes to a named Gossipsub topic.
    Publish { topic: String, data: Vec<u8> },

    /// Dial a peer at an explicit multiaddr, bypassing discovery.
    Dial { addr: Multiaddr },

    /// Send a shard request to a remote peer.
    ///
    /// `completion` is this request's private delivery channel. When it is
    /// `Some`, the response — or the failure — is handed to that one caller
    /// and is NOT published on the shared event stream, so two concurrent
    /// requests to the same peer cannot be confused for one another. `None`
    /// keeps the pre-correlation behaviour: the result is broadcast as
    /// [`OmniNetEvent::ShardReceived`] / [`OmniNetEvent::ShardRequestFailed`].
    RequestShard {
        peer_id: PeerId,
        request: ShardRequest,
        completion: Option<Completion<ShardResponse>>,
    },

    /// Send a shard response on a stored response channel.
    SendShardResponse { channel_id: u64, response: ShardResponse },

    /// Send a tensor (hidden-state activation) to a remote pipeline stage.
    ///
    /// `completion` carries the same meaning as on
    /// [`SwarmCommand::RequestShard`].
    RequestTensor {
        peer_id: PeerId,
        request: TensorRequest,
        completion: Option<Completion<TensorResponse>>,
    },

    /// Send a tensor acknowledgment on a stored response channel.
    SendTensorResponse { channel_id: u64, response: TensorResponse },

    /// Exit the event loop cleanly.
    Shutdown,
}

// ── PendingInbound ───────────────────────────────────────────────────────────

/// Tracks an inbound request's response channel alongside its libp2p
/// `InboundRequestId` so we can clean up state on `InboundFailure` or
/// `ResponseSent` — preventing a memory leak if the caller never responds.
struct PendingInbound<T> {
    channel: ResponseChannel<T>,
    request_id: InboundRequestId,
}

// ── OmniSwarm ─────────────────────────────────────────────────────────────────

/// Owns the [`libp2p::Swarm`] and the [`GossipManager`].
/// Constructed by [`OmniSwarm::build`] and consumed by [`OmniSwarm::run`].
pub struct OmniSwarm {
    inner:  libp2p::Swarm<OmniNodeBehaviour>,
    gossip: GossipManager,

    /// Inbound shard channels keyed by our monotonic channel_id.
    pending_shard_channels: HashMap<u64, PendingInbound<ShardResponse>>,
    /// Reverse index: InboundRequestId → channel_id for cleanup on failure/sent.
    pending_shard_by_req: HashMap<InboundRequestId, u64>,

    /// Inbound tensor channels keyed by our monotonic channel_id.
    pending_tensor_channels: HashMap<u64, PendingInbound<TensorResponse>>,
    /// Reverse index: InboundRequestId → channel_id for cleanup on failure/sent.
    pending_tensor_by_req: HashMap<InboundRequestId, u64>,

    /// Monotonic counter shared across shard and tensor channel IDs.
    next_channel_id: u64,

    // ── Outbound correlation ─────────────────────────────────────────────
    //
    // Keyed by the `OutboundRequestId` that `send_request` returns — the only
    // identifier that distinguishes two concurrent requests to one peer. Both
    // tables are emptied on every exit from `run`, so a caller is never left
    // waiting on a response the loop can no longer deliver.
    /// Callers waiting on a solicited shard response.
    pending_shard_requests: PendingRequests<OutboundRequestId, ShardResponse>,
    /// Callers waiting on a solicited tensor acknowledgment.
    pending_tensor_requests: PendingRequests<OutboundRequestId, TensorResponse>,

    // ── WAN state ────────────────────────────────────────────────────────

    /// Peers known to have open NATs — candidates for relay reservations.
    relay_peers: Vec<PeerId>,

    /// Current NAT status as determined by AutoNAT.
    nat_status: NatStatus,

    /// The relay peer we currently hold a reservation with, if any.
    /// Prevents spamming the same relay on repeated StatusChanged::Private.
    active_relay_reservation: Option<PeerId>,
}

impl OmniSwarm {
    /// Construct and configure the Swarm via the libp2p 0.55 `SwarmBuilder` API.
    ///
    /// Transport:  QUIC + relay-client (for /p2p-circuit addresses)
    /// Behaviour:  mDNS + Gossipsub + Identify + Kademlia + AutoNAT +
    ///             Relay (server) + Relay (client) + DCUtR +
    ///             shard xfer + tensor xfer
    /// Listener:   `0.0.0.0:<config.listen_port>` (0 = OS-assigned)
    pub fn build(config: &NetConfig) -> Result<Self> {
        let gossip_cfg = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(Duration::from_secs(10))
            .validation_mode(gossipsub::ValidationMode::Strict)
            .history_length(10)
            .history_gossip(3)
            .build()
            .map_err(|msg| anyhow::anyhow!("gossipsub config error: {msg}"))?;

        let relay_server_enabled = config.relay_server;

        // The `.with_relay_client()` call wraps the QUIC transport in a
        // relay-aware transport that can dial and listen on /p2p-circuit
        // multiaddrs. It injects a `relay::client::Behaviour` into the
        // behaviour closure.
        //
        // Relay circuits run over TCP-like streams, so they need Noise + Yamux
        // even though our primary transport is QUIC (which has encryption
        // and multiplexing baked in).
        // Stage 12.6 — branch on the operator's identity policy.
        // `Ephemeral` reproduces pre-12.6 behavior (fresh Ed25519
        // each `OmniNet::new`); `KeypairProtobufBytes(_)` decodes
        // an existing libp2p Keypair so `local_peer_id()` is
        // stable across restarts.
        use omni_types::config::NetIdentity;
        let identity_phase = match &config.identity {
            NetIdentity::Ephemeral => SwarmBuilder::with_new_identity(),
            NetIdentity::KeypairProtobufBytes(bytes) => {
                let kp = crate::identity::decode_keypair_protobuf(bytes)
                    .map_err(|e| anyhow::anyhow!(
                        "NetConfig.identity: malformed keypair protobuf bytes: {e}"
                    ))?;
                SwarmBuilder::with_existing_identity(kp)
            }
        };
        let mut swarm = identity_phase
            .with_tokio()
            .with_quic()
            .with_relay_client(
                libp2p::noise::Config::new,
                libp2p::yamux::Config::default,
            )?
            .with_behaviour(|key, relay_client| {
                let local_peer_id = key.public().to_peer_id();

                // ── Existing protocols ────────────────────────────────
                let mdns = mdns::tokio::Behaviour::new(
                    mdns::Config::default(),
                    local_peer_id,
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

                let shard_xfer = request_response::Behaviour::new(
                    [(SHARD_XFER_PROTOCOL.to_string(), ProtocolSupport::Full)],
                    request_response::Config::default()
                        .with_request_timeout(Duration::from_secs(120)),
                );

                let tensor_xfer = request_response::Behaviour::new(
                    [(TENSOR_XFER_PROTOCOL.to_string(), ProtocolSupport::Full)],
                    request_response::Config::default()
                        .with_request_timeout(Duration::from_secs(60)),
                );

                // ── New WAN protocols ─────────────────────────────────
                let kademlia = discovery::build_kademlia(local_peer_id);
                let autonat = nat::build_autonat(local_peer_id);
                let relay = nat::build_relay_server(local_peer_id, relay_server_enabled);
                let dcutr = dcutr::Behaviour::new(local_peer_id);

                Ok(OmniNodeBehaviour {
                    mdns,
                    gossipsub,
                    identify,
                    shard_xfer,
                    tensor_xfer,
                    kademlia,
                    autonat,
                    relay,
                    relay_client,
                    dcutr,
                })
            })?
            .with_swarm_config(|c: libp2p::swarm::Config| {
                // Keep idle QUIC connections alive between pipeline requests.
                c.with_idle_connection_timeout(Duration::from_secs(60))
            })
            .build();

        // ── Bind QUIC listener ───────────────────────────────────────────
        let listen_addr: Multiaddr =
            format!("/ip4/0.0.0.0/udp/{}/quic-v1", config.listen_port)
                .parse()
                .context("invalid QUIC listen multiaddr")?;

        swarm
            .listen_on(listen_addr)
            .context("failed to bind QUIC listener")?;

        // ── Seed DHT from bootstrap peers ────────────────────────────────
        if !config.bootstrap_peers.is_empty() {
            discovery::bootstrap_dht(&mut swarm, &config.bootstrap_peers)?;
        }

        Ok(Self {
            inner: swarm,
            gossip: GossipManager::new(),
            pending_shard_channels: HashMap::new(),
            pending_shard_by_req: HashMap::new(),
            pending_tensor_channels: HashMap::new(),
            pending_tensor_by_req: HashMap::new(),
            next_channel_id: 0,
            pending_shard_requests: PendingRequests::new(),
            pending_tensor_requests: PendingRequests::new(),
            relay_peers: Vec::new(),
            nat_status: NatStatus::Unknown,
            active_relay_reservation: None,
        })
    }

    /// Subscribe the node to all OmniNode Gossipsub topics.
    pub fn subscribe_all_topics(&mut self) -> Result<()> {
        self.gossip
            .subscribe_all(&mut self.inner.behaviour_mut().gossipsub)
    }

    /// Stage 12.5-pre — return the local libp2p `PeerId`. Cheap copy.
    /// Internally delegates to `Swarm::local_peer_id`; captured by
    /// `OmniNet::new` before the run loop is spawned so callers can
    /// read it without re-entering the swarm task.
    pub fn local_peer_id(&self) -> PeerId {
        *self.inner.local_peer_id()
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
    ///
    /// The loop never awaits a consumer. Unsolicited events go out through
    /// `event_tx.try_send`, and solicited results go out through a `oneshot`,
    /// which also never blocks. Awaiting either here would deadlock the node:
    /// consumers issue requests from inside their event handling, and those
    /// commands are serviced only by this loop.
    ///
    /// On every exit — clean shutdown or a dropped command channel — every
    /// retained request is completed with [`RequestError::RouterGone`].
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
                        Some(SwarmCommand::Dial { addr }) => {
                            if let Err(e) = self.inner.dial(addr.clone()) {
                                warn!(%e, %addr, "dial failed");
                            }
                        }
                        Some(SwarmCommand::RequestShard { peer_id, request, completion }) => {
                            // Capture the id the moment the request exists.
                            // This is the only point at which the caller's
                            // completion channel can be tied to the wire.
                            let request_id = self.inner.behaviour_mut().shard_xfer
                                .send_request(&peer_id, request);
                            if let Some(completion) = completion {
                                self.pending_shard_requests.insert(request_id, completion);
                                debug!(
                                    %request_id,
                                    inflight = self.pending_shard_requests.len(),
                                    "shard request correlated"
                                );
                            }
                        }
                        Some(SwarmCommand::SendShardResponse { channel_id, response }) => {
                            if let Some(pending) = self.pending_shard_channels.remove(&channel_id) {
                                self.pending_shard_by_req.remove(&pending.request_id);
                                if let Err(resp) = self.inner.behaviour_mut().shard_xfer
                                    .send_response(pending.channel, response)
                                {
                                    warn!(cid = %resp.cid, "failed to send shard response — channel closed");
                                }
                            } else {
                                warn!(channel_id, "no pending channel for shard response");
                            }
                        }
                        Some(SwarmCommand::RequestTensor { peer_id, request, completion }) => {
                            let request_id = self.inner.behaviour_mut().tensor_xfer
                                .send_request(&peer_id, request);
                            if let Some(completion) = completion {
                                self.pending_tensor_requests.insert(request_id, completion);
                                debug!(
                                    %request_id,
                                    inflight = self.pending_tensor_requests.len(),
                                    "tensor request correlated"
                                );
                            }
                        }
                        Some(SwarmCommand::SendTensorResponse { channel_id, response }) => {
                            if let Some(pending) = self.pending_tensor_channels.remove(&channel_id) {
                                self.pending_tensor_by_req.remove(&pending.request_id);
                                if let Err(resp) = self.inner.behaviour_mut().tensor_xfer
                                    .send_response(pending.channel, response)
                                {
                                    warn!(
                                        session = %resp.session_id,
                                        "failed to send tensor response — channel closed"
                                    );
                                }
                            } else {
                                warn!(channel_id, "no pending channel for tensor response");
                            }
                        }
                        Some(SwarmCommand::Shutdown) | None => {
                            let released = self.release_pending_requests();
                            info!(released, "swarm event loop shutting down");
                            return Ok(());
                        }
                    }
                }
            }
        }
    }

    // ── Private: outbound correlation teardown ───────────────────────────────

    /// Complete every retained outbound request with
    /// [`RequestError::RouterGone`] and empty both tables.
    ///
    /// Returns how many callers were released. Dropping the tables instead
    /// would also wake the callers — a dropped `oneshot::Sender` closes the
    /// channel — but doing it explicitly makes the reason legible to the
    /// caller and keeps the count observable.
    fn release_pending_requests(&mut self) -> usize {
        let shard = self
            .pending_shard_requests
            .fail_all(RequestError::RouterGone);
        let tensor = self
            .pending_tensor_requests
            .fail_all(RequestError::RouterGone);
        shard + tensor
    }

    // ── Private: inbound channel cleanup ─────────────────────────────────────

    /// Remove a shard inbound channel by its libp2p request ID.
    /// Called on InboundFailure and ResponseSent to prevent memory leaks.
    fn cleanup_shard_inbound(&mut self, request_id: &InboundRequestId) {
        if let Some(channel_id) = self.pending_shard_by_req.remove(request_id) {
            self.pending_shard_channels.remove(&channel_id);
            debug!(%channel_id, %request_id, "cleaned up shard inbound channel");
        }
    }

    /// Remove a tensor inbound channel by its libp2p request ID.
    fn cleanup_tensor_inbound(&mut self, request_id: &InboundRequestId) {
        if let Some(channel_id) = self.pending_tensor_by_req.remove(request_id) {
            self.pending_tensor_channels.remove(&channel_id);
            debug!(%channel_id, %request_id, "cleaned up tensor inbound channel");
        }
    }

    // ── Private event dispatcher ──────────────────────────────────────────────

    fn handle_swarm_event(
        &mut self,
        event:    SwarmEvent<OmniNodeBehaviourEvent>,
        event_tx: &mpsc::Sender<OmniNetEvent>,
    ) {
        match event {
            // ── mDNS ──────────────────────────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Mdns(e)) => {
                discovery::handle_mdns_event(
                    e,
                    &mut self.inner.behaviour_mut().gossipsub,
                    event_tx,
                );
            }

            // ── Gossipsub: incoming message ────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Gossipsub(
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
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Gossipsub(e)) => {
                debug!(?e, "gossipsub mesh event");
            }

            // ── Identify ──────────────────────────────────────────────────
            // CRITICAL: feed identified addresses into Kademlia so the DHT
            // routing table grows beyond just bootstrap peers.
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Identify(
                identify::Event::Received { peer_id, info, .. }
            )) => {
                // Feed every listen address into Kademlia.
                for addr in &info.listen_addrs {
                    self.inner
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                }

                // [Fix #2] Register the observed address as a local external
                // address candidate so AutoNAT and relay circuits use the
                // correct public-facing address.
                self.inner.add_external_address(info.observed_addr.clone());

                // If the peer supports the relay protocol, track it as a
                // candidate relay for NAT traversal.
                let supports_relay = info.protocols.iter().any(|p| {
                    p.as_ref().contains("relay")
                });
                if supports_relay && !self.relay_peers.contains(&peer_id) {
                    self.relay_peers.push(peer_id);
                    debug!(%peer_id, "identified as relay-capable peer");
                }

                debug!(
                    %peer_id,
                    observed_addr = %info.observed_addr,
                    protocols = ?info.protocols,
                    addrs = info.listen_addrs.len(),
                    "identify received — fed addresses into kademlia"
                );
            }

            // Other Identify events (Sent, Pushed, Error).
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Identify(e)) => {
                debug!(?e, "identify event");
            }

            // ── Kademlia DHT ──────────────────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Kademlia(e)) => {
                discovery::handle_kademlia_event(
                    e,
                    &mut self.inner.behaviour_mut().gossipsub,
                    event_tx,
                );
            }

            // ── AutoNAT ──────────────────────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Autonat(e)) => {
                // [Fix #4] Pass active_relay_reservation to prevent repeated
                // reservation requests to the same relay.
                nat::handle_autonat_event(
                    e,
                    &self.relay_peers,
                    &mut self.inner,
                    &mut self.nat_status,
                    &mut self.active_relay_reservation,
                    event_tx,
                );
            }

            // ── Relay server ──────────────────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Relay(e)) => {
                nat::handle_relay_server_event(e);
            }

            // ── Relay client ─────────────────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::RelayClient(e)) => {
                nat::handle_relay_client_event(e, event_tx);
            }

            // ── DCUtR (hole-punching) ─────────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::Dcutr(e)) => {
                nat::handle_dcutr_event(e, event_tx);
            }

            // ── Shard transfer (Phase 2) ──────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::ShardXfer(
                request_response::Event::Message { peer, message, .. }
            )) => {
                match message {
                    request_response::Message::Request { request_id, request, channel, .. } => {
                        let channel_id = self.next_channel_id;
                        self.next_channel_id += 1;
                        info!(
                            %peer,
                            cid = %request.cid,
                            channel_id,
                            "inbound shard request"
                        );
                        // [Fix #1] Only store the channel if the event was
                        // successfully delivered. If the channel is full the
                        // caller will never see the request, so holding the
                        // ResponseChannel would leak forever.
                        match event_tx.try_send(OmniNetEvent::ShardRequested {
                            peer_id: peer,
                            request,
                            channel_id,
                        }) {
                            Ok(()) => {
                                self.pending_shard_channels.insert(channel_id, PendingInbound {
                                    channel,
                                    request_id,
                                });
                                self.pending_shard_by_req.insert(request_id, channel_id);
                            }
                            Err(e) => {
                                warn!(%e, "event channel full — dropping ShardRequested");
                                drop(channel);
                            }
                        }
                    }
                    request_response::Message::Response {
                        request_id,
                        response,
                    } => {
                        info!(
                            %peer,
                            %request_id,
                            cid = %response.cid,
                            offset = response.offset,
                            bytes = response.data.len(),
                            "shard chunk received"
                        );
                        // A solicited response belongs to exactly one caller.
                        // Delivering it on the shared event stream instead
                        // would make it indistinguishable from the response
                        // to any other request to the same peer.
                        if self.pending_shard_requests.contains(&request_id) {
                            self.pending_shard_requests
                                .complete(&request_id, Ok(response));
                            return;
                        }
                        if let Err(e) = event_tx.try_send(OmniNetEvent::ShardReceived {
                            peer_id: peer,
                            response,
                        }) {
                            warn!(%e, "event channel full — dropping ShardReceived");
                        }
                    }
                }
            }

            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::ShardXfer(
                request_response::Event::OutboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                warn!(%peer, %request_id, %error, "shard request outbound failure");
                // Includes libp2p's own request timeout, so a caller whose
                // peer simply went silent is still completed rather than
                // left waiting.
                if self.pending_shard_requests.contains(&request_id) {
                    self.pending_shard_requests.complete(
                        &request_id,
                        Err(RequestError::Outbound {
                            peer: peer.to_string(),
                            error: error.to_string(),
                        }),
                    );
                    return;
                }
                if let Err(e) = event_tx.try_send(OmniNetEvent::ShardRequestFailed {
                    peer_id: peer,
                    error: error.to_string(),
                }) {
                    warn!(%e, "event channel full — dropping ShardRequestFailed");
                }
            }

            // [Fix #1] Clean up leaked channel on inbound failure.
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::ShardXfer(
                request_response::Event::InboundFailure { peer, request_id, error, .. }
            )) => {
                self.cleanup_shard_inbound(&request_id);
                debug!(%peer, %request_id, %error, "shard inbound failure — channel cleaned up");
            }

            // [Fix #1] Clean up channel after successful response send.
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::ShardXfer(
                request_response::Event::ResponseSent { peer, request_id, .. }
            )) => {
                self.cleanup_shard_inbound(&request_id);
                debug!(%peer, %request_id, "shard response sent — channel cleaned up");
            }

            // ── Tensor transfer (Phase 4) ─────────────────────────────────
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::TensorXfer(
                request_response::Event::Message { peer, message, .. }
            )) => {
                match message {
                    request_response::Message::Request { request_id, request, channel, .. } => {
                        let channel_id = self.next_channel_id;
                        self.next_channel_id += 1;
                        info!(
                            %peer,
                            session = %request.session_id,
                            micro_batch = request.micro_batch_index,
                            from_stage = request.from_stage,
                            to_stage = request.to_stage,
                            bytes = request.data.len(),
                            "inbound tensor request"
                        );
                        // [Fix #1] Only store the channel if the event was
                        // successfully delivered — same pattern as shard.
                        match event_tx.try_send(OmniNetEvent::TensorReceived {
                            peer_id: peer,
                            request,
                            channel_id,
                        }) {
                            Ok(()) => {
                                self.pending_tensor_channels.insert(channel_id, PendingInbound {
                                    channel,
                                    request_id,
                                });
                                self.pending_tensor_by_req.insert(request_id, channel_id);
                            }
                            Err(e) => {
                                warn!(%e, "event channel full — dropping TensorReceived");
                                drop(channel);
                            }
                        }
                    }
                    request_response::Message::Response {
                        request_id,
                        response,
                    } => {
                        info!(
                            %peer,
                            %request_id,
                            session = %response.session_id,
                            micro_batch = response.micro_batch_index,
                            stage = response.stage_index,
                            accepted = response.accepted,
                            "tensor response received"
                        );
                        if self.pending_tensor_requests.contains(&request_id) {
                            self.pending_tensor_requests
                                .complete(&request_id, Ok(response));
                            return;
                        }
                        if let Err(e) = event_tx.try_send(OmniNetEvent::TensorResponseReceived {
                            peer_id: peer,
                            response,
                        }) {
                            warn!(%e, "event channel full — dropping TensorResponseReceived");
                        }
                    }
                }
            }

            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::TensorXfer(
                request_response::Event::OutboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                warn!(%peer, %request_id, %error, "tensor request outbound failure");
                if self.pending_tensor_requests.contains(&request_id) {
                    self.pending_tensor_requests.complete(
                        &request_id,
                        Err(RequestError::Outbound {
                            peer: peer.to_string(),
                            error: error.to_string(),
                        }),
                    );
                    return;
                }
                if let Err(e) = event_tx.try_send(OmniNetEvent::TensorRequestFailed {
                    peer_id: peer,
                    error: error.to_string(),
                }) {
                    warn!(%e, "event channel full — dropping TensorRequestFailed");
                }
            }

            // [Fix #1] Clean up leaked channel on inbound failure.
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::TensorXfer(
                request_response::Event::InboundFailure { peer, request_id, error, .. }
            )) => {
                self.cleanup_tensor_inbound(&request_id);
                debug!(%peer, %request_id, %error, "tensor inbound failure — channel cleaned up");
            }

            // [Fix #1] Clean up channel after successful response send.
            SwarmEvent::Behaviour(OmniNodeBehaviourEvent::TensorXfer(
                request_response::Event::ResponseSent { peer, request_id, .. }
            )) => {
                self.cleanup_tensor_inbound(&request_id);
                debug!(%peer, %request_id, "tensor response sent — channel cleaned up");
            }

            // ── Transport ─────────────────────────────────────────────────
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

// ── Never-await invariant ─────────────────────────────────────────────────────
//
// There is a real deadlock cycle behind this: a consumer issues a request from
// inside its handling of an event (omni-store's `FetchManager::process` calls
// `request_chunk`, which blocks on `cmd_tx.send`), and that command is
// serviced only by this loop. If the loop ever awaited a congested consumer —
// an awaiting send on the event lane instead of `try_send` — the loop would be
// waiting on the consumer while the consumer waits on the loop, and the node
// stops.
//
// A test that *triggers* the deadlock cannot exist: it either hangs the suite
// or passes without proving anything. So the property is pinned structurally
// instead — once by the compiler, once by reading the source.

#[cfg(test)]
mod never_await_invariant {
    use super::*;

    /// Reading swarm.rs from disk would make the test pass vacuously if the
    /// file moved; `include_str!` is resolved by the compiler against this
    /// very file.
    const SWARM_SRC: &str = include_str!("swarm.rs");

    /// Needles are assembled at runtime so this test module's own text can
    /// never satisfy the scan it performs.
    fn needle(head: &str, tail: &str) -> String {
        format!("{head}{tail}")
    }

    #[test]
    fn handle_swarm_event_is_not_async() {
        // An `async fn` returns an opaque future, which cannot coerce to a
        // function pointer with a `()` return type. This line therefore stops
        // compiling the moment someone makes the dispatcher awaitable — which
        // is the only way an `.await` could be introduced inside it.
        let _: fn(&mut OmniSwarm, SwarmEvent<OmniNodeBehaviourEvent>, &mpsc::Sender<OmniNetEvent>) =
            OmniSwarm::handle_swarm_event;
    }

    /// What the swarm hands to a waiting caller.
    type ShardOutcome = Result<ShardResponse, RequestError>;
    /// The exact shape of a non-blocking, non-awaitable delivery call.
    type SyncDelivery = fn(Completion<ShardResponse>, ShardOutcome) -> Result<(), ShardOutcome>;

    #[test]
    fn completion_delivery_is_synchronous() {
        // The solicited path must be non-blocking too. `oneshot::Sender::send`
        // consumes self and returns a plain `Result`, so it can neither await
        // nor block; pin that in the type system.
        let _: SyncDelivery = Completion::<ShardResponse>::send;
    }

    #[test]
    fn the_swarm_loop_never_blocks_on_the_event_lane() {
        for (head, tail) in [
            // An awaiting send on the event lane — waits on a congested
            // consumer, which is the deadlock.
            ("event_tx.se", "nd("),
            // A blocking send — stalls the whole runtime worker instead.
            ("blocking_se", "nd("),
            // A cloned sender is the same hazard wearing a different name.
            ("event_tx.clone().se", "nd("),
        ] {
            let forbidden = needle(head, tail);
            assert!(
                !SWARM_SRC.contains(&forbidden),
                "swarm.rs contains `{forbidden}`: the swarm loop would then \
                 wait on a consumer that is itself waiting on the swarm loop"
            );
        }
    }

    #[test]
    fn unsolicited_events_are_delivered_with_try_send() {
        // The positive half of the invariant: delivery happens, and it happens
        // through the non-blocking call.
        let try_send = needle("event_tx.try_", "send(");
        assert!(
            SWARM_SRC.matches(&try_send).count() >= 8,
            "expected every unsolicited event to be delivered via `{try_send}`"
        );
    }
}
