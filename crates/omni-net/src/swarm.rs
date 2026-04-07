use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::StreamExt;
use libp2p::{
    dcutr, gossipsub, identify, mdns,
    request_response::{self, InboundRequestId, ProtocolSupport, ResponseChannel},
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
};

// ── SwarmCommand ──────────────────────────────────────────────────────────────

/// Commands sent from the [`crate::OmniNet`] handle into the running swarm loop.
#[derive(Debug)]
pub enum SwarmCommand {
    /// Publish bytes to a named Gossipsub topic.
    Publish { topic: String, data: Vec<u8> },

    /// Send a shard request to a remote peer.
    RequestShard { peer_id: PeerId, request: ShardRequest },

    /// Send a shard response on a stored response channel.
    SendShardResponse { channel_id: u64, response: ShardResponse },

    /// Send a tensor (hidden-state activation) to a remote pipeline stage.
    RequestTensor { peer_id: PeerId, request: TensorRequest },

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
        let mut swarm = SwarmBuilder::with_new_identity()
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
                        Some(SwarmCommand::RequestShard { peer_id, request }) => {
                            self.inner.behaviour_mut().shard_xfer
                                .send_request(&peer_id, request);
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
                        Some(SwarmCommand::RequestTensor { peer_id, request }) => {
                            self.inner.behaviour_mut().tensor_xfer
                                .send_request(&peer_id, request);
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
                            info!("swarm event loop shutting down");
                            return Ok(());
                        }
                    }
                }
            }
        }
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
                    request_response::Message::Response { response, .. } => {
                        info!(
                            %peer,
                            cid = %response.cid,
                            offset = response.offset,
                            bytes = response.data.len(),
                            "shard chunk received"
                        );
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
                request_response::Event::OutboundFailure { peer, error, .. }
            )) => {
                warn!(%peer, %error, "shard request outbound failure");
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
                    request_response::Message::Response { response, .. } => {
                        info!(
                            %peer,
                            session = %response.session_id,
                            micro_batch = response.micro_batch_index,
                            stage = response.stage_index,
                            accepted = response.accepted,
                            "tensor response received"
                        );
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
                request_response::Event::OutboundFailure { peer, error, .. }
            )) => {
                warn!(%peer, %error, "tensor request outbound failure");
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
