//! NAT traversal layer: AutoNAT probing, Circuit Relay v2, and DCUtR hole-punching.
//!
//! ## Flow
//!
//! 1. **AutoNAT** asks random DHT peers "can you dial me back?" to determine
//!    whether this node has a public address or is behind NAT.
//! 2. If **Private**, the node picks a relay peer and listens on a
//!    `/p2p-circuit` address so firewalled peers can reach it.
//! 3. **DCUtR** automatically upgrades relay circuits to direct QUIC
//!    connections via simultaneous UDP hole-punching when both ends support it.

use std::num::NonZeroU32;
use std::time::Duration;

use libp2p::{autonat, dcutr, relay, Multiaddr, PeerId, Swarm};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::behaviour::OmniNodeBehaviour;
use crate::events::OmniNetEvent;

// ── NAT status tracking ──────────────────────────────────────────────────────

/// Simplified NAT status for the swarm to reason about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NatStatus {
    Unknown,
    Public(Multiaddr),
    Private,
}

// ── AutoNAT construction ─────────────────────────────────────────────────────

/// Build the AutoNAT behaviour.
///
/// Probing cadence: every 60 s retry, 300 s refresh once confirmed.
/// Confidence threshold: 3 confirmations before declaring Public/Private.
pub fn build_autonat(local_peer_id: PeerId) -> autonat::Behaviour {
    let config = autonat::Config {
        retry_interval: Duration::from_secs(60),
        refresh_interval: Duration::from_secs(300),
        confidence_max: 3,
        throttle_server_period: Duration::from_secs(15),
        ..Default::default()
    };
    autonat::Behaviour::new(local_peer_id, config)
}

// ── Relay server construction ────────────────────────────────────────────────

/// Build the Circuit Relay v2 server behaviour.
///
/// When `enabled` is true, the node accepts relay reservations from firewalled
/// peers and relays short-lived circuits so DCUtR can upgrade them.
///
/// [Fix #3] Circuit bytes capped at 8 MiB — relay circuits are only meant to
/// last long enough for DCUtR to hole-punch (seconds, not minutes). Allowing
/// bulk tensor traffic (128 MiB) through the relay would DDoS volunteer nodes
/// before the direct connection upgrade succeeds.
///
/// Rate limits prevent a single peer from exhausting relay resources:
/// - Max 2 reservations per peer per 60 s
/// - Max 4 circuit opens per peer per 60 s
///
/// When `enabled` is false, `max_reservations: 0` means no clients are
/// accepted — the behaviour is inert but present in the NetworkBehaviour struct.
pub fn build_relay_server(local_peer_id: PeerId, enabled: bool) -> relay::Behaviour {
    let config = if enabled {
        relay::Config {
            max_reservations: 128,
            max_reservations_per_peer: 4,
            max_circuit_duration: Duration::from_secs(120),
            max_circuit_bytes: 1 << 23, // 8 MiB — tight cap for pre-upgrade relay
            ..Default::default()
        }
        // Rate-limit reservation requests: max 2 per peer per 60 s.
        .reservation_rate_per_peer(NonZeroU32::new(2).unwrap(), Duration::from_secs(60))
        // Rate-limit circuit opens: max 4 per source peer per 60 s.
        .circuit_src_per_peer(NonZeroU32::new(4).unwrap(), Duration::from_secs(60))
        // Rate-limit circuit opens per IP: prevents a single IP from DDoSing
        // the relay by spinning up multiple peer IDs.
        .circuit_src_per_ip(NonZeroU32::new(8).unwrap(), Duration::from_secs(60))
    } else {
        relay::Config {
            max_reservations: 0,
            ..Default::default()
        }
    };
    relay::Behaviour::new(local_peer_id, config)
}

// ── AutoNAT event handler ────────────────────────────────────────────────────

/// Process an AutoNAT event. When we transition to Private, attempt to
/// acquire a relay reservation from one of the known relay-capable peers.
///
/// [Fix #4] `active_relay` tracks whether we already hold a reservation,
/// preventing repeated requests on every `StatusChanged::Private` event.
pub fn handle_autonat_event(
    event: autonat::Event,
    relay_peers: &[PeerId],
    swarm: &mut Swarm<OmniNodeBehaviour>,
    nat_status: &mut NatStatus,
    active_relay: &mut Option<PeerId>,
    event_tx: &mpsc::Sender<OmniNetEvent>,
) {
    match event {
        autonat::Event::StatusChanged { old, new } => {
            info!(?old, ?new, "AutoNAT status changed");

            match new {
                autonat::NatStatus::Public(addr) => {
                    *nat_status = NatStatus::Public(addr.clone());
                    // We're public — clear any active relay reservation.
                    *active_relay = None;
                    let _ = event_tx.try_send(OmniNetEvent::NatStatusChanged {
                        is_public: true,
                        public_addr: Some(addr),
                    });
                }

                autonat::NatStatus::Private => {
                    *nat_status = NatStatus::Private;
                    let _ = event_tx.try_send(OmniNetEvent::NatStatusChanged {
                        is_public: false,
                        public_addr: None,
                    });

                    // [Fix #4] Only request a reservation if we don't already
                    // have one — avoids spamming the relay on repeated events.
                    if active_relay.is_some() {
                        debug!("NAT still private — relay reservation already active, skipping");
                        return;
                    }

                    if let Some(relay_peer) = relay_peers.first() {
                        request_relay_reservation(swarm, *relay_peer);
                        *active_relay = Some(*relay_peer);
                    } else {
                        warn!("NAT is private but no relay peers known — unreachable over WAN");
                    }
                }

                autonat::NatStatus::Unknown => {
                    *nat_status = NatStatus::Unknown;
                }
            }
        }

        other => {
            debug!(?other, "autonat event");
        }
    }
}

/// Ask the swarm to listen on a relay circuit address.
///
/// The address format is: `/p2p/<relay_peer_id>/p2p-circuit`
/// This tells libp2p's relay client transport to open a reservation with
/// the relay peer, giving us a publicly-reachable circuit address.
fn request_relay_reservation(
    swarm: &mut Swarm<OmniNodeBehaviour>,
    relay_peer: PeerId,
) {
    let relay_addr: Multiaddr = format!("/p2p/{}/p2p-circuit", relay_peer)
        .parse()
        .expect("static relay circuit multiaddr format is always valid");

    match swarm.listen_on(relay_addr.clone()) {
        Ok(_) => {
            info!(%relay_peer, "requesting relay reservation");
        }
        Err(e) => {
            warn!(%relay_peer, %e, "failed to listen on relay circuit");
        }
    }
}

// ── Relay server event handler ───────────────────────────────────────────────

/// Process relay server events. Mostly logging — the relay behaviour is
/// self-contained and doesn't need external state.
pub fn handle_relay_server_event(event: relay::Event) {
    match event {
        relay::Event::ReservationReqAccepted {
            src_peer_id,
            renewed,
            ..
        } => {
            info!(
                %src_peer_id,
                renewed,
                "relay reservation accepted for remote peer"
            );
        }
        relay::Event::ReservationReqDenied { src_peer_id, .. } => {
            debug!(%src_peer_id, "relay reservation denied (limit reached)");
        }
        relay::Event::CircuitReqDenied {
            src_peer_id,
            dst_peer_id,
            ..
        } => {
            debug!(%src_peer_id, %dst_peer_id, "relay circuit denied");
        }
        other => {
            debug!(?other, "relay server event");
        }
    }
}

// ── Relay client event handler ───────────────────────────────────────────────

/// Process relay client events. The critical one is `ReservationReqAccepted`
/// — it means we now have a publicly-reachable relay address.
pub fn handle_relay_client_event(
    event: relay::client::Event,
    event_tx: &mpsc::Sender<OmniNetEvent>,
) {
    match event {
        relay::client::Event::ReservationReqAccepted {
            relay_peer_id, ..
        } => {
            info!(
                %relay_peer_id,
                "relay reservation accepted — we are now reachable via circuit"
            );
            // Build the circuit address for the event.
            let relay_addr: Multiaddr = format!("/p2p/{}/p2p-circuit", relay_peer_id)
                .parse()
                .expect("static relay circuit multiaddr is always valid");

            let _ = event_tx.try_send(OmniNetEvent::RelayReservation {
                relay_peer_id,
                relay_addr,
            });
        }
        other => {
            debug!(?other, "relay client event");
        }
    }
}

// ── DCUtR event handler ──────────────────────────────────────────────────────

/// Process DCUtR (Direct Connection Upgrade through Relay) events.
///
/// dcutr::Event is a struct with `remote_peer_id` and `result: Result<ConnectionId, Error>`.
/// A successful upgrade means the relay circuit has been replaced with a
/// direct QUIC connection via UDP hole-punching.
///
/// [Fix #4] On failure, emit `HolePunchFailed` so higher layers can react
/// (e.g., fall back to relay-only mode, alert the user).
pub fn handle_dcutr_event(
    event: dcutr::Event,
    event_tx: &mpsc::Sender<OmniNetEvent>,
) {
    match event.result {
        Ok(_connection_id) => {
            info!(
                remote_peer_id = %event.remote_peer_id,
                "DCUtR hole-punch succeeded — direct connection established"
            );
            let _ = event_tx.try_send(OmniNetEvent::HolePunchSucceeded {
                peer_id: event.remote_peer_id,
            });
        }
        Err(ref error) => {
            warn!(
                remote_peer_id = %event.remote_peer_id,
                %error,
                "DCUtR hole-punch failed — staying on relay circuit"
            );
            let _ = event_tx.try_send(OmniNetEvent::HolePunchFailed {
                peer_id: event.remote_peer_id,
                error: error.to_string(),
            });
        }
    }
}
