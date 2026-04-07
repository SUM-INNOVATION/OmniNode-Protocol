use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::time::Duration;

use anyhow::{Context, Result};
use libp2p::{gossipsub, kad, mdns, Multiaddr, PeerId, StreamProtocol, Swarm};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::behaviour::OmniNodeBehaviour;
use crate::events::OmniNetEvent;

// ── Kademlia construction ────────────────────────────────────────────────────

/// Custom protocol string so OmniNode DHTs form their own namespace.
const KAD_PROTOCOL: &str = "/omni/kad/1.0.0";

/// Build a Kademlia behaviour with OmniNode-specific tuning.
///
/// - Server mode: every OmniNode is a full DHT participant.
/// - 20-way replication: aggressive for a young network.
/// - 1-hour record TTL with 10-minute republish keeps the DHT fresh.
pub fn build_kademlia(local_peer_id: PeerId) -> kad::Behaviour<kad::store::MemoryStore> {
    let store = kad::store::MemoryStore::new(local_peer_id);

    let mut config = kad::Config::new(StreamProtocol::new(KAD_PROTOCOL));
    config
        .set_query_timeout(Duration::from_secs(60))
        .set_record_ttl(Some(Duration::from_secs(3600)))
        .set_replication_factor(NonZeroUsize::new(20).unwrap())
        .set_publication_interval(Some(Duration::from_secs(600)))
        .set_provider_record_ttl(Some(Duration::from_secs(3600)));

    let mut behaviour = kad::Behaviour::with_config(local_peer_id, store, config);
    behaviour.set_mode(Some(kad::Mode::Server));
    behaviour
}

// ── DHT bootstrap ────────────────────────────────────────────────────────────

/// Extract PeerId from the trailing `/p2p/<peer_id>` component of a Multiaddr.
fn extract_peer_id(addr: &Multiaddr) -> Result<PeerId> {
    addr.iter()
        .find_map(|p| match p {
            libp2p::multiaddr::Protocol::P2p(peer_id) => Some(peer_id),
            _ => None,
        })
        .context("multiaddr missing /p2p/<peer_id> component")
}

/// Seed the Kademlia routing table from the bootstrap peer list, dial each
/// peer, and trigger a bootstrap query to discover the wider network.
///
/// Does nothing if `bootstrap_peers` is empty (LAN-only mode).
pub fn bootstrap_dht(
    swarm: &mut Swarm<OmniNodeBehaviour>,
    bootstrap_peers: &[String],
) -> Result<()> {
    if bootstrap_peers.is_empty() {
        return Ok(());
    }

    for addr_str in bootstrap_peers {
        let addr: Multiaddr = addr_str
            .parse()
            .with_context(|| format!("invalid bootstrap multiaddr: {addr_str}"))?;

        let peer_id = extract_peer_id(&addr)?;

        // Seed Kademlia routing table with this peer's address.
        swarm
            .behaviour_mut()
            .kademlia
            .add_address(&peer_id, addr.clone());

        // Dial so we get a live connection (Identify will fire, feeding more
        // addresses back into Kademlia).
        if let Err(e) = swarm.dial(addr.clone()) {
            warn!(%addr, %e, "failed to dial bootstrap peer — continuing");
        }
    }

    // Trigger the iterative Kademlia bootstrap query: finds k-closest peers
    // to our own PeerId, rapidly expanding the routing table.
    swarm
        .behaviour_mut()
        .kademlia
        .bootstrap()
        .context("kademlia bootstrap query failed to start")?;

    info!(
        count = bootstrap_peers.len(),
        "kademlia bootstrap initiated"
    );
    Ok(())
}

// ── Kademlia event handler ───────────────────────────────────────────────────

/// Process a single Kademlia event.
///
/// Peers added to the routing table are wired into Gossipsub and surfaced
/// as `PeerDiscovered` — the same event type mDNS uses, so downstream
/// consumers (omni-store, omni-pipeline) don't need WAN-specific logic.
pub fn handle_kademlia_event(
    event: kad::Event,
    gossipsub: &mut gossipsub::Behaviour,
    event_tx: &mpsc::Sender<OmniNetEvent>,
) {
    match event {
        kad::Event::RoutingUpdated {
            peer, addresses, ..
        } => {
            gossipsub.add_explicit_peer(&peer);
            let addrs = addresses.into_vec();
            info!(%peer, addr_count = addrs.len(), "kademlia routing table updated");
            if let Err(e) = event_tx.try_send(OmniNetEvent::PeerDiscovered {
                peer_id: peer,
                addrs,
            }) {
                warn!(%e, "event channel full — dropping PeerDiscovered (kad)");
            }
        }

        kad::Event::OutboundQueryProgressed { result, .. } => match result {
            kad::QueryResult::Bootstrap(Ok(kad::BootstrapOk {
                num_remaining, ..
            })) => {
                if num_remaining == 0 {
                    info!("kademlia bootstrap complete — routing table seeded");
                } else {
                    debug!(num_remaining, "kademlia bootstrap in progress");
                }
            }
            kad::QueryResult::Bootstrap(Err(e)) => {
                warn!(?e, "kademlia bootstrap query failed");
            }
            other => {
                debug!(?other, "kademlia query result");
            }
        },

        other => {
            debug!(?other, "kademlia event");
        }
    }
}

// ── mDNS event handler (unchanged from Phase 1) ─────────────────────────────

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
