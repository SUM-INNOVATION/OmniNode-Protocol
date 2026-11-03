use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::time::Duration;

use anyhow::{Context, Result};
use libp2p::{gossipsub, kad, mdns, Multiaddr, PeerId, StreamProtocol, Swarm};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::behaviour::OmniNodeBehaviour;
use crate::events::OmniNetEvent;
use crate::policy::{NetPolicy, PolicySource};

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
/// Does nothing if `bootstrap_peers` is empty (LAN-only mode). An entry the
/// outbound allow-list refuses (issue #158) is neither stored nor dialled; if
/// every entry is refused, no bootstrap query is started.
pub fn bootstrap_dht(
    swarm: &mut Swarm<OmniNodeBehaviour>,
    bootstrap_peers: &[String],
    policy: &NetPolicy,
) -> Result<()> {
    if bootstrap_peers.is_empty() {
        return Ok(());
    }

    let mut admitted = 0usize;
    for addr_str in bootstrap_peers {
        let addr: Multiaddr = addr_str
            .parse()
            .with_context(|| format!("invalid bootstrap multiaddr: {addr_str}"))?;

        let peer_id = extract_peer_id(&addr)?;
        if !policy.admit(PolicySource::Bootstrap, &addr, Some(peer_id)) {
            continue;
        }
        admitted += 1;

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

    if admitted == 0 {
        warn!("every bootstrap peer is outside the outbound allow-list; no kademlia bootstrap");
        return Ok(());
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
///
/// Issue #158: addresses the outbound allow-list refuses are removed from the
/// routing table at once (the last one removes the peer), so they are never
/// redialled. A peer left with no allowed address is not wired into Gossipsub
/// or reported.
pub fn handle_kademlia_event(
    event: kad::Event,
    behaviour: &mut OmniNodeBehaviour,
    policy: &NetPolicy,
    event_tx: &mpsc::Sender<OmniNetEvent>,
) {
    match event {
        kad::Event::RoutingUpdated {
            peer, addresses, ..
        } => {
            let mut addrs = addresses.into_vec();
            if policy.is_restricted() {
                let (allowed, refused): (Vec<_>, Vec<_>) = addrs
                    .into_iter()
                    .partition(|a| policy.admit(PolicySource::Kad, a, Some(peer)));
                for addr in &refused {
                    behaviour.kademlia.remove_address(&peer, addr);
                }
                if allowed.is_empty() {
                    debug!(%peer, "kademlia peer has no allowed address — not wired");
                    return;
                }
                addrs = allowed;
            }
            behaviour.gossipsub.add_explicit_peer(&peer);
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
///
/// Issue #158: a discovered address the outbound allow-list refuses is
/// dropped; a peer with no allowed address is neither made an explicit
/// gossipsub peer (which gossipsub would redial every heartbeat) nor reported.
pub fn handle_mdns_event(
    event: mdns::Event,
    gossipsub: &mut gossipsub::Behaviour,
    policy: &NetPolicy,
    event_tx: &mpsc::Sender<OmniNetEvent>,
) {
    match event {
        mdns::Event::Discovered(list) => {
            // Group all addresses by PeerId — a peer may be reachable on
            // multiple addresses, but we emit one event per peer.
            let mut by_peer: HashMap<PeerId, Vec<_>> = HashMap::new();
            for (peer_id, addr) in list {
                if !policy.admit(PolicySource::Mdns, &addr, Some(peer_id)) {
                    continue;
                }
                by_peer.entry(peer_id).or_default().push(addr);
            }
            for (peer_id, addrs) in by_peer {
                gossipsub.add_explicit_peer(&peer_id);
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

/// Issue #158: refused addresses never reach the peer stores.
#[cfg(test)]
mod policy_tests {
    use super::*;
    use crate::policy::{NetPolicy, OutboundAllowList};

    fn restricted(items: &[&str]) -> NetPolicy {
        NetPolicy::new(true, OutboundAllowList::parse(items).unwrap())
    }

    fn gossipsub() -> gossipsub::Behaviour {
        gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(libp2p::identity::Keypair::generate_ed25519()),
            gossipsub::Config::default(),
        )
        .unwrap()
    }

    fn discovered(policy: &NetPolicy, list: Vec<(PeerId, &str)>) -> Vec<(PeerId, Vec<Multiaddr>)> {
        let (tx, mut rx) = mpsc::channel(16);
        let list = list
            .into_iter()
            .map(|(p, a)| (p, a.parse().unwrap()))
            .collect();
        handle_mdns_event(mdns::Event::Discovered(list), &mut gossipsub(), policy, &tx);
        let mut out = Vec::new();
        while let Ok(OmniNetEvent::PeerDiscovered { peer_id, addrs }) = rx.try_recv() {
            out.push((peer_id, addrs));
        }
        out.sort_by_key(|(p, _)| *p);
        out
    }

    /// An mDNS discovery keeps only allowed addresses, and a peer with none is
    /// not reported (nor made an explicit gossipsub peer, which gossipsub
    /// would redial every heartbeat).
    #[test]
    fn mdns_drops_refused_addresses_and_peers_with_none_left() {
        let (mixed, refused_only) = (PeerId::random(), PeerId::random());
        let policy = restricted(&["127.0.0.1"]);
        let got = discovered(
            &policy,
            vec![
                (mixed, "/ip4/127.0.0.1/udp/4001/quic-v1"),
                (mixed, "/ip4/192.168.1.5/udp/4001/quic-v1"),
                (refused_only, "/ip4/192.168.1.6/udp/4001/quic-v1"),
            ],
        );
        assert_eq!(
            got,
            vec![(
                mixed,
                vec!["/ip4/127.0.0.1/udp/4001/quic-v1".parse().unwrap()]
            )]
        );
        assert_eq!(policy.refusals().mdns, 2);
    }

    #[test]
    fn mdns_without_an_allow_list_reports_every_address() {
        let (a, b) = (PeerId::random(), PeerId::random());
        let policy = NetPolicy::unrestricted();
        let got = discovered(
            &policy,
            vec![
                (a, "/ip4/127.0.0.1/udp/4001/quic-v1"),
                (a, "/ip4/192.168.1.5/udp/4001/quic-v1"),
                (b, "/ip4/192.168.1.6/udp/4001/quic-v1"),
            ],
        );
        assert_eq!(got.len(), 2);
        assert_eq!(got.iter().map(|(_, a)| a.len()).sum::<usize>(), 3);
        assert_eq!(policy.refusals().total(), 0);
    }
}
