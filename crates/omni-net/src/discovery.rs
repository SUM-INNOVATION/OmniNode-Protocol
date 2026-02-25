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
