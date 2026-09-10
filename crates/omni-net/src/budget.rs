//! Shadow byte accounting for the event router.
//!
//! The router already counts *events*. An event count cannot tell a flood of
//! empty keep-alives from a single peer streaming activation tensors at line
//! rate, and it is bytes — not events — that a future admission budget would
//! have to spend. This module measures those bytes.
//!
//! ## Shadow mode, and only shadow mode
//!
//! Nothing here refuses anything. [`ByteLedger::charge`] returns a receipt,
//! not a verdict: the router charges the copy, reads what a hypothetical
//! budget *would* have said, records that, and delivers the copy regardless.
//! Not one event's fate differs from the commit before this one. The point is
//! to answer "what would a budget have refused?" from production traffic
//! before anything is given the power to refuse, because a limit chosen
//! without that answer is a limit chosen by guessing.
//!
//! ## Seen, and retained: two different questions
//!
//! An event is *seen* once — the router takes it off the swarm lane exactly
//! once — and that is what [`ByteLedger::weigh`] records into the cumulative
//! totals ([`ByteCounts::events`], [`ByteCounts::bytes`]).
//!
//! What a budget would actually be spending is something else: memory held.
//! A copy is held for as long as it sits in a subscriber's queue waiting to
//! be read, and each interested subscriber gets its **own** copy with its own
//! payload allocation. So the in-flight figures ([`ByteCounts::bytes_in_flight`])
//! are charged per retained copy, by [`ByteLedger::charge`], and released when
//! that copy leaves the queue — dequeued, dropped when the queue was full,
//! dropped with a departed subscriber, or dropped with the subscription
//! itself. Three interested subscribers means three charges; nobody
//! interested means none at all, because nothing is being held.
//!
//! Charging the fan-out instead — one charge, released as soon as the
//! synchronous fan-out returned — would have measured the size of a single
//! routing call and called it "bytes in flight". Against a 32 MiB ceiling
//! that is not an admission budget; it is a test for whether one event is
//! larger than 32 MiB.
//!
//! ## Why the arithmetic is spelled out
//!
//! A counter that wraps is worse than no counter: it reports a small number
//! with total confidence at exactly the moment the number matters. So there
//! are no `as` casts on any path that could truncate — `usize` reaches `u64`
//! through [`u64::try_from`] with an explicit ceiling — and every accumulation
//! saturates rather than wraps.
//!
//! Saturation, not `checked_add`-and-give-up, because the two failure modes
//! are not symmetric. A saturated total is wrong in the safe direction: it
//! says "at least `u64::MAX` bytes", which a budget would refuse, and it stays
//! monotonically non-decreasing so a delta between two readings is never
//! negative. Every saturation is itself counted
//! ([`ByteCounts::bytes_saturated`]), so a reading that has hit the ceiling
//! announces itself instead of quietly lying. At the observed rates the
//! ceiling is unreachable — 2^64 bytes is roughly 16 exabytes, some 58 years
//! of a saturated 100 Gb/s link — so a non-zero `bytes_saturated` means a
//! weight bug, not a busy node.
//!
//! ## Cardinality
//!
//! The reported metrics ([`ByteCounts`]) are fixed struct fields. There is no
//! map keyed by peer, topic, CID or session in anything this module exports: a
//! remote peer picks those keys, so a map keyed by one is unbounded memory
//! growth wearing a metrics label, and an attacker needs only a fresh identity
//! per event to grow it.
//!
//! Per-peer observation is therefore structural, not keyed. The one map that
//! exists — [`InFlight::per_peer`] — is **bounded by retained work, never by
//! peers ever seen**: an entry is created when a peer's first byte is charged
//! and removed the instant its charge returns to zero. Its size is bounded by
//! how many charged copies are alive at once, and a charge is alive exactly
//! while a subscriber's bounded queue is holding the copy it paid for — so
//! the ceiling is the router's own structure, `subscribers ×
//! SUBSCRIBER_CAPACITY`, not traffic. A node that has seen ten million
//! distinct peers whose subscribers are all drained holds zero entries. That
//! bound is proved in
//! `the_peer_map_is_bounded_by_in_flight_not_by_peers_ever_seen`.
//!
//! What escapes into [`ByteCounts`] from that map is only aggregate: how many
//! entries there are, the high-water mark of that count, and the largest
//! single-peer charge ever seen. Fixed cardinality, whatever the peer set.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use libp2p::{Multiaddr, PeerId};

use crate::events::OmniNetEvent;

// ── Weights ───────────────────────────────────────────────────────────────────

/// Charged to every event before any payload is added.
///
/// Stands for what an event costs the node no matter how small it is: a slot
/// in the swarm lane, a slot in every interested subscriber's channel, the
/// enum itself, and the wake-ups on both sides. Without a floor, an infinite
/// flood of payload-free events — `PeerConnected`, `PeerExpired`,
/// `HolePunchSucceeded` — would be free, and "free" is precisely the property
/// a flood is looking for.
pub const EVENT_FLOOR_BYTES: u64 = 64;

/// Widen a length to `u64` without an `as` cast.
///
/// On every target this crate builds for `usize` is at most 64 bits, so the
/// fallback is unreachable; it exists so that a hypothetical 128-bit target
/// saturates instead of truncating, which is the direction that keeps the
/// number honest.
fn widen(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// Bytes attributable to a multiaddress.
fn addr_bytes(addr: &Multiaddr) -> u64 {
    widen(addr.len())
}

/// Bytes attributable to an optional error string.
fn error_bytes(error: &Option<String>) -> u64 {
    error.as_ref().map_or(0, |e| widen(e.len()))
}

/// What one event costs, in bytes, for accounting purposes.
///
/// Every payload-bearing variant weighs its payload; every variant carries
/// [`EVENT_FLOOR_BYTES`] on top, so no variant can weigh zero.
///
/// The match is exhaustive with no wildcard arm, deliberately. Adding a
/// variant to [`OmniNetEvent`] is then a compile error here rather than a new
/// kind of traffic that silently weighs nothing — the failure mode a `_ => 0`
/// arm would introduce is exactly the one this accounting exists to catch.
///
/// The figure approximates the wire payload, not `size_of` the enum: heap
/// payloads are what scale with traffic and what a budget would be spending.
/// Accumulation saturates, so a pathological event cannot wrap the total.
pub fn weight(event: &OmniNetEvent) -> u64 {
    let payload = match event {
        OmniNetEvent::Listening { addr } => addr_bytes(addr),

        OmniNetEvent::PeerDiscovered { addrs, .. } => addrs
            .iter()
            .fold(0u64, |acc, addr| acc.saturating_add(addr_bytes(addr))),

        OmniNetEvent::PeerExpired { .. }
        | OmniNetEvent::PeerConnected { .. }
        | OmniNetEvent::PeerDisconnected { .. } => 0,

        OmniNetEvent::MessageReceived { topic, data, .. } => {
            widen(topic.len()).saturating_add(widen(data.len()))
        }

        OmniNetEvent::ShardRequested { request, .. } => widen(request.cid.len()),

        OmniNetEvent::ShardReceived { response, .. } => widen(response.cid.len())
            .saturating_add(widen(response.data.len()))
            .saturating_add(error_bytes(&response.error)),

        OmniNetEvent::ShardRequestFailed { error, .. } => widen(error.len()),

        OmniNetEvent::TensorReceived { request, .. } => {
            widen(request.session_id.len()).saturating_add(widen(request.data.len()))
        }

        OmniNetEvent::TensorResponseReceived { response, .. } => {
            widen(response.session_id.len()).saturating_add(error_bytes(&response.error))
        }

        OmniNetEvent::TensorRequestFailed { error, .. } => widen(error.len()),

        OmniNetEvent::NatStatusChanged { public_addr, .. } => {
            public_addr.as_ref().map_or(0, addr_bytes)
        }

        OmniNetEvent::RelayReservation { relay_addr, .. } => addr_bytes(relay_addr),

        OmniNetEvent::HolePunchSucceeded { .. } => 0,

        OmniNetEvent::HolePunchFailed { error, .. } => widen(error.len()),
    };

    EVENT_FLOOR_BYTES.saturating_add(payload)
}

/// The remote peer an event is attributable to, if any.
///
/// `Listening` and `NatStatusChanged` describe this node rather than a remote
/// one, so they have no peer to charge; they are counted as unattributed. The
/// match is exhaustive with no wildcard for the same reason [`weight`]'s is.
pub fn attributed_peer(event: &OmniNetEvent) -> Option<PeerId> {
    match event {
        OmniNetEvent::Listening { .. } | OmniNetEvent::NatStatusChanged { .. } => None,

        OmniNetEvent::PeerDiscovered { peer_id, .. }
        | OmniNetEvent::PeerExpired { peer_id }
        | OmniNetEvent::PeerConnected { peer_id }
        | OmniNetEvent::PeerDisconnected { peer_id }
        | OmniNetEvent::ShardRequested { peer_id, .. }
        | OmniNetEvent::ShardReceived { peer_id, .. }
        | OmniNetEvent::ShardRequestFailed { peer_id, .. }
        | OmniNetEvent::TensorReceived { peer_id, .. }
        | OmniNetEvent::TensorResponseReceived { peer_id, .. }
        | OmniNetEvent::TensorRequestFailed { peer_id, .. }
        | OmniNetEvent::HolePunchSucceeded { peer_id }
        | OmniNetEvent::HolePunchFailed { peer_id, .. } => Some(*peer_id),

        OmniNetEvent::MessageReceived { from, .. } => Some(*from),

        OmniNetEvent::RelayReservation { relay_peer_id, .. } => Some(*relay_peer_id),
    }
}

// ── The shadow budget ─────────────────────────────────────────────────────────

/// Shadow ceiling on bytes the router is carrying at once, across all peers.
///
/// Not enforced. Crossing it increments [`ByteCounts::would_refuse_global`]
/// and nothing else happens.
pub const SHADOW_GLOBAL_BYTES: u64 = 32 * 1024 * 1024;

/// Shadow ceiling on bytes the router is carrying at once for any one peer.
///
/// Not enforced. Crossing it increments [`ByteCounts::would_refuse_peer`].
pub const SHADOW_PEER_BYTES: u64 = 4 * 1024 * 1024;

// ── Readings ──────────────────────────────────────────────────────────────────

/// A reading of the shadow accounting.
///
/// Fixed fields only — no per-peer, per-topic or per-session entries escape
/// here. See the module docs on cardinality.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ByteCounts {
    /// Events weighed.
    pub events: u64,
    /// Bytes weighed, cumulative. Saturating; see `bytes_saturated`.
    pub bytes: u64,
    /// How many times the cumulative total hit `u64::MAX` and stopped
    /// growing. Non-zero means the reading below is a floor, not a total —
    /// and, at any real traffic rate, means a weight bug.
    pub bytes_saturated: u64,
    /// Events with no remote peer to charge — this node's own listen
    /// addresses and NAT status.
    pub unattributed_events: u64,

    // ── In-flight, across all peers ──────────────────────────────────────
    /// Bytes currently charged and not yet released.
    pub bytes_in_flight: u64,
    /// Highest `bytes_in_flight` ever observed.
    pub peak_bytes_in_flight: u64,

    // ── In-flight, per peer, reported only in aggregate ──────────────────
    /// How many peers currently hold a non-zero charge. Bounded by concurrent
    /// routing work, not by peers ever seen.
    pub peers_in_flight: u64,
    /// Highest `peers_in_flight` ever observed. The measured bound on the
    /// per-peer map.
    pub peak_peers_in_flight: u64,
    /// Largest charge any single peer has ever held at once.
    pub peak_peer_bytes_in_flight: u64,

    // ── What a budget would have said ────────────────────────────────────
    /// Events that would have pushed total in-flight bytes past
    /// [`SHADOW_GLOBAL_BYTES`]. Recorded, never acted on.
    pub would_refuse_global: u64,
    /// Events that would have pushed one peer's in-flight bytes past
    /// [`SHADOW_PEER_BYTES`]. Recorded, never acted on.
    pub would_refuse_peer: u64,
}

impl ByteCounts {
    /// Whether a budget would have refused this traffic, either way.
    pub fn would_refuse_any(&self) -> bool {
        self.would_refuse_global > 0 || self.would_refuse_peer > 0
    }
}

// ── The ledger ────────────────────────────────────────────────────────────────

/// The in-flight state, behind one lock.
///
/// `total` and `per_peer` must move together or a reading could show a total
/// that no set of entries adds up to, so they share a lock rather than being
/// separate atomics.
#[derive(Debug, Default)]
struct InFlight {
    /// Bytes charged and not yet released, across all peers.
    total: u64,
    /// Bytes charged and not yet released, per peer.
    ///
    /// The one keyed map in this module. Bounded by in-flight work: an entry
    /// appears when a peer's charge goes above zero and is **removed** when it
    /// returns to zero, so its size tracks concurrent routing, not the set of
    /// peers ever seen. `BTreeMap` rather than `HashMap` because the keys come
    /// from remote peers and ordered lookup has no collision behaviour for
    /// them to choose.
    per_peer: BTreeMap<PeerId, u64>,
}

/// Shadow byte accounting: weighs events, records what a budget would say,
/// and refuses nothing.
///
/// Synchronous throughout. It sits inside the router's fan-out, which must
/// never await (see the router's `never_await_invariant`), so nothing here may
/// either.
#[derive(Debug, Default)]
pub struct ByteLedger {
    events: AtomicU64,
    bytes: AtomicU64,
    bytes_saturated: AtomicU64,
    unattributed_events: AtomicU64,
    peak_bytes_in_flight: AtomicU64,
    peak_peers_in_flight: AtomicU64,
    peak_peer_bytes_in_flight: AtomicU64,
    would_refuse_global: AtomicU64,
    would_refuse_peer: AtomicU64,
    in_flight: Mutex<InFlight>,
}

impl ByteLedger {
    /// A ledger with everything at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Take the in-flight state, recovering from a poisoned lock.
    ///
    /// A panic elsewhere in the router must not take byte accounting — or,
    /// through it, event delivery — down with it. The state behind the lock is
    /// a counter and a map with no invariant a panic could leave half-built,
    /// so recovering is safe and strictly better than propagating. Matches how
    /// the router treats its own subscriber table.
    fn in_flight(&self) -> std::sync::MutexGuard<'_, InFlight> {
        self.in_flight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record that `event` was *seen*, and return what it weighed.
    ///
    /// Cumulative totals only — [`ByteCounts::events`], [`ByteCounts::bytes`],
    /// [`ByteCounts::unattributed_events`]. No in-flight charge is applied,
    /// because being seen is not being held: an event nobody wants is weighed
    /// here and retained nowhere.
    ///
    /// Call this exactly once per event, where the event enters. Retention is
    /// [`ByteLedger::charge`], called once per copy that is actually kept.
    pub fn weigh(&self, event: &OmniNetEvent) -> u64 {
        let weight = weight(event);

        self.events.fetch_add(1, Ordering::Relaxed);
        if attributed_peer(event).is_none() {
            self.unattributed_events.fetch_add(1, Ordering::Relaxed);
        }

        // Cumulative bytes: saturating, with the saturation itself counted so
        // a pinned reading cannot pass for a real one.
        let previous = self
            .bytes
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
                Some(total.saturating_add(weight))
            })
            .expect("the update closure always returns Some");
        if previous.checked_add(weight).is_none() {
            self.bytes_saturated.fetch_add(1, Ordering::Relaxed);
        }

        weight
    }

    /// Charge one *retained copy* of `event`, and record what a budget would
    /// have said about admitting it.
    ///
    /// Returns a [`Charge`] receipt, never a verdict: the caller keeps the
    /// copy whatever the receipt says. The charge stays open until the receipt
    /// is dropped, so the receipt must live exactly as long as the copy it
    /// paid for — put it in the same envelope and let them die together. A
    /// receipt that outlives its copy is a leak that reads as permanent
    /// in-flight bytes; one that dies first under-reports memory actually
    /// held.
    ///
    /// The receipt owns a handle on the ledger rather than borrowing it,
    /// precisely so it can outlive the call that made it and sit in a queue.
    ///
    /// Cumulative totals are untouched here: an event is weighed once by
    /// [`ByteLedger::weigh`] however many copies of it are kept.
    #[must_use = "dropping the receipt immediately releases the charge; bind it \
                  into the envelope that holds the copy it paid for"]
    pub fn charge(self: &Arc<Self>, event: &OmniNetEvent) -> Charge {
        let weight = weight(event);
        let peer = attributed_peer(event);

        let mut in_flight = self.in_flight();

        // The shadow verdict: exactly the test a real budget would run —
        // "would admitting this take me over the ceiling?" — evaluated before
        // the charge is applied, and then ignored.
        let peer_before = peer
            .and_then(|p| in_flight.per_peer.get(&p).copied())
            .unwrap_or(0);
        if in_flight.total.saturating_add(weight) > SHADOW_GLOBAL_BYTES {
            self.would_refuse_global.fetch_add(1, Ordering::Relaxed);
        }
        if peer.is_some() && peer_before.saturating_add(weight) > SHADOW_PEER_BYTES {
            self.would_refuse_peer.fetch_add(1, Ordering::Relaxed);
        }

        // Charge. The receipt remembers the amount that was *actually* added
        // rather than the weight, so that even in the unreachable saturated
        // case release brings the counter back to exactly where it started
        // and the "entry removed at zero" invariant holds.
        let total_before = in_flight.total;
        in_flight.total = total_before.saturating_add(weight);
        let charged_total = in_flight.total - total_before;
        self.peak_bytes_in_flight
            .fetch_max(in_flight.total, Ordering::Relaxed);

        let charged_peer = match peer {
            Some(peer_id) => {
                let entry = in_flight.per_peer.entry(peer_id).or_insert(0);
                let after = entry.saturating_add(weight);
                let charged = after - *entry;
                *entry = after;
                self.peak_peer_bytes_in_flight
                    .fetch_max(after, Ordering::Relaxed);
                charged
            }
            None => 0,
        };
        self.peak_peers_in_flight
            .fetch_max(widen(in_flight.per_peer.len()), Ordering::Relaxed);
        drop(in_flight);

        Charge {
            ledger: Arc::clone(self),
            peer,
            weight,
            charged_total,
            charged_peer,
        }
    }

    /// Weigh `event` and charge a single retained copy of it.
    ///
    /// The whole transaction for a caller that sees an event once and keeps
    /// exactly one copy. A caller that keeps N copies — the router, which
    /// hands one to every interested subscriber — calls [`ByteLedger::weigh`]
    /// once and [`ByteLedger::charge`] N times instead.
    #[must_use = "dropping the receipt immediately releases the charge; bind it \
                  into the envelope that holds the copy it paid for"]
    pub fn observe(self: &Arc<Self>, event: &OmniNetEvent) -> Charge {
        self.weigh(event);
        self.charge(event)
    }

    /// Release a charge. Called only by [`Charge::drop`].
    fn release(&self, peer: Option<PeerId>, charged_total: u64, charged_peer: u64) {
        let mut in_flight = self.in_flight();
        in_flight.total = in_flight.total.saturating_sub(charged_total);
        if let Some(peer_id) = peer {
            if let std::collections::btree_map::Entry::Occupied(mut entry) =
                in_flight.per_peer.entry(peer_id)
            {
                let remaining = entry.get().saturating_sub(charged_peer);
                if remaining == 0 {
                    // The bound: the entry goes away the moment the peer stops
                    // owing anything, so the map holds in-flight peers rather
                    // than every peer ever accounted.
                    entry.remove();
                } else {
                    *entry.get_mut() = remaining;
                }
            }
        }
    }

    /// Read the counters.
    pub fn snapshot(&self) -> ByteCounts {
        let (bytes_in_flight, peers_in_flight) = {
            let in_flight = self.in_flight();
            (in_flight.total, widen(in_flight.per_peer.len()))
        };
        ByteCounts {
            events: self.events.load(Ordering::Relaxed),
            bytes: self.bytes.load(Ordering::Relaxed),
            bytes_saturated: self.bytes_saturated.load(Ordering::Relaxed),
            unattributed_events: self.unattributed_events.load(Ordering::Relaxed),
            bytes_in_flight,
            peak_bytes_in_flight: self.peak_bytes_in_flight.load(Ordering::Relaxed),
            peers_in_flight,
            peak_peers_in_flight: self.peak_peers_in_flight.load(Ordering::Relaxed),
            peak_peer_bytes_in_flight: self.peak_peer_bytes_in_flight.load(Ordering::Relaxed),
            would_refuse_global: self.would_refuse_global.load(Ordering::Relaxed),
            would_refuse_peer: self.would_refuse_peer.load(Ordering::Relaxed),
        }
    }

    /// How many peers currently hold a charge. The live size of the one keyed
    /// map in this module.
    pub fn peers_in_flight(&self) -> usize {
        self.in_flight().per_peer.len()
    }
}

/// A receipt for one *retained copy* of an event.
///
/// Holds the charge open for as long as the copy is held and releases it on
/// `Drop`. It owns a handle on the ledger, so it can travel with the copy it
/// paid for — into a subscriber's queue and out the other side — instead of
/// expiring with the call that issued it.
///
/// It carries no permission and grants none: a copy is delivered whether or
/// not [`Charge::would_be_refused`] is true.
#[derive(Debug)]
pub struct Charge {
    ledger: Arc<ByteLedger>,
    peer: Option<PeerId>,
    weight: u64,
    charged_total: u64,
    charged_peer: u64,
}

impl Charge {
    /// What the event this copy is of weighed.
    pub fn weight(&self) -> u64 {
        self.weight
    }

    /// Whether a budget at the shadow ceilings would have refused this event.
    ///
    /// Observation only. No caller acts on it, and the router delivers the
    /// copy either way — that is the whole meaning of "shadow".
    pub fn would_be_refused(&self) -> bool {
        self.charged_total > 0 && {
            let in_flight = self.ledger.in_flight();
            in_flight.total > SHADOW_GLOBAL_BYTES
                || self
                    .peer
                    .and_then(|p| in_flight.per_peer.get(&p).copied())
                    .is_some_and(|bytes| bytes > SHADOW_PEER_BYTES)
        }
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.ledger
            .release(self.peer, self.charged_total, self.charged_peer);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codec::{ShardRequest, ShardResponse};
    use crate::tensor_codec::{TensorRequest, TensorResponse};

    fn addr() -> Multiaddr {
        "/ip4/127.0.0.1/udp/4001/quic-v1"
            .parse()
            .expect("a well-formed loopback QUIC multiaddress")
    }

    fn gossip(topic: &str, bytes: usize) -> OmniNetEvent {
        OmniNetEvent::MessageReceived {
            from: PeerId::random(),
            topic: topic.to_string(),
            data: vec![0u8; bytes],
        }
    }

    fn gossip_from(peer: PeerId, bytes: usize) -> OmniNetEvent {
        OmniNetEvent::MessageReceived {
            from: peer,
            topic: "omni/weights/v1".to_string(),
            data: vec![0u8; bytes],
        }
    }

    fn tensor(bytes: usize) -> OmniNetEvent {
        OmniNetEvent::TensorReceived {
            peer_id: PeerId::random(),
            request: TensorRequest {
                session_id: "session".to_string(),
                micro_batch_index: 0,
                from_stage: 0,
                to_stage: 1,
                seq_len: 1,
                hidden_dim: 1,
                dtype: 0,
                data: vec![0u8; bytes],
            },
            channel_id: 1,
        }
    }

    fn shard_response(bytes: usize) -> OmniNetEvent {
        OmniNetEvent::ShardReceived {
            peer_id: PeerId::random(),
            response: ShardResponse {
                cid: "cid".to_string(),
                offset: 0,
                total_bytes: 0,
                data: vec![0u8; bytes],
                error: None,
            },
        }
    }

    /// One of every [`OmniNetEvent`] variant.
    ///
    /// Kept exhaustive by hand — the compiler cannot force a list to cover
    /// every variant — but the two functions it feeds *are* exhaustive
    /// matches, so a new variant breaks the build in `weight` and
    /// `attributed_peer` before it can reach here weighing nothing.
    fn one_of_every_variant() -> Vec<(&'static str, OmniNetEvent)> {
        vec![
            ("Listening", OmniNetEvent::Listening { addr: addr() }),
            (
                "PeerDiscovered",
                OmniNetEvent::PeerDiscovered {
                    peer_id: PeerId::random(),
                    addrs: vec![addr()],
                },
            ),
            (
                "PeerExpired",
                OmniNetEvent::PeerExpired {
                    peer_id: PeerId::random(),
                },
            ),
            (
                "PeerConnected",
                OmniNetEvent::PeerConnected {
                    peer_id: PeerId::random(),
                },
            ),
            (
                "PeerDisconnected",
                OmniNetEvent::PeerDisconnected {
                    peer_id: PeerId::random(),
                },
            ),
            ("MessageReceived", gossip("omni/topic/v1", 0)),
            (
                "ShardRequested",
                OmniNetEvent::ShardRequested {
                    peer_id: PeerId::random(),
                    request: ShardRequest {
                        cid: String::new(),
                        offset: None,
                        max_bytes: None,
                    },
                    channel_id: 1,
                },
            ),
            ("ShardReceived", shard_response(0)),
            (
                "ShardRequestFailed",
                OmniNetEvent::ShardRequestFailed {
                    peer_id: PeerId::random(),
                    error: String::new(),
                },
            ),
            ("TensorReceived", tensor(0)),
            (
                "TensorResponseReceived",
                OmniNetEvent::TensorResponseReceived {
                    peer_id: PeerId::random(),
                    response: TensorResponse {
                        session_id: String::new(),
                        micro_batch_index: 0,
                        stage_index: 0,
                        accepted: true,
                        error: None,
                    },
                },
            ),
            (
                "TensorRequestFailed",
                OmniNetEvent::TensorRequestFailed {
                    peer_id: PeerId::random(),
                    error: String::new(),
                },
            ),
            (
                "NatStatusChanged",
                OmniNetEvent::NatStatusChanged {
                    is_public: true,
                    public_addr: None,
                },
            ),
            (
                "RelayReservation",
                OmniNetEvent::RelayReservation {
                    relay_peer_id: PeerId::random(),
                    relay_addr: addr(),
                },
            ),
            (
                "HolePunchSucceeded",
                OmniNetEvent::HolePunchSucceeded {
                    peer_id: PeerId::random(),
                },
            ),
            (
                "HolePunchFailed",
                OmniNetEvent::HolePunchFailed {
                    peer_id: PeerId::random(),
                    error: String::new(),
                },
            ),
        ]
    }

    // ── Weights ──────────────────────────────────────────────────────────

    #[test]
    fn no_variant_is_free_even_with_an_empty_payload() {
        // A zero-weight variant is a free channel: an attacker who finds one
        // can flood the router forever and spend nothing.
        for (name, event) in one_of_every_variant() {
            assert!(
                weight(&event) >= EVENT_FLOOR_BYTES,
                "{name} weighs {} — below the floor, so a flood of it is free",
                weight(&event)
            );
        }
    }

    #[test]
    fn weight_grows_with_a_gossip_payload() {
        // Monotonic in payload size, and by exactly the payload: a gossip
        // message that carries a megabyte must not weigh what an empty one
        // weighs. A weight function that ignores its payload fails here.
        let empty = weight(&gossip("t", 0));
        let small = weight(&gossip("t", 1_024));
        let large = weight(&gossip("t", 1_048_576));

        assert!(empty < small, "empty {empty} vs 1KiB {small}");
        assert!(small < large, "1KiB {small} vs 1MiB {large}");
        assert_eq!(large - small, 1_048_576 - 1_024, "payload counted exactly");
    }

    #[test]
    fn weight_grows_with_a_tensor_payload() {
        // The variant that actually moves bulk on this network.
        let empty = weight(&tensor(0));
        let large = weight(&tensor(4_194_304));
        assert!(empty < large, "empty {empty} vs 4MiB {large}");
        assert_eq!(large - empty, 4_194_304);
    }

    #[test]
    fn weight_grows_with_a_shard_payload() {
        let empty = weight(&shard_response(0));
        let large = weight(&shard_response(65_536));
        assert!(empty < large, "empty {empty} vs 64KiB {large}");
        assert_eq!(large - empty, 65_536);
    }

    #[test]
    fn a_longer_topic_costs_more_than_a_short_one() {
        // The topic is remote-chosen text that the router carries and clones
        // per subscriber, so it is payload too.
        assert!(weight(&gossip("a", 0)) < weight(&gossip("aaaaaaaaaaaaaaaa", 0)));
    }

    #[test]
    fn weight_is_monotonic_across_a_sweep_of_payload_sizes() {
        let mut previous = 0;
        for size in [0usize, 1, 2, 16, 512, 4_096, 262_144, 1_048_576] {
            let now = weight(&tensor(size));
            assert!(
                now > previous,
                "weight fell or stalled at {size} bytes: {previous} -> {now}"
            );
            previous = now;
        }
    }

    #[test]
    fn only_this_nodes_own_events_are_unattributed() {
        for (name, event) in one_of_every_variant() {
            let attributed = attributed_peer(&event).is_some();
            let expected = !matches!(name, "Listening" | "NatStatusChanged");
            assert_eq!(
                attributed, expected,
                "{name}: attribution should be {expected}"
            );
        }
    }

    // ── Arithmetic ───────────────────────────────────────────────────────

    #[test]
    fn the_cumulative_total_saturates_instead_of_wrapping() {
        // The boundary. A wrapping counter would report ~1KiB here, with
        // total confidence, at the one moment the number matters.
        let ledger = Arc::new(ByteLedger::new());
        ledger.bytes.store(u64::MAX - 1_000, Ordering::Relaxed);

        drop(ledger.observe(&gossip("t", 1_048_576)));

        let counts = ledger.snapshot();
        assert_eq!(counts.bytes, u64::MAX, "the total pinned, it did not wrap");
        assert_eq!(counts.bytes_saturated, 1, "and it said so");

        // Still monotonic afterwards: a second reading cannot come back
        // smaller than the first.
        drop(ledger.observe(&gossip("t", 1)));
        assert_eq!(ledger.snapshot().bytes, u64::MAX);
        assert_eq!(ledger.snapshot().bytes_saturated, 2);
    }

    #[test]
    fn in_flight_bytes_saturate_and_still_return_to_zero() {
        // The receipt remembers what was actually charged, not the weight, so
        // even a saturated charge releases exactly what it added.
        let ledger = Arc::new(ByteLedger::new());
        ledger.in_flight().total = u64::MAX - 10;

        {
            let _charge = ledger.observe(&gossip("t", 4_096));
            assert_eq!(ledger.snapshot().bytes_in_flight, u64::MAX);
        }
        assert_eq!(
            ledger.snapshot().bytes_in_flight,
            u64::MAX - 10,
            "release must undo exactly the charge that was applied"
        );
    }

    #[test]
    fn a_single_event_cannot_wrap_the_weight_of_its_parts() {
        // Every accumulation inside `weight` saturates, so the largest
        // conceivable event is `u64::MAX`, never a small number.
        let huge = OmniNetEvent::PeerDiscovered {
            peer_id: PeerId::random(),
            addrs: vec![addr(); 64],
        };
        let one = OmniNetEvent::PeerDiscovered {
            peer_id: PeerId::random(),
            addrs: vec![addr()],
        };
        assert!(weight(&huge) > weight(&one));
    }

    // ── The per-peer map's bound ─────────────────────────────────────────

    #[test]
    fn the_peer_map_is_bounded_by_in_flight_not_by_peers_ever_seen() {
        // The property that makes a keyed map admissible here at all. Ten
        // thousand distinct remote identities are accounted; because each
        // charge is released, the map is empty at the end. A map bounded by
        // "peers ever seen" would hold ten thousand entries and grow forever
        // at one entry per attacker keypair.
        let ledger = Arc::new(ByteLedger::new());
        for _ in 0..10_000 {
            drop(ledger.observe(&gossip("omni/flood/v1", 8)));
            assert!(
                ledger.peers_in_flight() <= 1,
                "a released charge must leave no entry behind"
            );
        }

        assert_eq!(
            ledger.peers_in_flight(),
            0,
            "ten thousand peers seen, none in flight, so no entries"
        );
        let counts = ledger.snapshot();
        assert_eq!(counts.events, 10_000, "all of them were accounted");
        assert_eq!(
            counts.peak_peers_in_flight, 1,
            "the map never held more than the one event in flight"
        );

        // And the bound tracks concurrent work, not history: three charges
        // held open are three entries, and only three.
        let held: Vec<Charge> = (0..3)
            .map(|_| ledger.observe(&gossip("omni/flood/v1", 8)))
            .collect();
        assert_eq!(ledger.peers_in_flight(), 3);
        assert_eq!(ledger.snapshot().peak_peers_in_flight, 3);
        drop(held);
        assert_eq!(ledger.peers_in_flight(), 0);
    }

    #[test]
    fn a_peers_entry_is_removed_only_when_its_charge_returns_to_zero() {
        let ledger = Arc::new(ByteLedger::new());
        let peer = PeerId::random();

        let first = ledger.observe(&gossip_from(peer, 128));
        let second = ledger.observe(&gossip_from(peer, 256));
        assert_eq!(ledger.peers_in_flight(), 1, "one peer, two charges");

        drop(first);
        assert_eq!(
            ledger.peers_in_flight(),
            1,
            "the entry stays while the peer still owes something"
        );
        assert!(ledger.snapshot().bytes_in_flight > 0);

        drop(second);
        assert_eq!(ledger.peers_in_flight(), 0);
        assert_eq!(ledger.snapshot().bytes_in_flight, 0);
    }

    #[test]
    fn an_unattributed_event_creates_no_peer_entry() {
        let ledger = Arc::new(ByteLedger::new());
        let charge = ledger.observe(&OmniNetEvent::Listening { addr: addr() });
        assert_eq!(ledger.peers_in_flight(), 0, "this node is not a peer");
        assert!(charge.weight() >= EVENT_FLOOR_BYTES);
        drop(charge);
        assert_eq!(ledger.snapshot().unattributed_events, 1);
    }

    // ── Shadow verdicts ──────────────────────────────────────────────────

    #[test]
    fn a_payload_over_the_peer_ceiling_is_recorded_as_would_refuse() {
        let ledger = Arc::new(ByteLedger::new());
        // `try_from`, not `as`: the same rule the production paths follow.
        let over = usize::try_from(SHADOW_PEER_BYTES).expect("the ceiling fits a usize") + 1;
        let charge = ledger.observe(&tensor(over));
        assert!(
            charge.would_be_refused(),
            "a budget at the shadow ceiling would have turned this away"
        );
        drop(charge);

        let counts = ledger.snapshot();
        assert_eq!(counts.would_refuse_peer, 1);
        assert!(counts.would_refuse_any());
        assert_eq!(counts.events, 1, "and it was accounted, not skipped");
    }

    #[test]
    fn traffic_inside_the_ceilings_records_no_verdict() {
        let ledger = Arc::new(ByteLedger::new());
        for _ in 0..100 {
            drop(ledger.observe(&gossip("omni/topic/v1", 1_024)));
        }
        let counts = ledger.snapshot();
        assert_eq!(counts.would_refuse_global, 0);
        assert_eq!(counts.would_refuse_peer, 0);
        assert!(!counts.would_refuse_any());
        assert_eq!(counts.bytes_in_flight, 0);
    }

    #[test]
    fn the_ledger_has_no_way_to_refuse_anything() {
        // Structural, not behavioural: every entry point returns a receipt or
        // a number, and neither is an `Option`, a `Result`, or a bool the
        // caller must branch on. There is no shape of this API in which a
        // caller could be denied. These lines stop compiling if that changes.
        let _: fn(&ByteLedger, &OmniNetEvent) -> u64 = ByteLedger::weigh;
        let _: fn(&Arc<ByteLedger>, &OmniNetEvent) -> Charge = ByteLedger::charge;
        let _: fn(&Arc<ByteLedger>, &OmniNetEvent) -> Charge = ByteLedger::observe;
    }

    #[test]
    fn weighing_is_seen_and_charging_is_held() {
        // The split the router depends on: an event is weighed once however
        // many copies of it are kept, and a copy that is not kept costs no
        // in-flight bytes at all.
        let ledger = Arc::new(ByteLedger::new());
        let event = gossip("omni/topic/v1", 1_024);
        let weight = ledger.weigh(&event);

        let counts = ledger.snapshot();
        assert_eq!(counts.events, 1, "seen once");
        assert_eq!(counts.bytes, weight);
        assert_eq!(
            counts.bytes_in_flight, 0,
            "weighing records what arrived, not what is held"
        );

        // Three copies kept: three charges, still one event seen.
        let held: Vec<Charge> = (0..3).map(|_| ledger.charge(&event)).collect();
        let counts = ledger.snapshot();
        assert_eq!(counts.events, 1, "charging a copy is not seeing an event");
        assert_eq!(
            counts.bytes, weight,
            "and it is not cumulative bytes either"
        );
        assert_eq!(
            counts.bytes_in_flight,
            weight * 3,
            "each retained copy holds its own payload, so each pays: {counts:?}"
        );

        drop(held);
        assert_eq!(ledger.snapshot().bytes_in_flight, 0);
    }
}
