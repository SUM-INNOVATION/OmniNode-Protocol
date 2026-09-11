//! The single event router.
//!
//! The swarm loop produces one stream of [`OmniNetEvent`]s over one
//! `mpsc::Receiver`. A receiver has exactly one owner — `mpsc::Sender` is
//! `Clone` and `mpsc::Receiver` is not — so every consumer that wanted events
//! had to reach the same receiver, and the codebase did that by wrapping the
//! whole `OmniNet` in `Arc<tokio::sync::Mutex<_>>` and taking turns.
//!
//! Taking turns on a receiver is not sharing. `recv()` *removes* the event, so
//! whoever wins the lock consumes traffic the other consumer needed and throws
//! it away: `omni-contributor`'s relay drained the tensor transport's
//! `TensorReceived` events into `if let MessageReceived` and dropped them on
//! the floor, and the tensor transport did the same to every gossip message.
//! Both were "working" only because they were rarely awake at once.
//!
//! This module removes the shared receiver from every consumer's reach:
//!
//! * [`EventRouter`] owns the receiver and is the only thing that calls
//!   `recv()` on it.
//! * A consumer registers the *kinds of events it wants* — gossip topics,
//!   inbound shard traffic, inbound tensor traffic, control events — and gets
//!   back a [`Subscription`] with its own bounded channel. Two subscribers
//!   with overlapping interests each receive their own copy; neither can
//!   consume the other's.
//! * A [`Subscription`] deregisters itself on `Drop`, so a consumer that goes
//!   away releases its slot and its channel rather than accumulating events
//!   nobody will read.
//!
//! ## What does *not* come through here
//!
//! A solicited response — the answer to a request this node issued — is not
//! broadcast traffic and never enters the router. It is delivered to the one
//! caller that asked, through that request's private completion channel (see
//! [`crate::request`]). The router carries only what arrives unbidden.
//!
//! ## The router must not block the swarm loop
//!
//! The swarm loop hands events over with `try_send` and never awaits a
//! consumer (see the `never_await_invariant` module in `swarm.rs`). The router
//! sits directly behind that lane, so the same rule applies to it: admission
//! is [`Registry::route`], a plain synchronous function that fans out with
//! `try_send`. A slow subscriber loses events; it can never stall the node.
//! That property is pinned the same way — by the compiler, in
//! [`never_await_invariant`] below.
//!
//! ## Nothing is silently dropped
//!
//! An event with no interested subscriber is *counted*, and counted separately
//! from an event that had a subscriber the router failed to reach. Those are
//! different faults: the first says a consumer was never wired up, the second
//! says a wired-up consumer is falling behind or has vanished. The counters
//! are fixed struct fields, never a map keyed by topic or peer — a key chosen
//! by a remote peer is unbounded growth wearing a metrics label.
//!
//! ## Bytes, observed and never enforced
//!
//! Counting events cannot distinguish a flood of empty keep-alives from one
//! peer streaming activation tensors, and it is bytes a future admission
//! budget would have to spend. So every event the router sees is also weighed
//! by [`crate::budget`], in shadow mode: the ledger records what a budget
//! *would* have refused and the router then routes the event anyway. Nothing
//! in this file consults the ledger to decide anything. Read
//! [`RouterHandle::byte_counts`] to see what a budget would have said before
//! giving anything the power to say it.
//!
//! ### What "in flight" means here
//!
//! Bytes in flight are bytes *this node is still holding*, and what holds
//! them is a subscriber's queue. The router itself holds an event for the
//! length of one synchronous fan-out and then hands it on; the copies are
//! what sit in memory afterwards, one bounded queue at a time, until a
//! consumer reads them.
//!
//! So the charge travels with the copy. Every event admitted to a
//! subscriber's channel goes in wrapped in a [`Delivery`] — the event plus
//! its own [`budget::Charge`] — and the charge is released when that envelope
//! dies: on dequeue in [`Subscription::recv`], when `try_send` bounces it off
//! a full channel, when the subscriber has departed, when the subscription is
//! dropped with a backlog still in it, or when the router stops and the
//! queues go with it. The envelope is internal; `recv` yields the plain
//! [`OmniNetEvent`] the caller has always received.
//!
//! Two consequences follow, and both are the point. N interested subscribers
//! means N charges, because N copies exist, each with its own payload
//! allocation. An event nobody wants means no charge at all, because after
//! the counters are bumped nothing keeps it.
//!
//! The rejected alternative — charge on entry, release when fan-out returns —
//! is what makes a 32 MiB ceiling meaningless: it can only ever be crossed by
//! a single event larger than 32 MiB, whatever the backlog behind it.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tracing::debug;

use crate::budget::{ByteCounts, ByteLedger, Charge};
use crate::events::OmniNetEvent;

/// Per-subscriber channel depth.
///
/// Matches the swarm→router lane, so a subscriber that keeps up with the
/// router has the same burst tolerance the single shared receiver used to
/// give it.
pub const SUBSCRIBER_CAPACITY: usize = 256;

// ── Event classes ─────────────────────────────────────────────────────────────

/// What kind of traffic an event is, for the purpose of matching interests.
///
/// Gossip carries its topic because subscribers care about individual topics:
/// the contributor relay wants seven specific topics and nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventClass<'a> {
    /// A gossipsub message on the named topic.
    Gossip(&'a str),
    /// Inbound shard traffic — a request from a peer, or an unsolicited
    /// response/failure that no completion channel claimed.
    Shard,
    /// Inbound tensor traffic, same shape as [`EventClass::Shard`].
    Tensor,
    /// Everything else: listen addresses, peer lifecycle, NAT and relay state.
    Control,
}

/// Classify an event into the class subscribers register against.
pub fn classify(event: &OmniNetEvent) -> EventClass<'_> {
    match event {
        OmniNetEvent::MessageReceived { topic, .. } => EventClass::Gossip(topic.as_str()),

        OmniNetEvent::ShardRequested { .. }
        | OmniNetEvent::ShardReceived { .. }
        | OmniNetEvent::ShardRequestFailed { .. } => EventClass::Shard,

        OmniNetEvent::TensorReceived { .. }
        | OmniNetEvent::TensorResponseReceived { .. }
        | OmniNetEvent::TensorRequestFailed { .. } => EventClass::Tensor,

        OmniNetEvent::Listening { .. }
        | OmniNetEvent::PeerDiscovered { .. }
        | OmniNetEvent::PeerExpired { .. }
        | OmniNetEvent::PeerConnected { .. }
        | OmniNetEvent::PeerDisconnected { .. }
        | OmniNetEvent::NatStatusChanged { .. }
        | OmniNetEvent::RelayReservation { .. }
        | OmniNetEvent::HolePunchSucceeded { .. }
        | OmniNetEvent::HolePunchFailed { .. } => EventClass::Control,
    }
}

// ── Interests ─────────────────────────────────────────────────────────────────

/// What a consumer wants delivered to it.
///
/// Built by chaining; an empty `Interests` matches nothing, which is why
/// [`Interests::none`] exists but is only useful as a starting point.
///
/// Registering by interest rather than taking everything is what lets two
/// consumers of one mesh coexist: the relay asks for its gossip topics, the
/// tensor transport asks for tensor traffic, and neither is handed — or can
/// discard — the other's events.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Interests {
    /// Named gossip topics.
    topics: BTreeSet<String>,
    /// Every gossip topic, whatever its name.
    every_topic: bool,
    shard: bool,
    tensor: bool,
    control: bool,
}

impl Interests {
    /// Matches nothing. A starting point for the builder methods.
    pub fn none() -> Self {
        Self::default()
    }

    /// Matches every event of every class.
    ///
    /// This is the shape of a single-consumer caller — one CLI command that
    /// owns its own mesh — not of a consumer sharing a node with others.
    pub fn everything() -> Self {
        Self {
            topics: BTreeSet::new(),
            every_topic: true,
            shard: true,
            tensor: true,
            control: true,
        }
    }

    /// Also deliver gossip messages on `topic`.
    pub fn topic(mut self, topic: impl Into<String>) -> Self {
        self.topics.insert(topic.into());
        self
    }

    /// Also deliver gossip messages on each of `topics`.
    pub fn with_topics<I, S>(mut self, topics: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for topic in topics {
            self.topics.insert(topic.into());
        }
        self
    }

    /// Also deliver gossip messages on every topic.
    pub fn every_topic(mut self) -> Self {
        self.every_topic = true;
        self
    }

    /// Also deliver inbound shard traffic.
    pub fn shard(mut self) -> Self {
        self.shard = true;
        self
    }

    /// Also deliver inbound tensor traffic.
    pub fn tensor(mut self) -> Self {
        self.tensor = true;
        self
    }

    /// Also deliver control events — listen addresses, peer lifecycle, NAT.
    pub fn control(mut self) -> Self {
        self.control = true;
        self
    }

    /// Whether this set matches nothing at all.
    pub fn is_empty(&self) -> bool {
        self.topics.is_empty() && !self.every_topic && !self.shard && !self.tensor && !self.control
    }

    /// Whether an event of `class` should be delivered to this subscriber.
    pub fn wants(&self, class: EventClass<'_>) -> bool {
        match class {
            EventClass::Gossip(topic) => self.every_topic || self.topics.contains(topic),
            EventClass::Shard => self.shard,
            EventClass::Tensor => self.tensor,
            EventClass::Control => self.control,
        }
    }
}

// ── Counters ──────────────────────────────────────────────────────────────────

/// A reading of the router's counters.
///
/// Fixed fields. Never a map keyed by topic, peer, or subscriber name: those
/// keys are chosen by remote peers or by call sites, and a metrics map keyed
/// by an unbounded input is a memory leak that reports itself as telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RouterCounts {
    /// Events taken off the swarm lane.
    pub events: u64,
    /// Copies handed to a subscriber's channel.
    pub delivered: u64,

    // ── Nobody asked for this ────────────────────────────────────────────
    //
    // Deliberately distinct from the delivery failures below: an unwanted
    // event means no consumer ever registered that interest — a wiring gap,
    // not a runtime fault.
    /// Gossip messages on a topic no subscriber asked for.
    pub unwanted_gossip: u64,
    /// Shard traffic with no shard subscriber.
    pub unwanted_shard: u64,
    /// Tensor traffic with no tensor subscriber.
    pub unwanted_tensor: u64,
    /// Control events with no control subscriber.
    pub unwanted_control: u64,

    // ── Someone asked and we failed them ─────────────────────────────────
    /// A subscriber wanted the event but its channel was full — it is not
    /// keeping up, and this copy is gone.
    pub dropped_backlogged: u64,
    /// A subscriber wanted the event but its channel was already closed —
    /// it went away without deregistering.
    pub dropped_departed: u64,

    // ── Registry lifecycle ───────────────────────────────────────────────
    /// Subscriptions handed out.
    pub subscriptions_opened: u64,
    /// Subscriptions released, by `Drop` or by router shutdown.
    pub subscriptions_closed: u64,
}

/// The live counters behind [`RouterCounts`].
#[derive(Debug, Default)]
struct Counters {
    events: AtomicU64,
    delivered: AtomicU64,
    unwanted_gossip: AtomicU64,
    unwanted_shard: AtomicU64,
    unwanted_tensor: AtomicU64,
    unwanted_control: AtomicU64,
    dropped_backlogged: AtomicU64,
    dropped_departed: AtomicU64,
    subscriptions_opened: AtomicU64,
    subscriptions_closed: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> RouterCounts {
        RouterCounts {
            events: self.events.load(Ordering::Relaxed),
            delivered: self.delivered.load(Ordering::Relaxed),
            unwanted_gossip: self.unwanted_gossip.load(Ordering::Relaxed),
            unwanted_shard: self.unwanted_shard.load(Ordering::Relaxed),
            unwanted_tensor: self.unwanted_tensor.load(Ordering::Relaxed),
            unwanted_control: self.unwanted_control.load(Ordering::Relaxed),
            dropped_backlogged: self.dropped_backlogged.load(Ordering::Relaxed),
            dropped_departed: self.dropped_departed.load(Ordering::Relaxed),
            subscriptions_opened: self.subscriptions_opened.load(Ordering::Relaxed),
            subscriptions_closed: self.subscriptions_closed.load(Ordering::Relaxed),
        }
    }
}

// ── RouterStopped ─────────────────────────────────────────────────────────────

/// The router is no longer running, so a subscription cannot be served.
///
/// Returned rather than handing back a channel that would never yield an
/// event: a consumer must be able to tell "nothing has happened yet" from
/// "nothing will ever happen again".
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the event router has stopped — no new subscription can be served")]
pub struct RouterStopped;

// ── Registry ──────────────────────────────────────────────────────────────────

/// One event, and the byte charge its place in a queue is paying for.
///
/// Internal to the router. It exists so that a charge can outlive the fan-out
/// that created it and expire with the copy it accounts for: whichever way the
/// copy dies — dequeued, bounced off a full channel, dropped with a departed
/// subscriber, dropped with the subscription, dropped when the router stops —
/// this struct dies with it, and `Drop for Charge` releases the bytes.
///
/// It never reaches a caller. [`Subscription::recv`] unwraps it and returns
/// the unchanged public [`OmniNetEvent`].
pub(crate) struct Delivery {
    event: OmniNetEvent,
    /// Released on drop. Never read; holding it *is* the accounting.
    _charge: Charge,
}

impl Delivery {
    /// Take the event out, releasing the charge that its queue slot held.
    fn into_event(self) -> OmniNetEvent {
        self.event
    }
}

/// One registered consumer.
struct Subscriber {
    id: u64,
    interests: Interests,
    tx: mpsc::Sender<Delivery>,
}

/// The shared table of subscribers, plus the counters.
///
/// Guarded by a `std::sync::Mutex` rather than tokio's, deliberately: every
/// operation on it — registration, deregistration, fan-out — is synchronous
/// and finishes without awaiting, so the lock is never held across a suspend
/// point and `Drop` can take it.
pub(crate) struct Registry {
    subscribers: Mutex<Vec<Subscriber>>,
    next_id: AtomicU64,
    running: AtomicBool,
    counts: Counters,
    /// Shadow byte accounting. Observed on every event, consulted for no
    /// decision — see the module docs and [`crate::budget`].
    ///
    /// Held behind an `Arc` because a charge outlives the fan-out that issued
    /// it: it travels into a subscriber's queue inside a [`Delivery`] and may
    /// be released from a consumer's task, long after `route` returned.
    ledger: Arc<ByteLedger>,
}

impl Registry {
    fn new() -> Self {
        Self {
            subscribers: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            running: AtomicBool::new(true),
            counts: Counters::default(),
            ledger: Arc::new(ByteLedger::new()),
        }
    }

    /// Take the subscriber table, recovering from a poisoned lock.
    ///
    /// A panic inside fan-out would otherwise poison the router permanently
    /// and take the whole node's event delivery with it. The data behind the
    /// lock is a plain `Vec` of senders with no invariant a panic could break,
    /// so recovering is safe and strictly better than propagating.
    fn subscribers(&self) -> std::sync::MutexGuard<'_, Vec<Subscriber>> {
        self.subscribers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Register `interests` and return that consumer's own channel.
    fn register(self: &Arc<Self>, interests: Interests) -> Result<Subscription, RouterStopped> {
        if !self.running.load(Ordering::Acquire) {
            return Err(RouterStopped);
        }
        let (tx, rx) = mpsc::channel(SUBSCRIBER_CAPACITY);
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut subs = self.subscribers();
            // Re-check under the lock: a shutdown between the check above and
            // here would otherwise leave this subscriber in a table nobody
            // will ever clear, holding a channel that never yields.
            if !self.running.load(Ordering::Acquire) {
                return Err(RouterStopped);
            }
            subs.push(Subscriber { id, interests, tx });
        }
        self.counts
            .subscriptions_opened
            .fetch_add(1, Ordering::Relaxed);
        Ok(Subscription {
            id,
            rx,
            registry: Arc::clone(self),
        })
    }

    /// Drop the subscriber with `id`, releasing its channel.
    fn deregister(&self, id: u64) {
        let removed = {
            let mut subs = self.subscribers();
            let before = subs.len();
            subs.retain(|s| s.id != id);
            before - subs.len()
        };
        if removed > 0 {
            self.counts
                .subscriptions_closed
                .fetch_add(removed as u64, Ordering::Relaxed);
        }
    }

    /// Fan one event out to every subscriber that asked for its class.
    ///
    /// Synchronous by construction — see the module docs and
    /// [`never_await_invariant`]. Delivery is `try_send`, so a subscriber that
    /// has fallen behind loses this event instead of stalling the router and,
    /// through it, the swarm loop.
    fn route(&self, event: OmniNetEvent) {
        self.counts.events.fetch_add(1, Ordering::Relaxed);

        // Shadow accounting, wired here because here is where production
        // events actually pass — the ledger therefore weighs exactly what the
        // router routes, not a sampled or reconstructed shadow of it.
        //
        // Weighing is *seeing*: it happens once, before it is known whether
        // anyone wants this, so the cumulative totals cover the unwanted event
        // below as well. Retention is charged separately, per copy, at the
        // moment a copy is actually put somewhere that will hold it.
        //
        // Nothing below reads a verdict. There is deliberately no branch on
        // `Charge::would_be_refused` anywhere in this file — see
        // `observation_only_invariant` — so an event's fate is identical to
        // what it was before the ledger existed.
        self.ledger.weigh(&event);

        let mut subs = self.subscribers();

        // The interested set is resolved into indices while the event is
        // still borrowed for its class, because delivering the event *moves*
        // it into the last recipient and the class borrow cannot outlive that
        // move. One small allocation per event, against an event whose
        // payload was already heap-allocated upstream.
        let wanted: Vec<usize> = {
            let class = classify(&event);
            let wanted: Vec<usize> = subs
                .iter()
                .enumerate()
                .filter(|(_, sub)| sub.interests.wants(class))
                .map(|(index, _)| index)
                .collect();
            if wanted.is_empty() {
                // Nobody asked for this. Counted, never silently discarded —
                // and counted apart from the delivery failures below, because
                // this is a missing consumer, not a failing one.
                //
                // No in-flight charge was ever taken: the event was weighed,
                // and it is about to be dropped rather than held, so there are
                // no retained bytes to account for.
                self.counter_for(class).fetch_add(1, Ordering::Relaxed);
                debug!(?class, "event had no interested subscriber");
                return;
            }
            wanted
        };

        // The last interested subscriber takes the event by move, so the
        // common single-subscriber case never clones a tensor payload.
        let mut carrier = Some(event);
        let mut remaining = wanted.len();
        let mut saw_departed = false;
        for index in wanted {
            remaining -= 1;
            let copy = if remaining == 0 {
                carrier
                    .take()
                    .expect("carrier holds the event until the last delivery")
            } else {
                carrier
                    .as_ref()
                    .expect("carrier holds the event before the last delivery")
                    .clone()
            };
            // Charge this copy, then hand copy and receipt over together. If
            // the send fails the envelope comes straight back in the error and
            // is dropped at the end of the arm, which releases the charge: a
            // copy that was never taken is a copy nobody is holding.
            match subs[index].tx.try_send(self.envelope(copy)) {
                Ok(()) => {
                    // The charge is now in that subscriber's queue and stays
                    // open until the copy leaves it.
                    self.counts.delivered.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Full(bounced)) => {
                    // Someone asked and we failed them: this consumer is not
                    // keeping up. Deliberately not the "nobody asked" counter.
                    drop(bounced);
                    self.counts
                        .dropped_backlogged
                        .fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Closed(bounced)) => {
                    drop(bounced);
                    self.counts.dropped_departed.fetch_add(1, Ordering::Relaxed);
                    saw_departed = true;
                }
            }
        }

        if saw_departed {
            // A consumer that vanished without running `Subscription::drop` —
            // a leaked handle, or a task torn down mid-flight. Release its
            // slot here so the table cannot grow without bound.
            let before = subs.len();
            subs.retain(|sub| !sub.tx.is_closed());
            let closed = before - subs.len();
            if closed > 0 {
                self.counts
                    .subscriptions_closed
                    .fetch_add(closed as u64, Ordering::Relaxed);
            }
        }
    }

    /// Wrap one copy in its own charge, ready to be handed to a queue.
    ///
    /// The charge is taken here and released only when the returned
    /// [`Delivery`] dies, so the accounting follows the copy wherever it goes
    /// — including straight back out of a `try_send` that failed.
    fn envelope(&self, event: OmniNetEvent) -> Delivery {
        let charge = self.ledger.charge(&event);
        Delivery {
            event,
            _charge: charge,
        }
    }

    /// The "nobody subscribed" counter for `class`.
    fn counter_for(&self, class: EventClass<'_>) -> &AtomicU64 {
        match class {
            EventClass::Gossip(_) => &self.counts.unwanted_gossip,
            EventClass::Shard => &self.counts.unwanted_shard,
            EventClass::Tensor => &self.counts.unwanted_tensor,
            EventClass::Control => &self.counts.unwanted_control,
        }
    }

    /// Stop the router: refuse new subscriptions and release every existing
    /// one, so each consumer's `recv()` returns `None` instead of waiting on
    /// a stream that will never produce again.
    fn close(&self) {
        self.running.store(false, Ordering::Release);
        let released = {
            let mut subs = self.subscribers();
            let n = subs.len();
            // Dropping the senders is what wakes every waiting consumer.
            subs.clear();
            n
        };
        if released > 0 {
            self.counts
                .subscriptions_closed
                .fetch_add(released as u64, Ordering::Relaxed);
        }
        debug!(released, "event router stopped");
    }
}

// ── Subscription ──────────────────────────────────────────────────────────────

/// One consumer's private event stream.
///
/// Yields only the classes it registered for. Dropping it deregisters the
/// consumer — the router stops copying events into a channel nobody reads.
pub struct Subscription {
    id: u64,
    /// Carries `Delivery`, not the bare event: each queued item holds the
    /// charge for the bytes its own queue slot retains. Dequeuing releases it.
    rx: mpsc::Receiver<Delivery>,
    registry: Arc<Registry>,
}

impl std::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription")
            .field("id", &self.id)
            .finish()
    }
}

impl Subscription {
    /// This subscription's registry id. Stable for its lifetime.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Wait for the next matching event.
    ///
    /// Returns `None` once the router has stopped and the buffer is drained —
    /// the signal that no further event can arrive.
    ///
    /// The public type is unchanged: the delivery envelope is unwrapped here
    /// and its charge released, so a caller never sees the accounting. Holding
    /// the event past this point costs nothing, because the bytes are no longer
    /// sitting in a queue.
    pub async fn recv(&mut self) -> Option<OmniNetEvent> {
        self.rx.recv().await.map(Delivery::into_event)
    }

    /// Take the next matching event if one is already buffered.
    ///
    /// Never blocks. `None` covers both "nothing queued right now" and "the
    /// router has stopped"; a caller that needs to tell those apart should
    /// await [`Subscription::recv`] instead.
    pub fn try_recv(&mut self) -> Option<OmniNetEvent> {
        self.rx.try_recv().ok().map(Delivery::into_event)
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.registry.deregister(self.id);
    }
}

// ── RouterHandle ──────────────────────────────────────────────────────────────

/// Cheap, cloneable access to the router's registry.
///
/// Carried by every network handle. Cloning it does not clone any receiver —
/// that is the whole point: a clone can *ask for* a stream, never take one
/// out from under somebody else.
#[derive(Clone)]
pub struct RouterHandle {
    registry: Arc<Registry>,
}

impl std::fmt::Debug for RouterHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterHandle")
            .field("running", &self.is_running())
            .field("subscribers", &self.subscriber_count())
            .finish()
    }
}

impl RouterHandle {
    /// Register `interests` and receive this consumer's own stream.
    pub fn subscribe(&self, interests: Interests) -> Result<Subscription, RouterStopped> {
        self.registry.register(interests)
    }

    /// Read the router's counters.
    pub fn counts(&self) -> RouterCounts {
        self.registry.counts.snapshot()
    }

    /// Read the shadow byte accounting.
    ///
    /// Observation only: [`ByteCounts::would_refuse_global`] and
    /// [`ByteCounts::would_refuse_peer`] report what a budget at the shadow
    /// ceilings would have turned away. Nothing was turned away.
    pub fn byte_counts(&self) -> ByteCounts {
        self.registry.ledger.snapshot()
    }

    /// How many peers currently hold a byte charge.
    ///
    /// Test-only: this is the live size of the ledger's one keyed map, which
    /// production reads only in aggregate through [`RouterHandle::byte_counts`].
    #[cfg(test)]
    pub(crate) fn peers_in_flight_for_test(&self) -> usize {
        self.registry.ledger.peers_in_flight()
    }

    /// Whether the router is still draining the swarm lane.
    pub fn is_running(&self) -> bool {
        self.registry.running.load(Ordering::Acquire)
    }

    /// How many consumers are currently registered.
    pub fn subscriber_count(&self) -> usize {
        self.registry.subscribers().len()
    }

    /// Deliver `event` as if it had come off the swarm lane.
    ///
    /// Test-only: production events reach the registry solely through
    /// [`EventRouter::run`], which owns the receiver.
    #[cfg(test)]
    pub(crate) fn route_for_test(&self, event: OmniNetEvent) {
        self.registry.route(event);
    }

    /// Register `interests` and hand back only the raw receiver, with no
    /// [`Subscription`] to deregister it.
    ///
    /// Test-only. Models a consumer that vanished without running its `Drop` —
    /// a leaked handle, or a task torn down mid-flight — which is the only way
    /// the registry can be left holding a closed channel.
    #[cfg(test)]
    pub(crate) fn orphan_subscriber_for_test(
        &self,
        interests: Interests,
    ) -> mpsc::Receiver<Delivery> {
        let (tx, rx) = mpsc::channel(SUBSCRIBER_CAPACITY);
        let id = self.registry.next_id.fetch_add(1, Ordering::Relaxed);
        self.registry
            .subscribers()
            .push(Subscriber { id, interests, tx });
        self.registry
            .counts
            .subscriptions_opened
            .fetch_add(1, Ordering::Relaxed);
        rx
    }
}

// ── EventRouter ───────────────────────────────────────────────────────────────

/// The one owner of the swarm's event receiver.
///
/// Constructed with the receiver and never gives it up. Consumers get streams
/// of their own from the [`RouterHandle`] returned alongside it.
pub struct EventRouter {
    event_rx: mpsc::Receiver<OmniNetEvent>,
    registry: Arc<Registry>,
}

impl EventRouter {
    /// Take ownership of the swarm's event lane.
    ///
    /// The returned handle is the only way to reach the events from anywhere
    /// else, and it can only hand out per-consumer copies.
    pub fn new(event_rx: mpsc::Receiver<OmniNetEvent>) -> (Self, RouterHandle) {
        let registry = Arc::new(Registry::new());
        let handle = RouterHandle {
            registry: Arc::clone(&registry),
        };
        (Self { event_rx, registry }, handle)
    }

    /// Drain the swarm lane until it closes, fanning each event out.
    ///
    /// On exit — the swarm task stopped, or was shut down — every subscription
    /// is released, so no consumer is left awaiting a stream that can no
    /// longer produce.
    pub async fn run(mut self) {
        while let Some(event) = self.event_rx.recv().await {
            self.registry.route(event);
        }
        self.registry.close();
    }

    /// Run the router on its own task.
    pub fn spawn(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(self.run())
    }
}

// ── Never-await invariant ─────────────────────────────────────────────────────
//
// The router inherits the swarm loop's rule. `swarm.rs` refuses to await on
// the event lane because a consumer can issue a command from inside its own
// event handling, and the loop is the only thing that services commands — an
// awaiting handoff makes the loop wait on a consumer that is waiting on the
// loop. The router is that lane's reader: if it awaited a congested
// subscriber, it would stop calling `recv()`, the swarm's `try_send` would
// start failing, and the deadlock would simply have moved one hop downstream.
//
// So admission is synchronous, and that is pinned rather than trusted.

#[cfg(test)]
mod never_await_invariant {
    use super::*;

    /// Resolved by the compiler against this very file, so the scan cannot
    /// pass vacuously if the module is moved or renamed.
    const ROUTER_SRC: &str = include_str!("router.rs");

    /// Assembled at runtime so this module's own text cannot satisfy the scan.
    fn needle(head: &str, tail: &str) -> String {
        format!("{head}{tail}")
    }

    #[test]
    fn fan_out_is_not_async() {
        // An `async fn` returns an opaque future, which cannot coerce to a
        // function pointer returning `()`. This line stops compiling the
        // moment someone makes admission awaitable.
        let _: fn(&Registry, OmniNetEvent) = Registry::route;
    }

    #[test]
    fn registration_and_teardown_are_not_async() {
        // `Drop::drop` cannot await, so deregistration must stay synchronous
        // for a subscription to release its slot at all.
        let _: fn(&Registry, u64) = Registry::deregister;
        let _: fn(&Registry) = Registry::close;
    }

    #[test]
    fn the_router_never_blocks_on_a_subscriber() {
        for (head, tail) in [
            // An awaiting send to a subscriber — waits on a congested
            // consumer, which is the deadlock one hop downstream.
            ("sub.tx.se", "nd("),
            ("tx.clone().se", "nd("),
            // A blocking send stalls the runtime worker instead.
            ("blocking_se", "nd("),
            // Holding the subscriber table across a suspend point would
            // deadlock registration against fan-out.
            ("subscribers().a", "wait"),
        ] {
            let forbidden = needle(head, tail);
            assert!(
                !ROUTER_SRC.contains(&forbidden),
                "router.rs contains `{forbidden}`: the router would then wait \
                 on a consumer, and the swarm loop would wait on the router"
            );
        }
    }

    #[test]
    fn subscribers_are_admitted_with_try_send() {
        let try_send = needle("try_", "send(");
        assert!(
            ROUTER_SRC.contains(&try_send),
            "fan-out must use `{try_send}` so a slow consumer cannot stall the \
             swarm lane"
        );
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use libp2p::PeerId;

    use crate::codec::{ShardRequest, ShardResponse};
    use crate::tensor_codec::{TensorRequest, TensorResponse};

    const TOPIC_A: &str = "omni/topic-a/v1";
    const TOPIC_B: &str = "omni/topic-b/v1";

    fn gossip(topic: &str, body: &str) -> OmniNetEvent {
        OmniNetEvent::MessageReceived {
            from: PeerId::random(),
            topic: topic.to_string(),
            data: body.as_bytes().to_vec(),
        }
    }

    fn shard_request(cid: &str) -> OmniNetEvent {
        OmniNetEvent::ShardRequested {
            peer_id: PeerId::random(),
            request: ShardRequest {
                cid: cid.to_string(),
                offset: None,
                max_bytes: None,
            },
            channel_id: 1,
        }
    }

    fn shard_response(cid: &str) -> OmniNetEvent {
        OmniNetEvent::ShardReceived {
            peer_id: PeerId::random(),
            response: ShardResponse {
                cid: cid.to_string(),
                offset: 0,
                data: Vec::new(),
                total_bytes: 0,
                error: None,
            },
        }
    }

    fn tensor(session: &str) -> OmniNetEvent {
        OmniNetEvent::TensorReceived {
            peer_id: PeerId::random(),
            request: TensorRequest {
                session_id: session.to_string(),
                micro_batch_index: 0,
                from_stage: 0,
                to_stage: 1,
                seq_len: 1,
                hidden_dim: 1,
                dtype: 0,
                data: Vec::new(),
            },
            channel_id: 2,
        }
    }

    fn tensor_ack(session: &str) -> OmniNetEvent {
        OmniNetEvent::TensorResponseReceived {
            peer_id: PeerId::random(),
            response: TensorResponse {
                session_id: session.to_string(),
                micro_batch_index: 0,
                stage_index: 1,
                accepted: true,
                error: None,
            },
        }
    }

    fn control() -> OmniNetEvent {
        OmniNetEvent::PeerConnected {
            peer_id: PeerId::random(),
        }
    }

    /// A router with a live lane, so `route` can be driven directly.
    fn router() -> (mpsc::Sender<OmniNetEvent>, RouterHandle) {
        let (tx, rx) = mpsc::channel(SUBSCRIBER_CAPACITY);
        let (router, handle) = EventRouter::new(rx);
        router.spawn();
        (tx, handle)
    }

    /// Deadline on every await in this suite.
    ///
    /// A routing regression must *fail* these tests, not hang them. Every
    /// assertion below waits on a `recv()` that a broken router simply never
    /// satisfies — drop the interest filter, or the fan-out, and the awaiting
    /// task parks forever. Without a deadline the test never returns and CI
    /// reports a stalled job, or a killed one, instead of a named failing
    /// test; a regression that hangs the suite is a regression nobody reads.
    ///
    /// This is a failure detector, not an expected duration: the router hands
    /// events over synchronously and a passing run never comes near it.
    const TEST_DEADLINE: Duration = Duration::from_secs(5);

    /// Await the next event, failing by name if none arrives in time.
    async fn recv_within(sub: &mut Subscription, expected: &str) -> OmniNetEvent {
        match tokio::time::timeout(TEST_DEADLINE, sub.recv()).await {
            Ok(Some(event)) => event,
            Ok(None) => panic!("the stream ended before {expected} arrived"),
            Err(_) => {
                panic!("timed out after {TEST_DEADLINE:?} waiting for {expected}")
            }
        }
    }

    /// Await the *end* of a stream, failing by name if it neither ends nor
    /// yields. The shutdown counterpart of [`recv_within`].
    async fn end_within(sub: &mut Subscription, expected: &str) {
        match tokio::time::timeout(TEST_DEADLINE, sub.recv()).await {
            Ok(None) => {}
            Ok(Some(event)) => {
                panic!("expected {expected} to have ended, got {event:?}")
            }
            Err(_) => panic!(
                "timed out after {TEST_DEADLINE:?} waiting for {expected} to end — \
                 a consumer must learn the stream stopped, not wait forever"
            ),
        }
    }

    fn topic_of(event: &OmniNetEvent) -> String {
        match event {
            OmniNetEvent::MessageReceived { topic, .. } => topic.clone(),
            other => panic!("expected a gossip message, got {other:?}"),
        }
    }

    fn body_of(event: &OmniNetEvent) -> String {
        match event {
            OmniNetEvent::MessageReceived { data, .. } => {
                String::from_utf8(data.clone()).expect("utf8 body")
            }
            other => panic!("expected a gossip message, got {other:?}"),
        }
    }

    // ── Classification ───────────────────────────────────────────────────

    #[test]
    fn every_event_lands_in_exactly_one_class() {
        assert_eq!(classify(&gossip(TOPIC_A, "x")), EventClass::Gossip(TOPIC_A));
        assert_eq!(classify(&shard_request("cid")), EventClass::Shard);
        assert_eq!(classify(&shard_response("cid")), EventClass::Shard);
        assert_eq!(classify(&tensor("s")), EventClass::Tensor);
        assert_eq!(classify(&tensor_ack("s")), EventClass::Tensor);
        assert_eq!(classify(&control()), EventClass::Control);
    }

    #[test]
    fn interests_match_only_what_was_asked_for() {
        let only_a = Interests::none().topic(TOPIC_A);
        assert!(only_a.wants(EventClass::Gossip(TOPIC_A)));
        assert!(!only_a.wants(EventClass::Gossip(TOPIC_B)));
        assert!(!only_a.wants(EventClass::Shard));
        assert!(!only_a.wants(EventClass::Control));

        let everything = Interests::everything();
        assert!(everything.wants(EventClass::Gossip("anything/at/all")));
        assert!(everything.wants(EventClass::Shard));
        assert!(everything.wants(EventClass::Tensor));
        assert!(everything.wants(EventClass::Control));

        assert!(Interests::none().is_empty());
        assert!(!Interests::none().control().is_empty());
    }

    // ── The bug this commit exists to fix ────────────────────────────────

    #[tokio::test]
    async fn a_gossip_event_survives_a_tensor_consumer_draining() {
        // The `tensor_transport.rs` half of the old bug: it drained the shared
        // receiver looking for `TensorReceived` and dropped everything else,
        // including gossip the relay was waiting on.
        let (lane, handle) = router();
        let mut relay = handle
            .subscribe(Interests::none().topic(TOPIC_A))
            .expect("relay subscribes");
        let mut transport = handle
            .subscribe(Interests::none().tensor())
            .expect("transport subscribes");

        lane.send(gossip(TOPIC_A, "job-announcement"))
            .await
            .unwrap();
        lane.send(tensor("session-1")).await.unwrap();

        // The tensor consumer drains everything it can see...
        let drained = recv_within(&mut transport, "the tensor consumer's own event").await;
        assert!(matches!(drained, OmniNetEvent::TensorReceived { .. }));
        assert!(
            transport.try_recv().is_none(),
            "and nothing that isn't its own"
        );

        // ...and the gossip event is still there for the relay.
        let survived = recv_within(
            &mut relay,
            "the gossip event the tensor drain must not have taken",
        )
        .await;
        assert_eq!(body_of(&survived), "job-announcement");
    }

    #[tokio::test]
    async fn a_tensor_event_survives_a_gossip_consumer_draining() {
        // The `relay.rs` half: it drained the same receiver looking for
        // `MessageReceived` and silently discarded the transport's tensors.
        let (lane, handle) = router();
        let mut relay = handle
            .subscribe(Interests::none().every_topic())
            .expect("relay subscribes");
        let mut transport = handle
            .subscribe(Interests::none().tensor())
            .expect("transport subscribes");

        lane.send(tensor("session-2")).await.unwrap();
        lane.send(gossip(TOPIC_B, "result-announcement"))
            .await
            .unwrap();

        let drained = recv_within(&mut relay, "the gossip consumer's own event").await;
        assert_eq!(topic_of(&drained), TOPIC_B);
        assert!(relay.try_recv().is_none());

        let survived = recv_within(
            &mut transport,
            "the tensor the gossip drain must not have taken",
        )
        .await;
        match survived {
            OmniNetEvent::TensorReceived { request, .. } => {
                assert_eq!(request.session_id, "session-2");
            }
            other => panic!("expected TensorReceived, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn two_subscribers_on_one_topic_both_receive_it() {
        // Overlapping interests are a fan-out, not a race. Under the old
        // shared receiver exactly one of these would have seen the message.
        let (lane, handle) = router();
        let mut first = handle
            .subscribe(Interests::none().topic(TOPIC_A))
            .expect("first");
        let mut second = handle
            .subscribe(Interests::none().topic(TOPIC_A))
            .expect("second");

        lane.send(gossip(TOPIC_A, "broadcast")).await.unwrap();

        let to_first = recv_within(&mut first, "the first subscriber's copy").await;
        let to_second = recv_within(&mut second, "the second subscriber's copy").await;
        assert_eq!(body_of(&to_first), "broadcast");
        assert_eq!(body_of(&to_second), "broadcast");
    }

    #[tokio::test]
    async fn simultaneous_fetch_relay_and_tensor_traffic_all_arrive() {
        // The three consumers that used to fight over one receiver, all awake
        // at once, with their traffic interleaved.
        let (lane, handle) = router();
        let mut store = handle
            .subscribe(Interests::none().shard())
            .expect("store subscribes");
        let mut relay = handle
            .subscribe(Interests::none().topic(TOPIC_A).topic(TOPIC_B))
            .expect("relay subscribes");
        let mut transport = handle
            .subscribe(Interests::none().tensor())
            .expect("transport subscribes");
        let mut watcher = handle
            .subscribe(Interests::none().control())
            .expect("peer watcher subscribes");

        for round in 0..4 {
            lane.send(shard_request(&format!("cid-{round}")))
                .await
                .unwrap();
            lane.send(gossip(TOPIC_A, &format!("job-{round}")))
                .await
                .unwrap();
            lane.send(tensor(&format!("session-{round}")))
                .await
                .unwrap();
            lane.send(gossip(TOPIC_B, &format!("result-{round}")))
                .await
                .unwrap();
            lane.send(control()).await.unwrap();
        }

        for round in 0..4 {
            match recv_within(&mut store, &format!("shard request cid-{round}")).await {
                OmniNetEvent::ShardRequested { request, .. } => {
                    assert_eq!(request.cid, format!("cid-{round}"));
                }
                other => panic!("store received {other:?}"),
            }
            assert_eq!(
                body_of(&recv_within(&mut relay, &format!("gossip job-{round}")).await),
                format!("job-{round}")
            );
            match recv_within(&mut transport, &format!("tensor session-{round}")).await {
                OmniNetEvent::TensorReceived { request, .. } => {
                    assert_eq!(request.session_id, format!("session-{round}"));
                }
                other => panic!("transport received {other:?}"),
            }
            assert_eq!(
                body_of(&recv_within(&mut relay, &format!("gossip result-{round}")).await),
                format!("result-{round}")
            );
            assert!(matches!(
                recv_within(&mut watcher, &format!("control event {round}")).await,
                OmniNetEvent::PeerConnected { .. }
            ));
        }

        // Nobody stole anybody's traffic: each stream is empty, not short.
        assert!(store.try_recv().is_none());
        assert!(relay.try_recv().is_none());
        assert!(transport.try_recv().is_none());
        assert!(watcher.try_recv().is_none());
    }

    #[tokio::test]
    async fn a_control_watcher_does_not_swallow_the_traffic_it_ignores() {
        // `contributor_cli::wait_for_first_peer` awaited the shared stream for
        // a peer event and discarded every gossip message that arrived during
        // its window. Its replacement sees only control events.
        let (lane, handle) = router();
        let mut watcher = handle
            .subscribe(Interests::none().control())
            .expect("watcher");
        let mut relay = handle
            .subscribe(Interests::none().every_topic())
            .expect("relay");

        lane.send(gossip(TOPIC_A, "published-during-peer-wait"))
            .await
            .unwrap();
        lane.send(control()).await.unwrap();

        assert!(matches!(
            recv_within(&mut watcher, "the peer event the watcher is waiting on").await,
            OmniNetEvent::PeerConnected { .. }
        ));
        assert!(
            watcher.try_recv().is_none(),
            "the peer watcher must never be handed gossip it would discard"
        );
        assert_eq!(
            body_of(&recv_within(&mut relay, "the gossip published during the peer wait").await),
            "published-during-peer-wait"
        );
    }

    // ── Registration lifecycle ───────────────────────────────────────────

    #[tokio::test]
    async fn dropping_a_subscription_deregisters_it() {
        let (lane, handle) = router();
        let mut keeper = handle.subscribe(Interests::none().every_topic()).unwrap();
        let leaver = handle.subscribe(Interests::none().every_topic()).unwrap();
        assert_eq!(handle.subscriber_count(), 2);

        drop(leaver);
        assert_eq!(
            handle.subscriber_count(),
            1,
            "a dropped subscription must release its slot immediately"
        );
        assert_eq!(handle.counts().subscriptions_closed, 1);

        // The survivor is unaffected.
        lane.send(gossip(TOPIC_A, "after-drop")).await.unwrap();
        assert_eq!(
            body_of(&recv_within(&mut keeper, "the survivor's event after the drop").await),
            "after-drop"
        );
        // And the router never counts a departed-subscriber delivery for it.
        assert_eq!(handle.counts().dropped_departed, 0);
    }

    #[tokio::test]
    async fn a_dropped_subscriber_stops_costing_deliveries() {
        let (lane, handle) = router();
        let listener = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();
        lane.send(gossip(TOPIC_A, "one")).await.unwrap();
        // Let the router observe the first event before the drop.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(handle.counts().delivered, 1);

        drop(listener);
        lane.send(gossip(TOPIC_A, "two")).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let counts = handle.counts();
        assert_eq!(counts.delivered, 1, "no copy is made for a gone consumer");
        assert_eq!(
            counts.unwanted_gossip, 1,
            "the event is counted as unsubscribed, not silently discarded"
        );
    }

    #[tokio::test]
    async fn router_shutdown_ends_every_subscription() {
        let (lane, handle) = router();
        let mut relay = handle.subscribe(Interests::none().every_topic()).unwrap();
        let mut transport = handle.subscribe(Interests::none().tensor()).unwrap();

        // Closing the swarm lane is what a stopped swarm task looks like.
        drop(lane);

        end_within(&mut relay, "the relay's stream").await;
        end_within(&mut transport, "the tensor consumer's stream").await;
        assert!(!handle.is_running());
        assert_eq!(
            handle.subscribe(Interests::everything()).unwrap_err(),
            RouterStopped,
            "a late subscriber must be told the router is gone"
        );
    }

    #[tokio::test]
    async fn shutdown_releases_subscribers_still_holding_their_handles() {
        let (lane, handle) = router();
        let _relay = handle.subscribe(Interests::none().every_topic()).unwrap();
        let _transport = handle.subscribe(Interests::none().tensor()).unwrap();
        assert_eq!(handle.subscriber_count(), 2);

        drop(lane);
        // Give the router task its turn to notice the closed lane.
        for _ in 0..50 {
            if !handle.is_running() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(
            handle.subscriber_count(),
            0,
            "shutdown must release retained subscriber state"
        );
        assert_eq!(handle.counts().subscriptions_closed, 2);
    }

    // ── Counters ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn events_nobody_asked_for_are_counted_by_class() {
        let (_lane, handle) = router();
        // Only a gossip consumer is registered.
        let _relay = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        handle.route_for_test(gossip(TOPIC_B, "wrong topic"));
        handle.route_for_test(shard_request("cid"));
        handle.route_for_test(tensor("s"));
        handle.route_for_test(control());
        handle.route_for_test(control());

        let counts = handle.counts();
        assert_eq!(counts.unwanted_gossip, 1);
        assert_eq!(counts.unwanted_shard, 1);
        assert_eq!(counts.unwanted_tensor, 1);
        assert_eq!(counts.unwanted_control, 2);
        assert_eq!(counts.events, 5);
        assert_eq!(counts.delivered, 0);
        assert_eq!(
            counts.dropped_backlogged, 0,
            "nobody asked for these — that is not a delivery failure"
        );
        assert_eq!(counts.dropped_departed, 0);
    }

    #[tokio::test]
    async fn a_backlogged_subscriber_is_counted_apart_from_an_unwanted_event() {
        // The distinction the counters exist for: this consumer *did* ask, and
        // the router *did* fail it. Reporting that as "nobody subscribed"
        // would hide a consumer that is falling behind.
        let (_lane, handle) = router();
        let _slow = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        for i in 0..(SUBSCRIBER_CAPACITY + 5) {
            handle.route_for_test(gossip(TOPIC_A, &format!("m{i}")));
        }

        let counts = handle.counts();
        assert_eq!(counts.delivered, SUBSCRIBER_CAPACITY as u64);
        assert_eq!(counts.dropped_backlogged, 5);
        assert_eq!(
            counts.unwanted_gossip, 0,
            "a backlogged subscriber is not the same fault as no subscriber"
        );
    }

    #[tokio::test]
    async fn a_vanished_subscriber_is_counted_and_swept() {
        // A subscription whose receiver was leaked past its `Drop` — the
        // registry can only learn about it from a failed delivery.
        let (_lane, handle) = router();
        let rx = handle.orphan_subscriber_for_test(Interests::none().topic(TOPIC_A));
        drop(rx);
        assert_eq!(
            handle.subscriber_count(),
            1,
            "the orphaned slot survives until a delivery fails"
        );

        handle.route_for_test(gossip(TOPIC_A, "into the void"));

        let counts = handle.counts();
        assert_eq!(counts.dropped_departed, 1);
        assert_eq!(counts.delivered, 0);
        assert_eq!(
            handle.subscriber_count(),
            0,
            "a departed subscriber must be swept out of the table"
        );
    }

    // ── Shadow byte accounting ───────────────────────────────────────────

    #[tokio::test]
    async fn the_router_weighs_exactly_the_events_it_routes() {
        // The ledger is wired into `route`, so it sees production traffic
        // rather than a reconstruction of it. Both paths through `route` are
        // exercised: an event with a taker and an event with none.
        //
        // The property that changed: a charge is held for as long as the copy
        // sits in a subscriber's queue, not for the duration of the fan-out.
        // Releasing at the end of `route` would measure one routing call and
        // call it "in flight", which is what an earlier version of this did.
        let (_lane, handle) = router();
        let mut consumer = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        let wanted = gossip(TOPIC_A, "short");
        let unwanted = gossip(TOPIC_B, &"x".repeat(4_096));
        let wanted_weight = crate::budget::weight(&wanted);
        let expected = wanted_weight + crate::budget::weight(&unwanted);

        handle.route_for_test(wanted);
        handle.route_for_test(unwanted);

        let bytes = handle.byte_counts();
        assert_eq!(bytes.events, 2, "both events were weighed");
        assert_eq!(bytes.bytes, expected, "including the one nobody wanted");
        assert_eq!(bytes.bytes_saturated, 0);

        // The delivered copy is still sitting in the consumer's queue, so its
        // bytes are still retained. The unwanted one was never enqueued and so
        // never took a charge — the whole 4 KiB of it.
        assert_eq!(
            bytes.bytes_in_flight, wanted_weight,
            "only the queued copy is in flight; the undelivered event took no charge"
        );

        // Dequeuing releases it, and nothing else.
        let _ = recv_within(&mut consumer, "the queued gossip event").await;
        assert_eq!(
            handle.byte_counts().bytes_in_flight,
            0,
            "dequeuing releases exactly the charge its queue slot held"
        );
    }

    #[tokio::test]
    async fn accounting_observes_but_does_not_refuse() {
        // The load-bearing property of a shadow increment. This traffic would
        // exhaust any plausible budget — every event on its own is larger than
        // the shadow per-peer ceiling — and the ledger says so. Not one event's
        // fate differs: all of them are delivered, in order, intact, and the
        // router's own counters read exactly as they would with no ledger at
        // all. The day accounting starts refusing, this test fails.
        let (_lane, handle) = router();
        let mut consumer = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        let peer = PeerId::random();
        let over_ceiling = usize::try_from(crate::budget::SHADOW_PEER_BYTES)
            .expect("the shadow ceiling fits a usize")
            + 1;
        const FLOOD: u64 = 4;

        for marker in 1..=FLOOD {
            handle.route_for_test(OmniNetEvent::MessageReceived {
                from: peer,
                topic: TOPIC_A.to_string(),
                // A distinct first byte per event, so a reordering or a
                // swallowed event shows up as a wrong marker rather than a
                // right count.
                data: {
                    let mut data = vec![0u8; over_ceiling];
                    data[0] = u8::try_from(marker).expect("marker fits a byte");
                    data
                },
            });
        }

        let bytes = handle.byte_counts();
        assert_eq!(
            bytes.would_refuse_peer, FLOOD,
            "a budget at the shadow ceiling would have refused every one: {bytes:?}"
        );
        assert!(bytes.would_refuse_any());

        // And yet.
        for marker in 1..=FLOOD {
            let event =
                recv_within(&mut consumer, &format!("the flood event marked {marker}")).await;
            match event {
                OmniNetEvent::MessageReceived { data, .. } => {
                    assert_eq!(
                        u64::from(data[0]),
                        marker,
                        "the flood was delivered out of order or short"
                    );
                    assert_eq!(data.len(), over_ceiling, "payload arrived truncated");
                }
                other => panic!("the consumer received {other:?}"),
            }
        }

        let counts = handle.counts();
        assert_eq!(counts.events, FLOOD);
        assert_eq!(
            counts.delivered, FLOOD,
            "shadow accounting must not cost a single delivery"
        );
        assert_eq!(counts.dropped_backlogged, 0);
        assert_eq!(counts.dropped_departed, 0);
        assert_eq!(counts.unwanted_gossip, 0);
        assert_eq!(
            handle.byte_counts().bytes_in_flight,
            0,
            "and the ledger is back to zero, holding nothing"
        );
    }

    /// The load-bearing case for queue retention: a backlog of several events
    /// sitting in one subscriber's queue holds several charges at once.
    ///
    /// Releasing at the end of each fan-out — the behaviour this amends —
    /// leaves `bytes_in_flight` at zero here no matter how deep the backlog
    /// gets, which is exactly why the old accounting could only ever answer
    /// "does one event exceed the ceiling".
    #[tokio::test]
    async fn a_backlog_of_events_retains_every_charge_until_drained() {
        let (_lane, handle) = router();
        let mut consumer = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        let bodies = ["one", "two", "three", "four", "five"];
        let mut expected = 0u64;
        for body in bodies {
            let event = gossip(TOPIC_A, body);
            expected += crate::budget::weight(&event);
            handle.route_for_test(event);
        }

        assert_eq!(
            handle.byte_counts().bytes_in_flight,
            expected,
            "every queued event's charge is held at once, not just the last"
        );

        // Draining releases them one at a time, monotonically.
        let mut last = expected;
        for body in bodies {
            let _ = recv_within(&mut consumer, body).await;
            let now = handle.byte_counts().bytes_in_flight;
            assert!(now < last, "each dequeue must release its own charge");
            last = now;
        }
        assert_eq!(last, 0, "a fully drained queue retains nothing");
    }

    /// One event, several interested subscribers: each queue keeps its own
    /// copy, so each copy is charged.
    #[tokio::test]
    async fn each_delivered_clone_is_charged_separately() {
        let (_lane, handle) = router();
        let mut a = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();
        let mut b = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();
        let mut c = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        let event = gossip(TOPIC_A, "shared");
        let one = crate::budget::weight(&event);
        handle.route_for_test(event);

        assert_eq!(
            handle.byte_counts().bytes_in_flight,
            one * 3,
            "three queues retain three payloads, so three charges are open"
        );

        for (sub, name) in [(&mut a, "a"), (&mut b, "b"), (&mut c, "c")] {
            let _ = recv_within(sub, name).await;
        }
        assert_eq!(handle.byte_counts().bytes_in_flight, 0);
    }

    /// A copy that never reaches a queue is never charged for one. The
    /// envelope comes back out of the failed `try_send` and dies there.
    #[tokio::test]
    async fn a_full_subscriber_queue_releases_the_bounced_charge() {
        let (_lane, handle) = router();
        let _consumer = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        // Fill the queue exactly, then overflow it.
        for i in 0..SUBSCRIBER_CAPACITY {
            handle.route_for_test(gossip(TOPIC_A, &format!("fill-{i}")));
        }
        let full = handle.byte_counts().bytes_in_flight;
        assert!(full > 0, "the queue is holding its backlog");

        let overflow = gossip(TOPIC_A, "bounced");
        handle.route_for_test(overflow);

        assert_eq!(
            handle.byte_counts().bytes_in_flight,
            full,
            "a bounced copy adds no retained bytes"
        );
        assert_eq!(
            handle.counts().dropped_backlogged,
            1,
            "and it is counted as a delivery failure, not as unwanted"
        );
    }

    /// A subscriber that goes away still releases whatever its queue held.
    #[tokio::test]
    async fn dropping_a_subscription_releases_its_whole_backlog() {
        let (_lane, handle) = router();
        let consumer = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();

        for i in 0..8 {
            handle.route_for_test(gossip(TOPIC_A, &format!("queued-{i}")));
        }
        assert!(handle.byte_counts().bytes_in_flight > 0);

        drop(consumer);
        assert_eq!(
            handle.byte_counts().bytes_in_flight,
            0,
            "dropping the subscription drops its queue, which releases its charges"
        );
    }

    /// No charge outlives the router that opened it.
    ///
    /// Shutdown closes the sender side; a closed channel still hands over what
    /// it already buffered, so the queued copies — and their charges — survive
    /// until the consumer drains or drops. Both routes are asserted here,
    /// because "shutdown zeroes the ledger instantly" would be a nice claim and
    /// a false one.
    #[tokio::test]
    async fn shutdown_releases_every_retained_charge() {
        let (lane, handle) = router();
        let mut a = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();
        let b = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();
        for i in 0..4 {
            handle.route_for_test(gossip(TOPIC_A, &format!("q-{i}")));
        }
        assert!(handle.byte_counts().bytes_in_flight > 0);

        // A stopped swarm task is the lane closing.
        drop(lane);
        assert!(
            handle.byte_counts().bytes_in_flight > 0,
            "shutdown does not reach into a queue that already has copies in it"
        );

        // Route one: the consumer drains. Each dequeue releases its own charge,
        // and the stream then ends rather than hanging.
        for _ in 0..4 {
            let _ = recv_within(&mut a, "a buffered event after shutdown").await;
        }
        end_within(&mut a, "the drained subscriber's stream").await;

        // Route two: the consumer never drains and simply goes away.
        drop(b);
        assert_eq!(
            handle.byte_counts().bytes_in_flight,
            0,
            "between draining and dropping, no charge outlives the router"
        );
    }

    #[tokio::test]
    async fn routing_for_many_peers_leaves_no_per_peer_state_behind() {
        // The ledger's one keyed map, seen through the router: a thousand
        // distinct remote identities — one keypair each, which is free to
        // mint — leave nothing behind. A map keyed by "peers ever seen" would
        // be a thousand entries here and unbounded in production.
        let (_lane, handle) = router();

        // First the early-return path: nobody has subscribed, so every event
        // is counted as unwanted and `route` returns before any charge.
        for _ in 0..1_000 {
            handle.route_for_test(control());
        }
        assert_eq!(
            handle.peers_in_flight_for_test(),
            0,
            "an event with no interested subscriber must take no charge at all"
        );

        // Then the delivery path. Each copy is charged on enqueue and released
        // on dequeue, so the map is bounded by what is *queued*, never by how
        // many distinct peers have ever been seen.
        let mut watcher = handle.subscribe(Interests::none().control()).unwrap();
        for _ in 0..100 {
            handle.route_for_test(control());
        }
        assert!(
            handle.peers_in_flight_for_test() > 0,
            "queued copies are still retained"
        );

        for _ in 0..100 {
            let _ = recv_within(&mut watcher, "a queued control event").await;
        }

        assert_eq!(
            handle.peers_in_flight_for_test(),
            0,
            "draining the queue returns the map to empty"
        );
        let bytes = handle.byte_counts();
        assert_eq!(bytes.events, 1_100);
        assert_eq!(bytes.bytes_in_flight, 0);
        assert_eq!(bytes.peers_in_flight, 0);
        assert!(
            bytes.peak_peers_in_flight <= 100,
            "the map is bounded by queued copies, not by peers ever seen: {bytes:?}"
        );
    }

    #[tokio::test]
    async fn delivery_counts_one_copy_per_interested_subscriber() {
        let (_lane, handle) = router();
        let _a = handle.subscribe(Interests::none().topic(TOPIC_A)).unwrap();
        let _b = handle.subscribe(Interests::none().every_topic()).unwrap();
        let _c = handle.subscribe(Interests::none().shard()).unwrap();

        handle.route_for_test(gossip(TOPIC_A, "two takers"));
        handle.route_for_test(shard_request("cid"));

        let counts = handle.counts();
        assert_eq!(counts.events, 2);
        assert_eq!(counts.delivered, 3);
        assert_eq!(counts.subscriptions_opened, 3);
    }
}
