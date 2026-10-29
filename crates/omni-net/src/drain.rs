//! Bounded drain of recently published gossip before the swarm stops.
//!
//! `gossipsub::Behaviour::publish` does not send anything. It pushes the
//! message onto a per-peer queue, and each connection's handler writes it to
//! the wire the next time that connection task runs. If the swarm is dropped
//! first — the event loop returns, the process exits — the message never
//! leaves the node, even though `publish` succeeded and "message published"
//! was logged. A command that broadcasts and then exits lost its last
//! announcement this way.
//!
//! The send queue cannot be observed from outside the behaviour:
//! libp2p-gossipsub keeps it crate-private and reports no per-message send
//! completion, and libp2p-quic does not surface stream acknowledgements. So
//! the drain is a bound on time, not a flush acknowledgement. When the loop is
//! asked to stop, it keeps driving the swarm — connections stay open, handlers
//! keep writing, inbound traffic is still processed — until every gossip
//! message this node accepted within the last `grace` has had `grace` of
//! continued network service since it was accepted. It stops sooner if no peer
//! that message was queued for is still connected, and at once when nothing
//! was published within the window. The wait never exceeds `grace` past the
//! shutdown request, and `grace` itself is capped at [`MAX_SHUTDOWN_DRAIN`].
//!
//! Only messages this node originated are tracked. Gossip it was merely
//! forwarding for others is not held for.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use libp2p::{gossipsub::MessageId, PeerId};

/// Drain used by [`crate::NetHandle::shutdown`] and when every handle to the
/// node is dropped. Matches the `--propagation-wait-ms` default of the one-shot
/// announce commands.
pub const DEFAULT_SHUTDOWN_DRAIN: Duration = Duration::from_millis(200);

/// Upper bound on any drain, whatever the caller asks for. Keeps shutdown
/// bounded even if an operator passes an absurd wait.
pub const MAX_SHUTDOWN_DRAIN: Duration = Duration::from_secs(30);

/// How many recent publishes are remembered. Only the newest matter for the
/// deadline; the bound keeps a node that publishes continuously from growing
/// the record without limit.
pub(crate) const RECENT_PUBLISH_CAP: usize = 1024;

/// Why a drain ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainOutcome {
    /// The caller asked for no drain (`grace` zero): the pre-drain behaviour.
    Disabled,
    /// Nothing this node published was still inside the window when the stop
    /// was requested, so there was nothing to wait for.
    Idle,
    /// Every publish inside the window was given its full `grace`.
    Settled,
    /// No peer an outstanding publish was queued for was still connected, so
    /// nothing more could be delivered.
    RecipientsGone,
}

impl DrainOutcome {
    /// Stable lower-case label for logs.
    pub fn as_str(self) -> &'static str {
        match self {
            DrainOutcome::Disabled => "disabled",
            DrainOutcome::Idle => "idle",
            DrainOutcome::Settled => "settled",
            DrainOutcome::RecipientsGone => "recipients_gone",
        }
    }
}

/// What a drain did. Returned by [`crate::NetHandle::shutdown_with_drain`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrainReport {
    pub outcome: DrainOutcome,
    /// The bound actually applied (the request, capped at
    /// [`MAX_SHUTDOWN_DRAIN`]).
    pub grace: Duration,
    /// How long the loop kept the swarm running after the stop request.
    pub waited: Duration,
    /// Publishes that were inside the window when the stop was requested —
    /// the ones the drain waited for.
    pub publishes_in_window: usize,
    /// Of the peers those publishes were queued for, how many were still
    /// connected when the drain ended.
    pub recipients_connected: usize,
}

/// One accepted gossip publish.
#[derive(Debug, Clone)]
pub(crate) struct RecentPublish {
    pub at: Instant,
    pub topic: String,
    pub message_id: MessageId,
    /// Peers subscribed to the topic when the message was queued — the peers
    /// gossipsub could have queued it for.
    pub recipients: Vec<PeerId>,
}

/// The publishes the drain may have to wait for, newest last.
#[derive(Debug, Default)]
pub(crate) struct RecentPublishes {
    entries: VecDeque<RecentPublish>,
}

impl RecentPublishes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember a publish, forgetting what can no longer matter: anything
    /// older than [`MAX_SHUTDOWN_DRAIN`], and the oldest beyond
    /// [`RECENT_PUBLISH_CAP`].
    pub fn record(&mut self, publish: RecentPublish) {
        let now = publish.at;
        while let Some(front) = self.entries.front() {
            if now.saturating_duration_since(front.at) > MAX_SHUTDOWN_DRAIN {
                self.entries.pop_front();
            } else {
                break;
            }
        }
        while self.entries.len() >= RECENT_PUBLISH_CAP {
            self.entries.pop_front();
        }
        self.entries.push_back(publish);
    }

    /// Publishes still inside a `grace` window at `now`.
    pub fn in_window(&self, now: Instant, grace: Duration) -> impl Iterator<Item = &RecentPublish> {
        self.entries
            .iter()
            .filter(move |p| p.at + grace > now)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The bound applied for a requested grace.
pub(crate) fn effective_grace(requested: Duration) -> Duration {
    requested.min(MAX_SHUTDOWN_DRAIN)
}

/// What the drain should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DrainStep {
    /// Keep driving the swarm until this instant, re-evaluating after every
    /// swarm event.
    WaitUntil(Instant),
    /// Stop now.
    Exit(DrainOutcome),
}

/// Decide the next drain step.
///
/// `grace` must already be capped ([`effective_grace`]). `waited_for_any`
/// records whether the drain found anything to wait for when it started, which
/// is what separates [`DrainOutcome::Settled`] from [`DrainOutcome::Idle`].
///
/// Bound: every publish was accepted no later than `now`, so a returned
/// deadline is never later than `now + grace`.
pub(crate) fn next_step(
    now: Instant,
    grace: Duration,
    publishes: &RecentPublishes,
    waited_for_any: bool,
    is_connected: impl Fn(&PeerId) -> bool,
) -> DrainStep {
    if grace.is_zero() {
        return DrainStep::Exit(DrainOutcome::Disabled);
    }
    let mut deadline: Option<Instant> = None;
    let mut any_recipient_connected = false;
    for p in publishes.in_window(now, grace) {
        let until = p.at + grace;
        deadline = Some(deadline.map_or(until, |d| d.max(until)));
        if !any_recipient_connected {
            any_recipient_connected = p.recipients.iter().any(&is_connected);
        }
    }
    match deadline {
        None if waited_for_any => DrainStep::Exit(DrainOutcome::Settled),
        None => DrainStep::Exit(DrainOutcome::Idle),
        Some(_) if !any_recipient_connected => DrainStep::Exit(DrainOutcome::RecipientsGone),
        Some(d) => DrainStep::WaitUntil(d),
    }
}

/// Distinct recipients of the publishes inside the window at `now`, and how
/// many of them are connected.
pub(crate) fn recipients_in_window(
    now: Instant,
    grace: Duration,
    publishes: &RecentPublishes,
    is_connected: impl Fn(&PeerId) -> bool,
) -> (usize, usize) {
    let mut peers: Vec<PeerId> = publishes
        .in_window(now, grace)
        .flat_map(|p| p.recipients.iter().copied())
        .collect();
    peers.sort();
    peers.dedup();
    let connected = peers.iter().filter(|p| is_connected(p)).count();
    (peers.len(), connected)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRACE: Duration = Duration::from_millis(200);

    fn publish(at: Instant, recipients: Vec<PeerId>) -> RecentPublish {
        RecentPublish {
            at,
            topic: "omni/test/v1".into(),
            message_id: MessageId::from(b"id".to_vec()),
            recipients,
        }
    }

    fn record(entries: Vec<RecentPublish>) -> RecentPublishes {
        let mut r = RecentPublishes::new();
        for e in entries {
            r.record(e);
        }
        r
    }

    #[test]
    fn zero_grace_disables_the_drain_even_with_a_fresh_publish() {
        let now = Instant::now();
        let peer = PeerId::random();
        let r = record(vec![publish(now, vec![peer])]);
        assert_eq!(
            next_step(now, Duration::ZERO, &r, false, |_| true),
            DrainStep::Exit(DrainOutcome::Disabled)
        );
    }

    #[test]
    fn nothing_published_means_no_wait() {
        let now = Instant::now();
        assert_eq!(
            next_step(now, GRACE, &RecentPublishes::new(), false, |_| true),
            DrainStep::Exit(DrainOutcome::Idle)
        );
    }

    #[test]
    fn a_publish_older_than_the_window_is_not_waited_for() {
        let now = Instant::now() + Duration::from_secs(1);
        let r = record(vec![publish(now - GRACE, vec![PeerId::random()])]);
        assert_eq!(
            next_step(now, GRACE, &r, false, |_| true),
            DrainStep::Exit(DrainOutcome::Idle)
        );
    }

    #[test]
    fn a_fresh_publish_is_held_for_the_rest_of_its_window() {
        let now = Instant::now() + Duration::from_secs(1);
        let at = now - Duration::from_millis(50);
        let r = record(vec![publish(at, vec![PeerId::random()])]);
        assert_eq!(
            next_step(now, GRACE, &r, false, |_| true),
            DrainStep::WaitUntil(at + GRACE)
        );
    }

    #[test]
    fn the_newest_publish_sets_the_deadline() {
        let now = Instant::now() + Duration::from_secs(1);
        let older = now - Duration::from_millis(150);
        let newer = now - Duration::from_millis(10);
        let r = record(vec![
            publish(older, vec![PeerId::random()]),
            publish(newer, vec![PeerId::random()]),
        ]);
        assert_eq!(
            next_step(now, GRACE, &r, false, |_| true),
            DrainStep::WaitUntil(newer + GRACE)
        );
    }

    #[test]
    fn the_deadline_never_exceeds_grace_past_the_request() {
        let now = Instant::now() + Duration::from_secs(5);
        for age_ms in [0u64, 1, 7, 50, 199, 200, 201, 1_000] {
            for grace_ms in [1u64, 10, 200, 5_000] {
                let grace = Duration::from_millis(grace_ms);
                let r = record(vec![publish(
                    now - Duration::from_millis(age_ms),
                    vec![PeerId::random()],
                )]);
                if let DrainStep::WaitUntil(d) = next_step(now, grace, &r, false, |_| true) {
                    assert!(d <= now + grace, "age {age_ms} grace {grace_ms}");
                    assert!(d > now, "age {age_ms} grace {grace_ms}");
                }
            }
        }
    }

    #[test]
    fn the_window_ends_settled_once_it_has_been_waited_out() {
        let start = Instant::now() + Duration::from_secs(1);
        let r = record(vec![publish(start, vec![PeerId::random()])]);
        // Inside the window: wait.
        assert_eq!(
            next_step(start, GRACE, &r, false, |_| true),
            DrainStep::WaitUntil(start + GRACE)
        );
        // At the deadline: done, and it was something worth waiting for.
        assert_eq!(
            next_step(start + GRACE, GRACE, &r, true, |_| true),
            DrainStep::Exit(DrainOutcome::Settled)
        );
    }

    #[test]
    fn no_connected_recipient_ends_the_drain_early() {
        let now = Instant::now() + Duration::from_secs(1);
        let gone = PeerId::random();
        let r = record(vec![publish(now, vec![gone])]);
        assert_eq!(
            next_step(now, GRACE, &r, true, |_| false),
            DrainStep::Exit(DrainOutcome::RecipientsGone)
        );
    }

    #[test]
    fn one_connected_recipient_keeps_the_drain_running() {
        let now = Instant::now() + Duration::from_secs(1);
        let gone = PeerId::random();
        let here = PeerId::random();
        let r = record(vec![publish(now, vec![gone]), publish(now, vec![here])]);
        assert_eq!(
            next_step(now, GRACE, &r, true, |p| *p == here),
            DrainStep::WaitUntil(now + GRACE)
        );
        assert_eq!(recipients_in_window(now, GRACE, &r, |p| *p == here), (2, 1));
    }

    #[test]
    fn requested_grace_is_capped() {
        assert_eq!(effective_grace(Duration::from_millis(150)), Duration::from_millis(150));
        assert_eq!(effective_grace(Duration::from_secs(3_600)), MAX_SHUTDOWN_DRAIN);
        assert_eq!(effective_grace(Duration::MAX), MAX_SHUTDOWN_DRAIN);
    }

    #[test]
    fn the_record_is_bounded_in_count_and_age() {
        let start = Instant::now();
        let mut r = RecentPublishes::new();
        for i in 0..(RECENT_PUBLISH_CAP + 10) {
            r.record(publish(start + Duration::from_micros(i as u64), vec![]));
        }
        assert_eq!(r.len(), RECENT_PUBLISH_CAP);
        // A publish far past the cap's age forgets everything before it.
        r.record(publish(start + MAX_SHUTDOWN_DRAIN + Duration::from_secs(1), vec![]));
        assert_eq!(r.len(), 1);
    }
}
