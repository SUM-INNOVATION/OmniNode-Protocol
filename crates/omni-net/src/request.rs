//! Per-request correlation and completion delivery.
//!
//! A solicited request (shard chunk, tensor handoff) is issued from a caller
//! task but sent from the swarm task, which is the only owner of the libp2p
//! `Swarm`. The `OutboundRequestId` that identifies the request on the wire
//! therefore only exists inside the swarm task, and only from the moment
//! `send_request` returns.
//!
//! This module holds the machinery that keeps the two halves joined:
//!
//! * [`PendingRequests`] — the swarm-side table mapping a request key to the
//!   completion channel of the caller that is waiting for it.
//! * [`Pending`] — the caller-side handle: a future that resolves to the
//!   response, or to a [`RequestError`] explaining why one will never come.
//!
//! ## Why a table and not the event stream
//!
//! Responses used to be published on the shared broadcast event stream, which
//! made them indistinguishable from unsolicited traffic and impossible to
//! attribute: two requests to the same peer produced two `ShardReceived`
//! events with nothing to tell them apart. The table is keyed by the
//! `OutboundRequestId`, so two concurrent requests to one peer complete
//! independently and out-of-order responses land on the right caller.
//!
//! ## No caller may wait forever
//!
//! Every route out of the table completes the caller:
//!
//! | what happened                     | how the caller learns          |
//! |-----------------------------------|--------------------------------|
//! | response arrived                  | `Ok(response)`                 |
//! | libp2p reported an outbound error | [`RequestError::Outbound`]     |
//! | libp2p's own request timeout       | [`RequestError::Outbound`]     |
//! | caller-side deadline elapsed      | [`RequestError::Timeout`]      |
//! | swarm loop shut down              | [`RequestError::RouterGone`]   |
//! | swarm task died / channel closed  | [`RequestError::RouterGone`]   |
//! | command channel already closed    | [`RequestError::NotSent`]      |
//!
//! The last two are covered without any explicit code path: dropping the
//! `oneshot::Sender` (because the table was dropped with the swarm task)
//! closes the channel, and [`Pending`] maps that closure to `RouterGone`.

use std::collections::HashMap;
use std::hash::Hash;
use std::time::Duration;

use tokio::sync::oneshot;

/// Sweep the table for cancelled callers once it reaches this size.
///
/// A caller that drops its [`Pending`] before the response arrives leaves an
/// entry behind. libp2p's own request timeout eventually retires it, but that
/// is up to two minutes for shard transfers, so the table is swept whenever it
/// is already large enough for the linear scan to be worth its cost.
const SWEEP_THRESHOLD: usize = 64;

// ── RequestError ──────────────────────────────────────────────────────────────

/// Why a solicited request did not produce a response.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RequestError {
    /// libp2p reported a failure for this specific request — dial failure,
    /// connection closed, unsupported protocol, or its own request timeout.
    #[error("outbound request to {peer} failed: {error}")]
    Outbound {
        /// The peer the request was addressed to.
        peer: String,
        /// libp2p's rendering of the failure.
        error: String,
    },

    /// The caller's own deadline elapsed first. The request may still be in
    /// flight on the wire; the swarm retires its table entry when the response
    /// or libp2p's failure arrives.
    #[error("request timed out after {0:?}")]
    Timeout(Duration),

    /// The swarm loop stopped — either a clean `Shutdown` or the task dying —
    /// before the response arrived.
    #[error("swarm loop stopped before the response arrived")]
    RouterGone,

    /// The command never reached the swarm loop, so no request was sent.
    #[error("swarm loop has stopped — request was never sent")]
    NotSent,
}

// ── Completion channel ────────────────────────────────────────────────────────

/// Swarm-side half of one request's completion channel.
pub(crate) type Completion<V> = oneshot::Sender<Result<V, RequestError>>;

/// Caller-side handle to one in-flight solicited request.
///
/// Awaiting it yields the response or the reason there will not be one.
/// Dropping it cancels the caller's interest: the swarm notices the closed
/// channel and releases the table entry on its next sweep.
#[derive(Debug)]
pub struct Pending<V> {
    rx: oneshot::Receiver<Result<V, RequestError>>,
}

impl<V> Pending<V> {
    /// Wrap the receiving half of a completion channel.
    pub(crate) fn new(rx: oneshot::Receiver<Result<V, RequestError>>) -> Self {
        Self { rx }
    }

    /// A handle that is already resolved to `err` — used when the command
    /// could not even be handed to the swarm loop, so the caller sees the
    /// same shape whether the failure was early or late.
    pub(crate) fn failed(err: RequestError) -> Self {
        let (tx, rx) = oneshot::channel();
        // The receiver is alive right here, so this send cannot fail.
        let _ = tx.send(Err(err));
        Self { rx }
    }

    /// Wait for the response, however long libp2p's own request timeout takes.
    ///
    /// Resolves to [`RequestError::RouterGone`] if the swarm task drops the
    /// completion channel without sending — the case where a caller would
    /// otherwise wait forever.
    pub async fn response(self) -> Result<V, RequestError> {
        match self.rx.await {
            Ok(result) => result,
            Err(_closed) => Err(RequestError::RouterGone),
        }
    }

    /// Wait for the response, giving up after `within`.
    ///
    /// On expiry the handle is consumed, which closes the completion channel
    /// and lets the swarm release its table entry.
    pub async fn response_within(self, within: Duration) -> Result<V, RequestError> {
        match tokio::time::timeout(within, self.response()).await {
            Ok(result) => result,
            Err(_elapsed) => Err(RequestError::Timeout(within)),
        }
    }
}

// ── PendingRequests ───────────────────────────────────────────────────────────

/// Swarm-side table of in-flight solicited requests.
///
/// Generic over the key so the correlation property can be unit-tested:
/// production instantiates it with libp2p's `OutboundRequestId`, which has no
/// public constructor.
///
/// The key is what makes correlation correct. Keying by anything coarser than
/// the per-request id — the peer, say — collapses two concurrent requests to
/// one peer into a single entry, and the first response completes the wrong
/// caller while the second is delivered to nobody.
pub(crate) struct PendingRequests<K, V> {
    inflight: HashMap<K, Completion<V>>,
}

impl<K: Eq + Hash, V> PendingRequests<K, V> {
    /// An empty table.
    pub(crate) fn new() -> Self {
        Self {
            inflight: HashMap::new(),
        }
    }

    /// How many requests are currently retained.
    pub(crate) fn len(&self) -> usize {
        self.inflight.len()
    }

    /// Record `completion` as the destination for `key`'s response.
    ///
    /// Sweeps abandoned entries first once the table is large enough that a
    /// leak would matter, so a caller that walks away cannot pin state until
    /// libp2p's request timeout fires.
    pub(crate) fn insert(&mut self, key: K, completion: Completion<V>) {
        if self.inflight.len() >= SWEEP_THRESHOLD {
            self.sweep_cancelled();
        }
        self.inflight.insert(key, completion);
    }

    /// Deliver `result` to the caller waiting on `key` and release the entry.
    ///
    /// Returns `false` when no caller was waiting — the request was
    /// unsolicited, already completed, or the caller cancelled — in which case
    /// the caller of this method is free to fall back to the event stream.
    pub(crate) fn complete(&mut self, key: &K, result: Result<V, RequestError>) -> bool {
        match self.inflight.remove(key) {
            // A closed receiver means the caller cancelled between the sweep
            // and now. The entry is gone either way, which is the point.
            Some(completion) => completion.send(result).is_ok(),
            None => false,
        }
    }

    /// Whether `key` has a caller waiting on it.
    pub(crate) fn contains(&self, key: &K) -> bool {
        self.inflight.contains_key(key)
    }

    /// Drop every entry whose caller has gone away.
    ///
    /// Returns how many were released.
    pub(crate) fn sweep_cancelled(&mut self) -> usize {
        let before = self.inflight.len();
        self.inflight
            .retain(|_, completion| !completion.is_closed());
        before - self.inflight.len()
    }

    /// Complete every retained request with `err` and empty the table.
    ///
    /// Called when the swarm loop exits, so no caller is left waiting on a
    /// response that can no longer arrive.
    pub(crate) fn fail_all(&mut self, err: RequestError) -> usize {
        let failed = self.inflight.len();
        for (_, completion) in self.inflight.drain() {
            let _ = completion.send(Err(err.clone()));
        }
        failed
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Stand-in for `OutboundRequestId`, which libp2p does not let us build.
    /// Production keys the table by that type; the property under test — one
    /// entry per request, not per peer — is the same either way.
    type ReqId = u64;

    fn channel() -> (
        Completion<&'static str>,
        oneshot::Receiver<Result<&'static str, RequestError>>,
    ) {
        oneshot::channel()
    }

    #[tokio::test]
    async fn response_is_delivered_to_the_caller_that_asked() {
        let mut table = PendingRequests::<ReqId, &str>::new();
        let (tx, rx) = channel();
        table.insert(1, tx);

        assert!(table.complete(&1, Ok("chunk")));
        assert_eq!(table.len(), 0, "a completed request must not stay retained");
        assert_eq!(Pending::new(rx).response().await.unwrap(), "chunk");
    }

    #[tokio::test]
    async fn two_requests_to_one_peer_complete_independently() {
        // The correlation property. Both requests below are addressed to the
        // same peer; only the per-request key tells them apart. Keying the
        // table by PeerId collapses them and this test fails.
        let mut table = PendingRequests::<ReqId, &str>::new();
        let (tx_a, rx_a) = channel();
        let (tx_b, rx_b) = channel();
        table.insert(10, tx_a);
        table.insert(11, tx_b);
        assert_eq!(table.len(), 2, "one entry per request, not per peer");

        assert!(table.complete(&11, Ok("second")));
        assert!(table.complete(&10, Ok("first")));

        assert_eq!(Pending::new(rx_a).response().await.unwrap(), "first");
        assert_eq!(Pending::new(rx_b).response().await.unwrap(), "second");
    }

    #[tokio::test]
    async fn out_of_order_responses_land_on_the_right_caller() {
        let mut table = PendingRequests::<ReqId, u64>::new();
        let mut receivers = Vec::new();
        for id in 0..8u64 {
            let (tx, rx) = oneshot::channel::<Result<u64, RequestError>>();
            table.insert(id, tx);
            receivers.push((id, rx));
        }
        // Complete in reverse, then interleaved — the table must not care.
        for id in (0..8u64).rev() {
            assert!(table.complete(&id, Ok(id * 100)));
        }
        for (id, rx) in receivers {
            assert_eq!(Pending::new(rx).response().await.unwrap(), id * 100);
        }
    }

    #[tokio::test]
    async fn completing_an_unknown_key_reports_no_waiter() {
        let mut table = PendingRequests::<ReqId, &str>::new();
        assert!(
            !table.complete(&99, Ok("stray")),
            "an unsolicited response must be reported as uncorrelated so the \
             caller can fall back to the event stream"
        );
    }

    #[tokio::test]
    async fn outbound_failure_reaches_the_caller_as_an_error() {
        let mut table = PendingRequests::<ReqId, &str>::new();
        let (tx, rx) = channel();
        table.insert(4, tx);
        table.complete(
            &4,
            Err(RequestError::Outbound {
                peer: "peer-x".into(),
                error: "dial failed".into(),
            }),
        );
        let err = Pending::new(rx).response().await.unwrap_err();
        assert!(matches!(err, RequestError::Outbound { .. }), "got {err:?}");
        assert_eq!(table.len(), 0);
    }

    #[tokio::test]
    async fn cancelling_a_caller_releases_the_entry_on_sweep() {
        let mut table = PendingRequests::<ReqId, &str>::new();
        let (tx, rx) = channel();
        table.insert(7, tx);
        drop(rx); // caller walked away

        assert_eq!(table.len(), 1, "the entry survives until it is swept");
        assert_eq!(table.sweep_cancelled(), 1);
        assert_eq!(table.len(), 0, "cancellation must release retained state");
    }

    #[tokio::test]
    async fn a_crowded_table_sweeps_cancelled_entries_on_insert() {
        let mut table = PendingRequests::<ReqId, &str>::new();
        // Fill to the sweep threshold with callers that have all walked away.
        for id in 0..SWEEP_THRESHOLD as u64 {
            let (tx, rx) = channel();
            table.insert(id, tx);
            drop(rx);
        }
        assert_eq!(table.len(), SWEEP_THRESHOLD);

        // The next insert must not grow the table past the abandoned entries.
        let (tx, _rx_live) = channel();
        table.insert(9_999, tx);
        assert_eq!(table.len(), 1, "only the live caller should be retained");
    }

    #[tokio::test]
    async fn shutdown_fails_every_retained_request() {
        let mut table = PendingRequests::<ReqId, &str>::new();
        let mut receivers = Vec::new();
        for id in 0..5u64 {
            let (tx, rx) = channel();
            table.insert(id, tx);
            receivers.push(rx);
        }
        assert_eq!(table.fail_all(RequestError::RouterGone), 5);
        assert_eq!(table.len(), 0, "shutdown must release all retained state");
        for rx in receivers {
            assert_eq!(
                Pending::new(rx).response().await.unwrap_err(),
                RequestError::RouterGone
            );
        }
    }

    #[tokio::test]
    async fn dropping_the_table_completes_the_caller_rather_than_hanging() {
        // The swarm task dying without running any shutdown path. The caller
        // must still be woken — this is the "no path leaves a caller waiting
        // forever" guarantee at its weakest point.
        let (tx, rx) = channel();
        let mut table = PendingRequests::<ReqId, &str>::new();
        table.insert(1, tx);
        drop(table);
        assert_eq!(
            Pending::new(rx).response().await.unwrap_err(),
            RequestError::RouterGone
        );
    }

    #[tokio::test]
    async fn a_caller_deadline_expires_without_a_response() {
        let (tx, rx) = channel();
        let pending: Pending<&str> = Pending::new(rx);
        let err = pending
            .response_within(Duration::from_millis(20))
            .await
            .unwrap_err();
        assert_eq!(err, RequestError::Timeout(Duration::from_millis(20)));
        // The expired handle closed its side, so the swarm can now release it.
        assert!(
            tx.is_closed(),
            "an expired caller must be visible as cancelled"
        );
    }

    #[tokio::test]
    async fn a_response_beating_the_deadline_is_returned() {
        let (tx, rx) = channel();
        let pending: Pending<&str> = Pending::new(rx);
        tokio::spawn(async move {
            let _ = tx.send(Ok("in time"));
        });
        assert_eq!(
            pending
                .response_within(Duration::from_secs(5))
                .await
                .unwrap(),
            "in time"
        );
    }

    #[tokio::test]
    async fn a_request_that_was_never_sent_resolves_immediately() {
        let pending: Pending<&str> = Pending::failed(RequestError::NotSent);
        assert_eq!(pending.response().await.unwrap_err(), RequestError::NotSent);
    }

    #[tokio::test]
    async fn contains_tracks_retention() {
        let mut table = PendingRequests::<ReqId, &str>::new();
        let (tx, _rx) = channel();
        table.insert(3, tx);
        assert!(table.contains(&3));
        assert!(!table.contains(&4));
        table.complete(&3, Ok("done"));
        assert!(!table.contains(&3));
    }
}
