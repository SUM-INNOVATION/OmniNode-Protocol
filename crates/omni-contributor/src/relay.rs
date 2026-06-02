//! Stage 12.2 — contributor mesh relay abstraction + implementations.
//!
//! The `ContributorRelay` trait is the sync, transport-agnostic
//! interface the watch loop uses to publish + poll network
//! announcements. Two concrete impls:
//!
//!   - [`InMemoryRelay`] — vec-backed in-process queues for tests.
//!     Fully synchronous; no tokio runtime required.
//!
//!   - [`OmniNetRelay`] — production adapter wrapping
//!     `omni_net::OmniNet`. Uses Stage 12.2-pre's
//!     `try_next_event` for non-blocking event drain and
//!     `block_in_place + Handle::block_on` to bridge the
//!     async `OmniNet::publish` from the sync watch loop.
//!
//! Dedup-by-`posted_id` lives in the watch loop (mirrors 12.1's
//! filesystem dedup); the relay returns whatever the transport has
//! accumulated since the last poll.

use std::collections::VecDeque;

use crate::error::RelayError;
use crate::net::{
    NetworkAggregatedResultAnnouncement, NetworkContributorJoinedAnnouncement,
    NetworkPartialResultAnnouncement, NetworkPeerAdvertisementAnnouncement,
    NetworkPostedJobAnnouncement, NetworkPostedResultAnnouncement,
    NetworkSessionOpenedAnnouncement, NetworkWorkAssignedAnnouncement,
};

/// Minimal sync interface a contributor watch / announce loop uses.
/// Production calls go through `OmniNetRelay`; tests use
/// `InMemoryRelay`.
pub trait ContributorRelay {
    /// Broadcast a job announcement. Returns once the announcement
    /// has been handed to the transport (does NOT wait for peer
    /// receipt — gossipsub is best-effort).
    fn publish_job(
        &mut self,
        msg: &NetworkPostedJobAnnouncement,
    ) -> Result<(), RelayError>;

    /// Broadcast a result announcement.
    fn publish_result(
        &mut self,
        msg: &NetworkPostedResultAnnouncement,
    ) -> Result<(), RelayError>;

    /// Drain all job announcements accumulated since the last poll.
    /// Non-blocking; returns empty if nothing is pending.
    fn poll_jobs(
        &mut self,
    ) -> Result<Vec<NetworkPostedJobAnnouncement>, RelayError>;

    /// Drain all result announcements accumulated since the last
    /// poll. Non-blocking.
    fn poll_results(
        &mut self,
    ) -> Result<Vec<NetworkPostedResultAnnouncement>, RelayError>;

    // ── Stage 12.3 — session network surface ────────────────────────

    fn publish_session_opened(
        &mut self,
        msg: &NetworkSessionOpenedAnnouncement,
    ) -> Result<(), RelayError>;

    fn publish_contributor_joined(
        &mut self,
        msg: &NetworkContributorJoinedAnnouncement,
    ) -> Result<(), RelayError>;

    fn publish_work_assigned(
        &mut self,
        msg: &NetworkWorkAssignedAnnouncement,
    ) -> Result<(), RelayError>;

    fn publish_partial_result(
        &mut self,
        msg: &NetworkPartialResultAnnouncement,
    ) -> Result<(), RelayError>;

    fn publish_aggregated_result(
        &mut self,
        msg: &NetworkAggregatedResultAnnouncement,
    ) -> Result<(), RelayError>;

    fn poll_sessions_opened(
        &mut self,
    ) -> Result<Vec<NetworkSessionOpenedAnnouncement>, RelayError>;

    fn poll_contributors_joined(
        &mut self,
    ) -> Result<Vec<NetworkContributorJoinedAnnouncement>, RelayError>;

    fn poll_work_assigned(
        &mut self,
    ) -> Result<Vec<NetworkWorkAssignedAnnouncement>, RelayError>;

    fn poll_partial_results(
        &mut self,
    ) -> Result<Vec<NetworkPartialResultAnnouncement>, RelayError>;

    fn poll_aggregated_results(
        &mut self,
    ) -> Result<Vec<NetworkAggregatedResultAnnouncement>, RelayError>;

    // ── Stage 12.5 — peer advertisement surface ──────────────────

    fn publish_peer_advertisement(
        &mut self,
        msg: &NetworkPeerAdvertisementAnnouncement,
    ) -> Result<(), RelayError>;

    fn poll_peer_advertisements(
        &mut self,
    ) -> Result<Vec<NetworkPeerAdvertisementAnnouncement>, RelayError>;

    // ── Stage 12.11 — assignment supersession surface ────────────

    fn publish_assignment_supersession(
        &mut self,
        msg: &crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
    ) -> Result<(), RelayError>;

    fn poll_assignment_supersessions(
        &mut self,
    ) -> Result<
        Vec<crate::net::NetworkWorkAssignmentSupersessionAnnouncement>,
        RelayError,
    >;
}

// ── InMemoryRelay ─────────────────────────────────────────────────────────

/// Test-only relay. Two FIFO queues, no transport. `publish_*` push;
/// `poll_*` drain.
///
/// For end-to-end tests where one process plays both publisher and
/// subscriber, a single `InMemoryRelay` carries the announcements
/// between them.
pub struct InMemoryRelay {
    jobs: VecDeque<NetworkPostedJobAnnouncement>,
    results: VecDeque<NetworkPostedResultAnnouncement>,
    sessions_opened: VecDeque<NetworkSessionOpenedAnnouncement>,
    contributors_joined: VecDeque<NetworkContributorJoinedAnnouncement>,
    work_assigned: VecDeque<NetworkWorkAssignedAnnouncement>,
    partial_results: VecDeque<NetworkPartialResultAnnouncement>,
    aggregated_results: VecDeque<NetworkAggregatedResultAnnouncement>,
    peer_adverts: VecDeque<NetworkPeerAdvertisementAnnouncement>,
    assignment_supersessions:
        VecDeque<crate::net::NetworkWorkAssignmentSupersessionAnnouncement>,
}

impl InMemoryRelay {
    pub fn new() -> Self {
        Self {
            jobs: VecDeque::new(),
            results: VecDeque::new(),
            sessions_opened: VecDeque::new(),
            contributors_joined: VecDeque::new(),
            work_assigned: VecDeque::new(),
            partial_results: VecDeque::new(),
            aggregated_results: VecDeque::new(),
            peer_adverts: VecDeque::new(),
            assignment_supersessions: VecDeque::new(),
        }
    }

    /// Number of currently pending job announcements (queue length).
    /// Used by tests to assert publish/drain semantics.
    pub fn pending_jobs(&self) -> usize {
        self.jobs.len()
    }

    /// Number of currently pending result announcements.
    pub fn pending_results(&self) -> usize {
        self.results.len()
    }
}

impl Default for InMemoryRelay {
    fn default() -> Self {
        Self::new()
    }
}

impl ContributorRelay for InMemoryRelay {
    fn publish_job(
        &mut self,
        msg: &NetworkPostedJobAnnouncement,
    ) -> Result<(), RelayError> {
        self.jobs.push_back(msg.clone());
        Ok(())
    }

    fn publish_result(
        &mut self,
        msg: &NetworkPostedResultAnnouncement,
    ) -> Result<(), RelayError> {
        self.results.push_back(msg.clone());
        Ok(())
    }

    fn poll_jobs(
        &mut self,
    ) -> Result<Vec<NetworkPostedJobAnnouncement>, RelayError> {
        Ok(self.jobs.drain(..).collect())
    }

    fn poll_results(
        &mut self,
    ) -> Result<Vec<NetworkPostedResultAnnouncement>, RelayError> {
        Ok(self.results.drain(..).collect())
    }

    fn publish_session_opened(
        &mut self,
        msg: &NetworkSessionOpenedAnnouncement,
    ) -> Result<(), RelayError> {
        self.sessions_opened.push_back(msg.clone());
        Ok(())
    }

    fn publish_contributor_joined(
        &mut self,
        msg: &NetworkContributorJoinedAnnouncement,
    ) -> Result<(), RelayError> {
        self.contributors_joined.push_back(msg.clone());
        Ok(())
    }

    fn publish_work_assigned(
        &mut self,
        msg: &NetworkWorkAssignedAnnouncement,
    ) -> Result<(), RelayError> {
        self.work_assigned.push_back(msg.clone());
        Ok(())
    }

    fn publish_partial_result(
        &mut self,
        msg: &NetworkPartialResultAnnouncement,
    ) -> Result<(), RelayError> {
        self.partial_results.push_back(msg.clone());
        Ok(())
    }

    fn publish_aggregated_result(
        &mut self,
        msg: &NetworkAggregatedResultAnnouncement,
    ) -> Result<(), RelayError> {
        self.aggregated_results.push_back(msg.clone());
        Ok(())
    }

    fn poll_sessions_opened(
        &mut self,
    ) -> Result<Vec<NetworkSessionOpenedAnnouncement>, RelayError> {
        Ok(self.sessions_opened.drain(..).collect())
    }

    fn poll_contributors_joined(
        &mut self,
    ) -> Result<Vec<NetworkContributorJoinedAnnouncement>, RelayError> {
        Ok(self.contributors_joined.drain(..).collect())
    }

    fn poll_work_assigned(
        &mut self,
    ) -> Result<Vec<NetworkWorkAssignedAnnouncement>, RelayError> {
        Ok(self.work_assigned.drain(..).collect())
    }

    fn poll_partial_results(
        &mut self,
    ) -> Result<Vec<NetworkPartialResultAnnouncement>, RelayError> {
        Ok(self.partial_results.drain(..).collect())
    }

    fn poll_aggregated_results(
        &mut self,
    ) -> Result<Vec<NetworkAggregatedResultAnnouncement>, RelayError> {
        Ok(self.aggregated_results.drain(..).collect())
    }

    fn publish_peer_advertisement(
        &mut self,
        msg: &NetworkPeerAdvertisementAnnouncement,
    ) -> Result<(), RelayError> {
        self.peer_adverts.push_back(msg.clone());
        Ok(())
    }

    fn poll_peer_advertisements(
        &mut self,
    ) -> Result<Vec<NetworkPeerAdvertisementAnnouncement>, RelayError> {
        Ok(self.peer_adverts.drain(..).collect())
    }

    fn publish_assignment_supersession(
        &mut self,
        msg: &crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
    ) -> Result<(), RelayError> {
        self.assignment_supersessions.push_back(msg.clone());
        Ok(())
    }

    fn poll_assignment_supersessions(
        &mut self,
    ) -> Result<
        Vec<crate::net::NetworkWorkAssignmentSupersessionAnnouncement>,
        RelayError,
    > {
        Ok(self.assignment_supersessions.drain(..).collect())
    }
}

// ── OmniNetRelay ──────────────────────────────────────────────────────────

#[cfg(feature = "network")]
pub use omni_net_relay::OmniNetRelay;

#[cfg(feature = "network")]
mod omni_net_relay {
    use super::*;

    use std::sync::{Arc, Mutex as StdMutex};

    use omni_net::{
        OmniNet, OmniNetEvent, TOPIC_CONTRIBUTOR_JOB, TOPIC_CONTRIBUTOR_RESULT,
        TOPIC_CONTRIBUTOR_SESSION_AGGREGATED, TOPIC_CONTRIBUTOR_SESSION_ASSIGN,
        TOPIC_CONTRIBUTOR_SESSION_ASSIGNMENT_SUPERSESSION,
        TOPIC_CONTRIBUTOR_SESSION_JOIN, TOPIC_CONTRIBUTOR_SESSION_OPEN,
        TOPIC_CONTRIBUTOR_SESSION_PARTIAL, TOPIC_CONTRIBUTOR_SESSION_PEER_ADVERT,
    };
    use tokio::runtime::Handle;
    use tokio::sync::Mutex as AsyncMutex;

    /// Production relay over `omni_net::OmniNet`. Holds an
    /// `Arc<tokio::sync::Mutex<OmniNet>>` (so multiple consumers —
    /// e.g. a watch loop's `NetworkSource` AND a result broadcaster
    /// running off the same loop's events — can share the
    /// underlying mesh connection, and `OmniNet::shutdown().await`
    /// can be held across an await from the async CLI handlers) and
    /// a `tokio::runtime::Handle` to bridge the async
    /// `OmniNet::publish` from sync callers.
    ///
    /// `OmniNetRelay` is `Clone`-able; clones share `pending_jobs` /
    /// `pending_results` queues (via `Arc<std::sync::Mutex<_>>`) so
    /// the caller can drain events through one clone while
    /// publishing through another without dropping messages.
    ///
    /// Event-drain strategy: the OmniNet event stream is shared
    /// across all gossipsub topics (and other event kinds like
    /// PeerConnected etc.). On every `drain_events()` call we pull
    /// from `try_next_event` and route to the per-topic pending
    /// queue. `poll_jobs` and `poll_results` each drain first, so
    /// callers don't lose results-topic messages when polling jobs.
    #[derive(Clone)]
    pub struct OmniNetRelay {
        net: Arc<AsyncMutex<OmniNet>>,
        handle: Handle,
        pending_jobs: Arc<StdMutex<VecDeque<NetworkPostedJobAnnouncement>>>,
        pending_results: Arc<StdMutex<VecDeque<NetworkPostedResultAnnouncement>>>,
        // Stage 12.3 session-topic queues.
        pending_sessions_opened:
            Arc<StdMutex<VecDeque<NetworkSessionOpenedAnnouncement>>>,
        pending_contributors_joined:
            Arc<StdMutex<VecDeque<NetworkContributorJoinedAnnouncement>>>,
        pending_work_assigned:
            Arc<StdMutex<VecDeque<NetworkWorkAssignedAnnouncement>>>,
        pending_partial_results:
            Arc<StdMutex<VecDeque<NetworkPartialResultAnnouncement>>>,
        pending_aggregated_results:
            Arc<StdMutex<VecDeque<NetworkAggregatedResultAnnouncement>>>,
        // Stage 12.5 — peer-advert queue.
        pending_peer_adverts:
            Arc<StdMutex<VecDeque<NetworkPeerAdvertisementAnnouncement>>>,
        // Stage 12.11 — assignment-supersession queue.
        pending_assignment_supersessions: Arc<
            StdMutex<
                VecDeque<
                    crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
                >,
            >,
        >,
    }

    impl OmniNetRelay {
        /// Construct from a shared `OmniNet` handle + the current
        /// tokio runtime handle. The caller (`omni-node` CLI) wraps
        /// its `OmniNet::new` instance in
        /// `Arc<tokio::sync::Mutex<_>>` from its async main and
        /// passes a clone here.
        pub fn new(net: Arc<AsyncMutex<OmniNet>>, handle: Handle) -> Self {
            Self {
                net,
                handle,
                pending_jobs: Arc::new(StdMutex::new(VecDeque::new())),
                pending_results: Arc::new(StdMutex::new(VecDeque::new())),
                pending_sessions_opened: Arc::new(StdMutex::new(VecDeque::new())),
                pending_contributors_joined: Arc::new(StdMutex::new(VecDeque::new())),
                pending_work_assigned: Arc::new(StdMutex::new(VecDeque::new())),
                pending_partial_results: Arc::new(StdMutex::new(VecDeque::new())),
                pending_aggregated_results: Arc::new(StdMutex::new(VecDeque::new())),
                pending_peer_adverts: Arc::new(StdMutex::new(VecDeque::new())),
                pending_assignment_supersessions: Arc::new(StdMutex::new(VecDeque::new())),
            }
        }

        /// Drain the OmniNet event stream into per-topic buffers.
        /// Routes `MessageReceived` events by their topic string and
        /// silently ignores everything else (peer events, shard
        /// events, etc.). Malformed JSON on a contributor topic is
        /// also silently dropped — the higher-level watch loop's
        /// announcer-signature check is the load-bearing filter.
        fn drain_events(&self) {
            // Acquire the OmniNet lock via block_in_place + block_on
            // so the sync watch loop can call us without blocking
            // the runtime.
            let mut net = tokio::task::block_in_place(|| {
                self.handle.block_on(self.net.lock())
            });
            let mut jobs = self.pending_jobs.lock().expect("pending_jobs poisoned");
            let mut results = self.pending_results.lock().expect("pending_results poisoned");
            let mut s_open = self
                .pending_sessions_opened
                .lock()
                .expect("pending_sessions_opened poisoned");
            let mut s_join = self
                .pending_contributors_joined
                .lock()
                .expect("pending_contributors_joined poisoned");
            let mut s_assign = self
                .pending_work_assigned
                .lock()
                .expect("pending_work_assigned poisoned");
            let mut s_partial = self
                .pending_partial_results
                .lock()
                .expect("pending_partial_results poisoned");
            let mut s_agg = self
                .pending_aggregated_results
                .lock()
                .expect("pending_aggregated_results poisoned");
            let mut s_peer = self
                .pending_peer_adverts
                .lock()
                .expect("pending_peer_adverts poisoned");
            let mut s_super = self
                .pending_assignment_supersessions
                .lock()
                .expect("pending_assignment_supersessions poisoned");
            while let Some(ev) = net.try_next_event() {
                if let OmniNetEvent::MessageReceived { topic, data, .. } = ev {
                    match topic.as_str() {
                        TOPIC_CONTRIBUTOR_JOB => {
                            if let Ok(msg) =
                                serde_json::from_slice::<NetworkPostedJobAnnouncement>(&data)
                            {
                                jobs.push_back(msg);
                            }
                            // Malformed JSON on the contributor topic
                            // is dropped silently — it's spam.
                        }
                        TOPIC_CONTRIBUTOR_RESULT => {
                            if let Ok(msg) = serde_json::from_slice::<
                                NetworkPostedResultAnnouncement,
                            >(&data)
                            {
                                results.push_back(msg);
                            }
                        }
                        TOPIC_CONTRIBUTOR_SESSION_OPEN => {
                            if let Ok(msg) = serde_json::from_slice::<
                                NetworkSessionOpenedAnnouncement,
                            >(&data)
                            {
                                s_open.push_back(msg);
                            }
                        }
                        TOPIC_CONTRIBUTOR_SESSION_JOIN => {
                            if let Ok(msg) = serde_json::from_slice::<
                                NetworkContributorJoinedAnnouncement,
                            >(&data)
                            {
                                s_join.push_back(msg);
                            }
                        }
                        TOPIC_CONTRIBUTOR_SESSION_ASSIGN => {
                            if let Ok(msg) = serde_json::from_slice::<
                                NetworkWorkAssignedAnnouncement,
                            >(&data)
                            {
                                s_assign.push_back(msg);
                            }
                        }
                        TOPIC_CONTRIBUTOR_SESSION_PARTIAL => {
                            if let Ok(msg) = serde_json::from_slice::<
                                NetworkPartialResultAnnouncement,
                            >(&data)
                            {
                                s_partial.push_back(msg);
                            }
                        }
                        TOPIC_CONTRIBUTOR_SESSION_AGGREGATED => {
                            if let Ok(msg) = serde_json::from_slice::<
                                NetworkAggregatedResultAnnouncement,
                            >(&data)
                            {
                                s_agg.push_back(msg);
                            }
                        }
                        TOPIC_CONTRIBUTOR_SESSION_PEER_ADVERT => {
                            if let Ok(msg) = serde_json::from_slice::<
                                NetworkPeerAdvertisementAnnouncement,
                            >(&data)
                            {
                                s_peer.push_back(msg);
                            }
                        }
                        TOPIC_CONTRIBUTOR_SESSION_ASSIGNMENT_SUPERSESSION => {
                            if let Ok(msg) = serde_json::from_slice::<
                                crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
                            >(&data)
                            {
                                s_super.push_back(msg);
                            }
                        }
                        _ => {
                            // Other topics: not our concern.
                        }
                    }
                }
            }
        }

        fn publish_topic(
            &self,
            topic: &'static str,
            bytes: Vec<u8>,
        ) -> Result<(), RelayError> {
            // OmniNet::publish takes `&self`; acquire the async lock
            // and call publish inside a single block_on so we don't
            // hold the std-side guards across awaits.
            let result = tokio::task::block_in_place(|| {
                self.handle.block_on(async {
                    let g = self.net.lock().await;
                    g.publish(topic, bytes).await
                })
            });
            result.map_err(|e| RelayError::Publish(e.to_string()))?;
            Ok(())
        }
    }

    impl ContributorRelay for OmniNetRelay {
        fn publish_job(
            &mut self,
            msg: &NetworkPostedJobAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_JOB, bytes)
        }

        fn publish_result(
            &mut self,
            msg: &NetworkPostedResultAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_RESULT, bytes)
        }

        fn poll_jobs(
            &mut self,
        ) -> Result<Vec<NetworkPostedJobAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self.pending_jobs.lock().expect("pending_jobs poisoned");
            Ok(q.drain(..).collect())
        }

        fn poll_results(
            &mut self,
        ) -> Result<Vec<NetworkPostedResultAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self.pending_results.lock().expect("pending_results poisoned");
            Ok(q.drain(..).collect())
        }

        fn publish_session_opened(
            &mut self,
            msg: &NetworkSessionOpenedAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_SESSION_OPEN, bytes)
        }

        fn publish_contributor_joined(
            &mut self,
            msg: &NetworkContributorJoinedAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_SESSION_JOIN, bytes)
        }

        fn publish_work_assigned(
            &mut self,
            msg: &NetworkWorkAssignedAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_SESSION_ASSIGN, bytes)
        }

        fn publish_partial_result(
            &mut self,
            msg: &NetworkPartialResultAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_SESSION_PARTIAL, bytes)
        }

        fn publish_aggregated_result(
            &mut self,
            msg: &NetworkAggregatedResultAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_SESSION_AGGREGATED, bytes)
        }

        fn poll_sessions_opened(
            &mut self,
        ) -> Result<Vec<NetworkSessionOpenedAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self
                .pending_sessions_opened
                .lock()
                .expect("pending_sessions_opened poisoned");
            Ok(q.drain(..).collect())
        }

        fn poll_contributors_joined(
            &mut self,
        ) -> Result<Vec<NetworkContributorJoinedAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self
                .pending_contributors_joined
                .lock()
                .expect("pending_contributors_joined poisoned");
            Ok(q.drain(..).collect())
        }

        fn poll_work_assigned(
            &mut self,
        ) -> Result<Vec<NetworkWorkAssignedAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self
                .pending_work_assigned
                .lock()
                .expect("pending_work_assigned poisoned");
            Ok(q.drain(..).collect())
        }

        fn poll_partial_results(
            &mut self,
        ) -> Result<Vec<NetworkPartialResultAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self
                .pending_partial_results
                .lock()
                .expect("pending_partial_results poisoned");
            Ok(q.drain(..).collect())
        }

        fn poll_aggregated_results(
            &mut self,
        ) -> Result<Vec<NetworkAggregatedResultAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self
                .pending_aggregated_results
                .lock()
                .expect("pending_aggregated_results poisoned");
            Ok(q.drain(..).collect())
        }

        fn publish_peer_advertisement(
            &mut self,
            msg: &NetworkPeerAdvertisementAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_SESSION_PEER_ADVERT, bytes)
        }

        fn poll_peer_advertisements(
            &mut self,
        ) -> Result<Vec<NetworkPeerAdvertisementAnnouncement>, RelayError> {
            self.drain_events();
            let mut q = self
                .pending_peer_adverts
                .lock()
                .expect("pending_peer_adverts poisoned");
            Ok(q.drain(..).collect())
        }

        fn publish_assignment_supersession(
            &mut self,
            msg: &crate::net::NetworkWorkAssignmentSupersessionAnnouncement,
        ) -> Result<(), RelayError> {
            let bytes = serde_json::to_vec(msg)?;
            self.publish_topic(TOPIC_CONTRIBUTOR_SESSION_ASSIGNMENT_SUPERSESSION, bytes)
        }

        fn poll_assignment_supersessions(
            &mut self,
        ) -> Result<
            Vec<crate::net::NetworkWorkAssignmentSupersessionAnnouncement>,
            RelayError,
        > {
            self.drain_events();
            let mut q = self
                .pending_assignment_supersessions
                .lock()
                .expect("pending_assignment_supersessions poisoned");
            Ok(q.drain(..).collect())
        }
    }
}
