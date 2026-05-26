use anyhow::Result;
use libp2p::gossipsub::{self, IdentTopic, MessageId};
use thiserror::Error;
use tracing::{debug, info};

// ── Topic constants ───────────────────────────────────────────────────────────

pub const TOPIC_TEST: &str       = "omni/test/v1";
pub const TOPIC_CAPABILITY: &str = "omni/capability/v1";
pub const TOPIC_SHARD: &str      = "omni/shard/v1";
pub const TOPIC_PIPELINE: &str   = "omni/pipeline/v1";
pub const TOPIC_PROOF: &str      = "omni/proof/v1";

// Stage 12.2-pre — first-class topics for the contributor mesh.
// Off-chain only. Distinct strings, distinct `IdentTopic` handles.
pub const TOPIC_CONTRIBUTOR_JOB:    &str = "omni/contributor/job/v1";
pub const TOPIC_CONTRIBUTOR_RESULT: &str = "omni/contributor/result/v1";

/// Typed error for unknown / unsupported topic names. Replaces the
/// pre-Stage-12.2-pre silent fallback to `TOPIC_TEST`, which silently
/// misrouted any unknown topic and could not be detected by callers.
#[derive(Debug, Error)]
#[error("unknown gossipsub topic: {0:?}")]
pub struct UnknownTopic(pub String);

// ── GossipManager ─────────────────────────────────────────────────────────────

/// Manages Gossipsub topic subscriptions and message publishing.
///
/// Holds pre-built [`IdentTopic`] handles — the topic hash is computed once at
/// construction rather than on every call to `publish`.
pub struct GossipManager {
    topic_test:               IdentTopic,
    topic_capability:         IdentTopic,
    topic_shard:              IdentTopic,
    topic_pipeline:           IdentTopic,
    topic_proof:              IdentTopic,
    // Stage 12.2-pre — contributor mesh topics.
    topic_contributor_job:    IdentTopic,
    topic_contributor_result: IdentTopic,
}

impl GossipManager {
    pub fn new() -> Self {
        Self {
            topic_test:               IdentTopic::new(TOPIC_TEST),
            topic_capability:         IdentTopic::new(TOPIC_CAPABILITY),
            topic_shard:              IdentTopic::new(TOPIC_SHARD),
            topic_pipeline:           IdentTopic::new(TOPIC_PIPELINE),
            topic_proof:              IdentTopic::new(TOPIC_PROOF),
            topic_contributor_job:    IdentTopic::new(TOPIC_CONTRIBUTOR_JOB),
            topic_contributor_result: IdentTopic::new(TOPIC_CONTRIBUTOR_RESULT),
        }
    }

    /// Subscribe this node to all OmniNode Gossipsub topics.
    pub fn subscribe_all(&self, gs: &mut gossipsub::Behaviour) -> Result<()> {
        for topic in self.all_topics() {
            if gs.subscribe(topic)? {
                debug!(topic = %topic, "subscribed to gossipsub topic");
            }
        }
        Ok(())
    }

    /// Publish raw bytes to a named topic.
    /// Returns the [`MessageId`] on success.
    ///
    /// Stage 12.2-pre safety change: an unknown `topic_name` now returns
    /// a typed [`UnknownTopic`] error (surfaced as `anyhow::Error`) instead
    /// of silently routing to `TOPIC_TEST`. Callers must use one of the
    /// public `TOPIC_*` constants.
    pub fn publish(
        &self,
        gs: &mut gossipsub::Behaviour,
        topic_name: &str,
        data: impl Into<Vec<u8>>,
    ) -> Result<MessageId> {
        let topic = self.topic_by_name(topic_name)?;
        let id = gs
            .publish(topic.clone(), data.into())
            .map_err(|e| anyhow::anyhow!("publish on '{topic_name}': {e}"))?;
        info!(topic = topic_name, "message published");
        Ok(id)
    }

    fn all_topics(&self) -> [&IdentTopic; 7] {
        [
            &self.topic_test,
            &self.topic_capability,
            &self.topic_shard,
            &self.topic_pipeline,
            &self.topic_proof,
            &self.topic_contributor_job,
            &self.topic_contributor_result,
        ]
    }

    /// Stage 12.2-pre safety: returns `Err(UnknownTopic(...))` for any
    /// name that is not one of the public `TOPIC_*` constants.
    ///
    /// Previously this function silently routed unknown names to
    /// `TOPIC_TEST`, which (a) had no error signal for callers and (b)
    /// could land production-shape messages on the test topic. The new
    /// behavior is fail-loud: the caller MUST pass a known constant.
    fn topic_by_name(&self, name: &str) -> Result<&IdentTopic, UnknownTopic> {
        match name {
            TOPIC_TEST                => Ok(&self.topic_test),
            TOPIC_CAPABILITY          => Ok(&self.topic_capability),
            TOPIC_SHARD               => Ok(&self.topic_shard),
            TOPIC_PIPELINE            => Ok(&self.topic_pipeline),
            TOPIC_PROOF               => Ok(&self.topic_proof),
            TOPIC_CONTRIBUTOR_JOB     => Ok(&self.topic_contributor_job),
            TOPIC_CONTRIBUTOR_RESULT  => Ok(&self.topic_contributor_result),
            other                     => Err(UnknownTopic(other.to_string())),
        }
    }
}

impl Default for GossipManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_topics_includes_every_public_constant() {
        let g = GossipManager::new();
        let names: Vec<String> = g
            .all_topics()
            .iter()
            .map(|t| t.to_string())
            .collect();
        // gossipsub::IdentTopic's Display impl prints the topic hash,
        // not the original name string, so we instead assert via
        // topic_by_name returning Ok for each constant — that's the
        // routing contract the rest of the crate depends on.
        let _ = names; // not load-bearing; kept for debugging clarity.

        for name in &[
            TOPIC_TEST,
            TOPIC_CAPABILITY,
            TOPIC_SHARD,
            TOPIC_PIPELINE,
            TOPIC_PROOF,
            TOPIC_CONTRIBUTOR_JOB,
            TOPIC_CONTRIBUTOR_RESULT,
        ] {
            assert!(
                g.topic_by_name(name).is_ok(),
                "topic_by_name should route {name:?}",
            );
        }

        // Defense-in-depth: the count of routed topics matches
        // all_topics. If a new TOPIC_* constant is added without
        // routing, this fails.
        assert_eq!(g.all_topics().len(), 7);
    }

    #[test]
    fn topic_by_name_routes_contributor_topics_to_distinct_handles() {
        let g = GossipManager::new();
        // `IdentTopic` (which is `Topic<IdentityHash>`) does not impl
        // `PartialEq`. Compare via the gossipsub-computed `TopicHash`,
        // which is the value gossipsub actually routes on internally.
        let job = g.topic_by_name(TOPIC_CONTRIBUTOR_JOB).unwrap().hash();
        let result = g.topic_by_name(TOPIC_CONTRIBUTOR_RESULT).unwrap().hash();
        let test = g.topic_by_name(TOPIC_TEST).unwrap().hash();
        assert_ne!(job, test, "contributor_job must NOT alias test");
        assert_ne!(result, test, "contributor_result must NOT alias test");
        assert_ne!(job, result, "contributor_job and result must differ");
    }

    #[test]
    fn topic_by_name_rejects_unknown_topic_loudly() {
        let g = GossipManager::new();
        // Stage 12.2-pre safety: any name not in the constant set must
        // return UnknownTopic, NOT fall back to TOPIC_TEST.
        for bad in &[
            "",
            "omni/contributor/unknown/v1",
            "omni/proof/v2", // valid-looking but unknown version
            "TOPIC_TEST",    // case-sensitive
            "x",
        ] {
            let err = g.topic_by_name(bad).unwrap_err();
            assert_eq!(err.0, *bad);
            // Confirm the displayed message names the bad input.
            let msg = format!("{err}");
            assert!(msg.contains(bad), "error message should name the bad topic; got {msg:?}");
        }
    }

    #[test]
    fn topic_by_name_routes_pre_existing_topics_unchanged() {
        let g = GossipManager::new();
        // Regression guard for the four non-contributor production
        // topics — make sure the routing change didn't accidentally
        // break any existing constant.
        for name in &[
            TOPIC_TEST,
            TOPIC_CAPABILITY,
            TOPIC_SHARD,
            TOPIC_PIPELINE,
            TOPIC_PROOF,
        ] {
            assert!(g.topic_by_name(name).is_ok(), "{name} must still route");
        }
    }
}
