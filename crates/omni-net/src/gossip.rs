use anyhow::Result;
use libp2p::gossipsub::{self, IdentTopic, MessageId};
use tracing::{debug, info};

// ── Topic constants ───────────────────────────────────────────────────────────

pub const TOPIC_TEST: &str       = "omni/test/v1";
pub const TOPIC_CAPABILITY: &str = "omni/capability/v1";
pub const TOPIC_SHARD: &str      = "omni/shard/v1";
pub const TOPIC_PIPELINE: &str   = "omni/pipeline/v1";
pub const TOPIC_PROOF: &str      = "omni/proof/v1";

// ── GossipManager ─────────────────────────────────────────────────────────────

/// Manages Gossipsub topic subscriptions and message publishing.
///
/// Holds pre-built [`IdentTopic`] handles — the topic hash is computed once at
/// construction rather than on every call to `publish`.
pub struct GossipManager {
    topic_test:       IdentTopic,
    topic_capability: IdentTopic,
    topic_shard:      IdentTopic,
    topic_pipeline:   IdentTopic,
    topic_proof:      IdentTopic,
}

impl GossipManager {
    pub fn new() -> Self {
        Self {
            topic_test:       IdentTopic::new(TOPIC_TEST),
            topic_capability: IdentTopic::new(TOPIC_CAPABILITY),
            topic_shard:      IdentTopic::new(TOPIC_SHARD),
            topic_pipeline:   IdentTopic::new(TOPIC_PIPELINE),
            topic_proof:      IdentTopic::new(TOPIC_PROOF),
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
    pub fn publish(
        &self,
        gs: &mut gossipsub::Behaviour,
        topic_name: &str,
        data: impl Into<Vec<u8>>,
    ) -> Result<MessageId> {
        let topic = self.topic_by_name(topic_name);
        let id = gs
            .publish(topic.clone(), data.into())
            .map_err(|e| anyhow::anyhow!("publish on '{topic_name}': {e}"))?;
        info!(topic = topic_name, "message published");
        Ok(id)
    }

    fn all_topics(&self) -> [&IdentTopic; 5] {
        [
            &self.topic_test,
            &self.topic_capability,
            &self.topic_shard,
            &self.topic_pipeline,
            &self.topic_proof,
        ]
    }

    fn topic_by_name(&self, name: &str) -> &IdentTopic {
        match name {
            TOPIC_CAPABILITY => &self.topic_capability,
            TOPIC_SHARD      => &self.topic_shard,
            TOPIC_PIPELINE   => &self.topic_pipeline,
            TOPIC_PROOF      => &self.topic_proof,
            _                => &self.topic_test,
        }
    }
}

impl Default for GossipManager {
    fn default() -> Self {
        Self::new()
    }
}
