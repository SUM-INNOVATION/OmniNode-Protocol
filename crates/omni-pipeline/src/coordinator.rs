//! Pipeline session coordinator.
//!
//! Manages the lifecycle of pipeline sessions:
//! 1. Propose → broadcast on gossipsub
//! 2. Collect CapabilityOffers from peers
//! 3. Run planner to produce a schedule
//! 4. Broadcast ScheduleAssigned
//! 5. Collect StageReady from all stages
//! 6. Broadcast StartInference
//!
//! The coordinator does NOT perform network I/O directly. It produces
//! [`PipelineAction`] values that the caller (omni-node) executes via OmniNet.

use std::collections::HashMap;

use tracing::{info, warn};
use uuid::Uuid;

use omni_types::config::PipelineConfig;
use omni_types::pipeline::{PipelineCapability, PipelineMessage, PipelineSchedule};

use crate::error::{PipelineError, Result};
use crate::planner;
use crate::session::{PipelineSession, SessionState};
use crate::transport;

// ── Actions ──────────────────────────────────────────────────────────────────

/// Actions produced by the coordinator for the caller to execute via OmniNet.
#[derive(Debug)]
pub enum PipelineAction {
    /// Publish bytes on the `omni/pipeline/v1` gossipsub topic.
    PublishMessage { data: Vec<u8> },
}

// ── Coordinator ──────────────────────────────────────────────────────────────

pub struct PipelineCoordinator {
    config: PipelineConfig,
    sessions: HashMap<String, PipelineSession>,
}

impl PipelineCoordinator {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            sessions: HashMap::new(),
        }
    }

    /// Create a new pipeline session and produce a `Propose` broadcast.
    ///
    /// Returns `(session_id, action_to_publish)`.
    pub fn propose_session(
        &mut self,
        model_name: String,
        model_hash: String,
        total_layers: u32,
        local_peer_id: String,
    ) -> Result<(String, PipelineAction)> {
        let session_id = Uuid::new_v4().to_string();

        let session = PipelineSession::new(
            session_id.clone(),
            model_name.clone(),
            model_hash.clone(),
            total_layers,
        );
        self.sessions.insert(session_id.clone(), session);

        let msg = PipelineMessage::Propose {
            session_id: session_id.clone(),
            model_name,
            model_hash,
            total_layers,
            proposer_peer_id: local_peer_id,
        };
        let data = transport::encode_pipeline_message(&msg)?;

        info!(session_id = %session_id, "proposed pipeline session");

        Ok((session_id, PipelineAction::PublishMessage { data }))
    }

    /// Register a capability offer from a peer.
    pub fn handle_capability_offer(
        &mut self,
        session_id: &str,
        capability: PipelineCapability,
    ) -> Result<()> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| PipelineError::Session(format!("unknown session: {session_id}")))?;

        info!(
            session_id = %session_id,
            peer_id = %capability.peer_id,
            ram_mb = capability.available_ram_bytes / (1024 * 1024),
            "received capability offer"
        );

        session.add_capability(capability)
    }

    /// Run the planner on collected offers and produce a `ScheduleAssigned`
    /// broadcast.
    ///
    /// Returns the finalized schedule and the action to publish.
    pub fn finalize_schedule(
        &mut self,
        session_id: &str,
        hidden_dim: u32,
    ) -> Result<(PipelineSchedule, PipelineAction)> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| PipelineError::Session(format!("unknown session: {session_id}")))?;

        let capabilities: Vec<PipelineCapability> =
            session.capability_offers.values().cloned().collect();

        let schedule = planner::plan_stages(
            &capabilities,
            session.total_layers,
            &session.model_name,
            &session.model_hash,
            session_id,
            self.config.num_micro_batches,
            self.config.max_seq_len,
            hidden_dim,
        )?;

        session.set_schedule(schedule.clone())?;

        let msg = PipelineMessage::ScheduleAssigned {
            schedule: schedule.clone(),
        };
        let data = transport::encode_pipeline_message(&msg)?;

        info!(
            session_id = %session_id,
            num_stages = schedule.stages.len(),
            micro_batches = schedule.num_micro_batches,
            "schedule finalized and broadcast"
        );

        Ok((schedule, PipelineAction::PublishMessage { data }))
    }

    /// Handle a `StageReady` report. If all stages are ready, produces a
    /// `StartInference` broadcast.
    pub fn handle_stage_ready(
        &mut self,
        session_id: &str,
        stage_index: u32,
    ) -> Result<Option<PipelineAction>> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| PipelineError::Session(format!("unknown session: {session_id}")))?;

        let all_ready = session.mark_stage_ready(stage_index)?;

        if all_ready {
            session.start()?;

            let msg = PipelineMessage::StartInference {
                session_id: session_id.to_string(),
            };
            let data = transport::encode_pipeline_message(&msg)?;

            info!(session_id = %session_id, "all stages ready — starting inference");

            Ok(Some(PipelineAction::PublishMessage { data }))
        } else {
            Ok(None)
        }
    }

    /// Mark a session as completed and produce a `SessionComplete` broadcast.
    pub fn handle_session_complete(
        &mut self,
        session_id: &str,
        total_tokens: u64,
    ) -> Result<PipelineAction> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| PipelineError::Session(format!("unknown session: {session_id}")))?;

        session.complete();

        let msg = PipelineMessage::SessionComplete {
            session_id: session_id.to_string(),
            total_tokens_generated: total_tokens,
        };
        let data = transport::encode_pipeline_message(&msg)?;

        info!(session_id = %session_id, total_tokens, "session completed");

        Ok(PipelineAction::PublishMessage { data })
    }

    /// Handle a stage failure: abort the session and broadcast.
    pub fn handle_stage_failure(
        &mut self,
        session_id: &str,
        peer_id: &str,
        stage_index: u32,
        error: &str,
    ) -> Result<PipelineAction> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| PipelineError::Session(format!("unknown session: {session_id}")))?;

        let reason = format!("stage {stage_index} on {peer_id}: {error}");
        session.fail(&reason);

        let msg = PipelineMessage::SessionAborted {
            session_id: session_id.to_string(),
            reason: reason.clone(),
        };
        let data = transport::encode_pipeline_message(&msg)?;

        warn!(session_id = %session_id, %reason, "session aborted");

        Ok(PipelineAction::PublishMessage { data })
    }

    pub fn session(&self, session_id: &str) -> Option<&PipelineSession> {
        self.sessions.get(session_id)
    }

    pub fn session_mut(&mut self, session_id: &str) -> Option<&mut PipelineSession> {
        self.sessions.get_mut(session_id)
    }

    pub fn remove_session(&mut self, session_id: &str) -> Option<PipelineSession> {
        self.sessions.remove(session_id)
    }

    pub fn active_session_ids(&self) -> Vec<&str> {
        self.sessions
            .iter()
            .filter(|(_, s)| matches!(s.state, SessionState::Forming | SessionState::Scheduled | SessionState::Running))
            .map(|(id, _)| id.as_str())
            .collect()
    }
}
