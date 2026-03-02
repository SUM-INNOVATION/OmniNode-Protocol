use std::collections::HashMap;
use std::fmt;

use omni_types::pipeline::{PipelineCapability, PipelineSchedule};

use crate::error::{PipelineError, Result};

// ── Session State Machine ────────────────────────────────────────────────────

/// Forming → Scheduled → Running → Completed | Failed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Coordinator is collecting capability offers from peers.
    Forming,
    /// Schedule has been assigned; waiting for all stages to report ready.
    Scheduled,
    /// Inference is in progress.
    Running,
    /// All micro-batches completed successfully.
    Completed,
    /// A stage failed or the session was aborted.
    Failed,
}

impl fmt::Display for SessionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forming   => write!(f, "Forming"),
            Self::Scheduled => write!(f, "Scheduled"),
            Self::Running   => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed    => write!(f, "Failed"),
        }
    }
}

// ── Pipeline Session ─────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PipelineSession {
    pub session_id: String,
    pub model_name: String,
    pub model_hash: String,
    pub total_layers: u32,
    pub state: SessionState,
    pub capability_offers: HashMap<String, PipelineCapability>,
    pub schedule: Option<PipelineSchedule>,
    pub ready_stages: Vec<u32>,
    pub failure_reason: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl PipelineSession {
    pub fn new(
        session_id: String,
        model_name: String,
        model_hash: String,
        total_layers: u32,
    ) -> Self {
        Self {
            session_id,
            model_name,
            model_hash,
            total_layers,
            state: SessionState::Forming,
            capability_offers: HashMap::new(),
            schedule: None,
            ready_stages: Vec::new(),
            failure_reason: None,
            created_at: chrono::Utc::now(),
        }
    }

    /// Register a capability offer from a peer. Only valid in `Forming` state.
    pub fn add_capability(&mut self, capability: PipelineCapability) -> Result<()> {
        if self.state != SessionState::Forming {
            return Err(PipelineError::InvalidTransition {
                from: self.state.to_string(),
                to: "accepting capabilities".into(),
            });
        }
        self.capability_offers
            .insert(capability.peer_id.clone(), capability);
        Ok(())
    }

    /// Assign a finalized schedule. Transitions `Forming → Scheduled`.
    pub fn set_schedule(&mut self, schedule: PipelineSchedule) -> Result<()> {
        if self.state != SessionState::Forming {
            return Err(PipelineError::InvalidTransition {
                from: self.state.to_string(),
                to: SessionState::Scheduled.to_string(),
            });
        }
        self.schedule = Some(schedule);
        self.state = SessionState::Scheduled;
        Ok(())
    }

    /// Mark a stage as ready. Returns `true` if all stages are now ready.
    pub fn mark_stage_ready(&mut self, stage_index: u32) -> Result<bool> {
        if self.state != SessionState::Scheduled {
            return Err(PipelineError::InvalidTransition {
                from: self.state.to_string(),
                to: "marking stage ready".into(),
            });
        }
        if !self.ready_stages.contains(&stage_index) {
            self.ready_stages.push(stage_index);
        }
        Ok(self.all_stages_ready())
    }

    /// Transition `Scheduled → Running`. Requires all stages to be ready.
    pub fn start(&mut self) -> Result<()> {
        if self.state != SessionState::Scheduled {
            return Err(PipelineError::InvalidTransition {
                from: self.state.to_string(),
                to: SessionState::Running.to_string(),
            });
        }
        if !self.all_stages_ready() {
            return Err(PipelineError::Session("not all stages ready".into()));
        }
        self.state = SessionState::Running;
        Ok(())
    }

    /// Mark session as completed.
    pub fn complete(&mut self) {
        self.state = SessionState::Completed;
    }

    /// Mark session as failed with a reason.
    pub fn fail(&mut self, reason: &str) {
        self.state = SessionState::Failed;
        self.failure_reason = Some(reason.to_string());
    }

    /// Check if all stages in the schedule have reported ready.
    pub fn all_stages_ready(&self) -> bool {
        match self.schedule {
            Some(ref s) => self.ready_stages.len() == s.stages.len(),
            None => false,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use omni_types::model::LayerRange;
    use omni_types::pipeline::PipelineStage;

    fn test_schedule() -> PipelineSchedule {
        PipelineSchedule {
            session_id: "test".into(),
            model_name: "llama-7b".into(),
            model_hash: "abc".into(),
            total_layers: 32,
            stages: vec![
                PipelineStage {
                    stage_index: 0,
                    peer_id: "peer-a".into(),
                    layer_range: LayerRange { start: 0, end: 15 },
                    includes_embedding: true,
                    includes_output_head: false,
                },
                PipelineStage {
                    stage_index: 1,
                    peer_id: "peer-b".into(),
                    layer_range: LayerRange { start: 16, end: 31 },
                    includes_embedding: false,
                    includes_output_head: true,
                },
            ],
            num_micro_batches: 4,
            max_seq_len: 2048,
            hidden_dim: 4096,
            created_at: "2025-01-01T00:00:00Z".into(),
        }
    }

    #[test]
    fn state_machine_happy_path() {
        let mut s = PipelineSession::new(
            "sess-1".into(), "llama-7b".into(), "abc".into(), 32,
        );
        assert_eq!(s.state, SessionState::Forming);

        s.set_schedule(test_schedule()).unwrap();
        assert_eq!(s.state, SessionState::Scheduled);

        assert!(!s.mark_stage_ready(0).unwrap());
        assert!(s.mark_stage_ready(1).unwrap());

        s.start().unwrap();
        assert_eq!(s.state, SessionState::Running);

        s.complete();
        assert_eq!(s.state, SessionState::Completed);
    }

    #[test]
    fn invalid_transition_rejects() {
        let mut s = PipelineSession::new(
            "sess-1".into(), "llama-7b".into(), "abc".into(), 32,
        );
        // Cannot mark stage ready before scheduling
        assert!(s.mark_stage_ready(0).is_err());
        // Cannot start before scheduling
        assert!(s.start().is_err());
    }

    #[test]
    fn fail_sets_reason() {
        let mut s = PipelineSession::new(
            "sess-1".into(), "llama-7b".into(), "abc".into(), 32,
        );
        s.fail("node crashed");
        assert_eq!(s.state, SessionState::Failed);
        assert_eq!(s.failure_reason.as_deref(), Some("node crashed"));
    }
}
