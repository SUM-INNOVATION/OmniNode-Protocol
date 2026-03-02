#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("session error: {0}")]
    Session(String),

    #[error("planning error: {0}")]
    Planning(String),

    #[error("scheduling error: {0}")]
    Scheduling(String),

    #[error("execution error: {0}")]
    Execution(String),

    #[error("heartbeat timeout: stage {stage_index} on peer {peer_id}")]
    HeartbeatTimeout { stage_index: u32, peer_id: String },

    #[error("tensor transfer error: {0}")]
    TensorTransfer(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("invalid state transition: {from} → {to}")]
    InvalidTransition { from: String, to: String },
}

/// Convenience alias used throughout this crate.
pub type Result<T> = std::result::Result<T, PipelineError>;
