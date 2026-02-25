// Phase 1 — minimal error type. Extended per phase.
#[derive(Debug, thiserror::Error)]
pub enum OmniError {
    #[error("network error: {0}")]
    Network(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("internal error: {0}")]
    Internal(String),
}
