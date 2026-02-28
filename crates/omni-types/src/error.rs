// Global error type. Extended per phase.

#[derive(Debug, thiserror::Error)]
pub enum OmniError {
    // ── Phase 1: Networking ───────────────────────────────────────────────

    #[error("network error: {0}")]
    Network(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("internal error: {0}")]
    Internal(String),

    // ── Phase 2: Storage ──────────────────────────────────────────────────

    #[error("storage error: {0}")]
    Storage(String),

    #[error("GGUF parse error: {0}")]
    GgufParse(String),
}
