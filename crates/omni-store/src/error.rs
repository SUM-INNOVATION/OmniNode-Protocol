use std::io;

/// Crate-local error type for `omni-store` operations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("GGUF parse error: {0}")]
    GgufParse(String),

    #[error("integrity check failed: expected {expected}, got {actual}")]
    IntegrityMismatch { expected: String, actual: String },

    #[error("shard not found: {0}")]
    NotFound(String),

    #[error("SNIP V2 error: {0}")]
    SnipV2(#[from] crate::snip_v2::SnipV2Error),

    #[error("{0}")]
    Other(String),
}

/// Convenience alias used throughout this crate.
pub type Result<T> = std::result::Result<T, StoreError>;
