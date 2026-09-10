use std::io;
use std::path::PathBuf;

/// Crate-local error type for `omni-store` operations.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// Native Windows (and any other non-Unix target) is fail-closed for store
    /// publication: the confined-open, no-clobber link and directory-durability
    /// guarantees have no validated equivalent there. Tracked in Omni #119.
    #[error(
        "store publication is not supported on this platform: native Windows is \
         unsupported; run under WSL2 or another Unix target"
    )]
    UnsupportedPlatform,

    #[error("invalid store key: {0}")]
    InvalidKey(String),

    #[error("publication requires canonical CIDv1/raw/BLAKE3-256: {0}")]
    InvalidCid(String),

    #[error("publication may exist for {cid}, but durability is unconfirmed: {detail}")]
    DurabilityUnconfirmed { cid: String, detail: String },

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

    #[error("local shard file missing for cid {cid}: {}", path.display())]
    ShardFileMissing { cid: String, path: PathBuf },

    #[error("manifest shard {cid} has no SNIP V2 reference; cannot restore")]
    ShardLacksSnipRef { cid: String },

    #[error("{0}")]
    Other(String),
}

/// Convenience alias used throughout this crate.
pub type Result<T> = std::result::Result<T, StoreError>;
