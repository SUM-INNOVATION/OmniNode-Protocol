//! Shared tokio runtime singleton.
//!
//! Every async Rust method is called via `get_runtime().block_on(...)` to
//! provide synchronous Python wrappers. No pyo3-asyncio dependency needed.

use std::sync::OnceLock;
use tokio::runtime::Runtime;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Return (or lazily create) the global multi-threaded tokio runtime.
pub fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("failed to create tokio runtime")
    })
}
