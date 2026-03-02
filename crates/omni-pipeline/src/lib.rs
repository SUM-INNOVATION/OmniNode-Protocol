//! `omni-pipeline` — Distributed pipeline-parallel inference coordination.
//!
//! This crate is a **coordination layer**, not a compute layer. The actual
//! forward pass (matrix multiply, attention, etc.) executes in Python via
//! MLX/llama.cpp through `omni-bridge`. Rust orchestrates which node runs
//! which layers, manages GPipe micro-batch scheduling, and transports
//! hidden-state tensors between stages.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │   Stage 0   │────▶│   Stage 1   │────▶│   Stage 2   │
//! │ layers 0-10 │     │ layers 11-21│     │ layers 22-31│
//! │   Node A    │     │   Node B    │     │   Node C    │
//! └─────────────┘     └─────────────┘     └─────────────┘
//!       embed              ↑ tensor            lm_head
//!                          │ xfer
//! ```

pub mod coordinator;
pub mod error;
pub mod executor;
pub mod heartbeat;
pub mod planner;
pub mod scheduler;
pub mod session;
pub mod transport;

// ── Public re-exports ────────────────────────────────────────────────────────

pub use coordinator::{PipelineAction, PipelineCoordinator};
pub use error::{PipelineError, Result};
pub use executor::StageExecutor;
pub use heartbeat::HeartbeatMonitor;
pub use planner::plan_stages;
pub use scheduler::{MicroBatchSchedule, ScheduleCell};
pub use session::{PipelineSession, SessionState};
pub use transport::{decode_pipeline_message, encode_pipeline_message};
