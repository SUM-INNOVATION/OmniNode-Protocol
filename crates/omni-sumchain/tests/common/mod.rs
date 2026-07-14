//! Shared support for `omni-sumchain` integration tests.
//!
//! Pulled in via `mod common;` from individual test binaries. Each file
//! directly under `tests/` compiles as its own crate, so a subdirectory
//! module is the standard way to share helpers between them without
//! publishing anything from `src/`.

/// Chain ID of the shared OmniNode ecosystem devnet / local mirror.
///
/// Test-support only. No runtime code consumes a devnet default: the
/// operator's `--expect-chain-id` is a required flag with no runtime
/// fallback, so this lives here rather than in `src/` to keep the
/// production surface free of a hard-coded devnet id.
pub const DEVNET_CHAIN_ID: u64 = 1337;
