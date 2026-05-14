//! `omni-sumchain` — Phase 5 Stage 7a + 7b SUM Chain adapter.
//!
//! Implements the [`omni_zkml::ChainClient`] trait against SUM Chain's
//! JSON-RPC surface. Stage 7a shipped the read/query surface; Stage 7b
//! ships the real `submit_attestation` flow against vendored chain
//! primitives at rev `d83e45a4` (`sumchain-primitives` +
//! `sumchain-crypto`). The full chain-confirmed construction sequence
//! lives in [`tx::build_and_submit_signed_transaction`].
//!
//! ## Design points
//!
//! - **Generic over [`JsonRpcTransport`]** so default `cargo test` is
//!   fully hermetic against [`FakeJsonRpcTransport`]. Production
//!   construction via [`SumChainClient::new`] defaults to
//!   [`UreqTransport`] for sync HTTP.
//! - **No `Send + Sync` requirement on the transport trait.** Callers
//!   (the Stage 5.1 registry workflow) are single-threaded. Production
//!   [`UreqTransport`] happens to be `Send + Sync` via `ureq::Agent`'s
//!   internal pooling; the test fake's `Arc<Mutex<_>>` doesn't need
//!   the bound either.
//! - **Read DTOs are owned here**, not in `omni-zkml`. The Stage 6
//!   inner-payload types (`InferenceAttestationDigest`,
//!   `InferenceAttestationTxData`) stay in `omni-zkml::chain_wire` and
//!   are write-side canonical sources; the read RPCs return *view*
//!   shapes ([`InferenceAttestationInfo`],
//!   [`InferenceAttestationStatusInfo`]) that this crate parses out of
//!   JSON-RPC responses.
//! - **Submit boundary uses vendored chain types.** Stage 7b's
//!   [`tx`] module copies the Stage 6 digest field-by-field into
//!   `sumchain_primitives::InferenceAttestationDigest`; byte-equivalence
//!   under bincode 1.3 is pinned by
//!   `tests/parity_vendored_primitives.rs`.
//! - **`Unknown` is non-terminal.** [`crate::status::map_status_info`]
//!   maps the chain's `"unknown"` status to
//!   [`omni_zkml::AttestationStatus::Unknown`]; the Stage 5.1
//!   `query_attestation_workflow` already treats this as
//!   observation-only and leaves the record unchanged with a
//!   `tracing::warn!`. **`Dropped` is client-local only** and never
//!   produced by this crate.
//!
//! ## Gates
//!
//! - Hermetic tests (read + submit construction): default-on; no network.
//! - Live read tests: gated by `OMNINODE_SUMCHAIN_RPC_URL`; `#[ignore]`'d
//!   by default; auto-skip when unset.
//! - Live submit roundtrip: additionally gated by
//!   `OMNINODE_VERIFIER_SEED_HEX` (64 hex chars); auto-skips when unset.
//! - Runtime submit gates: `omninode_is_active()` AND `v2_is_active()`
//!   must both return `Ok(true)` and the configured seed must derive to
//!   the attestation's `verifier_address`; otherwise `sum_getNonce` /
//!   `sum_sendRawTransaction` are never reached.

pub mod client;
pub mod dto;
pub(crate) mod outer_sign;
pub mod rpc;
pub mod status;
pub mod tx;

pub use client::SumChainClient;
pub use dto::{
    BlockFinality, BlockHeightInfo, ChainParamsInfo, InferenceAttestationInfo,
    InferenceAttestationStatusInfo,
};
pub use rpc::{FakeJsonRpcTransport, JsonRpcTransport, UreqTransport};
pub use status::map_status_info;
