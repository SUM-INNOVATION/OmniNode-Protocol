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
// Stage 9a + 9c: outer_sign + tx depend on the public chain
// primitives (`sumchain-primitives` / `sumchain-crypto` v0.1.0,
// crates.io, MIT OR Apache-2.0). Gated behind `submit` so default
// builds resolve without ever touching the chain crates.
#[cfg(feature = "submit")]
pub(crate) mod outer_sign;
pub mod rpc;
// Issue #83 — settlement track read surface. Feature-gated so default
// builds compile zero settlement code. Dependency-neutral (no new
// external crates activated).
#[cfg(feature = "settlement-read")]
pub mod settlement;
pub mod status;
#[cfg(feature = "submit")]
pub mod tx;
// Issue #87 — settlement claim WRITE path. Superset of
// `settlement-read` + `submit`; adds no new crates on top of
// `submit`. See `settlement_submit/mod.rs` for the module contract.
#[cfg(feature = "settlement-submit")]
pub mod settlement_submit;
// Issue #81 — settlement dispute WRITE path (open + resolve).
// Superset of `settlement-submit`; adds no new crates.
#[cfg(feature = "settlement-dispute")]
pub mod settlement_dispute;

pub use client::SumChainClient;
pub use dto::{
    BlockFinality, BlockHeightInfo, ChainParamsInfo, InferenceAttestationInfo,
    InferenceAttestationStatusInfo,
};
pub use rpc::{
    classify_chain_client_error, error_prefixes, ChainErrorCategory, FakeJsonRpcTransport,
    JsonRpcTransport, UreqTransport,
};
pub use status::map_status_info;
