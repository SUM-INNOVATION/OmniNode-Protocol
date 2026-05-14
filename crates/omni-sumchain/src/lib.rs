//! `omni-sumchain` — Phase 5 Stage 7a SUM Chain adapter.
//!
//! Implements the [`omni_zkml::ChainClient`] trait against SUM Chain's
//! JSON-RPC surface. **Stage 7a is read/query only.** The submit path
//! returns a typed `ChainClientError::Other(_)` until Stage 7b lands
//! the outer `SignedTransaction` construction. Do not wire
//! [`SumChainClient`] into a production submit workflow before
//! Stage 7b.
//!
//! ## Design points
//!
//! - **Generic over [`JsonRpcTransport`]** so default `cargo test` is
//!   fully hermetic against [`FakeJsonRpcTransport`]. Production
//!   construction via [`SumChainClient::new`] defaults to
//!   [`UreqTransport`] for sync HTTP.
//! - **No `Send + Sync` requirement on the transport trait.** Stage 7a
//!   callers (the registry workflow) are single-threaded. Production
//!   [`UreqTransport`] happens to be `Send + Sync` via `ureq::Agent`'s
//!   internal pooling; the test fake's `Arc<Mutex<_>>` doesn't need
//!   the bound either.
//! - **Read DTOs are owned here**, not in `omni-zkml`. The Stage 6
//!   inner-payload types (`InferenceAttestationDigest`,
//!   `InferenceAttestationTxData`) stay in `omni-zkml::chain_wire` and
//!   are write-side concerns; the read RPCs return *view* shapes
//!   ([`InferenceAttestationInfo`], [`InferenceAttestationStatusInfo`])
//!   that this crate parses out of JSON-RPC responses.
//! - **`Unknown` is non-terminal.** [`crate::status::map_status_info`]
//!   maps the chain's `"unknown"` status to
//!   [`omni_zkml::AttestationStatus::Unknown`]; the Stage 5.1
//!   `query_attestation_workflow` already treats this as
//!   observation-only and leaves the record unchanged with a
//!   `tracing::warn!`. **`Dropped` is client-local only** and never
//!   produced by this crate.
//!
//! ## Gates (recap of the approved Stage 7 plan)
//!
//! - Stage 7a hermetic code: **unblocked**, this crate is the deliverable.
//! - Stage 7a live read tests: gated by `OMNINODE_SUMCHAIN_RPC_URL` env
//!   var; `#[ignore]`'d by default; auto-skip when unset.
//! - Stage 7a live activation detection (`omninode_is_active`
//!   returning `Ok(true)`): gated on the chain follow-up patch
//!   exposing `omninode_enabled_from_height` in `chain_getChainParams`.
//!   No code change needed once chain ships it.
//! - Stage 7b submit path: gated on the chain primitive vendoring
//!   decision. See [`tx`] module doc for the construction sequence.

pub mod client;
pub mod dto;
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
