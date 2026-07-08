//! Issue #83 — SUM Chain InferenceSettlement v1 read surface.
//!
//! Read-only RPC client for the settlement subprotocol introduced by
//! `sum-chain#76` and refined by `sum-chain#86`. Feature-gated behind
//! `settlement-read` so default `omni-sumchain` / `omni-node` builds
//! compile zero settlement code and the dependency tree is unchanged.
//!
//! ## Structure
//!
//! - [`wire`] — raw JSON-RPC response DTOs, matching the chain wire
//!   verbatim. Never invents fields; never composes across RPCs.
//! - [`view`] — normalized OmniNode-side views composed from one or
//!   more raw DTOs. Multi-verifier `Vec<_>` and consistency-group
//!   composition live here, not on the wire types.
//! - [`dormancy`] — typed errors ([`SettlementReadError`]) plus the
//!   local gate-active check. Three gates (settlement / consistency /
//!   bonding), each `Option<u64>` on `crate::dto::ChainParamsInfo`.
//! - [`adapter`] — extension methods on the existing generic
//!   `SumChainClient<T: JsonRpcTransport>`. Every gated method fetches
//!   `chain_getChainParams` and `chain_getBlockHeight` first and
//!   refuses to issue the gated RPC if the required gate is dormant.
//!   No parallel client; no writes; no signing.
//!
//! ## Non-goals
//!
//! - No write path. No key material. No claim submission.
//! - No CLI subcommand (that's Issue #84).
//! - No `event=` observability markers (that's Issue #85).
//! - No AI-semantic-correctness / on-chain zkML claim anywhere.
//! - Chain-side reward denial is never referred to as "slashing" on
//!   this read surface (or anywhere else in OmniNode).

pub mod adapter;
pub mod dormancy;
pub mod view;
pub mod wire;

pub use dormancy::{check_gate_active, SettlementGate, SettlementReadError};
pub use view::{
    BondState, BondSummary, ClaimState, DigestTuple, DisputeState, PerVerifierView,
    SessionLifecycle, SessionModeFlags, SettlementSessionView,
};
