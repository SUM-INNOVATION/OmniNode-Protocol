//! # R0 host-side foundation (OmniNode #101) — B0 wire adoption, NON-SELECTION
//!
//! An **isolated research crate** that adopts the published, frozen **B0**
//! candidate-neutral wire types from the SUM Chain team's `sumchain-wire 0.2.1`
//! crates.io release (namespace [`sumchain_wire::b0`]) and exercises the
//! host-side, crypto-free groundwork around them: object commitments, SNIP
//! Merkle witness/proof generation, the 996-byte candidate-neutral statement
//! *template*, the candidate-neutral partial-compute proof, the candidate-
//! specific production proof envelope, allowlist / verifier-material identity
//! binding, the integer reference workload, and a benchmark artifact schema.
//!
//! It is deliberately *outside* the OmniNode workspace (the root `Cargo.toml`
//! sets `exclude = ["tools"]`) and no production crate depends on it (enforced by
//! `tests/dependency_isolation.rs` + `scripts/check_no_prod_dep.sh`).
//!
//! ## Status: NON-SELECTION, host-side only — this is NOT a zkVM harness
//!
//! This crate **proves nothing** and integrates **no** zkVM SDK / toolchain /
//! container / build-script. There is no guest, no proving, no Groth16 wrapping,
//! no measurement, and no real receipt verification. The
//! [`verifier`](crate::verifier) models allowlist / identity / journal binding
//! **logic** over an explicitly *synthetic* [`verifier::CannedReceipt`] — it is
//! NOT proof verification.
//!
//! Non-selection is **structurally forced** (see [`classification`]): the only
//! run classification the crate can produce is
//! [`classification::RunClass::NonSelectionResearch`] (there is no
//! `Final`/`Official`/`Selection`/`Eligible` variant or code path), and every
//! emitted artifact carries the exact visible status string
//! [`classification::NON_SELECTION_STATUS`]. The statement is only ever built as
//! a zero-spec-hash **template** (via [`statement::SyntheticJournal`]).
//!
//! ## What this crate does and does not emit
//!
//! The one artifact it emits is a JSON research record ([`bench::BenchArtifact`])
//! that carries **no measured samples**, is **not a selection-valid artifact**,
//! and contains **no proof / finalization / protocol-hash data**. No public path
//! through this crate reaches B0's `materialize_final`, a raw `Writer`, or the
//! full B0 statement module: the B0 namespace is re-exported **crate-privately**,
//! and only the focused, safe adopted types are public (see the module list
//! below and `tests/api_surface.rs`).

#![forbid(unsafe_code)]

pub mod bench;
pub mod classification;
pub mod envelope;
pub mod fixture;
pub mod manifest;
pub mod merkle;
pub mod model_auth;
pub mod object;
pub mod statement;
pub mod verifier;
pub mod workload;

/// The adopted, frozen B0 wire family. Re-exported **crate-privately**: the full
/// `sumchain_wire::b0` namespace includes `statement::materialize_final`, the raw
/// `Writer`, and the complete statement module, none of which may be reachable
/// through this crate's public API. The safe adopted types are exposed instead
/// through the focused modules ([`object`], [`manifest`], [`statement`],
/// [`envelope`], [`merkle`], …); `tests/api_surface.rs` enforces the boundary.
pub(crate) use sumchain_wire::b0;
/// The B0 decode-error type, re-exported so callers can match rejection variants.
/// This is a read-only error enum — not a byte-construction primitive (the raw
/// `Reader`/`Writer` are deliberately NOT re-exported publicly).
pub use sumchain_wire::b0::codec::DecodeError;

/// Upper bound on any single committed object's declared `byte_len`, enforced by
/// the host-side Merkle witness helpers *before* allocating, so a hostile witness
/// cannot force an unbounded allocation. 4 GiB — well below the point at which
/// the B0 `chunk_count` (`u32`) could saturate.
pub const FROZEN_MAX_OBJECT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Frozen fixed-point scale exponent (`S = 2^8 = 256`), taken from the B0 frozen
/// scalar set so the reference workload matches the adopted numeric contract.
pub const FIXED_POINT_SCALE_LOG2: u8 = sumchain_wire::b0::consts::FIXED_POINT_SCALE_LOG2;

/// `BLAKE3` of `bytes`, returned as a raw 32-byte digest. Used for synthetic
/// fixture identities and model-byte derivation only — never for a protocol
/// identity (those go through the B0 domain-prefixed hash rules).
#[inline]
pub fn blake3_32(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}
