//! Phase 5 Stage 11d.2 — first production-grade fixed-point MLP
//! proof class.
//!
//! Scope of this crate (per the Stage 11d.2 plan and the Stage 11d.0
//! mainnet eligibility criteria):
//!
//! 1. A pure-Rust **canonical evaluator** for a `16 → 32 → 16 → 8`
//!    int16 fixed-point MLP (see [`canonical`]). This is the single
//!    source of truth — every framework manifest in
//!    [`tests/fixtures`] must reproduce its output byte-for-byte.
//! 2. Canonical byte encodings for input/output tensors (see
//!    [`encoding`]).
//! 3. The [`manifest::FrameworkManifest`] schema — the on-disk
//!    shape of each framework's production-tier fixture manifest.
//!    Structurally similar to the bounded reference's manifest but
//!    typed to the production tensor dims (16 / 8).
//! 4. The compile-time `EXPECTED_PRODUCTION_SPEC_HASH` constant
//!    (BLAKE3 of `assets/canonical_spec.json`), computed by
//!    `build.rs`. **Distinct const name** from the bounded
//!    reference's `EXPECTED_SPEC_HASH` per Stage 11d.0
//!    distinguishability hard rule H2.
//! 5. The halo2 [`circuit`] + [`verifier`] + (developer-host)
//!    [`prover`], gated behind `verify` / `prove` features.
//!
//! **Off-chain only.** This crate never appears in chain wire / tx /
//! `InferenceAttestationDigest` / RPC / validator-side verification
//! paths. It is reachable from `omni-node` only via the opt-in
//! `stage11d-production-verify` cargo feature.
//!
//! **Mainnet posture (Stage 11d.0/11d.1 invariants preserved):**
//! `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES` stays empty through
//! Stage 11d.2. Mainnet allowlist entry for this proof class is a
//! Stage 11d.3 deliverable (R1–R9 sign-off gate).

pub mod canonical;
pub mod encoding;
pub mod manifest;
pub mod shared;

// Stage 11d.2 — halo2 circuit + verifier + (developer-host) prover.
// Default build pulls none of these. CI's verifier job uses
// `--features verify`; the prover job uses `--features prove`.
#[cfg(feature = "verify")]
pub mod circuit;
#[cfg(feature = "verify")]
pub mod verifier;
#[cfg(feature = "prove")]
pub mod prover;

pub use canonical::{canonical_evaluate, CanonicalInput, CanonicalOutput};
pub use encoding::{
    decode_canonical_input, decode_canonical_output, encode_canonical_input,
    encode_canonical_output,
};
pub use manifest::{FrameworkManifest, GenerationMode, ManifestError};
pub use shared::{
    BACKEND_ID, CANONICAL_INPUT, CANONICAL_OUTPUT, EXPECTED_CIRCUIT_ID_HEX,
    EXPECTED_PRODUCTION_SPEC_HASH, EXPECTED_VK_HASH_HEX, HALO2_K, PRODUCTION_SPEC_NAME,
    PRODUCTION_SPEC_VERSION,
};

#[cfg(feature = "verify")]
pub use verifier::{
    derive_vk_identity_from_params, live_circuit_id_hex, live_vk_hash_hex, vk_canonical_bytes,
    Halo2ProductionMlpVerifier,
};

// Stage 14.5 — operator-reachable prover. The adapter exposes
// `prove_canonical` through `omni_zkml::ProofBackend` so
// `omni-node`'s `stage11d-production-prove` feature can dispatch
// proof generation through the same trait seam Stage 14.1 uses
// for `Halo2ReferenceProofBackend`.
#[cfg(feature = "prove")]
pub use prover::Halo2ProductionMlpProofBackend;
