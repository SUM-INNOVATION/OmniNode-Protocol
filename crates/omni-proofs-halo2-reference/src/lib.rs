//! Phase 5 Stage 11b.1.a — bounded multi-framework halo2 reference scaffold.
//!
//! This crate is the scaffold for the Stage 11b.1 "bounded
//! multi-framework halo2 reference backend." It ships:
//!
//! 1. A pure-Rust **canonical evaluator** for a 4 → 8 → 4 int16
//!    fixed-point MLP (see [`canonical`]). This is the single source
//!    of truth — every framework manifest in [`tests/fixtures`] must
//!    reproduce its output byte-for-byte.
//! 2. Canonical byte encodings for input/output tensors (see
//!    [`encoding`]).
//! 3. The [`FrameworkManifest`] schema (see [`manifest`]) — the
//!    on-disk shape of each framework's fixture manifest.
//! 4. The compile-time `EXPECTED_SPEC_HASH` constant (BLAKE3 of
//!    `assets/canonical_spec.json`), computed by `build.rs`.
//!
//! **What does NOT ship in Stage 11b.1.a:**
//!
//! - No halo2 circuit, no halo2 dependency. Stage 11b.1.b plugs the
//!   real prover/verifier in behind `verify` / `prove` features.
//! - No `ProofBackend` / `ProofVerifier` implementations. Stage 11b.1.b
//!   adds these, gated by the same feature flags.
//! - No operator-binary dispatch arm. Stage 11b.1.b wires
//!   `omni-node`'s `verify-proof` to route Stage11bHalo2Reference
//!   artifacts through this crate's verifier.
//!
//! **Mainnet posture (Stage 11b.0 invariants preserved):**
//! Stage11bHalo2Reference artifacts will set
//! `testnet_or_dev_only: Some(true)` when Stage 11b.1.b lands; layer
//! 1 + layer 3 + layer 6 of `omni_zkml::check_mainnet_eligible` all
//! hard-refuse. `MAINNET_APPROVED_PROOF_SYSTEMS` stays empty.

pub mod canonical;
pub mod encoding;
pub mod manifest;
pub mod shared;

pub use canonical::{canonical_evaluate, CanonicalInput, CanonicalOutput};
pub use encoding::{
    decode_canonical_input, decode_canonical_output, encode_canonical_input,
    encode_canonical_output,
};
pub use manifest::{FrameworkManifest, GenerationMode};
pub use shared::{
    BACKEND_ID, CANONICAL_INPUT, CANONICAL_OUTPUT, CANONICAL_SPEC_NAME, CANONICAL_SPEC_VERSION,
    EXPECTED_SPEC_HASH,
};
