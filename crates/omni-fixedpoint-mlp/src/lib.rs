//! #131 item 1(a) — the fixed-point MLP numeric contract of
//! `production-fixedpoint-mlp-v1`, evaluated over a checked layer set, so the
//! canonical model and a later loaded one share one evaluator.
//!
//! - [`contract`]: the arithmetic every dense layer uses (`i16 × i16 → i64`
//!   accumulation, the bias in the widened scale² domain, round half away
//!   from zero, saturation to `i16`, ReLU).
//! - [`canonical`]: the frozen canonical weights and the canonical model's
//!   identity, `BLAKE3(canonical_spec.json)`.
//! - [`model`]: [`FixedPointMlp`], a validated sequence of dense and ReLU
//!   layers, its tensor encoding (`i16` little-endian, no header), and
//!   [`MlpAccounting`], the counts a run reports, derived from the model.
//!
//! **No dependencies.** The halo2 proof crate
//! (`omni-proofs-halo2-production-mlp`) keeps its own canonical evaluator,
//! unchanged, and its tests check this crate against it: equal outputs on
//! the frozen corpus and a sweep, equal weights, and equal model identity.
//! Proofs exist for the canonical model only.
//!
//! Off-chain only.

pub mod canonical;
pub mod contract;
pub mod model;

pub use canonical::{CANONICAL_SPEC_BLAKE3_HEX, CANONICAL_SPEC_NAME, CANONICAL_SPEC_VERSION};
pub use model::{DenseLayer, FixedPointMlp, Layer, MlpAccounting, ModelError};
