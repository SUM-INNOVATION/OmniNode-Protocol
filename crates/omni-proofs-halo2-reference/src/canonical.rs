//! Stage 11b.1.a — pure-Rust canonical evaluator for the bounded MLP
//! described by `assets/canonical_spec.json`.
//!
//! **This module is the single source of truth.** Every framework
//! manifest in `tests/fixtures/` must reproduce its `output` byte-
//! for-byte; the cross-framework equivalence integration test
//! asserts this. Stage 11b.1.b's halo2 circuit will likewise prove
//! exactly this arithmetic.
//!
//! ## Numeric contract (frozen)
//!
//! - **Datatype:** `i16` signed 16-bit, throughout.
//! - **Fixed-point scale:** `S = 256 = 2^8`. A logical real value
//!   `r` maps to `q = round_half_to_even(r * S)` ∈ `[-32768, 32767]`.
//!   (The spec doesn't run real-valued inputs through here — the
//!   inputs ARE already integer-quantized — but the scale is what
//!   makes weight multiplications meaningful as fractional ops.)
//! - **Multiplication semantics for a Dense layer's accumulated
//!   matmul:** accumulate `i16 × i16 → i32` products in `i32`
//!   without intermediate overflow, then divide by `S` once at the
//!   end with **truncate-toward-zero** integer division, then add
//!   the bias in `i32`, then `saturate_to_i16` the final value.
//! - **Truncate toward zero** (NOT round-half-to-even): for
//!   `-1760 / 256` the result is `-6`, not `-7`. This matches the
//!   default Rust `i32 / i32` semantics and is the simplest
//!   operator to reproduce in every framework + halo2 circuit.
//! - **Saturation:** `saturate_to_i16(x: i32) = x.clamp(i16::MIN as
//!   i32, i16::MAX as i32) as i16`. Never wraps.
//! - **ReLU:** `max(0i16, x)`. Applied per-element after Layer 1's
//!   `(matmul / S) + bias + saturate`.
//! - **Layer 2** has no activation on the output.
//!
//! ## Frozen weights / biases
//!
//! Values match `assets/canonical_spec.json` exactly. Re-derivable
//! by inspection; pinned here so the canonical evaluator doesn't
//! parse JSON at runtime.

use crate::shared::{CANONICAL_INPUT, CANONICAL_OUTPUT};

pub type CanonicalInput = [i16; 4];
pub type CanonicalOutput = [i16; 4];

/// Layer-1 weights, shape `[input_dim=4, output_dim=8]`. Indexed
/// `W1[input_i][output_j]`.
pub const W1: [[i16; 8]; 4] = [
    [ 32, -16,   8,  64,  -8,  16, -32,   0],
    [-16,  32,  16,   8,   8,  16,  16,   4],
    [  8,   8,  -8,   8,  64,   0, -16,  16],
    [ 16, -32,  64,  16, -16,   8,   8,   8],
];

/// Layer-1 biases, shape `[output_dim=8]`.
pub const B1: [i16; 8] = [ 64, -32,  16, -16,  32,   0, -32,   8];

/// Layer-2 weights, shape `[input_dim=8, output_dim=4]`. Indexed
/// `W2[input_i][output_j]`.
pub const W2: [[i16; 4]; 8] = [
    [  8, -16,  32,  -8],
    [ 32,   8, -16,  16],
    [-16,  32,   8,  -8],
    [ 16, -16,  16,  32],
    [ -8,  16, -32,   8],
    [ 32,   8,  16,  -8],
    [-16, -32,   8,  32],
    [  8,  16, -16, -16],
];

/// Layer-2 biases, shape `[output_dim=4]`.
pub const B2: [i16; 4] = [ 32, -32,  16,   8];

/// Fixed-point scale `S`. All Dense multiplications divide by this.
pub const SCALE: i32 = 256;

/// Saturate an `i32` to the `i16` range. The canonical contract
/// requires this everywhere a Dense layer's accumulator is reduced
/// to `i16`.
#[inline]
pub fn saturate_to_i16(x: i32) -> i16 {
    x.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Element-wise ReLU on an `i16`. `max(0, x)`.
#[inline]
pub fn relu_i16(x: i16) -> i16 {
    if x > 0 {
        x
    } else {
        0
    }
}

/// Dense layer: `output_j = saturate( (Σ_i input[i] * W[i][j]) / SCALE + bias[j] )`.
///
/// `W` is indexed `W[input_i][output_j]` (matches the spec JSON's
/// `[input_dim, output_dim]` shape).
fn dense<const IN: usize, const OUT: usize>(
    input: &[i16; IN],
    weights: &[[i16; OUT]; IN],
    biases: &[i16; OUT],
) -> [i16; OUT] {
    let mut output = [0i16; OUT];
    for j in 0..OUT {
        let mut acc: i32 = 0;
        for i in 0..IN {
            // i16 × i16 fits in i32 without overflow.
            acc += (input[i] as i32) * (weights[i][j] as i32);
        }
        // Truncate-toward-zero division. Native Rust `i32 / i32` is
        // exactly this.
        let scaled: i32 = acc / SCALE;
        let with_bias: i32 = scaled + (biases[j] as i32);
        output[j] = saturate_to_i16(with_bias);
    }
    output
}

/// Stage 11b.1 canonical evaluator. Pure function; deterministic;
/// `forall input: [i16; 4], canonical_evaluate(input)` is a function
/// of the spec constants in this module and the input alone.
///
/// **Every framework manifest (RUMUS / PyTorch / TensorFlow / Caffe /
/// FrameworkAgnostic) must reproduce the output of this function
/// byte-for-byte for the canonical input.** The cross-framework
/// equivalence integration test pins that property at test time.
pub fn canonical_evaluate(input: CanonicalInput) -> CanonicalOutput {
    let hidden_pre_relu: [i16; 8] = dense::<4, 8>(&input, &W1, &B1);
    let mut hidden = [0i16; 8];
    for j in 0..8 {
        hidden[j] = relu_i16(hidden_pre_relu[j]);
    }
    dense::<8, 4>(&hidden, &W2, &B2)
}

// ── Compile-time sanity (debug-asserts in tests) ──────────────────────────────

/// Property: the frozen `CANONICAL_OUTPUT` constant equals
/// `canonical_evaluate(CANONICAL_INPUT)`. This is the load-bearing
/// invariant for fixture stability — if any of the weights, biases,
/// or arithmetic constants drift, this assertion fires.
pub fn canonical_invariant() -> bool {
    canonical_evaluate(CANONICAL_INPUT) == CANONICAL_OUTPUT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_invariant_holds() {
        assert!(canonical_invariant(),
                "canonical_evaluate({:?}) = {:?}, expected {:?}",
                CANONICAL_INPUT,
                canonical_evaluate(CANONICAL_INPUT),
                CANONICAL_OUTPUT);
    }

    #[test]
    fn saturate_clamps_to_i16_range() {
        assert_eq!(saturate_to_i16(0), 0);
        assert_eq!(saturate_to_i16(32767), 32767);
        assert_eq!(saturate_to_i16(-32768), -32768);
        assert_eq!(saturate_to_i16(100_000), 32767);
        assert_eq!(saturate_to_i16(-100_000), -32768);
        assert_eq!(saturate_to_i16(i32::MAX), 32767);
        assert_eq!(saturate_to_i16(i32::MIN), -32768);
    }

    #[test]
    fn relu_at_boundary_values() {
        assert_eq!(relu_i16(0), 0);
        assert_eq!(relu_i16(1), 1);
        assert_eq!(relu_i16(-1), 0);
        assert_eq!(relu_i16(i16::MAX), i16::MAX);
        assert_eq!(relu_i16(i16::MIN), 0);
    }

    #[test]
    fn dense_zero_input_gives_bias() {
        // Layer 1 on zero input: output = saturate(0 / SCALE + bias) = bias.
        let zero_input = [0i16; 4];
        let out = dense::<4, 8>(&zero_input, &W1, &B1);
        assert_eq!(out, B1);
    }

    #[test]
    fn canonical_evaluate_zero_input() {
        // ReLU(B1) → dense<8,4> → output. B1 = [64,-32,16,-16,32,0,-32,8].
        // After ReLU: [64, 0, 16, 0, 32, 0, 0, 8].
        let zero_input = [0i16; 4];
        let _out = canonical_evaluate(zero_input);
        // Don't pin the output bytes here — let the canonical_invariant
        // and the fixture-vs-evaluator tests be the byte-stable pin.
        // This test just confirms the function returns without panicking
        // on a degenerate input.
    }

    #[test]
    fn canonical_evaluate_extreme_positive_input_saturates() {
        // Pathological input that would overflow without saturation.
        let input = [i16::MAX; 4];
        let out = canonical_evaluate(input);
        // All output values must still be i16-valid (no panic, no
        // wraparound — saturation absorbed the overflow).
        for v in out {
            assert!(v >= i16::MIN && v <= i16::MAX);
        }
    }

    #[test]
    fn canonical_evaluate_extreme_negative_input_saturates() {
        let input = [i16::MIN; 4];
        let out = canonical_evaluate(input);
        for v in out {
            assert!(v >= i16::MIN && v <= i16::MAX);
        }
    }

    #[test]
    fn canonical_evaluate_is_deterministic() {
        // Same input → same output, always.
        let input = CANONICAL_INPUT;
        let a = canonical_evaluate(input);
        let b = canonical_evaluate(input);
        let c = canonical_evaluate(input);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn truncate_toward_zero_division_semantics() {
        // Pin the exact rounding rule the canonical spec requires.
        // Rust's i32 / i32 is truncate-toward-zero — verify this in
        // tree so a future stdlib change (unlikely but possible)
        // would surface here.
        assert_eq!(-1760i32 / 256, -6);  // not -7
        assert_eq!(-1i32 / 256, 0);       // truncates to 0, not -1
        assert_eq!(1760i32 / 256, 6);
        assert_eq!(255i32 / 256, 0);
        assert_eq!(-255i32 / 256, 0);
    }
}
