//! Stage 11b.1.a — pure-Rust canonical evaluator for the bounded MLP
//! described by `assets/canonical_spec.json`.
//!
//! **This module is the neutral reference implementation**, not a
//! "fifth framework." RUMUS, PyTorch, TensorFlow, and Caffe are all
//! equal-status primary compatibility targets; this evaluator gives
//! the canonical spec a runnable Rust pin so the cross-framework
//! integration test can assert byte-for-byte equivalence without
//! invoking any framework runtime.
//!
//! ## Numeric contract (frozen — matches `quantization` in the spec)
//!
//! - **Datatype:** `i16` signed 16-bit, throughout.
//! - **Fixed-point scale:** `S = 2^scale_log2 = 2^8 = 256`. A logical
//!   real value `r` maps to `q = round_half_away_from_zero(r * S)`
//!   clamped to `[i16::MIN, i16::MAX]`.
//! - **Dense layer arithmetic per output `j`:**
//!   1. `acc_i64 = Σ_i (input[i] as i64) * (W[i][j] as i64)` —
//!      products are exact in `i64` because `i16 × i16` fits in
//!      `i32`, and the per-layer fanin is bounded.
//!   2. `with_bias_i64 = acc_i64 + (bias[j] as i64) * S` — the bias
//!      is promoted into the widened scale² domain by left-shifting
//!      `scale_log2` bits **before** requantization, so saturation
//!      sees the full pre-quantized magnitude.
//!   3. `q_i64 = round_half_away_from_zero_div(with_bias_i64, S)` —
//!      shifts back to scale `S` using the nearest-ties-away-from-zero
//!      rule. **Not** Rust's default truncate-toward-zero.
//!   4. `output[j] = saturate_to_i16(q_i64)`.
//! - **Round half away from zero:** ties round in the direction of
//!   sign. `0.5 → 1`, `-0.5 → -1`, `2.5 → 3`, `-2.5 → -3`.
//! - **Saturation:** `saturate_to_i16(x: i64) = x.clamp(i16::MIN as
//!   i64, i16::MAX as i64) as i16`. Never wraps.
//! - **ReLU:** `max(0i16, x)`. Applied per-element after Layer 1.
//! - **Layer 2** has no activation on the output.
//!
//! These are the same semantics RUMUS 0.4.0's `fixed::FixedLinear` /
//! `fixed::requantize` implement.

use crate::shared::{CANONICAL_INPUT, CANONICAL_OUTPUT};

pub type CanonicalInput = [i16; 4];
pub type CanonicalOutput = [i16; 4];

pub const W1: [[i16; 8]; 4] = [
    [ 32, -16,   8,  64,  -8,  16, -32,   0],
    [-16,  32,  16,   8,   8,  16,  16,   4],
    [  8,   8,  -8,   8,  64,   0, -16,  16],
    [ 16, -32,  64,  16, -16,   8,   8,   8],
];

pub const B1: [i16; 8] = [ 64, -32,  16, -16,  32,   0, -32,   8];

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

pub const B2: [i16; 4] = [ 32, -32,  16,   8];

/// `scale_log2` exponent. The global scale is `S = 1 << SCALE_LOG2 = 256`.
pub const SCALE_LOG2: u32 = 8;

/// Saturate an `i64` to the `i16` range.
#[inline]
pub fn saturate_to_i16(x: i64) -> i16 {
    x.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

/// Element-wise ReLU on an `i16`. `max(0, x)`.
#[inline]
pub fn relu_i16(x: i16) -> i16 {
    if x > 0 { x } else { 0 }
}

/// Round-half-away-from-zero integer division.
///
/// `n` is the numerator in the widened scale² domain (sum of i16×i16
/// products + scaled bias); `scale_log2` is the exponent of the
/// shift back down to scale `S`. Returns the requantized value
/// **before** i16 saturation.
///
/// Ties (e.g. `0.5`, `-0.5`) round in the direction of sign.
#[inline]
pub fn round_half_away_from_zero_div(n: i64, scale_log2: u32) -> i64 {
    if scale_log2 == 0 {
        return n;
    }
    let half: i64 = 1i64 << (scale_log2 - 1);
    if n >= 0 {
        (n + half) >> scale_log2
    } else {
        -(((-n) + half) >> scale_log2)
    }
}

/// Dense layer per the §"Numeric contract" steps 1–4.
///
/// `W` is indexed `W[input_i][output_j]` (matches the spec's
/// `[input_dim, output_dim]` shape).
fn dense<const IN: usize, const OUT: usize>(
    input: &[i16; IN],
    weights: &[[i16; OUT]; IN],
    biases: &[i16; OUT],
) -> [i16; OUT] {
    let mut output = [0i16; OUT];
    for j in 0..OUT {
        // Step 1: widened i64 accumulator. i16×i16 fits in i32, but
        // we accumulate directly into i64 to mirror RUMUS's contract.
        let mut acc: i64 = 0;
        for i in 0..IN {
            let xv = input[i] as i64;
            let wv = weights[i][j] as i64;
            acc += xv * wv;
        }
        // Step 2: bias enters the widened scale² domain BEFORE
        // saturation — left-shifted by SCALE_LOG2.
        let with_bias: i64 = acc + ((biases[j] as i64) << SCALE_LOG2);
        // Step 3: round-half-away-from-zero requantization back to
        // scale S.
        let q: i64 = round_half_away_from_zero_div(with_bias, SCALE_LOG2);
        // Step 4: saturate to i16.
        output[j] = saturate_to_i16(q);
    }
    output
}

/// Stage 11b.1 canonical evaluator. Pure function; deterministic.
///
/// Every framework manifest (RUMUS / PyTorch / TensorFlow / Caffe /
/// FrameworkAgnostic) must reproduce the output of this function
/// byte-for-byte for the canonical input — the cross-framework
/// equivalence integration test enforces it.
pub fn canonical_evaluate(input: CanonicalInput) -> CanonicalOutput {
    let hidden_pre_relu: [i16; 8] = dense::<4, 8>(&input, &W1, &B1);
    let mut hidden = [0i16; 8];
    for j in 0..8 {
        hidden[j] = relu_i16(hidden_pre_relu[j]);
    }
    dense::<8, 4>(&hidden, &W2, &B2)
}

/// Property: the frozen `CANONICAL_OUTPUT` constant equals
/// `canonical_evaluate(CANONICAL_INPUT)`.
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
    fn canonical_hidden_after_relu_matches_spec() {
        // Hand-pinned: after Layer 1 + ReLU on CANONICAL_INPUT under
        // the new arithmetic contract. Same values as the spec's
        // `canonical_evaluation.hidden_after_relu` and as RUMUS
        // 0.4.0's `FixedLinear` produces.
        let hidden_pre = dense::<4, 8>(&CANONICAL_INPUT, &W1, &B1);
        assert_eq!(hidden_pre, [57, -17, -9, -23, 44, -3, -35, 6]);
        let hidden_post: [i16; 8] = std::array::from_fn(|j| relu_i16(hidden_pre[j]));
        assert_eq!(hidden_post, [57, 0, 0, 0, 44, 0, 0, 6]);
    }

    #[test]
    fn saturate_clamps_to_i16_range() {
        assert_eq!(saturate_to_i16(0), 0);
        assert_eq!(saturate_to_i16(32767), 32767);
        assert_eq!(saturate_to_i16(-32768), -32768);
        assert_eq!(saturate_to_i16(100_000), 32767);
        assert_eq!(saturate_to_i16(-100_000), -32768);
        assert_eq!(saturate_to_i16(i64::MAX), 32767);
        assert_eq!(saturate_to_i16(i64::MIN), -32768);
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
        // On zero input: acc=0, with_bias = bias[j]<<8 = bias[j]*256;
        // requantize(bias*256, 8) = bias; saturate(bias) = bias.
        let zero_input = [0i16; 4];
        let out = dense::<4, 8>(&zero_input, &W1, &B1);
        assert_eq!(out, B1);
    }

    #[test]
    fn canonical_evaluate_extreme_positive_input_saturates() {
        let input = [i16::MAX; 4];
        let out = canonical_evaluate(input);
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
        let input = CANONICAL_INPUT;
        let a = canonical_evaluate(input);
        let b = canonical_evaluate(input);
        let c = canonical_evaluate(input);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn round_half_away_from_zero_division_semantics() {
        // scale_log2 = 1 → divide-by-2. Pins ties on both signs.
        assert_eq!(round_half_away_from_zero_div(2, 1), 1);   //  1.0 →  1
        assert_eq!(round_half_away_from_zero_div(3, 1), 2);   //  1.5 →  2 (tie, away)
        assert_eq!(round_half_away_from_zero_div(5, 1), 3);   //  2.5 →  3 (tie, away)
        assert_eq!(round_half_away_from_zero_div(1, 1), 1);   //  0.5 →  1
        assert_eq!(round_half_away_from_zero_div(-1, 1), -1); // -0.5 → -1
        assert_eq!(round_half_away_from_zero_div(-3, 1), -2); // -1.5 → -2
        assert_eq!(round_half_away_from_zero_div(-5, 1), -3); // -2.5 → -3

        // scale_log2 = 8 → divide-by-256. Pins canonical scale.
        assert_eq!(round_half_away_from_zero_div(128, 8), 1);   //  0.5 →  1
        assert_eq!(round_half_away_from_zero_div(-128, 8), -1); // -0.5 → -1
        assert_eq!(round_half_away_from_zero_div(127, 8), 0);
        assert_eq!(round_half_away_from_zero_div(-127, 8), 0);
        assert_eq!(round_half_away_from_zero_div(256, 8), 1);
        assert_eq!(round_half_away_from_zero_div(-256, 8), -1);

        // scale_log2 = 0 is a no-op.
        assert_eq!(round_half_away_from_zero_div(123, 0), 123);
        assert_eq!(round_half_away_from_zero_div(-123, 0), -123);
    }

    #[test]
    fn bias_is_applied_before_saturation() {
        // Construct a case where the unbiased requantized accumulator
        // would already exceed i16::MAX, but the (negative) bias
        // brings it back into range — this only works if the bias is
        // added in the widened-scale² domain BEFORE saturation.
        //
        // Pre-bias `acc` of 50_000 * 256 = 12_800_000 sits well above
        // saturate_to_i16(50_000) = 32_767. With a bias of -20_000
        // applied first: acc + (-20_000 << 8) = 12_800_000 - 5_120_000
        // = 7_680_000 → requantize/8 = 30_000 → saturate → 30_000.
        // If bias were applied AFTER saturation, the answer would be
        // 32_767 - 20_000 = 12_767. We expect 30_000.
        let acc: i64 = 50_000i64 * 256;
        let with_bias: i64 = acc + ((-20_000i64) << SCALE_LOG2);
        let q = round_half_away_from_zero_div(with_bias, SCALE_LOG2);
        assert_eq!(saturate_to_i16(q), 30_000);
    }
}
