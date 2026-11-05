//! The numeric contract of `production-fixedpoint-mlp-v1`, as frozen by
//! `omni-proofs-halo2-production-mlp::canonical` (`quantization` in
//! `canonical_spec.json`):
//!
//! - `i16 × i16 → i64` accumulation per dense output;
//! - the bias added in the widened scale² domain, `acc + (bias << 8)`,
//!   before requantization;
//! - round half away from zero when dividing by `2^8`;
//! - saturation to `i16`;
//! - ReLU as `max(x, 0)`.
//!
//! The functions are copies of the proof crate's; its tests pin the two
//! together.

/// `log2` of the fixed-point scale (`scale = 256`).
pub const SCALE_LOG2: u32 = 8;

/// Saturate an `i64` to the `i16` range.
#[inline]
pub fn saturate_to_i16(x: i64) -> i16 {
    x.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

/// ReLU: `max(x, 0)`.
#[inline]
pub fn relu_i16(x: i16) -> i16 {
    if x > 0 { x } else { 0 }
}

/// `n / 2^scale_log2`, rounded half away from zero.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_ties_go_away_from_zero() {
        assert_eq!(round_half_away_from_zero_div(128, 8), 1); // +0.5
        assert_eq!(round_half_away_from_zero_div(-128, 8), -1); // -0.5
        assert_eq!(round_half_away_from_zero_div(384, 8), 2); // +1.5
        assert_eq!(round_half_away_from_zero_div(-384, 8), -2); // -1.5
        assert_eq!(round_half_away_from_zero_div(127, 8), 0);
        assert_eq!(round_half_away_from_zero_div(-127, 8), 0);
        assert_eq!(round_half_away_from_zero_div(7, 0), 7);
    }

    #[test]
    fn saturation_and_relu() {
        assert_eq!(saturate_to_i16(i64::MAX), i16::MAX);
        assert_eq!(saturate_to_i16(i64::MIN), i16::MIN);
        assert_eq!(saturate_to_i16(40_000), i16::MAX);
        assert_eq!(saturate_to_i16(-40_000), i16::MIN);
        assert_eq!(saturate_to_i16(-5), -5);
        assert_eq!(relu_i16(-1), 0);
        assert_eq!(relu_i16(0), 0);
        assert_eq!(relu_i16(9), 9);
    }
}
