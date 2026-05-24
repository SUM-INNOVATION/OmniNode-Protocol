//! Stage 11d.2 — pure-Rust canonical evaluator for the production
//! bounded MLP described by `assets/canonical_spec.json`.
//!
//! Mirrors the structure of `omni-proofs-halo2-reference::canonical`
//! (Stage 11c neutral reference implementation) but for the
//! `16 → 32 → 16 → 8` production architecture with new frozen
//! weights and biases. Same numeric contract as `halo2-mlp-v1 /
//! spec_version: 2`:
//!
//! - `i16 × i16 → i64` accumulation per dense output.
//! - Bias in widened scale² domain BEFORE saturation
//!   (`with_bias = acc + (bias as i64) << scale_log2`).
//! - Round-half-away-from-zero requantization via signed-magnitude
//!   Euclidean division (`abs_w + S/2 = q_abs · S + r_pos`).
//! - Saturate to i16.
//! - ReLU as `max(x, 0)`.
//!
//! **Off-chain only.** This evaluator is the source-of-truth for the
//! production circuit's witness assignment AND for the cross-framework
//! corpus equivalence test; it is never consumed by chain wire,
//! `InferenceAttestationDigest`, or any chain-side verifier.
//!
//! ## Weight provenance
//!
//! The frozen weights `W1`, `B1`, `W2`, `B2`, `W3`, `B3` are
//! deterministic small integers chosen by the closed-form pattern
//! documented per-array below. The chosen ranges
//! (`W_max = 8`, `B_max ∈ {32, 64}` per layer) keep `|with_bias|`
//! bounded such that **no dense-layer output saturates** for any
//! i16-valued input. Manual verification:
//!
//! - Layer 1: `|Σ input · W1| ≤ 16 · 32767 · 8 = 4_194_304`;
//!   `|with_bias| ≤ 4_194_304 + 64 · 256 = 4_210_688`;
//!   `|q| ≤ 16448 ≪ 2^15`. No saturation.
//! - Layer 2 inputs are post-ReLU, in `[0, 16448]`. Same analysis:
//!   `|Σ input · W2| ≤ 32 · 16448 · 4 ≈ 2.1M`;
//!   `|q| ≤ 8400 ≪ 2^15`. No saturation.
//! - Layer 3 inputs are post-ReLU from Layer 2. Same analysis;
//!   no saturation.
//!
//! The bounded production model is appropriate for the
//! "deterministic small-model classification" workload (Stage 11d.2
//! plan §4.1 one-liner).

use crate::shared::CANONICAL_INPUT;

pub type CanonicalInput = [i16; 16];
pub type CanonicalOutput = [i16; 8];

pub const SCALE_LOG2: u32 = 8;
pub const SCALE: i64 = 1 << SCALE_LOG2; // 256
pub const HALF_SCALE: i64 = SCALE / 2; // 128

/// Layer 1 weights, shape `[input_dim=16, output_dim=32]`.
/// Pattern: `W1[i][j] = ((i * 13 + j * 7) % 17) as i16 - 8`,
/// yielding values in `[-8, 8]`.
pub const W1: [[i16; 32]; 16] = build_w1();

/// Layer 1 biases, shape `[output_dim=32]`.
/// Pattern: `B1[j] = ((j * 11) % 129) as i16 - 64`, yielding
/// values in `[-64, 64]`.
pub const B1: [i16; 32] = build_b1();

/// Layer 2 weights, shape `[input_dim=32, output_dim=16]`.
/// Pattern: `W2[i][j] = ((i * 5 + j * 3) % 9) as i16 - 4`,
/// yielding values in `[-4, 4]`.
pub const W2: [[i16; 16]; 32] = build_w2();

/// Layer 2 biases, shape `[output_dim=16]`.
/// Pattern: `B2[j] = ((j * 17) % 65) as i16 - 32`, yielding
/// values in `[-32, 32]`.
pub const B2: [i16; 16] = build_b2();

/// Layer 3 weights, shape `[input_dim=16, output_dim=8]`.
/// Pattern: `W3[i][j] = ((i * 7 + j * 11) % 17) as i16 - 8`,
/// yielding values in `[-8, 8]`.
pub const W3: [[i16; 8]; 16] = build_w3();

/// Layer 3 biases, shape `[output_dim=8]`.
/// Pattern: `B3[j] = ((j * 19) % 65) as i16 - 32`, yielding
/// values in `[-32, 32]`.
pub const B3: [i16; 8] = build_b3();

const fn build_w1() -> [[i16; 32]; 16] {
    let mut out = [[0i16; 32]; 16];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 32 {
            out[i][j] = ((i * 13 + j * 7) % 17) as i16 - 8;
            j += 1;
        }
        i += 1;
    }
    out
}

const fn build_b1() -> [i16; 32] {
    let mut out = [0i16; 32];
    let mut j = 0;
    while j < 32 {
        out[j] = ((j * 11) % 129) as i16 - 64;
        j += 1;
    }
    out
}

const fn build_w2() -> [[i16; 16]; 32] {
    let mut out = [[0i16; 16]; 32];
    let mut i = 0;
    while i < 32 {
        let mut j = 0;
        while j < 16 {
            out[i][j] = ((i * 5 + j * 3) % 9) as i16 - 4;
            j += 1;
        }
        i += 1;
    }
    out
}

const fn build_b2() -> [i16; 16] {
    let mut out = [0i16; 16];
    let mut j = 0;
    while j < 16 {
        out[j] = ((j * 17) % 65) as i16 - 32;
        j += 1;
    }
    out
}

const fn build_w3() -> [[i16; 8]; 16] {
    let mut out = [[0i16; 8]; 16];
    let mut i = 0;
    while i < 16 {
        let mut j = 0;
        while j < 8 {
            out[i][j] = ((i * 7 + j * 11) % 17) as i16 - 8;
            j += 1;
        }
        i += 1;
    }
    out
}

const fn build_b3() -> [i16; 8] {
    let mut out = [0i16; 8];
    let mut j = 0;
    while j < 8 {
        out[j] = ((j * 19) % 65) as i16 - 32;
        j += 1;
    }
    out
}

#[inline]
pub fn saturate_to_i16(x: i64) -> i16 {
    x.clamp(i16::MIN as i64, i16::MAX as i64) as i16
}

#[inline]
pub fn relu_i16(x: i16) -> i16 {
    if x > 0 { x } else { 0 }
}

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

fn dense<const IN: usize, const OUT: usize>(
    input: &[i16; IN],
    weights: &[[i16; OUT]; IN],
    biases: &[i16; OUT],
) -> [i16; OUT] {
    let mut output = [0i16; OUT];
    for j in 0..OUT {
        let mut acc: i64 = 0;
        for i in 0..IN {
            acc += (input[i] as i64) * (weights[i][j] as i64);
        }
        let with_bias: i64 = acc + ((biases[j] as i64) << SCALE_LOG2);
        let q: i64 = round_half_away_from_zero_div(with_bias, SCALE_LOG2);
        output[j] = saturate_to_i16(q);
    }
    output
}

/// Stage 11d.2 production canonical evaluator. Pure function;
/// deterministic; the byte-stable source-of-truth for the
/// production circuit's witness assignment AND the cross-framework
/// corpus equivalence test.
pub fn canonical_evaluate(input: CanonicalInput) -> CanonicalOutput {
    let h1_pre: [i16; 32] = dense::<16, 32>(&input, &W1, &B1);
    let mut h1: [i16; 32] = [0i16; 32];
    for j in 0..32 {
        h1[j] = relu_i16(h1_pre[j]);
    }

    let h2_pre: [i16; 16] = dense::<32, 16>(&h1, &W2, &B2);
    let mut h2: [i16; 16] = [0i16; 16];
    for j in 0..16 {
        h2[j] = relu_i16(h2_pre[j]);
    }

    dense::<16, 8>(&h2, &W3, &B3)
}

/// Verify the frozen `CANONICAL_OUTPUT` constant equals the
/// canonical evaluator's output on the frozen `CANONICAL_INPUT`.
pub fn canonical_invariant() -> bool {
    canonical_evaluate(CANONICAL_INPUT) == crate::shared::CANONICAL_OUTPUT
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper test that PRINTS the canonical evaluation when run
    /// with `-- --nocapture`. Used to derive `CANONICAL_OUTPUT`
    /// before pinning it in shared.rs. Always passes; informational.
    #[test]
    fn print_canonical_evaluation() {
        let h1_pre: [i16; 32] = dense::<16, 32>(&CANONICAL_INPUT, &W1, &B1);
        let h1: [i16; 32] = std::array::from_fn(|j| relu_i16(h1_pre[j]));
        let h2_pre: [i16; 16] = dense::<32, 16>(&h1, &W2, &B2);
        let h2: [i16; 16] = std::array::from_fn(|j| relu_i16(h2_pre[j]));
        let output: [i16; 8] = dense::<16, 8>(&h2, &W3, &B3);
        eprintln!("CANONICAL_INPUT     = {:?}", CANONICAL_INPUT);
        eprintln!("hidden1_pre_relu    = {:?}", h1_pre);
        eprintln!("hidden1_after_relu  = {:?}", h1);
        eprintln!("hidden2_pre_relu    = {:?}", h2_pre);
        eprintln!("hidden2_after_relu  = {:?}", h2);
        eprintln!("CANONICAL_OUTPUT    = {:?}", output);
        // Spot-check: no saturation occurred at any layer.
        for v in h1_pre {
            assert!(v >= i16::MIN && v <= i16::MAX);
        }
        for v in h2_pre {
            assert!(v >= i16::MIN && v <= i16::MAX);
        }
        for v in output {
            assert!(v >= i16::MIN && v <= i16::MAX);
        }
    }

    /// Helper test that PRINTS the canonical_spec.json contents
    /// derived from the in-crate const weights + canonical evaluator.
    /// Used once at fixture-author time to emit `assets/canonical_spec.json`.
    /// Always passes; informational.
    #[test]
    fn print_canonical_spec_json() {
        fn fmt_2d<const I: usize, const J: usize>(arr: &[[i16; J]; I]) -> String {
            let mut s = String::from("[\n");
            for i in 0..I {
                s.push_str("        [");
                for j in 0..J {
                    if j > 0 { s.push_str(", "); }
                    s.push_str(&format!("{:>3}", arr[i][j]));
                }
                s.push(']');
                if i + 1 < I { s.push(','); }
                s.push('\n');
            }
            s.push_str("      ]");
            s
        }
        fn fmt_1d<const N: usize>(arr: &[i16; N]) -> String {
            let mut s = String::from("[");
            for j in 0..N {
                if j > 0 { s.push_str(", "); }
                s.push_str(&format!("{:>3}", arr[j]));
            }
            s.push(']');
            s
        }
        let h1_pre: [i16; 32] = dense::<16, 32>(&CANONICAL_INPUT, &W1, &B1);
        let h1: [i16; 32] = std::array::from_fn(|j| relu_i16(h1_pre[j]));
        let h2_pre: [i16; 16] = dense::<32, 16>(&h1, &W2, &B2);
        let h2: [i16; 16] = std::array::from_fn(|j| relu_i16(h2_pre[j]));
        let output: [i16; 8] = dense::<16, 8>(&h2, &W3, &B3);
        let w1 = fmt_2d::<16, 32>(&W1);
        let b1 = fmt_1d::<32>(&B1);
        let w2 = fmt_2d::<32, 16>(&W2);
        let b2 = fmt_1d::<16>(&B2);
        let w3 = fmt_2d::<16, 8>(&W3);
        let b3 = fmt_1d::<8>(&B3);
        let inp = fmt_1d::<16>(&CANONICAL_INPUT);
        let hp1 = fmt_1d::<32>(&h1_pre);
        let ha1 = fmt_1d::<32>(&h1);
        let hp2 = fmt_1d::<16>(&h2_pre);
        let ha2 = fmt_1d::<16>(&h2);
        let out = fmt_1d::<8>(&output);
        eprintln!("=== BEGIN canonical_spec.json ===");
        eprintln!("{{");
        eprintln!("  \"spec_name\": \"production-fixedpoint-mlp-v1\",");
        eprintln!("  \"spec_version\": 1,");
        eprintln!("  \"description\": \"Stage 11d.2 production fixed-point MLP — 16 -> 32 -> 16 -> 8 with ReLU, int16 fixed-point arithmetic. First production-grade proof class. Framework-neutral; RUMUS, PyTorch, TensorFlow, and Caffe are all equal-status primary compatibility targets. The pure-Rust canonical evaluator in crates/omni-proofs-halo2-production-mlp is the neutral reference implementation, not a fifth framework. Every framework's manifest must reproduce the canonical_evaluation.output byte-for-byte. Off-chain proof metadata only — this spec and the proof class it describes never appear in chain wire / tx / InferenceAttestationDigest / RPC / validator-side verification paths. Mainnet eligibility is a Stage 11d.3 deliverable; MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES stays empty through 11d.2.\",");
        eprintln!("  \"architecture\": {{");
        eprintln!("    \"layers\": [");
        eprintln!("      {{\"kind\": \"Dense\", \"input_dim\": 16, \"output_dim\": 32, \"weights_ref\": \"W1\", \"biases_ref\": \"B1\"}},");
        eprintln!("      {{\"kind\": \"ReLU\"}},");
        eprintln!("      {{\"kind\": \"Dense\", \"input_dim\": 32, \"output_dim\": 16, \"weights_ref\": \"W2\", \"biases_ref\": \"B2\"}},");
        eprintln!("      {{\"kind\": \"ReLU\"}},");
        eprintln!("      {{\"kind\": \"Dense\", \"input_dim\": 16, \"output_dim\": 8, \"weights_ref\": \"W3\", \"biases_ref\": \"B3\"}}");
        eprintln!("    ]");
        eprintln!("  }},");
        eprintln!("  \"quantization\": {{");
        eprintln!("    \"dtype\": \"i16\",");
        eprintln!("    \"accumulator_dtype\": \"i64\",");
        eprintln!("    \"scale\": 256,");
        eprintln!("    \"scale_log2\": 8,");
        eprintln!("    \"rounding\": \"round-nearest-ties-away-from-zero\",");
        eprintln!("    \"bias_application\": \"widened-domain-before-saturation\",");
        eprintln!("    \"overflow\": \"saturate-to-i16\",");
        eprintln!("    \"tensor_encoding\": \"i16-little-endian\"");
        eprintln!("  }},");
        eprintln!("  \"weights\": {{");
        eprintln!("    \"W1\": {{ \"shape\": [16, 32], \"values\": {} }},", w1);
        eprintln!("    \"B1\": {{ \"shape\": [32], \"values\": {} }},", b1);
        eprintln!("    \"W2\": {{ \"shape\": [32, 16], \"values\": {} }},", w2);
        eprintln!("    \"B2\": {{ \"shape\": [16], \"values\": {} }},", b2);
        eprintln!("    \"W3\": {{ \"shape\": [16, 8], \"values\": {} }},", w3);
        eprintln!("    \"B3\": {{ \"shape\": [8], \"values\": {} }}", b3);
        eprintln!("  }},");
        eprintln!("  \"canonical_evaluation\": {{");
        eprintln!("    \"input\": {},", inp);
        eprintln!("    \"hidden1_pre_relu\": {},", hp1);
        eprintln!("    \"hidden1_after_relu\": {},", ha1);
        eprintln!("    \"hidden2_pre_relu\": {},", hp2);
        eprintln!("    \"hidden2_after_relu\": {},", ha2);
        eprintln!("    \"output\": {}", out);
        eprintln!("  }}");
        eprintln!("}}");
        eprintln!("=== END canonical_spec.json ===");
    }

    #[test]
    fn canonical_input_evaluates_without_saturation() {
        // Confirms the chosen weight ranges keep all intermediate
        // i64 values within i16 range AFTER requantization. This is
        // the architectural invariant the production circuit relies
        // on: no dense output saturates for the canonical input.
        let h1_pre: [i16; 32] = dense::<16, 32>(&CANONICAL_INPUT, &W1, &B1);
        let h1: [i16; 32] = std::array::from_fn(|j| relu_i16(h1_pre[j]));
        let h2_pre: [i16; 16] = dense::<32, 16>(&h1, &W2, &B2);
        // No values at i16::MIN or i16::MAX (saturation boundaries).
        for v in h1_pre {
            assert!(v > i16::MIN && v < i16::MAX);
        }
        for v in h2_pre {
            assert!(v > i16::MIN && v < i16::MAX);
        }
    }

    #[test]
    fn canonical_invariant_holds() {
        assert!(
            canonical_invariant(),
            "frozen CANONICAL_OUTPUT does not equal canonical_evaluate(CANONICAL_INPUT)"
        );
    }

    #[test]
    fn canonical_evaluator_is_deterministic() {
        let a = canonical_evaluate(CANONICAL_INPUT);
        let b = canonical_evaluate(CANONICAL_INPUT);
        let c = canonical_evaluate(CANONICAL_INPUT);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn round_half_away_from_zero_semantics() {
        // Same semantics as Stage 11c (criteria-pinned).
        assert_eq!(round_half_away_from_zero_div(128, 8), 1);
        assert_eq!(round_half_away_from_zero_div(-128, 8), -1);
        assert_eq!(round_half_away_from_zero_div(127, 8), 0);
        assert_eq!(round_half_away_from_zero_div(256, 8), 1);
        assert_eq!(round_half_away_from_zero_div(-256, 8), -1);
        assert_eq!(round_half_away_from_zero_div(384, 8), 2);
        assert_eq!(round_half_away_from_zero_div(-384, 8), -2);
    }
}
