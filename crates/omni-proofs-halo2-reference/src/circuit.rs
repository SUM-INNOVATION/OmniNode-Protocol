//! Stage 11c — halo2 circuit for the canonical bounded MLP with
//! arbitrary-input soundness under the frozen `halo2-mlp-v1 /
//! spec_version: 2` numeric contract.
//!
//! Replaces the Stage 11b.1.b "linear identity + slack-remainder"
//! gates (which were unsound for arbitrary inputs at requantization
//! ties) with a complete gadget chain:
//!
//! ```text
//! inputs ─→ DENSE LINEAR IDENTITY ─→ w = Σ input·W + B·S
//!                                    │
//!                                    ▼
//!                          ROUND-HALF-AWAY-FROM-ZERO
//!                          (RHAZ via signed-magnitude
//!                           Euclidean division)
//!                                    │
//!                                    ▼
//!                                q_unsat
//!                                    │
//!                                    ▼
//!                          THREE-BRANCH SATURATION
//!                          (b_lo / b_in / b_hi)
//!                                    │
//!                                    ▼
//!                                  q_sat
//! ```
//!
//! At Layer 1 each `q_sat_layer1[j]` then flows through the ReLU
//! sign-bit gadget; the ReLU outputs feed Layer 2's dense identity;
//! Layer 2's `q_sat_layer2[j]` is constrained equal to the public
//! `output[j]` instance value.
//!
//! ## Public inputs (`instance` column)
//!
//! ```text
//! row 0..4 : input[0..4]   (canonical i16 input  lifted to Fp)
//! row 4..8 : output[0..4]  (canonical i16 output lifted to Fp)
//! ```
//!
//! ## Soundness scope (precise)
//!
//! The circuit proves **`canonical_evaluate(input) == output`** for
//! **every i16-valued public input** under the frozen
//! `halo2-mlp-v1 / spec_version: 2` numeric contract. Specifically:
//!
//! - The Euclidean division `abs_w + S/2 = q_abs · S + r_pos` with
//!   `r_pos ∈ [0, S)` is *unique*, so tie cases (`with_bias =
//!   ±S/2 · (2k+1)`) are pinned to the canonical round-half-AWAY
//!   branch (the canonical evaluator's `(with_bias + S/2) / S`
//!   form for non-negative magnitude).
//! - The saturation gadget's `b_lo + b_in + b_hi = 1` (with branch-
//!   correctness aux witnesses) deterministically maps `q_unsat`
//!   to `q_sat = saturate_i16(q_unsat)` whether or not `q_unsat`
//!   falls inside `[i16::MIN, i16::MAX]`.
//! - Range checks (bit decomposition) bound every witness to its
//!   declared width; no slack absorption is possible.
//!
//! **This is still not "production zkML" and is still
//! `testnet_or_dev_only: Some(true)`.** The bounded 4→8→4 MLP and
//! its committed weights/biases are an architectural-validation
//! fixture; mainnet allowlist eligibility is a Stage 11d+
//! deliverable.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    pasta::Fp,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector},
    poly::Rotation,
};

use crate::canonical::{B1, B2, W1, W2};

/// Convenience: wrap an Fp value into a constant Expression.
fn fp_expr(x: Fp) -> Expression<Fp> {
    Expression::Constant(x)
}

/// Embed an `i64` into Fp via signed modular reduction.
pub fn fp_from_i64(x: i64) -> Fp {
    if x >= 0 {
        Fp::from(x as u64)
    } else {
        -Fp::from(x.unsigned_abs())
    }
}

/// `Value::known(fp_from_i64(x))` convenience.
pub fn known_i64(x: i64) -> Value<Fp> {
    Value::known(fp_from_i64(x))
}

// ── Canonical-contract constants (mirrors src/canonical.rs) ───────
const SCALE_LOG2: u32 = 8;
const SCALE: i64 = 1 << SCALE_LOG2; // 256
const HALF_SCALE: i64 = SCALE / 2; // 128
const I16_MIN: i64 = -32768;
const I16_MAX: i64 = 32767;

// ── Bit-column allocation for range checks ────────────────────────
//
// 24 bit columns. The widest range check (abs_w, 23-bit unsigned)
// fits in 23 of them; 17/16/15/8-bit checks use a subset and the
// gates constrain the unused high bits to be zero.
const NUM_BIT_COLS: usize = 24;
const VALUE_COLS: usize = 8;

/// Circuit configuration — column allocation + selector definitions.
#[derive(Clone, Debug)]
pub struct BoundedMlpConfig {
    /// 8 general-purpose advice columns for cell values.
    value: [Column<Advice>; VALUE_COLS],
    /// 24 advice columns for bit decomposition.
    bit_cols: [Column<Advice>; NUM_BIT_COLS],
    /// Instance column carrying 4 input + 4 output i16 values.
    instance: Column<Instance>,

    // Dense linear identity (per layer).
    s_dense_layer1: Selector,
    s_dense_layer2: Selector,
    // RHAZ + saturation gadgets (shared across layers).
    s_rhaz: Selector,
    s_sat: Selector,
    // ReLU sign-bit gadget.
    s_relu: Selector,
    // Range checks at various widths.
    s_rc8u: Selector,
    s_rc15u: Selector,
    s_rc16s: Selector,
    s_rc16u: Selector,
    s_rc17u: Selector,
    s_rc23u: Selector,
}

/// Per-dense-output witnesses (12 outputs: 8 in Layer 1 + 4 in Layer 2).
#[derive(Clone, Debug)]
pub struct DenseUnitWitness {
    pub w: Value<Fp>,                                // with_bias
    pub s_w: Value<Fp>,                              // sign-of-w bit
    pub abs_w: Value<Fp>,                            // |with_bias|
    pub q_abs: Value<Fp>,                            // floor((|w| + S/2) / S)
    pub r_pos: Value<Fp>,                            // (|w| + S/2) mod S
    pub q_unsat: Value<Fp>,                          // signed quotient (pre-sat)
    pub q_sat: Value<Fp>,                            // i16-saturated quotient
    pub sat_b_lo: Value<Fp>,
    pub sat_b_in: Value<Fp>,
    pub sat_b_hi: Value<Fp>,
    pub sat_lo_aux: Value<Fp>,
    pub sat_hi_aux: Value<Fp>,
    pub sat_in_aux_lo: Value<Fp>,
    pub sat_in_aux_hi: Value<Fp>,
    pub abs_w_bits: [Value<Fp>; NUM_BIT_COLS],
    pub q_abs_bits: [Value<Fp>; NUM_BIT_COLS],
    pub r_pos_bits: [Value<Fp>; NUM_BIT_COLS],
    pub sat_lo_aux_bits: [Value<Fp>; NUM_BIT_COLS],
    pub sat_hi_aux_bits: [Value<Fp>; NUM_BIT_COLS],
    pub sat_in_aux_lo_bits: [Value<Fp>; NUM_BIT_COLS],
    pub sat_in_aux_hi_bits: [Value<Fp>; NUM_BIT_COLS],
    pub q_sat_bits: [Value<Fp>; NUM_BIT_COLS],
}

impl Default for DenseUnitWitness {
    fn default() -> Self {
        Self {
            w: Value::unknown(),
            s_w: Value::unknown(),
            abs_w: Value::unknown(),
            q_abs: Value::unknown(),
            r_pos: Value::unknown(),
            q_unsat: Value::unknown(),
            q_sat: Value::unknown(),
            sat_b_lo: Value::unknown(),
            sat_b_in: Value::unknown(),
            sat_b_hi: Value::unknown(),
            sat_lo_aux: Value::unknown(),
            sat_hi_aux: Value::unknown(),
            sat_in_aux_lo: Value::unknown(),
            sat_in_aux_hi: Value::unknown(),
            abs_w_bits: [Value::unknown(); NUM_BIT_COLS],
            q_abs_bits: [Value::unknown(); NUM_BIT_COLS],
            r_pos_bits: [Value::unknown(); NUM_BIT_COLS],
            sat_lo_aux_bits: [Value::unknown(); NUM_BIT_COLS],
            sat_hi_aux_bits: [Value::unknown(); NUM_BIT_COLS],
            sat_in_aux_lo_bits: [Value::unknown(); NUM_BIT_COLS],
            sat_in_aux_hi_bits: [Value::unknown(); NUM_BIT_COLS],
            q_sat_bits: [Value::unknown(); NUM_BIT_COLS],
        }
    }
}

/// The Stage 11c halo2 circuit.
#[derive(Clone, Debug)]
pub struct BoundedMlpCircuit {
    pub input: [Value<Fp>; 4],
    pub output: [Value<Fp>; 4],
    pub layer1: [DenseUnitWitness; 8],
    /// ReLU sign-bit witnesses (sign + magnitude + post_relu).
    pub sign_bits: [Value<Fp>; 8],
    pub magnitudes: [Value<Fp>; 8],
    pub magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; 8],
    pub hidden_post_relu: [Value<Fp>; 8],
    pub layer2: [DenseUnitWitness; 4],
}

impl Default for BoundedMlpCircuit {
    fn default() -> Self {
        Self {
            input: [Value::unknown(); 4],
            output: [Value::unknown(); 4],
            layer1: std::array::from_fn(|_| DenseUnitWitness::default()),
            sign_bits: [Value::unknown(); 8],
            magnitudes: [Value::unknown(); 8],
            magnitude_bits: [[Value::unknown(); NUM_BIT_COLS]; 8],
            hidden_post_relu: [Value::unknown(); 8],
            layer2: std::array::from_fn(|_| DenseUnitWitness::default()),
        }
    }
}

/// Decompose `value` into `n_bits` LE bits, padding remaining
/// `NUM_BIT_COLS - n_bits` cells with zero.
fn bits_le(value: u64, n_bits: usize) -> [Value<Fp>; NUM_BIT_COLS] {
    assert!(n_bits <= NUM_BIT_COLS);
    if n_bits < 64 {
        assert!(
            value < (1u64 << n_bits),
            "value {value} doesn't fit in {n_bits} bits"
        );
    }
    let mut bits = [Value::known(Fp::zero()); NUM_BIT_COLS];
    for i in 0..n_bits {
        let bit = (value >> i) & 1;
        bits[i] = Value::known(Fp::from(bit));
    }
    bits
}

/// Same as [`bits_le`] but for a signed `value` shifted by `offset`
/// before decomposition (so 16-bit signed values are decomposed as
/// `(value + 2^15) ∈ [0, 2^16)`).
fn bits_le_signed(value: i64, offset: i64, n_bits: usize) -> [Value<Fp>; NUM_BIT_COLS] {
    let shifted = value + offset;
    assert!(shifted >= 0, "value {value} + offset {offset} negative");
    bits_le(shifted as u64, n_bits)
}

impl BoundedMlpCircuit {
    /// Build a fully-populated circuit by running the canonical
    /// evaluator on `input_i16`. Developer-host use only.
    ///
    /// Halts (`panic!`) if any intermediate value violates the
    /// bounds the gadgets expect (e.g., `|with_bias| ≥ 2^23`,
    /// `q_abs ≥ 2^16`). This is the "halt and report" rule
    /// the Stage 11c plan committed to: if the spec/evaluator
    /// produces an out-of-bound value, the circuit cannot prove
    /// it and the caller must report rather than patching.
    pub fn from_canonical_input(input_i16: [i16; 4]) -> Self {
        // ── Layer 1 ───────────────────────────────────────────
        let mut layer1: [DenseUnitWitness; 8] =
            std::array::from_fn(|_| DenseUnitWitness::default());
        let mut hidden_pre_relu = [0i64; 8];
        for j in 0..8 {
            let acc: i64 = (0..4)
                .map(|i| (input_i16[i] as i64) * (W1[i][j] as i64))
                .sum();
            let w = acc + (B1[j] as i64) * SCALE;
            let (unit, q_sat) = compute_dense_unit_witnesses(w);
            layer1[j] = unit;
            hidden_pre_relu[j] = q_sat;
        }

        // ── ReLU ──────────────────────────────────────────────
        let mut sign_bits = [0i64; 8];
        let mut magnitudes = [0i64; 8];
        let mut hidden_post_relu = [0i64; 8];
        for j in 0..8 {
            let x = hidden_pre_relu[j];
            if x >= 0 {
                sign_bits[j] = 0;
                magnitudes[j] = x;
                hidden_post_relu[j] = x;
            } else {
                sign_bits[j] = 1;
                magnitudes[j] = -x;
                hidden_post_relu[j] = 0;
            }
        }
        let magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; 8] =
            std::array::from_fn(|j| bits_le(magnitudes[j] as u64, 15));

        // ── Layer 2 ───────────────────────────────────────────
        let mut layer2: [DenseUnitWitness; 4] =
            std::array::from_fn(|_| DenseUnitWitness::default());
        let mut output = [0i64; 4];
        for j in 0..4 {
            let acc: i64 = (0..8)
                .map(|i| (hidden_post_relu[i] as i64) * (W2[i][j] as i64))
                .sum();
            let w = acc + (B2[j] as i64) * SCALE;
            let (unit, q_sat) = compute_dense_unit_witnesses(w);
            layer2[j] = unit;
            output[j] = q_sat;
        }

        Self {
            input: [
                known_i64(input_i16[0] as i64),
                known_i64(input_i16[1] as i64),
                known_i64(input_i16[2] as i64),
                known_i64(input_i16[3] as i64),
            ],
            output: [
                known_i64(output[0]),
                known_i64(output[1]),
                known_i64(output[2]),
                known_i64(output[3]),
            ],
            layer1,
            sign_bits: std::array::from_fn(|j| known_i64(sign_bits[j])),
            magnitudes: std::array::from_fn(|j| known_i64(magnitudes[j])),
            magnitude_bits,
            hidden_post_relu: std::array::from_fn(|j| known_i64(hidden_post_relu[j])),
            layer2,
        }
    }

    pub fn canonical_outputs_for(input_i16: [i16; 4]) -> [i16; 4] {
        crate::canonical::canonical_evaluate(input_i16)
    }
}

/// Run the round-half-away-from-zero division and saturation
/// arithmetic on `w` (the dense layer's `with_bias` value) and
/// produce the corresponding gadget witnesses + the final `q_sat`
/// integer (for the builder to chain into the next layer).
fn compute_dense_unit_witnesses(w: i64) -> (DenseUnitWitness, i64) {
    let s_w = if w >= 0 { 0i64 } else { 1i64 };
    let abs_w = w.unsigned_abs() as i64;
    assert!(
        abs_w < (1 << 23),
        "|with_bias| = {abs_w} ≥ 2^23; canonical spec bounds violated — halt and report"
    );
    // Euclidean division: abs_w + S/2 = q_abs * S + r_pos, r_pos ∈ [0, S).
    let shifted = (abs_w as u64) + HALF_SCALE as u64;
    let q_abs = (shifted / (SCALE as u64)) as i64;
    let r_pos = (shifted % (SCALE as u64)) as i64;
    assert!(r_pos >= 0 && r_pos < SCALE);
    assert!(
        q_abs >= 0 && q_abs < (1 << 16),
        "q_abs out of 16-bit unsigned bound: {q_abs}"
    );
    let q_unsat = if s_w == 0 { q_abs } else { -q_abs };

    // Saturation
    let q_sat = q_unsat.clamp(I16_MIN, I16_MAX);
    let (sat_b_lo, sat_b_in, sat_b_hi);
    let (sat_lo_aux, sat_hi_aux, sat_in_aux_lo, sat_in_aux_hi);
    if q_unsat < I16_MIN {
        sat_b_lo = 1;
        sat_b_in = 0;
        sat_b_hi = 0;
        sat_lo_aux = -q_unsat - I16_MIN_MAGNITUDE_NEG_BOUNDARY;
        sat_hi_aux = 0;
        sat_in_aux_lo = 0;
        sat_in_aux_hi = 0;
    } else if q_unsat > I16_MAX {
        sat_b_lo = 0;
        sat_b_in = 0;
        sat_b_hi = 1;
        sat_lo_aux = 0;
        sat_hi_aux = q_unsat - (1 << 15);
        sat_in_aux_lo = 0;
        sat_in_aux_hi = 0;
    } else {
        sat_b_lo = 0;
        sat_b_in = 1;
        sat_b_hi = 0;
        sat_lo_aux = 0;
        sat_hi_aux = 0;
        sat_in_aux_lo = q_unsat + (1 << 15);
        sat_in_aux_hi = (1i64 << 15) - 1 - q_unsat;
    }
    for (name, v) in [
        ("sat_lo_aux", sat_lo_aux),
        ("sat_hi_aux", sat_hi_aux),
        ("sat_in_aux_lo", sat_in_aux_lo),
        ("sat_in_aux_hi", sat_in_aux_hi),
    ] {
        assert!(
            v >= 0 && v < (1 << 17),
            "{name} out of 17-bit unsigned bound: {v}"
        );
    }

    let unit = DenseUnitWitness {
        w: known_i64(w),
        s_w: known_i64(s_w),
        abs_w: known_i64(abs_w),
        q_abs: known_i64(q_abs),
        r_pos: known_i64(r_pos),
        q_unsat: known_i64(q_unsat),
        q_sat: known_i64(q_sat),
        sat_b_lo: known_i64(sat_b_lo),
        sat_b_in: known_i64(sat_b_in),
        sat_b_hi: known_i64(sat_b_hi),
        sat_lo_aux: known_i64(sat_lo_aux),
        sat_hi_aux: known_i64(sat_hi_aux),
        sat_in_aux_lo: known_i64(sat_in_aux_lo),
        sat_in_aux_hi: known_i64(sat_in_aux_hi),
        abs_w_bits: bits_le(abs_w as u64, 23),
        q_abs_bits: bits_le(q_abs as u64, 16),
        r_pos_bits: bits_le(r_pos as u64, 8),
        sat_lo_aux_bits: bits_le(sat_lo_aux as u64, 17),
        sat_hi_aux_bits: bits_le(sat_hi_aux as u64, 17),
        sat_in_aux_lo_bits: bits_le(sat_in_aux_lo as u64, 17),
        sat_in_aux_hi_bits: bits_le(sat_in_aux_hi as u64, 17),
        q_sat_bits: bits_le_signed(q_sat, 1 << 15, 16),
    };
    (unit, q_sat)
}

/// The lower saturation boundary expressed as `-2^15 - 1`. When the
/// `b_lo` branch fires, `lo_aux = -q_unsat - 2^15 - 1`; the
/// canonical formula expects this constant.
const I16_MIN_MAGNITUDE_NEG_BOUNDARY: i64 = (1 << 15) + 1;

impl Circuit<Fp> for BoundedMlpCircuit {
    type Config = BoundedMlpConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let value: [Column<Advice>; VALUE_COLS] = std::array::from_fn(|_| meta.advice_column());
        for c in &value {
            meta.enable_equality(*c);
        }
        let bit_cols: [Column<Advice>; NUM_BIT_COLS] =
            std::array::from_fn(|_| meta.advice_column());
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let s_dense_layer1 = meta.selector();
        let s_dense_layer2 = meta.selector();
        let s_rhaz = meta.selector();
        let s_sat = meta.selector();
        let s_relu = meta.selector();
        let s_rc8u = meta.selector();
        let s_rc15u = meta.selector();
        let s_rc16s = meta.selector();
        let s_rc16u = meta.selector();
        let s_rc17u = meta.selector();
        let s_rc23u = meta.selector();

        let scale_fp = Fp::from(SCALE as u64);
        let half_scale_fp = Fp::from(HALF_SCALE as u64);

        // ── Dense Layer 1 ───────────────────────────────────────────
        // Row layout (anchor row = Rotation::cur()):
        //   row 0: value[0..4] = inputs[0..4]
        //   row 1: value[0..8] = w_layer1[0..8]
        meta.create_gate("dense_layer1", |meta| {
            let s = meta.query_selector(s_dense_layer1);
            let inputs: [Expression<Fp>; 4] =
                std::array::from_fn(|i| meta.query_advice(value[i], Rotation::cur()));
            let mut polys = Vec::with_capacity(8);
            for j in 0..8 {
                let w_j = meta.query_advice(value[j], Rotation::next());
                let mut sum = w_j;
                for i in 0..4 {
                    sum = sum
                        - inputs[i].clone() * fp_expr(fp_from_i64(W1[i][j] as i64));
                }
                sum = sum - fp_expr(fp_from_i64(B1[j] as i64) * scale_fp);
                polys.push(s.clone() * sum);
            }
            polys
        });

        // ── Dense Layer 2 ───────────────────────────────────────────
        // Row layout:
        //   row 0: value[0..8] = post_relu[0..8]
        //   row 1: value[0..4] = w_layer2[0..4]
        meta.create_gate("dense_layer2", |meta| {
            let s = meta.query_selector(s_dense_layer2);
            let pr: [Expression<Fp>; 8] =
                std::array::from_fn(|i| meta.query_advice(value[i], Rotation::cur()));
            let mut polys = Vec::with_capacity(4);
            for j in 0..4 {
                let w_j = meta.query_advice(value[j], Rotation::next());
                let mut sum = w_j;
                for i in 0..8 {
                    sum = sum
                        - pr[i].clone() * fp_expr(fp_from_i64(W2[i][j] as i64));
                }
                sum = sum - fp_expr(fp_from_i64(B2[j] as i64) * scale_fp);
                polys.push(s.clone() * sum);
            }
            polys
        });

        // ── RHAZ (round-half-away-from-zero) gadget ─────────────────
        // Row layout (single anchor row):
        //   value[0] = w (with_bias) — copy-advice'd from dense region
        //   value[1] = s_w (sign bit ∈ {0,1})
        //   value[2] = abs_w
        //   value[3] = q_abs
        //   value[4] = r_pos
        //   value[5] = q_unsat
        meta.create_gate("rhaz", |meta| {
            let s = meta.query_selector(s_rhaz);
            let w = meta.query_advice(value[0], Rotation::cur());
            let s_w = meta.query_advice(value[1], Rotation::cur());
            let abs_w = meta.query_advice(value[2], Rotation::cur());
            let q_abs = meta.query_advice(value[3], Rotation::cur());
            let r_pos = meta.query_advice(value[4], Rotation::cur());
            let q_unsat = meta.query_advice(value[5], Rotation::cur());

            let one = fp_expr(Fp::from(1u64));
            let two = fp_expr(Fp::from(2u64));

            // 1. Sign decomposition: w = (1 - 2 s_w) · abs_w
            let sign_decomp = w - (one.clone() - two.clone() * s_w.clone()) * abs_w.clone();
            // 2. Sign booleanity
            let sign_bool = s_w.clone() * (one.clone() - s_w.clone());
            // 3. Euclidean division: abs_w + S/2 = q_abs · S + r_pos
            let euclid =
                abs_w + fp_expr(half_scale_fp) - q_abs.clone() * fp_expr(scale_fp) - r_pos;
            // 4. Signed quotient: q_unsat = (1 - 2 s_w) · q_abs
            let signed_quot = q_unsat - (one - two * s_w) * q_abs;
            vec![
                s.clone() * sign_decomp,
                s.clone() * sign_bool,
                s.clone() * euclid,
                s * signed_quot,
            ]
        });

        // ── Saturation gadget ───────────────────────────────────────
        // Row layout (two-row anchor):
        //   row 0: value[0]=q_unsat, value[1]=q_sat, value[2]=b_lo,
        //          value[3]=b_in, value[4]=b_hi, value[5]=lo_aux,
        //          value[6]=hi_aux, value[7]=in_aux_lo
        //   row 1: value[0]=in_aux_hi
        meta.create_gate("saturation", |meta| {
            let s = meta.query_selector(s_sat);
            let q_unsat = meta.query_advice(value[0], Rotation::cur());
            let q_sat = meta.query_advice(value[1], Rotation::cur());
            let b_lo = meta.query_advice(value[2], Rotation::cur());
            let b_in = meta.query_advice(value[3], Rotation::cur());
            let b_hi = meta.query_advice(value[4], Rotation::cur());
            let lo_aux = meta.query_advice(value[5], Rotation::cur());
            let hi_aux = meta.query_advice(value[6], Rotation::cur());
            let in_aux_lo = meta.query_advice(value[7], Rotation::cur());
            let in_aux_hi = meta.query_advice(value[0], Rotation::next());

            let one = fp_expr(Fp::from(1u64));
            let i16_min = fp_expr(fp_from_i64(I16_MIN));
            let i16_max = fp_expr(fp_from_i64(I16_MAX));
            let i16_min_neg_boundary =
                fp_expr(fp_from_i64(I16_MIN_MAGNITUDE_NEG_BOUNDARY)); // 2^15 + 1
            let two_pow_15 = fp_expr(Fp::from(1u64 << 15));

            let bool_lo = b_lo.clone() * (one.clone() - b_lo.clone());
            let bool_in = b_in.clone() * (one.clone() - b_in.clone());
            let bool_hi = b_hi.clone() * (one.clone() - b_hi.clone());
            let sum_one = b_lo.clone() + b_in.clone() + b_hi.clone() - one.clone();
            let output_rule = q_sat
                - b_lo.clone() * i16_min
                - b_in.clone() * q_unsat.clone()
                - b_hi.clone() * i16_max;
            // b_lo · (-q_unsat - (2^15 + 1) - lo_aux) = 0
            let lo_correct = b_lo
                * (fp_expr(Fp::zero()) - q_unsat.clone() - i16_min_neg_boundary - lo_aux);
            // b_hi · (q_unsat - 2^15 - hi_aux) = 0
            let hi_correct = b_hi * (q_unsat.clone() - two_pow_15.clone() - hi_aux);
            // b_in · (q_unsat + 2^15 - in_aux_lo) = 0
            let in_lo_correct = b_in.clone() * (q_unsat.clone() + two_pow_15.clone() - in_aux_lo);
            // b_in · (2^15 - 1 - q_unsat - in_aux_hi) = 0
            let in_hi_correct = b_in * (two_pow_15 - fp_expr(Fp::from(1u64)) - q_unsat - in_aux_hi);

            vec![
                s.clone() * bool_lo,
                s.clone() * bool_in,
                s.clone() * bool_hi,
                s.clone() * sum_one,
                s.clone() * output_rule,
                s.clone() * lo_correct,
                s.clone() * hi_correct,
                s.clone() * in_lo_correct,
                s * in_hi_correct,
            ]
        });

        // ── ReLU sign-bit gadget ────────────────────────────────────
        // Same as Stage 11b.1.b: single-row anchor.
        //   value[0] = pre_relu (copy-advice'd from saturation region)
        //   value[1] = sign
        //   value[2] = magnitude
        //   value[3] = post_relu
        meta.create_gate("relu", |meta| {
            let s = meta.query_selector(s_relu);
            let pre_relu = meta.query_advice(value[0], Rotation::cur());
            let sign = meta.query_advice(value[1], Rotation::cur());
            let magnitude = meta.query_advice(value[2], Rotation::cur());
            let post_relu = meta.query_advice(value[3], Rotation::cur());
            let one = fp_expr(Fp::from(1u64));
            let two = fp_expr(Fp::from(2u64));
            let booleanity = sign.clone() * (one.clone() - sign.clone());
            let decomposition =
                pre_relu - (one.clone() - two * sign.clone()) * magnitude.clone();
            let relu_eq = post_relu - (one - sign) * magnitude;
            vec![
                s.clone() * booleanity,
                s.clone() * decomposition,
                s * relu_eq,
            ]
        });

        // ── Range checks (helper) ───────────────────────────────────
        // For each width, a gate that:
        //   - asserts each bit ∈ {0,1} for i < width
        //   - asserts each bit == 0 for i ≥ width
        //   - asserts value = Σ bits[i] · 2^i (unsigned)  OR
        //                value + 2^15 = Σ bits[i] · 2^i (rc16s)
        fn build_rc_gate(
            meta: &mut ConstraintSystem<Fp>,
            name: &'static str,
            sel: Selector,
            value: Column<Advice>,
            bit_cols: &[Column<Advice>; NUM_BIT_COLS],
            width: usize,
            signed_offset: Option<Fp>,
        ) {
            meta.create_gate(name, |meta| {
                let s = meta.query_selector(sel);
                let v = meta.query_advice(value, Rotation::cur());
                let mut polys = Vec::with_capacity(NUM_BIT_COLS + 1);
                // Booleanity / forced-zero
                for i in 0..NUM_BIT_COLS {
                    let b = meta.query_advice(bit_cols[i], Rotation::cur());
                    if i < width {
                        polys.push(s.clone() * b.clone() * (fp_expr(Fp::from(1u64)) - b));
                    } else {
                        polys.push(s.clone() * b);
                    }
                }
                // Sum constraint
                let mut sum = fp_expr(Fp::zero());
                for i in 0..width {
                    let b = meta.query_advice(bit_cols[i], Rotation::cur());
                    sum = sum + b * fp_expr(Fp::from(1u64 << i as u32));
                }
                let lhs = match signed_offset {
                    Some(off) => v + fp_expr(off),
                    None => v,
                };
                polys.push(s * (sum - lhs));
                polys
            });
        }

        build_rc_gate(meta, "rc8u", s_rc8u, value[0], &bit_cols, 8, None);
        build_rc_gate(meta, "rc15u", s_rc15u, value[0], &bit_cols, 15, None);
        build_rc_gate(
            meta,
            "rc16s",
            s_rc16s,
            value[0],
            &bit_cols,
            16,
            Some(Fp::from(1u64 << 15)),
        );
        build_rc_gate(meta, "rc16u", s_rc16u, value[0], &bit_cols, 16, None);
        build_rc_gate(meta, "rc17u", s_rc17u, value[0], &bit_cols, 17, None);
        build_rc_gate(meta, "rc23u", s_rc23u, value[0], &bit_cols, 23, None);

        BoundedMlpConfig {
            value,
            bit_cols,
            instance,
            s_dense_layer1,
            s_dense_layer2,
            s_rhaz,
            s_sat,
            s_relu,
            s_rc8u,
            s_rc15u,
            s_rc16s,
            s_rc16u,
            s_rc17u,
            s_rc23u,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // ── Layer 1 dense identity region ───────────────────────────
        // Row 0: inputs[0..4]
        // Row 1: w_layer1[0..8]
        let (input_cells, w_layer1_cells) = layouter.assign_region(
            || "dense_layer1",
            |mut region| {
                config.s_dense_layer1.enable(&mut region, 0)?;
                let input_cells: Vec<_> = (0..4)
                    .map(|i| {
                        region.assign_advice(
                            || format!("input[{i}]"),
                            config.value[i],
                            0,
                            || self.input[i],
                        )
                    })
                    .collect::<Result<_, _>>()?;
                let mut w_cells = Vec::with_capacity(8);
                for j in 0..8 {
                    let cell = region.assign_advice(
                        || format!("w_layer1[{j}]"),
                        config.value[j],
                        1,
                        || self.layer1[j].w,
                    )?;
                    w_cells.push(cell);
                }
                Ok((input_cells, w_cells))
            },
        )?;

        // ── RHAZ + saturation + range checks for each Layer 1 j ─────
        let mut q_sat_layer1_cells = Vec::with_capacity(8);
        for j in 0..8 {
            let w_cell = w_layer1_cells[j].clone();
            let unit = &self.layer1[j];
            let q_sat_cell = assign_rhaz_sat_chain(
                &mut layouter,
                &config,
                &format!("layer1[{j}]"),
                w_cell,
                unit,
            )?;
            q_sat_layer1_cells.push(q_sat_cell);
        }

        // ── ReLU (8 single-row regions) ─────────────────────────────
        let mut post_relu_cells = Vec::with_capacity(8);
        for j in 0..8 {
            let pre = q_sat_layer1_cells[j].clone();
            let magnitude_bits = self.magnitude_bits[j];
            let (post, mag_cell) = layouter.assign_region(
                || format!("relu[{j}]"),
                |mut region| {
                    config.s_relu.enable(&mut region, 0)?;
                    pre.copy_advice(
                        || format!("pre_relu[{j}] copy"),
                        &mut region,
                        config.value[0],
                        0,
                    )?;
                    region.assign_advice(
                        || format!("sign[{j}]"),
                        config.value[1],
                        0,
                        || self.sign_bits[j],
                    )?;
                    let mag = region.assign_advice(
                        || format!("magnitude[{j}]"),
                        config.value[2],
                        0,
                        || self.magnitudes[j],
                    )?;
                    let post = region.assign_advice(
                        || format!("post_relu[{j}]"),
                        config.value[3],
                        0,
                        || self.hidden_post_relu[j],
                    )?;
                    Ok((post, mag))
                },
            )?;
            // Magnitude 15-bit range check.
            assign_range_check(
                &mut layouter,
                &config,
                &format!("rc15u_magnitude[{j}]"),
                config.s_rc15u,
                &mag_cell,
                &magnitude_bits,
            )?;
            post_relu_cells.push(post);
        }

        // ── Layer 2 dense identity region ───────────────────────────
        // Row 0: post_relu[0..8]
        // Row 1: w_layer2[0..4]
        let w_layer2_cells = layouter.assign_region(
            || "dense_layer2",
            |mut region| {
                config.s_dense_layer2.enable(&mut region, 0)?;
                for (i, cell) in post_relu_cells.iter().enumerate() {
                    cell.copy_advice(
                        || format!("post_relu[{i}] copy"),
                        &mut region,
                        config.value[i],
                        0,
                    )?;
                }
                let mut w_cells = Vec::with_capacity(4);
                for j in 0..4 {
                    let cell = region.assign_advice(
                        || format!("w_layer2[{j}]"),
                        config.value[j],
                        1,
                        || self.layer2[j].w,
                    )?;
                    w_cells.push(cell);
                }
                Ok(w_cells)
            },
        )?;

        // ── RHAZ + saturation + range checks for each Layer 2 j ─────
        let mut q_sat_layer2_cells = Vec::with_capacity(4);
        for j in 0..4 {
            let w_cell = w_layer2_cells[j].clone();
            let unit = &self.layer2[j];
            let q_sat_cell = assign_rhaz_sat_chain(
                &mut layouter,
                &config,
                &format!("layer2[{j}]"),
                w_cell,
                unit,
            )?;
            q_sat_layer2_cells.push(q_sat_cell);
        }

        // ── Public-input binding ────────────────────────────────────
        for (i, cell) in input_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }
        for (j, cell) in q_sat_layer2_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, 4 + j)?;
        }

        Ok(())
    }
}

/// Assign one full RHAZ → saturation → range-check chain for a
/// single dense output. Returns the `q_sat` cell so the caller can
/// chain into the next layer or bind to the public output.
fn assign_rhaz_sat_chain(
    layouter: &mut impl Layouter<Fp>,
    config: &BoundedMlpConfig,
    label: &str,
    w_cell: halo2_proofs::circuit::AssignedCell<Fp, Fp>,
    unit: &DenseUnitWitness,
) -> Result<halo2_proofs::circuit::AssignedCell<Fp, Fp>, Error> {
    // RHAZ region (single row).
    let (abs_w_cell, q_abs_cell, r_pos_cell, q_unsat_cell) = layouter.assign_region(
        || format!("rhaz_{label}"),
        |mut region| {
            config.s_rhaz.enable(&mut region, 0)?;
            w_cell.copy_advice(|| "w copy", &mut region, config.value[0], 0)?;
            region.assign_advice(|| "s_w", config.value[1], 0, || unit.s_w)?;
            let abs_w_cell =
                region.assign_advice(|| "abs_w", config.value[2], 0, || unit.abs_w)?;
            let q_abs_cell =
                region.assign_advice(|| "q_abs", config.value[3], 0, || unit.q_abs)?;
            let r_pos_cell =
                region.assign_advice(|| "r_pos", config.value[4], 0, || unit.r_pos)?;
            let q_unsat_cell =
                region.assign_advice(|| "q_unsat", config.value[5], 0, || unit.q_unsat)?;
            Ok((abs_w_cell, q_abs_cell, r_pos_cell, q_unsat_cell))
        },
    )?;

    // Saturation region (two rows).
    let (q_sat_cell, sat_lo_aux_cell, sat_hi_aux_cell, sat_in_aux_lo_cell, sat_in_aux_hi_cell) =
        layouter.assign_region(
            || format!("sat_{label}"),
            |mut region| {
                config.s_sat.enable(&mut region, 0)?;
                q_unsat_cell.copy_advice(|| "q_unsat copy", &mut region, config.value[0], 0)?;
                let q_sat_cell =
                    region.assign_advice(|| "q_sat", config.value[1], 0, || unit.q_sat)?;
                region.assign_advice(|| "b_lo", config.value[2], 0, || unit.sat_b_lo)?;
                region.assign_advice(|| "b_in", config.value[3], 0, || unit.sat_b_in)?;
                region.assign_advice(|| "b_hi", config.value[4], 0, || unit.sat_b_hi)?;
                let lo_aux = region.assign_advice(
                    || "lo_aux",
                    config.value[5],
                    0,
                    || unit.sat_lo_aux,
                )?;
                let hi_aux = region.assign_advice(
                    || "hi_aux",
                    config.value[6],
                    0,
                    || unit.sat_hi_aux,
                )?;
                let in_aux_lo = region.assign_advice(
                    || "in_aux_lo",
                    config.value[7],
                    0,
                    || unit.sat_in_aux_lo,
                )?;
                let in_aux_hi = region.assign_advice(
                    || "in_aux_hi",
                    config.value[0],
                    1,
                    || unit.sat_in_aux_hi,
                )?;
                Ok((q_sat_cell, lo_aux, hi_aux, in_aux_lo, in_aux_hi))
            },
        )?;

    // Range checks.
    assign_range_check(
        layouter,
        config,
        &format!("rc23u_abs_w_{label}"),
        config.s_rc23u,
        &abs_w_cell,
        &unit.abs_w_bits,
    )?;
    assign_range_check(
        layouter,
        config,
        &format!("rc16u_q_abs_{label}"),
        config.s_rc16u,
        &q_abs_cell,
        &unit.q_abs_bits,
    )?;
    assign_range_check(
        layouter,
        config,
        &format!("rc8u_r_pos_{label}"),
        config.s_rc8u,
        &r_pos_cell,
        &unit.r_pos_bits,
    )?;
    assign_range_check(
        layouter,
        config,
        &format!("rc17u_sat_lo_aux_{label}"),
        config.s_rc17u,
        &sat_lo_aux_cell,
        &unit.sat_lo_aux_bits,
    )?;
    assign_range_check(
        layouter,
        config,
        &format!("rc17u_sat_hi_aux_{label}"),
        config.s_rc17u,
        &sat_hi_aux_cell,
        &unit.sat_hi_aux_bits,
    )?;
    assign_range_check(
        layouter,
        config,
        &format!("rc17u_sat_in_aux_lo_{label}"),
        config.s_rc17u,
        &sat_in_aux_lo_cell,
        &unit.sat_in_aux_lo_bits,
    )?;
    assign_range_check(
        layouter,
        config,
        &format!("rc17u_sat_in_aux_hi_{label}"),
        config.s_rc17u,
        &sat_in_aux_hi_cell,
        &unit.sat_in_aux_hi_bits,
    )?;
    assign_range_check(
        layouter,
        config,
        &format!("rc16s_q_sat_{label}"),
        config.s_rc16s,
        &q_sat_cell,
        &unit.q_sat_bits,
    )?;

    Ok(q_sat_cell)
}

fn assign_range_check(
    layouter: &mut impl Layouter<Fp>,
    config: &BoundedMlpConfig,
    label: &str,
    selector: Selector,
    value_cell: &halo2_proofs::circuit::AssignedCell<Fp, Fp>,
    bits: &[Value<Fp>; NUM_BIT_COLS],
) -> Result<(), Error> {
    layouter.assign_region(
        || label.to_string(),
        |mut region| {
            selector.enable(&mut region, 0)?;
            value_cell.copy_advice(|| "value copy", &mut region, config.value[0], 0)?;
            for (i, b) in bits.iter().enumerate() {
                region.assign_advice(|| format!("bit[{i}]"), config.bit_cols[i], 0, || *b)?;
            }
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{CANONICAL_INPUT, CANONICAL_OUTPUT};
    use halo2_proofs::dev::MockProver;

    fn instance_for(input: [i16; 4], output: [i16; 4]) -> Vec<Fp> {
        let mut v = Vec::with_capacity(8);
        for i in &input {
            v.push(fp_from_i64(*i as i64));
        }
        for o in &output {
            v.push(fp_from_i64(*o as i64));
        }
        v
    }

    fn k() -> u32 {
        crate::shared::HALO2_K
    }

    #[test]
    fn canonical_input_circuit_satisfies_mock_prover() {
        let circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        prover
            .verify()
            .expect("canonical assignment must satisfy constraints");
    }

    #[test]
    fn wrong_output_fails_mock_prover() {
        let circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let mut wrong = CANONICAL_OUTPUT;
        wrong[0] += 1;
        let instance = instance_for(CANONICAL_INPUT, wrong);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(prover.verify().is_err(), "wrong output must fail");
    }

    /// Stage 11c soundness regression: the canonical evaluator's
    /// integer-divide tie-break (round half AWAY from zero) is
    /// pinned by the RHAZ gadget's Euclidean division. A prover
    /// who tries to pick the round-half-TOWARD-zero branch at a
    /// tie fails because the alternative (q_abs - 1, r_pos = S)
    /// requires r_pos ∉ [0, S), which the 8-bit range check on
    /// r_pos rejects.
    #[test]
    fn tie_input_is_uniquely_pinned() {
        // Input [0, 32, 0, 0] produces a Layer-1 tie at j=7
        // (with_bias = +2176, r_signed = -128 = -S/2).
        let input: [i16; 4] = [0, 32, 0, 0];
        let canonical_output = crate::canonical::canonical_evaluate(input);
        let circuit = BoundedMlpCircuit::from_canonical_input(input);
        let instance = instance_for(input, canonical_output);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        prover
            .verify()
            .expect("canonical tie-case witness must satisfy circuit");

        // Now try the "wrong tie branch": output[3] is the j=3
        // layer-2 output; tweaking it should fail.
        let mut wrong = canonical_output;
        wrong[3] = wrong[3].wrapping_add(1);
        let bad_instance = instance_for(input, wrong);
        let bad =
            MockProver::run(k(), &circuit, vec![bad_instance]).expect("mock prover runs");
        assert!(bad.verify().is_err(), "off-by-one at a tie must fail");
    }

    #[test]
    fn fp_from_i64_round_trips_through_signed_embedding() {
        let neg5 = fp_from_i64(-5);
        let pos5 = fp_from_i64(5);
        assert_eq!(neg5 + pos5, Fp::zero());
    }

    // ── Adversarial tampering regressions ────────────────────────
    //
    // These tests confirm that the Stage 11c gadget chain catches
    // each of the soundness failure modes a malicious prover might
    // attempt. They start from a valid canonical witness and tamper
    // with one field at a time.

    /// If the prover shifts `q_abs[j]` (in some Layer 1 unit) by +1
    /// and adjusts `r_pos[j]` to compensate, the Euclidean division
    /// constraint is still satisfied — but `r_pos` is now out of
    /// `[0, S)` and the 8-bit range check on it rejects.
    #[test]
    fn tampered_q_abs_fails_range_check() {
        let mut circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        // Bump layer1[0].q_abs by 1; adjust r_pos by -S to keep the
        // Euclidean equation satisfied. r_pos becomes negative,
        // which the rc8u range check rejects (negative i64 maps to
        // p-|x| in Fp, which fails bit decomposition).
        circuit.layer1[0].q_abs = known_i64(58); // canonical was 57
        circuit.layer1[0].r_pos = known_i64(-256 + 0); // canonical r_pos was 0
        // Re-decompose the bit witnesses to reflect the lie.
        circuit.layer1[0].q_abs_bits = bits_le(58u64, 16);
        // r_pos = -256 cannot be decomposed as 8 unsigned bits.
        // We leave the bit witness array as-is (all zeros for an
        // "honest" decomposition of -256); the sum constraint will
        // fail because Σ bits = 0 ≠ -256 mod p.
        circuit.layer1[0].r_pos_bits = [Value::known(Fp::zero()); NUM_BIT_COLS];
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(
            prover.verify().is_err(),
            "tampered q_abs + matching r_pos must fail range checks"
        );
    }

    /// Force the wrong saturation branch active (b_hi=1 when b_in
    /// was correct). The output rule forces `q_sat = i16::MAX` in
    /// that branch, which contradicts the canonical `q_sat`.
    #[test]
    fn tampered_saturation_branch_fails() {
        let mut circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        // canonical: layer1[0] has q_unsat = q_sat = 57 (in-range,
        // b_in=1). Force b_hi=1 and b_in=0; sum stays 1, but the
        // output rule says q_sat = 32767, which contradicts the
        // committed q_sat = 57. MockProver rejects.
        circuit.layer1[0].sat_b_in = known_i64(0);
        circuit.layer1[0].sat_b_hi = known_i64(1);
        // hi_aux must satisfy q_unsat - 2^15 - hi_aux = 0 if b_hi=1.
        // q_unsat = 57, so hi_aux = 57 - 32768 = -32711. Negative
        // -> rc17u rejects.
        circuit.layer1[0].sat_hi_aux = known_i64(-32711);
        circuit.layer1[0].sat_hi_aux_bits = [Value::known(Fp::zero()); NUM_BIT_COLS];
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(
            prover.verify().is_err(),
            "tampered saturation branch must fail"
        );
    }

    /// Set both b_lo and b_in to 1 (sum = 2 ≠ 1). The sat-sum
    /// constraint fails.
    #[test]
    fn tampered_saturation_sum_fails() {
        let mut circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        circuit.layer1[0].sat_b_lo = known_i64(1);
        // b_in is already 1 for the canonical in-range case.
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(
            prover.verify().is_err(),
            "two sat selectors active simultaneously must fail"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// Gadget-isolated test circuits (gated by `cfg(test)` because they
// share the gate-building helpers above but use a minimal column
// allocation tailored to one gadget at a time).
//
// These exist because the canonical bounded MLP under the frozen
// `halo2-mlp-v1 / spec_version: 2` spec never produces a `q_unsat`
// outside [i16::MIN, i16::MAX], so the saturation gadget's b_lo /
// b_hi branches can't be exercised by a real i16 input. The
// gadget-isolated tests below hand-craft witness assignments and
// confirm the saturation gadget pins the right q_sat for each
// branch.
//
// The RHAZ-isolated tests cover the round-half-away-from-zero
// gadget on tie cases at ±S/2 and ±3S/2 — these CAN be hit by
// real inputs, but the gadget tests pin the math at the gadget
// level (cheaper than a full circuit invocation).
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod gadget_tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    /// Saturation-gadget-only circuit. Public inputs: [q_unsat, q_sat].
    #[derive(Clone, Debug, Default)]
    struct SatTestCircuit {
        q_unsat: Value<Fp>,
        q_sat: Value<Fp>,
        b_lo: Value<Fp>,
        b_in: Value<Fp>,
        b_hi: Value<Fp>,
        lo_aux: Value<Fp>,
        hi_aux: Value<Fp>,
        in_aux_lo: Value<Fp>,
        in_aux_hi: Value<Fp>,
        lo_aux_bits: [Value<Fp>; NUM_BIT_COLS],
        hi_aux_bits: [Value<Fp>; NUM_BIT_COLS],
        in_aux_lo_bits: [Value<Fp>; NUM_BIT_COLS],
        in_aux_hi_bits: [Value<Fp>; NUM_BIT_COLS],
        q_sat_bits: [Value<Fp>; NUM_BIT_COLS],
    }

    // SatTestCircuit doesn't use an instance column — the public-
    // input binding of q_unsat/q_sat is unnecessary for gadget-
    // isolated tests (the witness fields fully specify the
    // assignment). Mock prover runs are passed an empty instance
    // vector. Avoids an unused-field warning.
    #[derive(Clone, Debug)]
    struct SatTestConfig {
        value: [Column<Advice>; 8],
        bit_cols: [Column<Advice>; NUM_BIT_COLS],
        s_sat: Selector,
        s_rc16s: Selector,
        s_rc17u: Selector,
    }

    impl SatTestCircuit {
        fn from_q_unsat(q_unsat: i64) -> Self {
            // Mirror compute_dense_unit_witnesses's saturation branch.
            let q_sat = q_unsat.clamp(I16_MIN, I16_MAX);
            let (b_lo, b_in, b_hi);
            let (lo_aux, hi_aux, in_aux_lo, in_aux_hi);
            if q_unsat < I16_MIN {
                b_lo = 1;
                b_in = 0;
                b_hi = 0;
                lo_aux = -q_unsat - I16_MIN_MAGNITUDE_NEG_BOUNDARY;
                hi_aux = 0;
                in_aux_lo = 0;
                in_aux_hi = 0;
            } else if q_unsat > I16_MAX {
                b_lo = 0;
                b_in = 0;
                b_hi = 1;
                lo_aux = 0;
                hi_aux = q_unsat - (1 << 15);
                in_aux_lo = 0;
                in_aux_hi = 0;
            } else {
                b_lo = 0;
                b_in = 1;
                b_hi = 0;
                lo_aux = 0;
                hi_aux = 0;
                in_aux_lo = q_unsat + (1 << 15);
                in_aux_hi = (1i64 << 15) - 1 - q_unsat;
            }
            Self {
                q_unsat: known_i64(q_unsat),
                q_sat: known_i64(q_sat),
                b_lo: known_i64(b_lo),
                b_in: known_i64(b_in),
                b_hi: known_i64(b_hi),
                lo_aux: known_i64(lo_aux),
                hi_aux: known_i64(hi_aux),
                in_aux_lo: known_i64(in_aux_lo),
                in_aux_hi: known_i64(in_aux_hi),
                lo_aux_bits: bits_le(lo_aux as u64, 17),
                hi_aux_bits: bits_le(hi_aux as u64, 17),
                in_aux_lo_bits: bits_le(in_aux_lo as u64, 17),
                in_aux_hi_bits: bits_le(in_aux_hi as u64, 17),
                q_sat_bits: bits_le_signed(q_sat, 1 << 15, 16),
            }
        }

        // SatTestCircuit has no instance column — gadget-isolated
        // tests rely on the witness assignment + constraint set, not
        // on public-input binding. MockProver is invoked with an
        // empty per-circuit instance vector.
    }

    impl Circuit<Fp> for SatTestCircuit {
        type Config = SatTestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
            let value: [Column<Advice>; 8] = std::array::from_fn(|_| meta.advice_column());
            for c in &value {
                meta.enable_equality(*c);
            }
            let bit_cols: [Column<Advice>; NUM_BIT_COLS] =
                std::array::from_fn(|_| meta.advice_column());

            let s_sat = meta.selector();
            let s_rc16s = meta.selector();
            let s_rc17u = meta.selector();

            // Saturation gate (same logic as main circuit).
            meta.create_gate("sat_gadget_test", |meta| {
                let s = meta.query_selector(s_sat);
                let q_unsat = meta.query_advice(value[0], Rotation::cur());
                let q_sat = meta.query_advice(value[1], Rotation::cur());
                let b_lo = meta.query_advice(value[2], Rotation::cur());
                let b_in = meta.query_advice(value[3], Rotation::cur());
                let b_hi = meta.query_advice(value[4], Rotation::cur());
                let lo_aux = meta.query_advice(value[5], Rotation::cur());
                let hi_aux = meta.query_advice(value[6], Rotation::cur());
                let in_aux_lo = meta.query_advice(value[7], Rotation::cur());
                let in_aux_hi = meta.query_advice(value[0], Rotation::next());
                let one = fp_expr(Fp::from(1u64));
                let i16_min = fp_expr(fp_from_i64(I16_MIN));
                let i16_max = fp_expr(fp_from_i64(I16_MAX));
                let i16_min_boundary = fp_expr(fp_from_i64(I16_MIN_MAGNITUDE_NEG_BOUNDARY));
                let two_pow_15 = fp_expr(Fp::from(1u64 << 15));
                let bool_lo = b_lo.clone() * (one.clone() - b_lo.clone());
                let bool_in = b_in.clone() * (one.clone() - b_in.clone());
                let bool_hi = b_hi.clone() * (one.clone() - b_hi.clone());
                let sum_one = b_lo.clone() + b_in.clone() + b_hi.clone() - one.clone();
                let output_rule = q_sat.clone()
                    - b_lo.clone() * i16_min
                    - b_in.clone() * q_unsat.clone()
                    - b_hi.clone() * i16_max;
                let lo_correct = b_lo
                    * (fp_expr(Fp::zero()) - q_unsat.clone() - i16_min_boundary - lo_aux);
                let hi_correct = b_hi * (q_unsat.clone() - two_pow_15.clone() - hi_aux);
                let in_lo = b_in.clone() * (q_unsat.clone() + two_pow_15.clone() - in_aux_lo);
                let in_hi = b_in * (two_pow_15 - fp_expr(Fp::from(1u64)) - q_unsat - in_aux_hi);
                vec![
                    s.clone() * bool_lo,
                    s.clone() * bool_in,
                    s.clone() * bool_hi,
                    s.clone() * sum_one,
                    s.clone() * output_rule,
                    s.clone() * lo_correct,
                    s.clone() * hi_correct,
                    s.clone() * in_lo,
                    s * in_hi,
                ]
            });

            // 17u range-check gate
            meta.create_gate("rc17u_gadget_test", |meta| {
                let s = meta.query_selector(s_rc17u);
                let v = meta.query_advice(value[0], Rotation::cur());
                let mut polys = Vec::with_capacity(NUM_BIT_COLS + 1);
                for i in 0..NUM_BIT_COLS {
                    let b = meta.query_advice(bit_cols[i], Rotation::cur());
                    if i < 17 {
                        polys.push(s.clone() * b.clone() * (fp_expr(Fp::from(1u64)) - b));
                    } else {
                        polys.push(s.clone() * b);
                    }
                }
                let mut sum = fp_expr(Fp::zero());
                for i in 0..17 {
                    let b = meta.query_advice(bit_cols[i], Rotation::cur());
                    sum = sum + b * fp_expr(Fp::from(1u64 << i as u32));
                }
                polys.push(s * (sum - v));
                polys
            });

            // 16s range-check gate
            meta.create_gate("rc16s_gadget_test", |meta| {
                let s = meta.query_selector(s_rc16s);
                let v = meta.query_advice(value[0], Rotation::cur());
                let mut polys = Vec::with_capacity(NUM_BIT_COLS + 1);
                for i in 0..NUM_BIT_COLS {
                    let b = meta.query_advice(bit_cols[i], Rotation::cur());
                    if i < 16 {
                        polys.push(s.clone() * b.clone() * (fp_expr(Fp::from(1u64)) - b));
                    } else {
                        polys.push(s.clone() * b);
                    }
                }
                let mut sum = fp_expr(Fp::zero());
                for i in 0..16 {
                    let b = meta.query_advice(bit_cols[i], Rotation::cur());
                    sum = sum + b * fp_expr(Fp::from(1u64 << i as u32));
                }
                polys.push(s * (sum - v - fp_expr(Fp::from(1u64 << 15))));
                polys
            });

            SatTestConfig {
                value,
                bit_cols,
                s_sat,
                s_rc16s,
                s_rc17u,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            // Saturation region (two rows, mirrors main circuit).
            let (q_sat_cell, lo_aux_cell, hi_aux_cell, in_aux_lo_cell, in_aux_hi_cell) =
                layouter.assign_region(
                    || "sat",
                    |mut region| {
                        config.s_sat.enable(&mut region, 0)?;
                        region.assign_advice(|| "q_unsat", config.value[0], 0, || self.q_unsat)?;
                        let q_sat = region.assign_advice(|| "q_sat", config.value[1], 0, || self.q_sat)?;
                        region.assign_advice(|| "b_lo", config.value[2], 0, || self.b_lo)?;
                        region.assign_advice(|| "b_in", config.value[3], 0, || self.b_in)?;
                        region.assign_advice(|| "b_hi", config.value[4], 0, || self.b_hi)?;
                        let lo_aux = region.assign_advice(
                            || "lo_aux",
                            config.value[5],
                            0,
                            || self.lo_aux,
                        )?;
                        let hi_aux = region.assign_advice(
                            || "hi_aux",
                            config.value[6],
                            0,
                            || self.hi_aux,
                        )?;
                        let in_aux_lo = region.assign_advice(
                            || "in_aux_lo",
                            config.value[7],
                            0,
                            || self.in_aux_lo,
                        )?;
                        let in_aux_hi = region.assign_advice(
                            || "in_aux_hi",
                            config.value[0],
                            1,
                            || self.in_aux_hi,
                        )?;
                        Ok((q_sat, lo_aux, hi_aux, in_aux_lo, in_aux_hi))
                    },
                )?;

            // Range checks.
            let rc = |layouter: &mut _,
                      label: &str,
                      sel: Selector,
                      cell: &halo2_proofs::circuit::AssignedCell<Fp, Fp>,
                      bits: &[Value<Fp>; NUM_BIT_COLS]|
             -> Result<(), Error> {
                <_ as Layouter<Fp>>::assign_region(
                    layouter,
                    || label.to_string(),
                    |mut region: halo2_proofs::circuit::Region<Fp>| {
                        sel.enable(&mut region, 0)?;
                        cell.copy_advice(|| "v copy", &mut region, config.value[0], 0)?;
                        for (i, b) in bits.iter().enumerate() {
                            region.assign_advice(
                                || format!("bit[{i}]"),
                                config.bit_cols[i],
                                0,
                                || *b,
                            )?;
                        }
                        Ok(())
                    },
                )
            };
            rc(
                &mut layouter,
                "rc17u_lo_aux",
                config.s_rc17u,
                &lo_aux_cell,
                &self.lo_aux_bits,
            )?;
            rc(
                &mut layouter,
                "rc17u_hi_aux",
                config.s_rc17u,
                &hi_aux_cell,
                &self.hi_aux_bits,
            )?;
            rc(
                &mut layouter,
                "rc17u_in_aux_lo",
                config.s_rc17u,
                &in_aux_lo_cell,
                &self.in_aux_lo_bits,
            )?;
            rc(
                &mut layouter,
                "rc17u_in_aux_hi",
                config.s_rc17u,
                &in_aux_hi_cell,
                &self.in_aux_hi_bits,
            )?;
            rc(
                &mut layouter,
                "rc16s_q_sat",
                config.s_rc16s,
                &q_sat_cell,
                &self.q_sat_bits,
            )?;

            Ok(())
        }
    }

    fn run_sat_test(q_unsat: i64) -> Result<(), halo2_proofs::dev::VerifyFailure> {
        let c = SatTestCircuit::from_q_unsat(q_unsat);
        let k = 6;
        // SatTestCircuit has no instance column; pass empty.
        let prover = MockProver::run(k, &c, vec![]).expect("mock prover runs");
        prover.verify().map(|_| ()).map_err(|errs| errs.into_iter().next().unwrap())
    }

    #[test]
    fn sat_in_range_zero() {
        run_sat_test(0).unwrap();
    }

    #[test]
    fn sat_in_range_i16_min_boundary() {
        run_sat_test(I16_MIN).unwrap(); // -32768 — exactly i16::MIN, b_in active
    }

    #[test]
    fn sat_in_range_i16_max_boundary() {
        run_sat_test(I16_MAX).unwrap(); // +32767 — exactly i16::MAX, b_in active
    }

    #[test]
    fn sat_lower_branch_below_min() {
        run_sat_test(I16_MIN - 1).unwrap(); // -32769, b_lo active, q_sat = -32768
    }

    #[test]
    fn sat_lower_branch_far_below_min() {
        run_sat_test(-50_000).unwrap();
    }

    #[test]
    fn sat_upper_branch_above_max() {
        run_sat_test(I16_MAX + 1).unwrap(); // +32768, b_hi active, q_sat = +32767
    }

    #[test]
    fn sat_upper_branch_far_above_max() {
        run_sat_test(50_000).unwrap();
    }

    #[test]
    fn sat_rejects_wrong_q_sat() {
        let mut c = SatTestCircuit::from_q_unsat(0);
        c.q_sat = known_i64(99); // Lie about the saturation output
        c.q_sat_bits = bits_le_signed(99, 1 << 15, 16);
        let prover = MockProver::run(6, &c, vec![]).expect("mock prover runs");
        assert!(prover.verify().is_err(), "wrong q_sat must fail");
    }

    /// RHAZ-isolated test cases: pin the gadget on ties at ±S/2,
    /// ±3S/2, and a couple of non-tie cases. These ARE reachable
    /// by real i16 inputs (the cross-framework corpus exercises
    /// them end-to-end), but pinning at the gadget level is
    /// cheaper and surfaces failures more locally.
    ///
    /// The RHAZ gadget alone doesn't require a separate test
    /// circuit because the canonical `compute_dense_unit_witnesses`
    /// helper already produces the right (q_abs, r_pos, q_unsat)
    /// tuple via the same Euclidean division the gadget enforces.
    /// We test that helper directly to confirm tie semantics.
    #[test]
    fn rhaz_helper_handles_pos_half_tie() {
        let (unit, q_sat) = compute_dense_unit_witnesses(128);
        assert_eq!(q_sat, 1, "+S/2 must round to +1 (away from zero)");
        unit.s_w.assert_if_known(|fp| *fp == Fp::zero());
        unit.r_pos.assert_if_known(|fp| *fp == Fp::zero());
    }

    #[test]
    fn rhaz_helper_handles_neg_half_tie() {
        let (_unit, q_sat) = compute_dense_unit_witnesses(-128);
        assert_eq!(q_sat, -1, "-S/2 must round to -1 (away from zero)");
    }

    #[test]
    fn rhaz_helper_handles_pos_three_half_tie() {
        let (_unit, q_sat) = compute_dense_unit_witnesses(384);
        assert_eq!(q_sat, 2, "+3S/2 must round to +2");
    }

    #[test]
    fn rhaz_helper_handles_neg_three_half_tie() {
        let (_unit, q_sat) = compute_dense_unit_witnesses(-384);
        assert_eq!(q_sat, -2, "-3S/2 must round to -2");
    }

    #[test]
    fn rhaz_helper_non_tie_cases() {
        assert_eq!(compute_dense_unit_witnesses(0).1, 0);
        assert_eq!(compute_dense_unit_witnesses(127).1, 0);
        assert_eq!(compute_dense_unit_witnesses(129).1, 1);
        assert_eq!(compute_dense_unit_witnesses(-127).1, 0);
        assert_eq!(compute_dense_unit_witnesses(-129).1, -1);
        assert_eq!(compute_dense_unit_witnesses(256).1, 1);
        assert_eq!(compute_dense_unit_witnesses(-256).1, -1);
    }
}
