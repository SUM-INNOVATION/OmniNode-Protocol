//! Stage 11d.2 — halo2 circuit for the production fixed-point MLP
//! (`16 → 32 → 16 → 8`) with arbitrary-input soundness under the
//! frozen `production-fixedpoint-mlp-v1 / spec_version: 1` numeric
//! contract.
//!
//! Adapted from `omni-proofs-halo2-reference::circuit` (Stage 11c
//! bounded reference). The gadget chain is identical — RHAZ
//! (round-half-away-from-zero), three-branch saturation, ReLU
//! sign-bit, and bit-decomposition range checks — but:
//!
//! - The dense layout uses `VALUE_COLS = 16` (vs the reference's
//!   `8`) so Layer-1's 16-wide input fits in a single row.
//! - Three dense layers (vs the reference's two), connected by
//!   two ReLU layers (32 then 16 units).
//! - Public-input shape: 16 inputs + 8 outputs = 24 instance rows.
//!
//! The gadgets themselves (RHAZ + saturation + ReLU + range-checks)
//! are **byte-identical** to Stage 11c per the Stage 11d.2 plan
//! §5.1 Option A — copy-pasted into this crate for clean
//! distinguishability from the bounded reference and zero risk to
//! Stage 11c byte-stability.
//!
//! **Off-chain only.** This circuit is reachable from `omni-node`
//! exclusively under the `stage11d-production-verify` opt-in
//! feature. Mainnet posture: `MAINNET_APPROVED_PROOF_SYSTEM_ENTRIES`
//! is empty through Stage 11d.2; mainnet eligibility registry
//! membership for this proof class is a Stage 11d.3 deliverable.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    pasta::Fp,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector},
    poly::Rotation,
};

use crate::canonical::{B1, B2, B3, W1, W2, W3};

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

/// The lower saturation boundary expressed as `2^15 + 1`. When the
/// `b_lo` branch fires, `lo_aux = -q_unsat - 2^15 - 1`; the
/// canonical formula expects this constant.
const I16_MIN_MAGNITUDE_NEG_BOUNDARY: i64 = (1 << 15) + 1;

// ── Column allocation ─────────────────────────────────────────────
//
// 16 general-purpose value columns (wide enough to fit Layer-1's
// 16-input row in a single anchor row; also wide enough to hold the
// 16 Layer-2 weights in a single row alongside Layer-1's 32
// post-ReLU outputs split across two anchor rows).
const NUM_BIT_COLS: usize = 24;
const VALUE_COLS: usize = 16;

// Layer dims (mirror canonical_spec.json's architecture).
const L1_IN: usize = 16;
const L1_OUT: usize = 32;
const L2_IN: usize = 32;
const L2_OUT: usize = 16;
const L3_IN: usize = 16;
const L3_OUT: usize = 8;

/// Circuit configuration — column allocation + selector definitions.
#[derive(Clone, Debug)]
pub struct ProductionMlpConfig {
    value: [Column<Advice>; VALUE_COLS],
    bit_cols: [Column<Advice>; NUM_BIT_COLS],
    instance: Column<Instance>,

    // Dense linear identity (per layer).
    s_dense_layer1: Selector,
    s_dense_layer2: Selector,
    s_dense_layer3: Selector,
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

/// Per-dense-output witnesses. Same shape as Stage 11c — one
/// witness bundle per dense output `j` across all three layers
/// (Layer 1 has 32, Layer 2 has 16, Layer 3 has 8 — total 56
/// units in the production circuit).
#[derive(Clone, Debug)]
pub struct DenseUnitWitness {
    pub w: Value<Fp>,
    pub s_w: Value<Fp>,
    pub abs_w: Value<Fp>,
    pub q_abs: Value<Fp>,
    pub r_pos: Value<Fp>,
    pub q_unsat: Value<Fp>,
    pub q_sat: Value<Fp>,
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

/// The Stage 11d.2 halo2 circuit.
#[derive(Clone, Debug)]
pub struct ProductionMlpCircuit {
    pub input: [Value<Fp>; L1_IN],
    pub output: [Value<Fp>; L3_OUT],
    pub layer1: [DenseUnitWitness; L1_OUT],
    pub relu1_sign_bits: [Value<Fp>; L1_OUT],
    pub relu1_magnitudes: [Value<Fp>; L1_OUT],
    pub relu1_magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; L1_OUT],
    pub hidden1_post_relu: [Value<Fp>; L1_OUT],
    pub layer2: [DenseUnitWitness; L2_OUT],
    pub relu2_sign_bits: [Value<Fp>; L2_OUT],
    pub relu2_magnitudes: [Value<Fp>; L2_OUT],
    pub relu2_magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; L2_OUT],
    pub hidden2_post_relu: [Value<Fp>; L2_OUT],
    pub layer3: [DenseUnitWitness; L3_OUT],
}

impl Default for ProductionMlpCircuit {
    fn default() -> Self {
        Self {
            input: [Value::unknown(); L1_IN],
            output: [Value::unknown(); L3_OUT],
            layer1: std::array::from_fn(|_| DenseUnitWitness::default()),
            relu1_sign_bits: [Value::unknown(); L1_OUT],
            relu1_magnitudes: [Value::unknown(); L1_OUT],
            relu1_magnitude_bits: [[Value::unknown(); NUM_BIT_COLS]; L1_OUT],
            hidden1_post_relu: [Value::unknown(); L1_OUT],
            layer2: std::array::from_fn(|_| DenseUnitWitness::default()),
            relu2_sign_bits: [Value::unknown(); L2_OUT],
            relu2_magnitudes: [Value::unknown(); L2_OUT],
            relu2_magnitude_bits: [[Value::unknown(); NUM_BIT_COLS]; L2_OUT],
            hidden2_post_relu: [Value::unknown(); L2_OUT],
            layer3: std::array::from_fn(|_| DenseUnitWitness::default()),
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

impl ProductionMlpCircuit {
    /// Build a fully-populated circuit by running the canonical
    /// evaluator on `input_i16`. Developer-host use only.
    ///
    /// Halts (`panic!`) if any intermediate value violates the
    /// bounds the gadgets expect (e.g., `|with_bias| ≥ 2^23`,
    /// `q_abs ≥ 2^16`). This is the "halt and report" rule the
    /// Stage 11c plan committed to and is preserved here: if the
    /// spec/evaluator produces an out-of-bound value, the circuit
    /// cannot prove it and the caller must report rather than
    /// patching.
    pub fn from_canonical_input(input_i16: [i16; L1_IN]) -> Self {
        // ── Layer 1 ───────────────────────────────────────────
        let mut layer1: [DenseUnitWitness; L1_OUT] =
            std::array::from_fn(|_| DenseUnitWitness::default());
        let mut hidden1_pre_relu = [0i64; L1_OUT];
        for j in 0..L1_OUT {
            let acc: i64 = (0..L1_IN)
                .map(|i| (input_i16[i] as i64) * (W1[i][j] as i64))
                .sum();
            let w = acc + (B1[j] as i64) * SCALE;
            let (unit, q_sat) = compute_dense_unit_witnesses(w);
            layer1[j] = unit;
            hidden1_pre_relu[j] = q_sat;
        }

        // ── ReLU 1 ────────────────────────────────────────────
        let mut relu1_sign_bits = [0i64; L1_OUT];
        let mut relu1_magnitudes = [0i64; L1_OUT];
        let mut hidden1_post_relu = [0i64; L1_OUT];
        for j in 0..L1_OUT {
            let x = hidden1_pre_relu[j];
            if x >= 0 {
                relu1_sign_bits[j] = 0;
                relu1_magnitudes[j] = x;
                hidden1_post_relu[j] = x;
            } else {
                relu1_sign_bits[j] = 1;
                relu1_magnitudes[j] = -x;
                hidden1_post_relu[j] = 0;
            }
        }
        let relu1_magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; L1_OUT] =
            std::array::from_fn(|j| bits_le(relu1_magnitudes[j] as u64, 15));

        // ── Layer 2 ───────────────────────────────────────────
        let mut layer2: [DenseUnitWitness; L2_OUT] =
            std::array::from_fn(|_| DenseUnitWitness::default());
        let mut hidden2_pre_relu = [0i64; L2_OUT];
        for j in 0..L2_OUT {
            let acc: i64 = (0..L2_IN)
                .map(|i| (hidden1_post_relu[i] as i64) * (W2[i][j] as i64))
                .sum();
            let w = acc + (B2[j] as i64) * SCALE;
            let (unit, q_sat) = compute_dense_unit_witnesses(w);
            layer2[j] = unit;
            hidden2_pre_relu[j] = q_sat;
        }

        // ── ReLU 2 ────────────────────────────────────────────
        let mut relu2_sign_bits = [0i64; L2_OUT];
        let mut relu2_magnitudes = [0i64; L2_OUT];
        let mut hidden2_post_relu = [0i64; L2_OUT];
        for j in 0..L2_OUT {
            let x = hidden2_pre_relu[j];
            if x >= 0 {
                relu2_sign_bits[j] = 0;
                relu2_magnitudes[j] = x;
                hidden2_post_relu[j] = x;
            } else {
                relu2_sign_bits[j] = 1;
                relu2_magnitudes[j] = -x;
                hidden2_post_relu[j] = 0;
            }
        }
        let relu2_magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; L2_OUT] =
            std::array::from_fn(|j| bits_le(relu2_magnitudes[j] as u64, 15));

        // ── Layer 3 ───────────────────────────────────────────
        let mut layer3: [DenseUnitWitness; L3_OUT] =
            std::array::from_fn(|_| DenseUnitWitness::default());
        let mut output = [0i64; L3_OUT];
        for j in 0..L3_OUT {
            let acc: i64 = (0..L3_IN)
                .map(|i| (hidden2_post_relu[i] as i64) * (W3[i][j] as i64))
                .sum();
            let w = acc + (B3[j] as i64) * SCALE;
            let (unit, q_sat) = compute_dense_unit_witnesses(w);
            layer3[j] = unit;
            output[j] = q_sat;
        }

        Self {
            input: std::array::from_fn(|i| known_i64(input_i16[i] as i64)),
            output: std::array::from_fn(|j| known_i64(output[j])),
            layer1,
            relu1_sign_bits: std::array::from_fn(|j| known_i64(relu1_sign_bits[j])),
            relu1_magnitudes: std::array::from_fn(|j| known_i64(relu1_magnitudes[j])),
            relu1_magnitude_bits,
            hidden1_post_relu: std::array::from_fn(|j| known_i64(hidden1_post_relu[j])),
            layer2,
            relu2_sign_bits: std::array::from_fn(|j| known_i64(relu2_sign_bits[j])),
            relu2_magnitudes: std::array::from_fn(|j| known_i64(relu2_magnitudes[j])),
            relu2_magnitude_bits,
            hidden2_post_relu: std::array::from_fn(|j| known_i64(hidden2_post_relu[j])),
            layer3,
        }
    }

    pub fn canonical_outputs_for(input_i16: [i16; L1_IN]) -> [i16; L3_OUT] {
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
    let shifted = (abs_w as u64) + HALF_SCALE as u64;
    let q_abs = (shifted / (SCALE as u64)) as i64;
    let r_pos = (shifted % (SCALE as u64)) as i64;
    assert!(r_pos >= 0 && r_pos < SCALE);
    assert!(
        q_abs >= 0 && q_abs < (1 << 16),
        "q_abs out of 16-bit unsigned bound: {q_abs}"
    );
    let q_unsat = if s_w == 0 { q_abs } else { -q_abs };

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

impl Circuit<Fp> for ProductionMlpCircuit {
    type Config = ProductionMlpConfig;
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
        let s_dense_layer3 = meta.selector();
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
        //   row 0: value[0..16] = inputs[0..16]
        //   row 1: value[0..16] = w_layer1[0..16]
        //   row 2: value[0..16] = w_layer1[16..32]
        meta.create_gate("dense_layer1", |meta| {
            let s = meta.query_selector(s_dense_layer1);
            let inputs: [Expression<Fp>; L1_IN] =
                std::array::from_fn(|i| meta.query_advice(value[i], Rotation::cur()));
            let mut polys = Vec::with_capacity(L1_OUT);
            for j in 0..L1_OUT {
                let (col, rot) = if j < VALUE_COLS {
                    (j, Rotation(1))
                } else {
                    (j - VALUE_COLS, Rotation(2))
                };
                let w_j = meta.query_advice(value[col], rot);
                let mut sum = w_j;
                for i in 0..L1_IN {
                    sum = sum - inputs[i].clone() * fp_expr(fp_from_i64(W1[i][j] as i64));
                }
                sum = sum - fp_expr(fp_from_i64(B1[j] as i64) * scale_fp);
                polys.push(s.clone() * sum);
            }
            polys
        });

        // ── Dense Layer 2 ───────────────────────────────────────────
        // Row layout:
        //   row 0: value[0..16] = post_relu1[0..16]
        //   row 1: value[0..16] = post_relu1[16..32]
        //   row 2: value[0..16] = w_layer2[0..16]
        meta.create_gate("dense_layer2", |meta| {
            let s = meta.query_selector(s_dense_layer2);
            let pr: [Expression<Fp>; L2_IN] = std::array::from_fn(|i| {
                let (col, rot) = if i < VALUE_COLS {
                    (i, Rotation::cur())
                } else {
                    (i - VALUE_COLS, Rotation(1))
                };
                meta.query_advice(value[col], rot)
            });
            let mut polys = Vec::with_capacity(L2_OUT);
            for j in 0..L2_OUT {
                let w_j = meta.query_advice(value[j], Rotation(2));
                let mut sum = w_j;
                for i in 0..L2_IN {
                    sum = sum - pr[i].clone() * fp_expr(fp_from_i64(W2[i][j] as i64));
                }
                sum = sum - fp_expr(fp_from_i64(B2[j] as i64) * scale_fp);
                polys.push(s.clone() * sum);
            }
            polys
        });

        // ── Dense Layer 3 ───────────────────────────────────────────
        // Row layout:
        //   row 0: value[0..16] = post_relu2[0..16]
        //   row 1: value[0..8]  = w_layer3[0..8]
        meta.create_gate("dense_layer3", |meta| {
            let s = meta.query_selector(s_dense_layer3);
            let pr: [Expression<Fp>; L3_IN] =
                std::array::from_fn(|i| meta.query_advice(value[i], Rotation::cur()));
            let mut polys = Vec::with_capacity(L3_OUT);
            for j in 0..L3_OUT {
                let w_j = meta.query_advice(value[j], Rotation::next());
                let mut sum = w_j;
                for i in 0..L3_IN {
                    sum = sum - pr[i].clone() * fp_expr(fp_from_i64(W3[i][j] as i64));
                }
                sum = sum - fp_expr(fp_from_i64(B3[j] as i64) * scale_fp);
                polys.push(s.clone() * sum);
            }
            polys
        });

        // ── RHAZ gadget ─────────────────────────────────────────────
        // Single-row anchor:
        //   value[0] = w, value[1] = s_w, value[2] = abs_w,
        //   value[3] = q_abs, value[4] = r_pos, value[5] = q_unsat
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

            let sign_decomp = w - (one.clone() - two.clone() * s_w.clone()) * abs_w.clone();
            let sign_bool = s_w.clone() * (one.clone() - s_w.clone());
            let euclid =
                abs_w + fp_expr(half_scale_fp) - q_abs.clone() * fp_expr(scale_fp) - r_pos;
            let signed_quot = q_unsat - (one - two * s_w) * q_abs;
            vec![
                s.clone() * sign_decomp,
                s.clone() * sign_bool,
                s.clone() * euclid,
                s * signed_quot,
            ]
        });

        // ── Saturation gadget ───────────────────────────────────────
        // Two-row anchor (row 0 + row 1 for in_aux_hi).
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
            let i16_min_neg_boundary = fp_expr(fp_from_i64(I16_MIN_MAGNITUDE_NEG_BOUNDARY));
            let two_pow_15 = fp_expr(Fp::from(1u64 << 15));

            let bool_lo = b_lo.clone() * (one.clone() - b_lo.clone());
            let bool_in = b_in.clone() * (one.clone() - b_in.clone());
            let bool_hi = b_hi.clone() * (one.clone() - b_hi.clone());
            let sum_one = b_lo.clone() + b_in.clone() + b_hi.clone() - one.clone();
            let output_rule = q_sat
                - b_lo.clone() * i16_min
                - b_in.clone() * q_unsat.clone()
                - b_hi.clone() * i16_max;
            let lo_correct =
                b_lo * (fp_expr(Fp::zero()) - q_unsat.clone() - i16_min_neg_boundary - lo_aux);
            let hi_correct = b_hi * (q_unsat.clone() - two_pow_15.clone() - hi_aux);
            let in_lo_correct = b_in.clone() * (q_unsat.clone() + two_pow_15.clone() - in_aux_lo);
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
        // Single-row anchor:
        //   value[0] = pre_relu, value[1] = sign,
        //   value[2] = magnitude, value[3] = post_relu
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
            vec![s.clone() * booleanity, s.clone() * decomposition, s * relu_eq]
        });

        // ── Range checks (helper) ───────────────────────────────────
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
                for i in 0..NUM_BIT_COLS {
                    let b = meta.query_advice(bit_cols[i], Rotation::cur());
                    if i < width {
                        polys.push(s.clone() * b.clone() * (fp_expr(Fp::from(1u64)) - b));
                    } else {
                        polys.push(s.clone() * b);
                    }
                }
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

        ProductionMlpConfig {
            value,
            bit_cols,
            instance,
            s_dense_layer1,
            s_dense_layer2,
            s_dense_layer3,
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
        // Row 0: inputs[0..16] | Row 1: w_layer1[0..16] | Row 2: w_layer1[16..32]
        let (input_cells, w_layer1_cells) = layouter.assign_region(
            || "dense_layer1",
            |mut region| {
                config.s_dense_layer1.enable(&mut region, 0)?;
                let input_cells: Vec<_> = (0..L1_IN)
                    .map(|i| {
                        region.assign_advice(
                            || format!("input[{i}]"),
                            config.value[i],
                            0,
                            || self.input[i],
                        )
                    })
                    .collect::<Result<_, _>>()?;
                let mut w_cells = Vec::with_capacity(L1_OUT);
                for j in 0..L1_OUT {
                    let (col, row) = if j < VALUE_COLS {
                        (j, 1)
                    } else {
                        (j - VALUE_COLS, 2)
                    };
                    let cell = region.assign_advice(
                        || format!("w_layer1[{j}]"),
                        config.value[col],
                        row,
                        || self.layer1[j].w,
                    )?;
                    w_cells.push(cell);
                }
                Ok((input_cells, w_cells))
            },
        )?;

        // ── RHAZ + saturation + range checks for each Layer 1 j ─────
        let mut q_sat_layer1_cells = Vec::with_capacity(L1_OUT);
        for j in 0..L1_OUT {
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

        // ── ReLU 1 (32 single-row regions) ──────────────────────────
        let mut post_relu1_cells = Vec::with_capacity(L1_OUT);
        for j in 0..L1_OUT {
            let pre = q_sat_layer1_cells[j].clone();
            let magnitude_bits = self.relu1_magnitude_bits[j];
            let (post, mag_cell) = layouter.assign_region(
                || format!("relu1[{j}]"),
                |mut region| {
                    config.s_relu.enable(&mut region, 0)?;
                    pre.copy_advice(
                        || format!("pre_relu1[{j}] copy"),
                        &mut region,
                        config.value[0],
                        0,
                    )?;
                    region.assign_advice(
                        || format!("sign1[{j}]"),
                        config.value[1],
                        0,
                        || self.relu1_sign_bits[j],
                    )?;
                    let mag = region.assign_advice(
                        || format!("magnitude1[{j}]"),
                        config.value[2],
                        0,
                        || self.relu1_magnitudes[j],
                    )?;
                    let post = region.assign_advice(
                        || format!("post_relu1[{j}]"),
                        config.value[3],
                        0,
                        || self.hidden1_post_relu[j],
                    )?;
                    Ok((post, mag))
                },
            )?;
            assign_range_check(
                &mut layouter,
                &config,
                &format!("rc15u_magnitude1[{j}]"),
                config.s_rc15u,
                &mag_cell,
                &magnitude_bits,
            )?;
            post_relu1_cells.push(post);
        }

        // ── Layer 2 dense identity region ───────────────────────────
        // Row 0: post_relu1[0..16] | Row 1: post_relu1[16..32] | Row 2: w_layer2[0..16]
        let w_layer2_cells = layouter.assign_region(
            || "dense_layer2",
            |mut region| {
                config.s_dense_layer2.enable(&mut region, 0)?;
                for (i, cell) in post_relu1_cells.iter().enumerate() {
                    let (col, row) = if i < VALUE_COLS {
                        (i, 0)
                    } else {
                        (i - VALUE_COLS, 1)
                    };
                    cell.copy_advice(
                        || format!("post_relu1[{i}] copy"),
                        &mut region,
                        config.value[col],
                        row,
                    )?;
                }
                let mut w_cells = Vec::with_capacity(L2_OUT);
                for j in 0..L2_OUT {
                    let cell = region.assign_advice(
                        || format!("w_layer2[{j}]"),
                        config.value[j],
                        2,
                        || self.layer2[j].w,
                    )?;
                    w_cells.push(cell);
                }
                Ok(w_cells)
            },
        )?;

        // ── RHAZ + saturation + range checks for each Layer 2 j ─────
        let mut q_sat_layer2_cells = Vec::with_capacity(L2_OUT);
        for j in 0..L2_OUT {
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

        // ── ReLU 2 (16 single-row regions) ──────────────────────────
        let mut post_relu2_cells = Vec::with_capacity(L2_OUT);
        for j in 0..L2_OUT {
            let pre = q_sat_layer2_cells[j].clone();
            let magnitude_bits = self.relu2_magnitude_bits[j];
            let (post, mag_cell) = layouter.assign_region(
                || format!("relu2[{j}]"),
                |mut region| {
                    config.s_relu.enable(&mut region, 0)?;
                    pre.copy_advice(
                        || format!("pre_relu2[{j}] copy"),
                        &mut region,
                        config.value[0],
                        0,
                    )?;
                    region.assign_advice(
                        || format!("sign2[{j}]"),
                        config.value[1],
                        0,
                        || self.relu2_sign_bits[j],
                    )?;
                    let mag = region.assign_advice(
                        || format!("magnitude2[{j}]"),
                        config.value[2],
                        0,
                        || self.relu2_magnitudes[j],
                    )?;
                    let post = region.assign_advice(
                        || format!("post_relu2[{j}]"),
                        config.value[3],
                        0,
                        || self.hidden2_post_relu[j],
                    )?;
                    Ok((post, mag))
                },
            )?;
            assign_range_check(
                &mut layouter,
                &config,
                &format!("rc15u_magnitude2[{j}]"),
                config.s_rc15u,
                &mag_cell,
                &magnitude_bits,
            )?;
            post_relu2_cells.push(post);
        }

        // ── Layer 3 dense identity region ───────────────────────────
        // Row 0: post_relu2[0..16] | Row 1: w_layer3[0..8]
        let w_layer3_cells = layouter.assign_region(
            || "dense_layer3",
            |mut region| {
                config.s_dense_layer3.enable(&mut region, 0)?;
                for (i, cell) in post_relu2_cells.iter().enumerate() {
                    cell.copy_advice(
                        || format!("post_relu2[{i}] copy"),
                        &mut region,
                        config.value[i],
                        0,
                    )?;
                }
                let mut w_cells = Vec::with_capacity(L3_OUT);
                for j in 0..L3_OUT {
                    let cell = region.assign_advice(
                        || format!("w_layer3[{j}]"),
                        config.value[j],
                        1,
                        || self.layer3[j].w,
                    )?;
                    w_cells.push(cell);
                }
                Ok(w_cells)
            },
        )?;

        // ── RHAZ + saturation + range checks for each Layer 3 j ─────
        let mut q_sat_layer3_cells = Vec::with_capacity(L3_OUT);
        for j in 0..L3_OUT {
            let w_cell = w_layer3_cells[j].clone();
            let unit = &self.layer3[j];
            let q_sat_cell = assign_rhaz_sat_chain(
                &mut layouter,
                &config,
                &format!("layer3[{j}]"),
                w_cell,
                unit,
            )?;
            q_sat_layer3_cells.push(q_sat_cell);
        }

        // ── Public-input binding ────────────────────────────────────
        for (i, cell) in input_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }
        for (j, cell) in q_sat_layer3_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, L1_IN + j)?;
        }

        Ok(())
    }
}

/// Assign one full RHAZ → saturation → range-check chain for a
/// single dense output. Returns the `q_sat` cell so the caller can
/// chain into the next layer or bind to the public output.
fn assign_rhaz_sat_chain(
    layouter: &mut impl Layouter<Fp>,
    config: &ProductionMlpConfig,
    label: &str,
    w_cell: halo2_proofs::circuit::AssignedCell<Fp, Fp>,
    unit: &DenseUnitWitness,
) -> Result<halo2_proofs::circuit::AssignedCell<Fp, Fp>, Error> {
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
                let lo_aux =
                    region.assign_advice(|| "lo_aux", config.value[5], 0, || unit.sat_lo_aux)?;
                let hi_aux =
                    region.assign_advice(|| "hi_aux", config.value[6], 0, || unit.sat_hi_aux)?;
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
    config: &ProductionMlpConfig,
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

    fn instance_for(input: [i16; L1_IN], output: [i16; L3_OUT]) -> Vec<Fp> {
        let mut v = Vec::with_capacity(L1_IN + L3_OUT);
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
        let circuit = ProductionMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        prover
            .verify()
            .expect("canonical assignment must satisfy constraints");
    }

    #[test]
    fn wrong_output_fails_mock_prover() {
        let circuit = ProductionMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let mut wrong = CANONICAL_OUTPUT;
        wrong[0] = wrong[0].wrapping_add(1);
        let instance = instance_for(CANONICAL_INPUT, wrong);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(prover.verify().is_err(), "wrong output must fail");
    }

    #[test]
    fn fp_from_i64_round_trips_through_signed_embedding() {
        let neg5 = fp_from_i64(-5);
        let pos5 = fp_from_i64(5);
        assert_eq!(neg5 + pos5, Fp::zero());
    }

    /// Adversarial regression: tamper with Layer-1's q_abs and r_pos
    /// to make the Euclidean equation balance, then verify the
    /// 8-bit range check on r_pos still catches the lie.
    #[test]
    fn tampered_q_abs_fails_range_check() {
        let mut circuit = ProductionMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let original_q_abs = circuit.layer1[0].q_abs;
        // Bump q_abs by 1, set r_pos to -S so Euclidean stays balanced.
        // r_pos = -256 cannot be decomposed as 8 unsigned bits.
        circuit.layer1[0].q_abs = original_q_abs.map(|v| v + Fp::from(1u64));
        circuit.layer1[0].r_pos = known_i64(-256);
        circuit.layer1[0].r_pos_bits = [Value::known(Fp::zero()); NUM_BIT_COLS];
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(
            prover.verify().is_err(),
            "tampered q_abs + matching r_pos must fail range checks"
        );
    }

    /// Adversarial regression: force the wrong saturation branch
    /// active. Canonical Layer-1 output 0 is in-range (b_in=1);
    /// flipping to b_hi=1 forces q_sat = i16::MAX, which the
    /// canonical witness contradicts.
    #[test]
    fn tampered_saturation_branch_fails() {
        let mut circuit = ProductionMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        circuit.layer1[0].sat_b_in = known_i64(0);
        circuit.layer1[0].sat_b_hi = known_i64(1);
        // hi_aux must satisfy q_unsat - 2^15 - hi_aux = 0 if b_hi=1.
        // For an in-range q_unsat, hi_aux becomes negative -> rc17u rejects.
        circuit.layer1[0].sat_hi_aux = known_i64(-32711);
        circuit.layer1[0].sat_hi_aux_bits = [Value::known(Fp::zero()); NUM_BIT_COLS];
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(
            prover.verify().is_err(),
            "tampered saturation branch must fail"
        );
    }

    #[test]
    fn tampered_saturation_sum_fails() {
        let mut circuit = ProductionMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        // Canonical Layer-1 output 0 has b_in=1; setting b_lo=1 too
        // makes the sum 2, which the sat-sum constraint rejects.
        circuit.layer1[0].sat_b_lo = known_i64(1);
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let prover = MockProver::run(k(), &circuit, vec![instance]).expect("mock prover runs");
        assert!(
            prover.verify().is_err(),
            "two sat selectors active simultaneously must fail"
        );
    }

    #[test]
    fn rhaz_helper_handles_pos_half_tie() {
        let (_unit, q_sat) = compute_dense_unit_witnesses(128);
        assert_eq!(q_sat, 1, "+S/2 must round to +1 (away from zero)");
    }

    #[test]
    fn rhaz_helper_handles_neg_half_tie() {
        let (_unit, q_sat) = compute_dense_unit_witnesses(-128);
        assert_eq!(q_sat, -1, "-S/2 must round to -1 (away from zero)");
    }

    #[test]
    fn rhaz_helper_handles_pos_three_half_tie() {
        let (_unit, q_sat) = compute_dense_unit_witnesses(384);
        assert_eq!(q_sat, 2);
    }

    #[test]
    fn rhaz_helper_handles_neg_three_half_tie() {
        let (_unit, q_sat) = compute_dense_unit_witnesses(-384);
        assert_eq!(q_sat, -2);
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
