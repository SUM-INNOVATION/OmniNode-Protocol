//! Stage 11b.1.b — halo2 circuit for the canonical bounded MLP.
//!
//! Gated by feature `verify` because both the verifier and the
//! developer-host prover share this `Circuit<Fp>` definition.
//!
//! ## Public inputs (`instance` column)
//!
//! ```text
//! row 0..4 : input[0..4]   (canonical i16 input  lifted to Fp)
//! row 4..8 : output[0..4]  (canonical i16 output lifted to Fp)
//! ```
//!
//! ## Witnesses
//!
//! - `hidden_pre_relu[0..8]`  — Layer 1 dense outputs *after*
//!   bias-pre-saturation + round-half-away requantization.
//! - `hidden_post_relu[0..8]` — ReLU outputs.
//! - `r1[0..8]`, `r2[0..4]`  — requantization remainders satisfying
//!   `with_bias = pre_relu * S + r`.
//! - `sign_bits[0..8]`, `magnitudes[0..8]` — ReLU sign-bit gadget
//!   witnesses (`x = (1 - 2*s) * mag`, `out = (1 - s) * mag`).
//! - Bit decompositions for the range checks (assigned by the
//!   prover from the canonical witness values).
//!
//! ## Constraints
//!
//! **Dense + requantize (Layer 1 / Layer 2):**
//!   `pre_relu[j] * S + r[j] = Σ_i (input[i] * W[i][j]) + B[j] * S`
//!   plus a 9-bit signed range check on `r[j]` ensuring
//!   `r[j] ∈ [-S/2, S/2]`. This is what makes the dense gate
//!   sound — without the range check on `r`, the equation is
//!   satisfiable for any `pre_relu` by absorbing slack into `r`.
//!
//! **ReLU sign-bit gadget:**
//!   `s[j] * (1 - s[j]) = 0`
//!   `pre_relu[j] = (1 - 2*s[j]) * magnitude[j]`
//!   `post_relu[j] = (1 - s[j]) * magnitude[j]`
//!   plus a 15-bit unsigned range check on `magnitude[j]` ensuring
//!   `magnitude[j] ∈ [0, 2^15)`. This bounds both `pre_relu[j]`
//!   (via `(1 - 2s) * mag`) and `post_relu[j]` (via `(1 - s) * mag`)
//!   to the i15 range, implicitly asserting Layer 1 did NOT
//!   saturate.
//!
//! **Output range check:**
//!   16-bit signed range check on `output[j]` ensuring
//!   `output[j] ∈ [-2^15, 2^15)`. This implicitly asserts Layer 2
//!   did NOT saturate.
//!
//! **Public boundary:** input[0..4] and output[0..4] equality-
//! constrained to instance column rows 0..8.
//!
//! ## Soundness scope (now documented honestly)
//!
//! The circuit proves `canonical_evaluate(input) == output` for
//! inputs that:
//!   (a) produce no requantization tie cases (i.e. no
//!       `with_bias = ±S/2 · (2k+1)` exact-half-integer outputs);
//!   (b) produce no i16 saturation at the layer outputs.
//!
//! The committed canonical input `[-5, 10, 20, -100]` satisfies
//! both conditions:
//!   * Layer 1 with_biases are 14624, -4432, -2344, -5776, 11192,
//!     -720, -8992, 1608 — none half-integer multiples of S.
//!   * Layer 2 with_biases are 8344, -8304, 4416, 1848 — same.
//!   * No pre_relu or output value reaches ±32768.
//!
//! Stage 11c+ deliverables for general-input soundness:
//!   * Explicit round-half-away tie-break gadget.
//!   * Explicit saturation gadget with three-branch selector.
//!
//! The verifier defends-in-depth by independently running the
//! pure-Rust `canonical_evaluate` and refusing any artifact whose
//! claimed output disagrees — see `src/verifier.rs` step 7.5.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    pasta::Fp,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector},
    poly::Rotation,
};

use crate::canonical::{B1, B2, W1, W2};

/// Convenience: wrap an Fp value into a constant Expression for use
/// inside `create_gate` polynomials.
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

/// Convenience: known `Value<Fp>` from an i64.
pub fn known_i64(x: i64) -> Value<Fp> {
    Value::known(fp_from_i64(x))
}

const NUM_BIT_COLS: usize = 16;

/// Circuit configuration.
#[derive(Clone, Debug)]
pub struct BoundedMlpConfig {
    /// Advice columns for the dense+ReLU regions.
    /// 4 columns: typically holds value/op/result/aux cells.
    advice: [Column<Advice>; 4],
    /// Bit-decomposition advice columns. 16 columns; for range
    /// checks <16 bits the high bits are constrained to be zero
    /// by the gate's polynomial expressions.
    bit_cols: [Column<Advice>; NUM_BIT_COLS],
    /// Instance column carrying the 8 public inputs (4 in + 4 out).
    instance: Column<Instance>,

    /// Layer 1 dense + requantize linear equation.
    s_layer1: Selector,
    /// ReLU sign-bit gadget.
    s_relu: Selector,
    /// Layer 2 dense + requantize linear equation.
    s_layer2: Selector,
    /// 9-bit signed range check (r ∈ [-S/2, S/2]).
    s_rc9: Selector,
    /// 15-bit unsigned range check (magnitude ∈ [0, 2^15)).
    s_rc15: Selector,
    /// 16-bit signed range check (i16 range).
    s_rc16: Selector,
}

/// The Stage 11b.1.b halo2 circuit.
///
/// Holds all witness values. During key generation, all
/// `Value<Fp>` fields are `Value::unknown()`; during proving they're
/// populated by the developer-host prover via
/// [`Self::from_canonical_input`].
#[derive(Clone, Debug)]
pub struct BoundedMlpCircuit {
    pub input: [Value<Fp>; 4],
    pub output: [Value<Fp>; 4],
    pub hidden_pre_relu: [Value<Fp>; 8],
    pub hidden_post_relu: [Value<Fp>; 8],
    pub r1: [Value<Fp>; 8],
    pub r2: [Value<Fp>; 4],
    pub sign_bits: [Value<Fp>; 8],
    pub magnitudes: [Value<Fp>; 8],
    /// Bit decomposition of each `r1[j] + 128` (8 instances × 9 bits).
    pub r1_bits: [[Value<Fp>; NUM_BIT_COLS]; 8],
    /// Bit decomposition of each `r2[j] + 128` (4 instances × 9 bits).
    pub r2_bits: [[Value<Fp>; NUM_BIT_COLS]; 4],
    /// Bit decomposition of each `magnitude[j]` (8 instances × 15 bits).
    pub magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; 8],
    /// Bit decomposition of each `output[j] + 2^15` (4 instances × 16 bits).
    pub output_bits: [[Value<Fp>; NUM_BIT_COLS]; 4],
}

impl Default for BoundedMlpCircuit {
    fn default() -> Self {
        Self {
            input: [Value::unknown(); 4],
            output: [Value::unknown(); 4],
            hidden_pre_relu: [Value::unknown(); 8],
            hidden_post_relu: [Value::unknown(); 8],
            r1: [Value::unknown(); 8],
            r2: [Value::unknown(); 4],
            sign_bits: [Value::unknown(); 8],
            magnitudes: [Value::unknown(); 8],
            r1_bits: [[Value::unknown(); NUM_BIT_COLS]; 8],
            r2_bits: [[Value::unknown(); NUM_BIT_COLS]; 4],
            magnitude_bits: [[Value::unknown(); NUM_BIT_COLS]; 8],
            output_bits: [[Value::unknown(); NUM_BIT_COLS]; 4],
        }
    }
}

/// Decompose `(value + offset)` into `n_bits` LE bits, padding the
/// remaining `NUM_BIT_COLS - n_bits` cells with explicit zeros.
fn bits_with_offset(value: i64, offset: i64, n_bits: usize) -> [Value<Fp>; NUM_BIT_COLS] {
    let shifted = value + offset;
    assert!(
        shifted >= 0 && shifted < (1i64 << n_bits as u32) + ((n_bits == 9) as i64) * 0,
        "value {value} + offset {offset} = {shifted} doesn't fit in {n_bits} bits"
    );
    let mut bits = [Value::known(Fp::zero()); NUM_BIT_COLS];
    for i in 0..n_bits {
        let bit = (shifted >> i) & 1;
        bits[i] = Value::known(Fp::from(bit as u64));
    }
    // For the 9-bit-with-overflow case: shifted can also equal exactly 2^8 = 256
    // (i.e., r = 128 in our canonical scale). bit_8 then = 1, all lower bits 0.
    // Both cases are covered by the `>> i & 1` extraction.
    bits
}

impl BoundedMlpCircuit {
    /// Build a fully-populated circuit from the canonical i16 input.
    /// Runs the canonical evaluator (in i64) to derive every witness.
    /// Developer-host use only.
    pub fn from_canonical_input(input_i16: [i16; 4]) -> Self {
        const S: i64 = 256;

        let mut hidden_pre_relu = [0i64; 8];
        let mut r1 = [0i64; 8];
        for j in 0..8 {
            let acc: i64 = (0..4)
                .map(|i| (input_i16[i] as i64) * (W1[i][j] as i64))
                .sum();
            let with_bias = acc + (B1[j] as i64) * S;
            let q = if with_bias >= 0 {
                (with_bias + S / 2) / S
            } else {
                -(((-with_bias) + S / 2) / S)
            };
            let r = with_bias - q * S;
            hidden_pre_relu[j] = q;
            r1[j] = r;
        }

        let mut hidden_post_relu = [0i64; 8];
        let mut sign_bits = [0i64; 8];
        let mut magnitudes = [0i64; 8];
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

        let mut output = [0i64; 4];
        let mut r2 = [0i64; 4];
        for j in 0..4 {
            let acc: i64 = (0..8)
                .map(|i| (hidden_post_relu[i] as i64) * (W2[i][j] as i64))
                .sum();
            let with_bias = acc + (B2[j] as i64) * S;
            let q = if with_bias >= 0 {
                (with_bias + S / 2) / S
            } else {
                -(((-with_bias) + S / 2) / S)
            };
            let r = with_bias - q * S;
            output[j] = q;
            r2[j] = r;
        }

        // Bit decompositions. Offsets:
        //   r-bits: +128  (range [-128, 128] → [0, 256])
        //   magnitude-bits: 0  (range [0, 2^15))
        //   output-bits: +2^15  (range [-2^15, 2^15) → [0, 2^16))
        let r1_bits: [[Value<Fp>; NUM_BIT_COLS]; 8] =
            std::array::from_fn(|j| bits_with_offset(r1[j], 128, 9));
        let r2_bits: [[Value<Fp>; NUM_BIT_COLS]; 4] =
            std::array::from_fn(|j| bits_with_offset(r2[j], 128, 9));
        let magnitude_bits: [[Value<Fp>; NUM_BIT_COLS]; 8] =
            std::array::from_fn(|j| bits_with_offset(magnitudes[j], 0, 15));
        let output_bits: [[Value<Fp>; NUM_BIT_COLS]; 4] =
            std::array::from_fn(|j| bits_with_offset(output[j], 1 << 15, 16));

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
            hidden_pre_relu: std::array::from_fn(|j| known_i64(hidden_pre_relu[j])),
            hidden_post_relu: std::array::from_fn(|j| known_i64(hidden_post_relu[j])),
            r1: std::array::from_fn(|j| known_i64(r1[j])),
            r2: std::array::from_fn(|j| known_i64(r2[j])),
            sign_bits: std::array::from_fn(|j| known_i64(sign_bits[j])),
            magnitudes: std::array::from_fn(|j| known_i64(magnitudes[j])),
            r1_bits,
            r2_bits,
            magnitude_bits,
            output_bits,
        }
    }

    pub fn canonical_outputs_for(input_i16: [i16; 4]) -> [i16; 4] {
        crate::canonical::canonical_evaluate(input_i16)
    }
}

impl Circuit<Fp> for BoundedMlpCircuit {
    type Config = BoundedMlpConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        for a in &advice {
            meta.enable_equality(*a);
        }

        let bit_cols: [Column<Advice>; NUM_BIT_COLS] =
            std::array::from_fn(|_| meta.advice_column());

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let s_layer1 = meta.selector();
        let s_relu = meta.selector();
        let s_layer2 = meta.selector();
        let s_rc9 = meta.selector();
        let s_rc15 = meta.selector();
        let s_rc16 = meta.selector();

        // ── Layer 1 dense + requantize ──────────────────────────────
        // Row layout (rotation-relative):
        //   row 0: advice[0..4] = input[0..4]
        //   row 1..9: advice[0] = pre_relu[j], advice[1] = r1[j]
        meta.create_gate("layer1_dense", |meta| {
            let s = meta.query_selector(s_layer1);
            let inputs: [_; 4] = std::array::from_fn(|i| {
                meta.query_advice(advice[i], Rotation::cur())
            });
            let scale = Fp::from(256u64);
            let mut polys = Vec::with_capacity(8);
            for j in 0..8 {
                let off = (j as i32) + 1;
                let pre_relu = meta.query_advice(advice[0], Rotation(off));
                let r1 = meta.query_advice(advice[1], Rotation(off));
                let mut sum = pre_relu * fp_expr(scale)
                    + r1
                    - inputs[0].clone() * fp_expr(fp_from_i64(W1[0][j] as i64))
                    - inputs[1].clone() * fp_expr(fp_from_i64(W1[1][j] as i64))
                    - inputs[2].clone() * fp_expr(fp_from_i64(W1[2][j] as i64))
                    - inputs[3].clone() * fp_expr(fp_from_i64(W1[3][j] as i64));
                sum = sum - fp_expr(fp_from_i64(B1[j] as i64) * scale);
                polys.push(s.clone() * sum);
            }
            polys
        });

        // ── ReLU sign-bit gadget ────────────────────────────────────
        meta.create_gate("relu", |meta| {
            let s = meta.query_selector(s_relu);
            let pre_relu = meta.query_advice(advice[0], Rotation::cur());
            let sign = meta.query_advice(advice[1], Rotation::cur());
            let magnitude = meta.query_advice(advice[2], Rotation::cur());
            let post_relu = meta.query_advice(advice[3], Rotation::cur());

            let one = Fp::from(1u64);
            let two = Fp::from(2u64);
            let booleanity = sign.clone() * (fp_expr(one) - sign.clone());
            let decomposition =
                pre_relu - (fp_expr(one) - fp_expr(two) * sign.clone()) * magnitude.clone();
            let relu_eq = post_relu - (fp_expr(one) - sign) * magnitude;
            vec![
                s.clone() * booleanity,
                s.clone() * decomposition,
                s * relu_eq,
            ]
        });

        // ── Layer 2 dense + requantize ──────────────────────────────
        meta.create_gate("layer2_dense", |meta| {
            let s = meta.query_selector(s_layer2);
            let pr0_3: [_; 4] = std::array::from_fn(|i| {
                meta.query_advice(advice[i], Rotation::cur())
            });
            let pr4_7: [_; 4] = std::array::from_fn(|i| {
                meta.query_advice(advice[i], Rotation::next())
            });
            let scale = Fp::from(256u64);
            let mut polys = Vec::with_capacity(4);
            for j in 0..4 {
                let off = 2 + (j as i32);
                let out = meta.query_advice(advice[0], Rotation(off));
                let r2 = meta.query_advice(advice[1], Rotation(off));
                let mut sum = out * fp_expr(scale)
                    + r2
                    - pr0_3[0].clone() * fp_expr(fp_from_i64(W2[0][j] as i64))
                    - pr0_3[1].clone() * fp_expr(fp_from_i64(W2[1][j] as i64))
                    - pr0_3[2].clone() * fp_expr(fp_from_i64(W2[2][j] as i64))
                    - pr0_3[3].clone() * fp_expr(fp_from_i64(W2[3][j] as i64))
                    - pr4_7[0].clone() * fp_expr(fp_from_i64(W2[4][j] as i64))
                    - pr4_7[1].clone() * fp_expr(fp_from_i64(W2[5][j] as i64))
                    - pr4_7[2].clone() * fp_expr(fp_from_i64(W2[6][j] as i64))
                    - pr4_7[3].clone() * fp_expr(fp_from_i64(W2[7][j] as i64));
                sum = sum - fp_expr(fp_from_i64(B2[j] as i64) * scale);
                polys.push(s.clone() * sum);
            }
            polys
        });

        // ── 9-bit signed range check (r ∈ [-S/2, S/2]) ──────────────
        // Layout (1 row):
        //   advice[0] = value (the r value)
        //   bit_cols[0..16] = bits
        // Constraints:
        //   bits[0..9] boolean; bits[9..16] = 0
        //   value + 128 = Σ bits[i] * 2^i  (i=0..8)
        //   bits[8] * (Σ_{i=0..7} bits[i]) = 0   (overflow: if high bit set, all lower must be 0)
        meta.create_gate("rc9_signed", |meta| {
            let s = meta.query_selector(s_rc9);
            let value = meta.query_advice(advice[0], Rotation::cur());
            let mut polys = Vec::new();

            // Booleanity for bits 0..8.
            for i in 0..9 {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                polys.push(s.clone() * b.clone() * (fp_expr(Fp::one()) - b));
            }
            // bits 9..16 must be zero.
            for i in 9..NUM_BIT_COLS {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                polys.push(s.clone() * b);
            }
            // Sum constraint: value + 128 = Σ_{i=0..8} bits[i] * 2^i.
            let mut sum = fp_expr(Fp::zero());
            for i in 0..9 {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                sum = sum + b * fp_expr(Fp::from(1u64 << i as u32));
            }
            polys.push(s.clone() * (sum - value - fp_expr(Fp::from(128u64))));
            // Overflow constraint: if bit_8 = 1 then all lower bits must be 0.
            let b_8 = meta.query_advice(bit_cols[8], Rotation::cur());
            let mut lower_sum = fp_expr(Fp::zero());
            for i in 0..8 {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                lower_sum = lower_sum + b;
            }
            polys.push(s.clone() * b_8 * lower_sum);

            polys
        });

        // ── 15-bit unsigned range check (magnitude ∈ [0, 2^15)) ─────
        meta.create_gate("rc15_unsigned", |meta| {
            let s = meta.query_selector(s_rc15);
            let value = meta.query_advice(advice[0], Rotation::cur());
            let mut polys = Vec::new();

            // Booleanity for bits 0..15.
            for i in 0..15 {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                polys.push(s.clone() * b.clone() * (fp_expr(Fp::one()) - b));
            }
            // bits 15 must be zero.
            for i in 15..NUM_BIT_COLS {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                polys.push(s.clone() * b);
            }
            // Sum constraint: value = Σ_{i=0..14} bits[i] * 2^i.
            let mut sum = fp_expr(Fp::zero());
            for i in 0..15 {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                sum = sum + b * fp_expr(Fp::from(1u64 << i as u32));
            }
            polys.push(s * (sum - value));

            polys
        });

        // ── 16-bit signed range check (output ∈ [-2^15, 2^15)) ──────
        meta.create_gate("rc16_signed", |meta| {
            let s = meta.query_selector(s_rc16);
            let value = meta.query_advice(advice[0], Rotation::cur());
            let mut polys = Vec::new();

            // Booleanity for all 16 bits.
            for i in 0..NUM_BIT_COLS {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                polys.push(s.clone() * b.clone() * (fp_expr(Fp::one()) - b));
            }
            // Sum constraint: value + 2^15 = Σ_{i=0..15} bits[i] * 2^i.
            let mut sum = fp_expr(Fp::zero());
            for i in 0..NUM_BIT_COLS {
                let b = meta.query_advice(bit_cols[i], Rotation::cur());
                sum = sum + b * fp_expr(Fp::from(1u64 << i as u32));
            }
            polys.push(s * (sum - value - fp_expr(Fp::from(1u64 << 15))));

            polys
        });

        BoundedMlpConfig {
            advice,
            bit_cols,
            instance,
            s_layer1,
            s_relu,
            s_layer2,
            s_rc9,
            s_rc15,
            s_rc16,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // ── Region 1: Layer 1 dense ─────────────────────────────────
        let (input_cells, pre_relu_cells, r1_cells) = layouter.assign_region(
            || "layer1_dense",
            |mut region| {
                config.s_layer1.enable(&mut region, 0)?;

                let input_cells: Vec<_> = (0..4)
                    .map(|i| {
                        region.assign_advice(
                            || format!("input[{i}]"),
                            config.advice[i],
                            0,
                            || self.input[i],
                        )
                    })
                    .collect::<Result<_, _>>()?;

                let mut pre_relu_cells = Vec::with_capacity(8);
                let mut r1_cells = Vec::with_capacity(8);
                for j in 0..8 {
                    let pre = region.assign_advice(
                        || format!("pre_relu[{j}]"),
                        config.advice[0],
                        j + 1,
                        || self.hidden_pre_relu[j],
                    )?;
                    let r1c = region.assign_advice(
                        || format!("r1[{j}]"),
                        config.advice[1],
                        j + 1,
                        || self.r1[j],
                    )?;
                    pre_relu_cells.push(pre);
                    r1_cells.push(r1c);
                }
                Ok((input_cells, pre_relu_cells, r1_cells))
            },
        )?;

        // ── Region 2: ReLU (8 single-row regions) + magnitude cells ─
        let mut post_relu_cells = Vec::with_capacity(8);
        let mut magnitude_cells = Vec::with_capacity(8);
        for j in 0..8 {
            let pre = pre_relu_cells[j].clone();
            let (post, mag) = layouter.assign_region(
                || format!("relu[{j}]"),
                |mut region| {
                    config.s_relu.enable(&mut region, 0)?;
                    pre.copy_advice(
                        || format!("pre_relu[{j}] copy"),
                        &mut region,
                        config.advice[0],
                        0,
                    )?;
                    region.assign_advice(
                        || format!("sign[{j}]"),
                        config.advice[1],
                        0,
                        || self.sign_bits[j],
                    )?;
                    let mag = region.assign_advice(
                        || format!("magnitude[{j}]"),
                        config.advice[2],
                        0,
                        || self.magnitudes[j],
                    )?;
                    let post = region.assign_advice(
                        || format!("post_relu[{j}]"),
                        config.advice[3],
                        0,
                        || self.hidden_post_relu[j],
                    )?;
                    Ok((post, mag))
                },
            )?;
            post_relu_cells.push(post);
            magnitude_cells.push(mag);
        }

        // ── Region 3: Layer 2 dense ─────────────────────────────────
        let (output_cells, r2_cells) = layouter.assign_region(
            || "layer2_dense",
            |mut region| {
                config.s_layer2.enable(&mut region, 0)?;
                for (i, cell) in post_relu_cells.iter().take(4).enumerate() {
                    cell.copy_advice(
                        || format!("post_relu[{i}] copy"),
                        &mut region,
                        config.advice[i],
                        0,
                    )?;
                }
                for (i, cell) in post_relu_cells.iter().skip(4).enumerate() {
                    cell.copy_advice(
                        || format!("post_relu[{}] copy", i + 4),
                        &mut region,
                        config.advice[i],
                        1,
                    )?;
                }
                let mut output_cells = Vec::with_capacity(4);
                let mut r2_cells = Vec::with_capacity(4);
                for j in 0..4 {
                    let out = region.assign_advice(
                        || format!("output[{j}]"),
                        config.advice[0],
                        2 + j,
                        || self.output[j],
                    )?;
                    let r2c = region.assign_advice(
                        || format!("r2[{j}]"),
                        config.advice[1],
                        2 + j,
                        || self.r2[j],
                    )?;
                    output_cells.push(out);
                    r2_cells.push(r2c);
                }
                Ok((output_cells, r2_cells))
            },
        )?;

        // ── Range-check regions ─────────────────────────────────────
        // Each range check is a single-row region: the value at
        // advice[0] (copied from the source region), bits at
        // bit_cols[0..16].

        // 9-bit signed range checks for r1[j].
        for j in 0..8 {
            let r1c = r1_cells[j].clone();
            let bits = self.r1_bits[j];
            layouter.assign_region(
                || format!("rc9_r1[{j}]"),
                |mut region| {
                    config.s_rc9.enable(&mut region, 0)?;
                    r1c.copy_advice(
                        || format!("r1[{j}] copy"),
                        &mut region,
                        config.advice[0],
                        0,
                    )?;
                    for (i, b) in bits.iter().enumerate() {
                        region.assign_advice(
                            || format!("r1[{j}] bit {i}"),
                            config.bit_cols[i],
                            0,
                            || *b,
                        )?;
                    }
                    Ok(())
                },
            )?;
        }

        // 9-bit signed range checks for r2[j].
        for j in 0..4 {
            let r2c = r2_cells[j].clone();
            let bits = self.r2_bits[j];
            layouter.assign_region(
                || format!("rc9_r2[{j}]"),
                |mut region| {
                    config.s_rc9.enable(&mut region, 0)?;
                    r2c.copy_advice(
                        || format!("r2[{j}] copy"),
                        &mut region,
                        config.advice[0],
                        0,
                    )?;
                    for (i, b) in bits.iter().enumerate() {
                        region.assign_advice(
                            || format!("r2[{j}] bit {i}"),
                            config.bit_cols[i],
                            0,
                            || *b,
                        )?;
                    }
                    Ok(())
                },
            )?;
        }

        // 15-bit unsigned range checks for magnitude[j].
        for j in 0..8 {
            let magc = magnitude_cells[j].clone();
            let bits = self.magnitude_bits[j];
            layouter.assign_region(
                || format!("rc15_magnitude[{j}]"),
                |mut region| {
                    config.s_rc15.enable(&mut region, 0)?;
                    magc.copy_advice(
                        || format!("magnitude[{j}] copy"),
                        &mut region,
                        config.advice[0],
                        0,
                    )?;
                    for (i, b) in bits.iter().enumerate() {
                        region.assign_advice(
                            || format!("magnitude[{j}] bit {i}"),
                            config.bit_cols[i],
                            0,
                            || *b,
                        )?;
                    }
                    Ok(())
                },
            )?;
        }

        // 16-bit signed range checks for output[j].
        for j in 0..4 {
            let outc = output_cells[j].clone();
            let bits = self.output_bits[j];
            layouter.assign_region(
                || format!("rc16_output[{j}]"),
                |mut region| {
                    config.s_rc16.enable(&mut region, 0)?;
                    outc.copy_advice(
                        || format!("output[{j}] copy"),
                        &mut region,
                        config.advice[0],
                        0,
                    )?;
                    for (i, b) in bits.iter().enumerate() {
                        region.assign_advice(
                            || format!("output[{j}] bit {i}"),
                            config.bit_cols[i],
                            0,
                            || *b,
                        )?;
                    }
                    Ok(())
                },
            )?;
        }

        // ── Public-input binding ────────────────────────────────────
        for (i, cell) in input_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, i)?;
        }
        for (j, cell) in output_cells.iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, 4 + j)?;
        }

        Ok(())
    }
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

    #[test]
    fn canonical_input_circuit_satisfies_mock_prover() {
        let circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let k = 9;
        let prover = MockProver::run(k, &circuit, vec![instance]).expect("mock prover runs");
        prover.verify().expect("canonical assignment must satisfy constraints");
    }

    #[test]
    fn wrong_output_fails_mock_prover() {
        let circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let mut wrong_output = CANONICAL_OUTPUT;
        wrong_output[0] += 1;
        let instance = instance_for(CANONICAL_INPUT, wrong_output);
        let k = 9;
        let prover = MockProver::run(k, &circuit, vec![instance]).expect("mock prover runs");
        assert!(prover.verify().is_err(), "wrong output must fail constraints");
    }

    /// Soundness regression: prove that the prover cannot satisfy
    /// the circuit by lying about a single hidden_pre_relu value
    /// and absorbing the slack into the corresponding r1[j]. Pre-
    /// range-check, this attack worked (constraint
    /// `pre*S + r = with_bias` was trivially satisfiable for any
    /// pre by choosing r = with_bias - pre*S). The 9-bit range
    /// check on r1[j] closes this attack — the lying r value is
    /// out of [-128, 128] and bit decomposition fails.
    #[test]
    fn slack_absorption_attack_is_rejected_by_range_check() {
        let mut circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        // Tamper: shift hidden_pre_relu[0] by +1 and adjust r1[0] by -S
        // (the absorbed slack). The dense equation still holds with
        // these adjusted values, but |r1[0]| now exceeds S/2 = 128.
        let pre_orig = 57i64;
        let r_orig = circuit.r1[0];
        let _ = r_orig;
        circuit.hidden_pre_relu[0] = known_i64(pre_orig + 1);
        circuit.r1[0] = known_i64(
            // r1[0] must now equal (with_bias - (pre+1)*S) = original_r - S = original_r - 256
            // For our canonical: with_bias=14624, original_r=14624 - 57*256 = 14624 - 14592 = 32.
            // New r = 32 - 256 = -224. Out of [-128, 128].
            32 - 256,
        );
        // Adjust the bit decomposition to "honestly" reflect the new
        // r value — i.e., decompose -224 + 128 = -96 (negative, can't
        // be decomposed in unsigned bits). Use 0 bits to force the
        // sum mismatch.
        circuit.r1_bits[0] = [Value::known(Fp::zero()); NUM_BIT_COLS];

        let instance = instance_for(CANONICAL_INPUT, CANONICAL_OUTPUT);
        let k = 9;
        let prover = MockProver::run(k, &circuit, vec![instance]).expect("mock prover runs");
        assert!(
            prover.verify().is_err(),
            "slack-absorption attack must fail under the 9-bit range check on r1"
        );
    }

    #[test]
    fn fp_from_i64_round_trips_through_signed_embedding() {
        let neg5 = fp_from_i64(-5);
        let pos5 = fp_from_i64(5);
        assert_eq!(neg5 + pos5, Fp::zero());
    }
}
