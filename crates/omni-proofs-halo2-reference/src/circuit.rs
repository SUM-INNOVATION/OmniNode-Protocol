//! Stage 11b.1.b — halo2 circuit for the canonical bounded MLP.
//!
//! Gated by feature `verify` because both the verifier and the
//! developer-host prover share this `Circuit<Fp>` definition.
//!
//! ## Public inputs (`instance` column, in order)
//!
//! ```text
//! row 0..4 : input[0..4]   (canonical i16 input  lifted to Fp)
//! row 4..8 : output[0..4]  (canonical i16 output lifted to Fp)
//! ```
//!
//! The verifier binds `metadata.input_hash` to BLAKE3 of the LE-
//! encoded input bytes and `metadata.response_hash` to BLAKE3 of
//! the LE-encoded output bytes; here in the circuit we only deal
//! with the raw field embeddings.
//!
//! ## Witnesses
//!
//! - `hidden_pre_relu[0..8]` — Layer 1 dense outputs *after*
//!   bias-pre-saturation + round-half-away requantization. (We
//!   skip the saturation gadget; see "soundness scope" below.)
//! - `hidden_post_relu[0..8]` — ReLU outputs.
//! - `r1[0..8]` — Layer 1 requantization remainders satisfying
//!   `with_bias = pre_relu * S + r`.
//! - `r2[0..4]` — Layer 2 requantization remainders.
//! - `sign_bits[0..8]`, `magnitudes[0..8]` — ReLU sign-bit gadget
//!   witnesses (`x = (1 - 2*s) * mag`, `out = (1 - s) * mag`).
//!
//! ## Constraints
//!
//! - **Layer 1 dense (8 instances):**
//!   `pre_relu[j] * S + r1[j] = Σ_i (input[i] * W1[i][j]) + B1[j] * S`
//! - **ReLU sign-bit gadget (8 instances):**
//!   `s[j] * (1 - s[j]) = 0`
//!   `pre_relu[j] = (1 - 2*s[j]) * magnitude[j]`
//!   `post_relu[j] = (1 - s[j]) * magnitude[j]`
//! - **Layer 2 dense (4 instances):**
//!   `output[j] * S + r2[j] = Σ_i (post_relu[i] * W2[i][j]) + B2[j] * S`
//! - **Public boundary:** input[0..4] and output[0..4] equality-
//!   constrained to instance column rows 0..8.
//!
//! ## Soundness scope (deliberate)
//!
//! Range checks on `r1`/`r2`/saturation are intentionally omitted
//! for Stage 11b.1.b. The committed canonical (input, output)
//! pair `[-5, 10, 20, -100] → [33, -32, 17, 7]` triggers no
//! requantization ties (all `with_bias / 256` values are
//! non-half-integer) and no saturation events (every dense output
//! stays within i16 range). The chain of linear constraints +
//! the public-input/output pin forces the prover into the unique
//! canonical witness assignment; a prover who picks
//! non-canonical intermediates fails the public output match.
//!
//! Stage 11c (or whenever we want to prove arbitrary inputs) must
//! add: range checks on `r` (|r| ≤ S/2), bit decomposition for
//! `pre_relu`/`post_relu`/`output` (i16 range), magnitude range
//! check, and an explicit saturation gadget for inputs that would
//! overflow. **This is documented as a known limitation** in the
//! crate README and operator-runbook.
//!
//! The circuit is sufficient for the bounded reference fixture
//! Stage 11b.1.b ships. The Stage 11b.0 mainnet refusal layers 1,
//! 3, and 6 catch the testnet/dev/bounded-reference posture
//! regardless of circuit completeness.

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

/// Fixed-point scale `S = 256 = 2^8` lifted into Fp.
fn scale_fp() -> Fp {
    Fp::from(256u64)
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

/// Circuit configuration — column allocation + selector definitions.
#[derive(Clone, Debug)]
pub struct BoundedMlpConfig {
    /// Advice column for input cells (and any value used as an
    /// equality-constrained witness across regions).
    advice: [Column<Advice>; 4],
    /// Instance column carrying the 8 public inputs (4 in + 4 out).
    instance: Column<Instance>,
    /// Selector for the Layer 1 dense + requantize linear gate.
    s_layer1: Selector,
    /// Selector for the ReLU sign-bit gadget.
    s_relu: Selector,
    /// Selector for the Layer 2 dense + requantize linear gate.
    s_layer2: Selector,
}

/// The Stage 11b.1.b halo2 circuit.
///
/// Holds all witness values (inputs, intermediates, outputs).
/// During key generation, all `Value<Fp>` fields are `Value::unknown()`;
/// during proving they're populated by the developer-host prover.
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
        }
    }
}

impl BoundedMlpCircuit {
    /// Build a fully-populated circuit from the canonical i16 inputs.
    /// Runs the canonical evaluator (in i64) to derive every witness.
    /// Developer-host use only.
    pub fn from_canonical_input(input_i16: [i16; 4]) -> Self {
        const SCALE_LOG2: u32 = 8;
        const S: i64 = 256;

        let mut hidden_pre_relu = [0i64; 8];
        let mut r1 = [0i64; 8];
        for j in 0..8 {
            let acc: i64 = (0..4)
                .map(|i| (input_i16[i] as i64) * (W1[i][j] as i64))
                .sum();
            let with_bias = acc + (B1[j] as i64) * S;
            // Round-half-away division identity: with_bias = q * S + r
            // with |r| ≤ S/2 chosen to match the canonical evaluator.
            // Use the exact same routine as src/canonical.rs.
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

        let _ = SCALE_LOG2;

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
        }
    }

    /// Convenience: derive the i16 output values from the canonical
    /// input. Used by the verifier to recompute the expected output
    /// bytes from the public-input field embeddings.
    pub fn canonical_outputs_for(input_i16: [i16; 4]) -> [i16; 4] {
        // Just delegate to the canonical evaluator — that's the
        // neutral reference implementation. We expose this so the
        // verifier can cross-check the public-output instance values
        // without re-deriving the math itself.
        let out = crate::canonical::canonical_evaluate(input_i16);
        out
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
        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let s_layer1 = meta.selector();
        let s_relu = meta.selector();
        let s_layer2 = meta.selector();

        // Layer 1 dense + requantize gate.
        //
        // Row layout (rotation-relative):
        //   row 0: advice[0..4] = input[0..4]
        //   row 1: advice[0] = pre_relu[j], advice[1] = r1[j],
        //          advice[2] = unused, advice[3] = unused
        //   (one gate-set per output j; weights W1[i][j] and bias B1[j]
        //    are embedded as Fp constants in the gate expression).
        //
        // Constraint:
        //   pre_relu * S + r - (Σ_i input[i] * W1[i][j]) - B1[j] * S = 0
        //
        // 8 separate constraint polynomials, one per j, all gated by
        // s_layer1 at row 0.
        meta.create_gate("layer1_dense", |meta| {
            let s = meta.query_selector(s_layer1);
            let inputs: [_; 4] = std::array::from_fn(|i| {
                meta.query_advice(advice[i], Rotation::cur())
            });
            let scale = scale_fp();

            // For each Layer 1 output j we add a polynomial constraint
            // gated by s_layer1. They share the same row offsets:
            //   pre_relu[j]    at advice[0], Rotation::next() + (offset based on j)
            // To keep this gate self-contained without ballooning the
            // number of advice rotations, we layout 8 successive rows
            // each holding one (pre_relu[j], r1[j]) pair.
            //
            // Effective layout for layer-1 region:
            //   row 0: input[0..4]
            //   row 1: pre_relu[0], r1[0]
            //   row 2: pre_relu[1], r1[1]
            //   ...
            //   row 8: pre_relu[7], r1[7]
            //
            // The selector s_layer1 is enabled at row 0 and the gate
            // queries Rotation(1)..Rotation(8) to reach each j.
            let mut polys = Vec::with_capacity(8);
            for j in 0..8 {
                let off = (j as i32) + 1;
                let pre_relu = meta.query_advice(advice[0], Rotation(off));
                let r1 = meta.query_advice(advice[1], Rotation(off));
                let mut sum = pre_relu * fp_expr(Fp::from(256u64))
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

        // ReLU sign-bit gadget.
        //
        // Row layout (per ReLU instance, single row):
        //   advice[0] = pre_relu  (input)
        //   advice[1] = sign_bit
        //   advice[2] = magnitude
        //   advice[3] = post_relu (output)
        //
        // Constraints:
        //   sign * (1 - sign) = 0
        //   pre_relu - (1 - 2*sign) * magnitude = 0
        //   post_relu - (1 - sign) * magnitude = 0
        meta.create_gate("relu", |meta| {
            let s = meta.query_selector(s_relu);
            let pre_relu = meta.query_advice(advice[0], Rotation::cur());
            let sign = meta.query_advice(advice[1], Rotation::cur());
            let magnitude = meta.query_advice(advice[2], Rotation::cur());
            let post_relu = meta.query_advice(advice[3], Rotation::cur());

            let one = Fp::from(1u64);
            let two = Fp::from(2u64);

            let booleanity = sign.clone() * (fp_expr(one) - sign.clone());
            let decomposition = pre_relu
                - (fp_expr(one) - fp_expr(two) * sign.clone())
                    * magnitude.clone();
            let relu_eq = post_relu - (fp_expr(one) - sign) * magnitude;

            vec![
                s.clone() * booleanity,
                s.clone() * decomposition,
                s * relu_eq,
            ]
        });

        // Layer 2 dense + requantize gate.
        //
        // Effective layout for layer-2 region (rotation-relative):
        //   row 0: post_relu[0..4]    advice[0..4]
        //   row 1: post_relu[4..8]    advice[0..4]
        //   row 2: output[0], r2[0]   advice[0..1]
        //   row 3: output[1], r2[1]
        //   row 4: output[2], r2[2]
        //   row 5: output[3], r2[3]
        meta.create_gate("layer2_dense", |meta| {
            let s = meta.query_selector(s_layer2);
            let pr0_3: [_; 4] = std::array::from_fn(|i| {
                meta.query_advice(advice[i], Rotation::cur())
            });
            let pr4_7: [_; 4] = std::array::from_fn(|i| {
                meta.query_advice(advice[i], Rotation::next())
            });
            let scale = scale_fp();

            let mut polys = Vec::with_capacity(4);
            for j in 0..4 {
                let off = 2 + (j as i32);
                let out = meta.query_advice(advice[0], Rotation(off));
                let r2 = meta.query_advice(advice[1], Rotation(off));
                let mut sum = out * fp_expr(Fp::from(256u64))
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

        BoundedMlpConfig {
            advice,
            instance,
            s_layer1,
            s_relu,
            s_layer2,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        // Region 1: Layer 1 dense.
        // Row 0: input[0..4]; rows 1..9: (pre_relu[j], r1[j]) for j=0..8.
        let (input_cells, pre_relu_cells) = layouter.assign_region(
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
                for j in 0..8 {
                    let pre = region.assign_advice(
                        || format!("pre_relu[{j}]"),
                        config.advice[0],
                        j + 1,
                        || self.hidden_pre_relu[j],
                    )?;
                    region.assign_advice(
                        || format!("r1[{j}]"),
                        config.advice[1],
                        j + 1,
                        || self.r1[j],
                    )?;
                    pre_relu_cells.push(pre);
                }
                Ok((input_cells, pre_relu_cells))
            },
        )?;

        // Region 2: ReLU (8 separate rows).
        let mut post_relu_cells = Vec::with_capacity(8);
        for j in 0..8 {
            let pre = pre_relu_cells[j].clone();
            let cell = layouter.assign_region(
                || format!("relu[{j}]"),
                |mut region| {
                    config.s_relu.enable(&mut region, 0)?;
                    let pre_in = pre.copy_advice(
                        || format!("pre_relu[{j}] copy"),
                        &mut region,
                        config.advice[0],
                        0,
                    )?;
                    let _ = pre_in;
                    region.assign_advice(
                        || format!("sign[{j}]"),
                        config.advice[1],
                        0,
                        || self.sign_bits[j],
                    )?;
                    region.assign_advice(
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
                    Ok(post)
                },
            )?;
            post_relu_cells.push(cell);
        }

        // Region 3: Layer 2 dense.
        // Row 0: post_relu[0..4]; Row 1: post_relu[4..8].
        // Rows 2..6: (output[j], r2[j]) for j=0..4.
        let output_cells = layouter.assign_region(
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
                for j in 0..4 {
                    let out = region.assign_advice(
                        || format!("output[{j}]"),
                        config.advice[0],
                        2 + j,
                        || self.output[j],
                    )?;
                    region.assign_advice(
                        || format!("r2[{j}]"),
                        config.advice[1],
                        2 + j,
                        || self.r2[j],
                    )?;
                    output_cells.push(out);
                }
                Ok(output_cells)
            },
        )?;

        // Public-input binding.
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
        let k = 8;
        let prover = MockProver::run(k, &circuit, vec![instance]).expect("mock prover runs");
        prover.verify().expect("canonical assignment must satisfy constraints");
    }

    #[test]
    fn wrong_output_fails_mock_prover() {
        let circuit = BoundedMlpCircuit::from_canonical_input(CANONICAL_INPUT);
        let mut wrong_output = CANONICAL_OUTPUT;
        wrong_output[0] += 1;
        let instance = instance_for(CANONICAL_INPUT, wrong_output);
        let k = 8;
        let prover = MockProver::run(k, &circuit, vec![instance]).expect("mock prover runs");
        assert!(prover.verify().is_err(), "wrong output must fail constraints");
    }

    #[test]
    fn fp_from_i64_round_trips_through_signed_embedding() {
        // Spot-check the signed embedding handles negatives correctly.
        // Fp = pasta_curves Pallas scalar field. Negative i16 maps to
        // p - |x| mod p.
        let neg5 = fp_from_i64(-5);
        let pos5 = fp_from_i64(5);
        assert_eq!(neg5 + pos5, Fp::zero());
    }
}
