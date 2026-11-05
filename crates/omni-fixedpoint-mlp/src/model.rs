//! [`FixedPointMlp`]: a validated sequence of dense and ReLU layers under
//! the [`crate::contract`], its tensor encoding, and the accounting a run
//! of it reports.
//!
//! **Accounting.** A run reports the model's
//! element counts, derived from the validated model, never configured:
//! - measured input = `input_dim` (elements), measured output = `output_dim`;
//! - base units = input + output;
//! - a `FlopsEstimate` stage contribution of `2 ×` the multiply-accumulates
//!   one evaluation performs (Σ `input_dim × output_dim` over the dense
//!   layers).
//!
//! For the canonical model: 16 / 8 elements, 24 base units, 2,304 FLOPs.

use std::fmt;

use crate::canonical::{B1, B2, B3, W1, W2, W3};
use crate::contract::{SCALE_LOG2, relu_i16, round_half_away_from_zero_div, saturate_to_i16};

/// Why a layer set or an input was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// The layer list is empty.
    NoDenseLayer,
    /// The first layer is not dense (only a dense layer fixes the input size).
    FirstLayerNotDense { layer: usize },
    /// A dimension is outside `1..=MAX_DIMENSION`.
    BadDimension {
        layer: usize,
        input_dim: usize,
        output_dim: usize,
        max: usize,
    },
    /// The weight count is not `input_dim × output_dim`.
    WeightCount {
        layer: usize,
        input_dim: usize,
        output_dim: usize,
        expected: usize,
        got: usize,
    },
    /// The bias count is not `output_dim`.
    BiasCount {
        layer: usize,
        expected: usize,
        got: usize,
    },
    /// A dense layer does not take as many values as the layer before gives.
    DimensionMismatch {
        layer: usize,
        input_dim: usize,
        previous_output: usize,
    },
    /// The input does not hold `input_dim` values.
    InputLength { expected: usize, got: usize },
    /// The encoded input is not `2 × input_dim` bytes.
    InputBytes {
        expected: usize,
        values: usize,
        got: usize,
    },
    /// The model's counts do not fit in `u64`.
    AccountingOverflow,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::NoDenseLayer => write!(f, "a model needs at least one dense layer"),
            ModelError::FirstLayerNotDense { layer } => write!(
                f,
                "layer {layer}: the first layer must be dense (it fixes the input size)"
            ),
            ModelError::BadDimension {
                layer,
                input_dim,
                output_dim,
                max,
            } => write!(
                f,
                "layer {layer}: dimensions must be between 1 and {max}, got {input_dim} x {output_dim}"
            ),
            ModelError::WeightCount {
                layer,
                input_dim,
                output_dim,
                expected,
                got,
            } => write!(
                f,
                "layer {layer}: expected {expected} weights ({input_dim} x {output_dim}), got {got}"
            ),
            ModelError::BiasCount {
                layer,
                expected,
                got,
            } => write!(f, "layer {layer}: expected {expected} biases, got {got}"),
            ModelError::DimensionMismatch {
                layer,
                input_dim,
                previous_output,
            } => write!(
                f,
                "layer {layer}: takes {input_dim} values, but the previous layer gives {previous_output}"
            ),
            ModelError::InputLength { expected, got } => {
                write!(f, "expected {expected} input values, got {got}")
            }
            ModelError::InputBytes {
                expected,
                values,
                got,
            } => write!(
                f,
                "expected {expected} input bytes ({values} x i16 little-endian), got {got}"
            ),
            ModelError::AccountingOverflow => {
                write!(f, "the model's element or FLOP counts do not fit in u64")
            }
        }
    }
}

impl std::error::Error for ModelError {}

/// The largest dimension a dense layer may have. With at most `2^32`
/// products of magnitude at most `2^30` each, plus a bias of magnitude at
/// most `2^23`, the `i64` accumulator cannot overflow. (A loader applies
/// much smaller limits of its own.)
pub const MAX_DIMENSION: usize = u32::MAX as usize;

/// One dense layer:
/// `output[j] = sat_i16(rhaz((Σ_i input[i] · W[i][j]) + (B[j] << 8), 8))`.
///
/// `weights` is row-major `[input_dim][output_dim]`, the layout of the `W`
/// arrays in `canonical_spec.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLayer {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<i16>,
    biases: Vec<i16>,
}

impl DenseLayer {
    /// A dense layer from flat row-major weights. Refuses a dimension outside
    /// `1..=MAX_DIMENSION` and weight or bias counts that do not match the
    /// dimensions. `layer` only labels the error.
    pub fn new(
        layer: usize,
        input_dim: usize,
        output_dim: usize,
        weights: Vec<i16>,
        biases: Vec<i16>,
    ) -> Result<Self, ModelError> {
        let bad = ModelError::BadDimension {
            layer,
            input_dim,
            output_dim,
            max: MAX_DIMENSION,
        };
        let in_range = |d: usize| (1..=MAX_DIMENSION).contains(&d);
        if !in_range(input_dim) || !in_range(output_dim) {
            return Err(bad);
        }
        let expected = input_dim.checked_mul(output_dim).ok_or(bad)?;
        if weights.len() != expected {
            return Err(ModelError::WeightCount {
                layer,
                input_dim,
                output_dim,
                expected,
                got: weights.len(),
            });
        }
        if biases.len() != output_dim {
            return Err(ModelError::BiasCount {
                layer,
                expected: output_dim,
                got: biases.len(),
            });
        }
        Ok(Self {
            input_dim,
            output_dim,
            weights,
            biases,
        })
    }

    /// A dense layer from fixed-size arrays, as the canonical constants are
    /// declared. Cannot fail: the shapes come from the types.
    pub fn from_arrays<const IN: usize, const OUT: usize>(
        weights: &[[i16; OUT]; IN],
        biases: &[i16; OUT],
    ) -> Self {
        Self {
            input_dim: IN,
            output_dim: OUT,
            weights: weights.iter().flatten().copied().collect(),
            biases: biases.to_vec(),
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// The row-major `[input_dim][output_dim]` weights.
    pub fn weights(&self) -> &[i16] {
        &self.weights
    }

    pub fn biases(&self) -> &[i16] {
        &self.biases
    }

    /// `input_dim × output_dim`: one multiply-accumulate per weight.
    pub fn multiply_accumulates(&self) -> Option<u64> {
        (self.input_dim as u64).checked_mul(self.output_dim as u64)
    }

    /// `input.len()` must equal `input_dim` (checked by the caller).
    fn apply(&self, input: &[i16]) -> Vec<i16> {
        debug_assert_eq!(input.len(), self.input_dim);
        (0..self.output_dim)
            .map(|j| {
                let acc: i64 = input
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| i64::from(x) * i64::from(self.weights[i * self.output_dim + j]))
                    .sum();
                let with_bias = acc + (i64::from(self.biases[j]) << SCALE_LOG2);
                saturate_to_i16(round_half_away_from_zero_div(with_bias, SCALE_LOG2))
            })
            .collect()
    }
}

/// A layer of the contract: dense, or ReLU (`max(x, 0)` per value; the size
/// is unchanged).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Layer {
    Dense(DenseLayer),
    Relu,
}

/// The counts one run of a model reports, derived from the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlpAccounting {
    /// Measured input count: the model's input elements.
    pub input_elements: u64,
    /// Measured output count: the model's output elements.
    pub output_elements: u64,
    /// `input_elements + output_elements`.
    pub base_units: u64,
    /// The `FlopsEstimate` stage contribution: `2 ×` the multiply-accumulates.
    pub flops_estimate: u64,
}

/// A checked sequence of layers. The first layer is dense, and each dense
/// layer takes as many values as the layer before it gives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedPointMlp {
    layers: Vec<Layer>,
    input_dim: usize,
    output_dim: usize,
}

impl FixedPointMlp {
    pub fn new(layers: Vec<Layer>) -> Result<Self, ModelError> {
        let input_dim = match layers.first() {
            Some(Layer::Dense(d)) => d.input_dim,
            Some(Layer::Relu) => return Err(ModelError::FirstLayerNotDense { layer: 0 }),
            None => return Err(ModelError::NoDenseLayer),
        };
        let mut width = input_dim;
        for (index, layer) in layers.iter().enumerate() {
            if let Layer::Dense(d) = layer {
                if d.input_dim != width {
                    return Err(ModelError::DimensionMismatch {
                        layer: index,
                        input_dim: d.input_dim,
                        previous_output: width,
                    });
                }
                width = d.output_dim;
            }
        }
        Ok(Self {
            layers,
            input_dim,
            output_dim: width,
        })
    }

    /// The frozen `production-fixedpoint-mlp-v1` model:
    /// Dense(16 → 32), ReLU, Dense(32 → 16), ReLU, Dense(16 → 8), with the
    /// weights and biases of [`crate::canonical`].
    pub fn canonical() -> Self {
        Self::new(vec![
            Layer::Dense(DenseLayer::from_arrays(&W1, &B1)),
            Layer::Relu,
            Layer::Dense(DenseLayer::from_arrays(&W2, &B2)),
            Layer::Relu,
            Layer::Dense(DenseLayer::from_arrays(&W3, &B3)),
        ])
        .expect("the canonical layer shapes chain")
    }

    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Evaluate on `input`, which must hold exactly `input_dim` values.
    pub fn evaluate(&self, input: &[i16]) -> Result<Vec<i16>, ModelError> {
        if input.len() != self.input_dim {
            return Err(ModelError::InputLength {
                expected: self.input_dim,
                got: input.len(),
            });
        }
        let mut values = input.to_vec();
        for layer in &self.layers {
            values = match layer {
                Layer::Dense(d) => d.apply(&values),
                Layer::Relu => values.into_iter().map(relu_i16).collect(),
            };
        }
        Ok(values)
    }

    /// Evaluate on the contract's tensor encoding: `input_dim` values as
    /// `i16` little-endian with no header, and the output encoded the same
    /// way. For the canonical model, a 32-byte input and a 16-byte output.
    pub fn evaluate_bytes(&self, input: &[u8]) -> Result<Vec<u8>, ModelError> {
        let expected = self.input_dim * 2;
        if input.len() != expected {
            return Err(ModelError::InputBytes {
                expected,
                values: self.input_dim,
                got: input.len(),
            });
        }
        let (pairs, _) = input.as_chunks::<2>();
        let values: Vec<i16> = pairs.iter().map(|b| i16::from_le_bytes(*b)).collect();
        Ok(self
            .evaluate(&values)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect())
    }

    /// The number of dense layers.
    pub fn dense_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, Layer::Dense(_)))
            .count()
    }

    /// Σ `input_dim × output_dim` over the dense layers: the exact number of
    /// multiply-accumulates one evaluation performs. `None` if it does not
    /// fit in `u64`.
    pub fn multiply_accumulates(&self) -> Option<u64> {
        self.layers.iter().try_fold(0u64, |sum, l| match l {
            Layer::Dense(d) => sum.checked_add(d.multiply_accumulates()?),
            Layer::Relu => Some(sum),
        })
    }

    /// What one run of this model reports: its element counts, their
    /// sum, and `2 ×` its multiply-accumulates as a FLOP estimate.
    pub fn accounting(&self) -> Result<MlpAccounting, ModelError> {
        let input_elements = self.input_dim as u64;
        let output_elements = self.output_dim as u64;
        let base_units = input_elements
            .checked_add(output_elements)
            .ok_or(ModelError::AccountingOverflow)?;
        let flops_estimate = self
            .multiply_accumulates()
            .and_then(|m| m.checked_mul(2))
            .ok_or(ModelError::AccountingOverflow)?;
        Ok(MlpAccounting {
            input_elements,
            output_elements,
            base_units,
            flops_estimate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen input and output of `canonical_spec.json`
    /// (`canonical_evaluation`), as the proof crate pins them.
    const CANONICAL_INPUT: [i16; 16] = [
        -5, 10, 20, -100, 7, -3, 14, 25, -8, 1, 11, -22, 4, -1, 17, 9,
    ];
    const CANONICAL_OUTPUT: [i16; 8] = [-32, -12, 6, 25, -21, -3, 16, -29];

    fn dense(input_dim: usize, output_dim: usize, w: &[i16], b: &[i16]) -> DenseLayer {
        DenseLayer::new(0, input_dim, output_dim, w.to_vec(), b.to_vec()).unwrap()
    }

    fn le(values: &[i16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn the_canonical_model_gives_the_frozen_output() {
        let m = FixedPointMlp::canonical();
        assert_eq!(
            m.evaluate(&CANONICAL_INPUT).unwrap(),
            CANONICAL_OUTPUT.to_vec()
        );
        assert_eq!(
            m.evaluate_bytes(&le(&CANONICAL_INPUT)).unwrap(),
            le(&CANONICAL_OUTPUT)
        );
        assert_eq!((m.input_dim(), m.output_dim()), (16, 8));
    }

    /// For the canonical model: 16 / 8 elements, 24 base units,
    /// 1,152 multiply-accumulates, so a 2,304 FLOP estimate.
    #[test]
    fn the_canonical_accounting() {
        let m = FixedPointMlp::canonical();
        assert_eq!(m.multiply_accumulates(), Some(16 * 32 + 32 * 16 + 16 * 8));
        assert_eq!(m.multiply_accumulates(), Some(1_152));
        assert_eq!(m.dense_layer_count(), 3);
        assert_eq!(
            m.accounting().unwrap(),
            MlpAccounting {
                input_elements: 16,
                output_elements: 8,
                base_units: 24,
                flops_estimate: 2_304,
            }
        );
    }

    /// Another shape's counts come from its own layers, ReLU adding none.
    #[test]
    fn another_shapes_accounting_derives_from_its_layers() {
        let m = FixedPointMlp::new(vec![
            Layer::Dense(dense(3, 5, &[0; 15], &[0; 5])),
            Layer::Relu,
            Layer::Dense(dense(5, 2, &[0; 10], &[0; 2])),
        ])
        .unwrap();
        assert_eq!(
            m.accounting().unwrap(),
            MlpAccounting {
                input_elements: 3,
                output_elements: 2,
                base_units: 5,
                flops_estimate: 2 * (15 + 10),
            }
        );
    }

    #[test]
    fn a_dense_layer_rounds_half_away_from_zero_and_saturates() {
        // with_bias = x·w + (b << 8). One input, one output.
        let cases: [(i16, i16, i16, i16); 8] = [
            (128, 1, 0, 1),                           //  0.5 ->  1
            (-128, 1, 0, -1),                         // -0.5 -> -1
            (127, 1, 0, 0),                           // < 0.5 -> 0
            (384, 1, 0, 2),                           //  1.5 ->  2
            (-384, 1, 0, -2),                         // -1.5 -> -2
            (0, 1, 3, 3),                             // the bias is in units of 1.0
            (i16::MAX, i16::MAX, i16::MAX, i16::MAX), // saturates high
            (i16::MIN, i16::MAX, i16::MIN, i16::MIN), // saturates low
        ];
        for (x, w, b, want) in cases {
            let m = FixedPointMlp::new(vec![Layer::Dense(dense(1, 1, &[w], &[b]))]).unwrap();
            assert_eq!(m.evaluate(&[x]).unwrap(), vec![want], "x={x} w={w} b={b}");
        }
    }

    #[test]
    fn relu_zeroes_negatives_and_keeps_the_size() {
        let id = dense(3, 3, &[256, 0, 0, 0, 256, 0, 0, 0, 256], &[0, 0, 0]);
        let m = FixedPointMlp::new(vec![Layer::Dense(id), Layer::Relu]).unwrap();
        assert_eq!(m.evaluate(&[-5, 0, 7]).unwrap(), vec![0, 0, 7]);
        assert_eq!(m.output_dim(), 3);
    }

    #[test]
    fn weights_are_row_major_input_by_output() {
        // W[i][j] at i·out + j: input 0 feeds output 1 only.
        let w = [0, 256, 0, 0, 0, 0]; // 2 inputs x 3 outputs
        let m = FixedPointMlp::new(vec![Layer::Dense(dense(2, 3, &w, &[0, 0, 0]))]).unwrap();
        assert_eq!(m.evaluate(&[9, 4]).unwrap(), vec![0, 9, 0]);
    }

    #[test]
    fn a_layer_set_whose_shapes_do_not_fit_is_refused() {
        assert_eq!(
            DenseLayer::new(2, 2, 3, vec![0; 5], vec![0; 3]),
            Err(ModelError::WeightCount {
                layer: 2,
                input_dim: 2,
                output_dim: 3,
                expected: 6,
                got: 5
            })
        );
        assert_eq!(
            DenseLayer::new(1, 2, 3, vec![0; 6], vec![0; 2]),
            Err(ModelError::BiasCount {
                layer: 1,
                expected: 3,
                got: 2
            })
        );
        assert!(matches!(
            DenseLayer::new(0, 0, 3, vec![], vec![0; 3]),
            Err(ModelError::BadDimension { input_dim: 0, .. })
        ));
        assert!(matches!(
            DenseLayer::new(0, 3, 0, vec![], vec![]),
            Err(ModelError::BadDimension { output_dim: 0, .. })
        ));
        assert_eq!(FixedPointMlp::new(vec![]), Err(ModelError::NoDenseLayer));
        assert_eq!(
            FixedPointMlp::new(vec![Layer::Relu, Layer::Dense(dense(1, 1, &[1], &[0]))]),
            Err(ModelError::FirstLayerNotDense { layer: 0 })
        );
        assert_eq!(
            FixedPointMlp::new(vec![
                Layer::Dense(dense(2, 3, &[0; 6], &[0; 3])),
                Layer::Relu,
                Layer::Dense(dense(4, 1, &[0; 4], &[0])),
            ]),
            Err(ModelError::DimensionMismatch {
                layer: 2,
                input_dim: 4,
                previous_output: 3
            })
        );
    }

    #[test]
    fn an_input_of_the_wrong_size_is_refused() {
        let m = FixedPointMlp::canonical();
        assert_eq!(
            m.evaluate(&[0; 15]),
            Err(ModelError::InputLength {
                expected: 16,
                got: 15
            })
        );
        for len in [0, 31, 33, 64] {
            assert_eq!(
                m.evaluate_bytes(&vec![0; len]),
                Err(ModelError::InputBytes {
                    expected: 32,
                    values: 16,
                    got: len
                })
            );
        }
    }

    #[test]
    fn errors_display_their_cause() {
        assert_eq!(
            ModelError::InputBytes {
                expected: 32,
                values: 16,
                got: 31
            }
            .to_string(),
            "expected 32 input bytes (16 x i16 little-endian), got 31"
        );
        let e: Box<dyn std::error::Error> = Box::new(ModelError::NoDenseLayer);
        assert_eq!(e.to_string(), "a model needs at least one dense layer");
    }
}
