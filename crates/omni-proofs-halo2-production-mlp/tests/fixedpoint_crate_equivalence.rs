//! OmniNode #131 item 1(a): the dependency-free evaluator crate
//! (`omni-fixedpoint-mlp`, used by omni-contributor's in-process runner)
//! against this crate's frozen canonical artefacts, which are unchanged.
//!
//! - Its canonical weights are this crate's `W1`..`B3`, and the arrays in
//!   `assets/canonical_spec.json`.
//! - Its canonical model equals `canonical_evaluate` on the frozen input and
//!   a 25,008-vector sweep, and every ground-truth corpus entry evaluates to
//!   its output and hashes.
//! - Its model identity is `EXPECTED_PRODUCTION_SPEC_HASH`, BLAKE3 of the
//!   spec file's exact bytes.
//!
//! Pure Rust; no `verify` / `prove` feature needed.

use omni_fixedpoint_mlp::{
    CANONICAL_SPEC_BLAKE3_HEX, CANONICAL_SPEC_NAME, CANONICAL_SPEC_VERSION, DenseLayer,
    FixedPointMlp, Layer, canonical as fx,
};
use omni_proofs_halo2_production_mlp::{
    CANONICAL_INPUT, CANONICAL_OUTPUT, EXPECTED_PRODUCTION_SPEC_HASH, PRODUCTION_SPEC_NAME,
    PRODUCTION_SPEC_VERSION, canonical, canonical_evaluate, encode_canonical_input,
    encode_canonical_output,
};
use serde_json::Value;

fn read(rel: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn read_json(rel: &str) -> Value {
    serde_json::from_slice(&read(rel)).unwrap_or_else(|e| panic!("parse {rel}: {e}"))
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_i64().expect("integer"))
        .collect()
}

fn i16s(v: &Value) -> Vec<i16> {
    ints(v)
        .into_iter()
        .map(|x| i16::try_from(x).expect("i16"))
        .collect()
}

#[test]
fn the_weights_are_this_crates_weights() {
    assert_eq!(fx::W1, canonical::W1);
    assert_eq!(fx::B1, canonical::B1);
    assert_eq!(fx::W2, canonical::W2);
    assert_eq!(fx::B2, canonical::B2);
    assert_eq!(fx::W3, canonical::W3);
    assert_eq!(fx::B3, canonical::B3);
}

#[test]
fn the_model_identity_is_this_crates_spec_hash() {
    let spec = read("assets/canonical_spec.json");
    let hex: String = EXPECTED_PRODUCTION_SPEC_HASH
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(CANONICAL_SPEC_BLAKE3_HEX, hex);
    assert_eq!(
        CANONICAL_SPEC_BLAKE3_HEX,
        blake3::hash(&spec).to_hex().as_str()
    );
    assert_eq!(CANONICAL_SPEC_NAME, PRODUCTION_SPEC_NAME);
    assert_eq!(CANONICAL_SPEC_VERSION, PRODUCTION_SPEC_VERSION);
}

#[test]
fn the_canonical_model_gives_the_frozen_output() {
    let m = FixedPointMlp::canonical();
    assert_eq!(
        m.evaluate(&CANONICAL_INPUT).unwrap(),
        CANONICAL_OUTPUT.to_vec()
    );
    assert_eq!(
        m.evaluate_bytes(&encode_canonical_input(&CANONICAL_INPUT))
            .unwrap(),
        encode_canonical_output(&CANONICAL_OUTPUT)
    );
}

/// A deterministic sweep over the whole i16 range, with the extremes and
/// saturating inputs: the two evaluators agree on every vector, values and
/// bytes.
#[test]
fn the_canonical_model_equals_canonical_evaluate_on_a_sweep() {
    let m = FixedPointMlp::canonical();
    let mut inputs: Vec<[i16; 16]> = vec![
        [0; 16],
        [i16::MAX; 16],
        [i16::MIN; 16],
        [1; 16],
        [-1; 16],
        std::array::from_fn(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN }),
        std::array::from_fn(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX }),
        std::array::from_fn(|i| (i as i16 - 8) * 128),
    ];
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..20_000 {
        inputs.push(std::array::from_fn(|_| next() as u16 as i16));
    }
    for _ in 0..5_000 {
        // Small magnitudes, where ReLU zeroing and rounding ties are frequent.
        inputs.push(std::array::from_fn(|_| (next() % 513) as i16 - 256));
    }
    assert_eq!(inputs.len(), 25_008);
    for input in inputs {
        let want = canonical_evaluate(input);
        assert_eq!(
            m.evaluate(&input).unwrap(),
            want.to_vec(),
            "input {input:?}"
        );
        assert_eq!(
            m.evaluate_bytes(&encode_canonical_input(&input)).unwrap(),
            encode_canonical_output(&want),
            "input {input:?}"
        );
    }
}

#[test]
fn every_ground_truth_corpus_entry_evaluates_to_its_output() {
    let corpus = read_json("tests/fixtures/corpus.json");
    let entries = corpus["entries"].as_array().expect("entries");
    assert_eq!(entries.len(), 16);
    let m = FixedPointMlp::canonical();
    for e in entries {
        let input = i16s(&e["input"]);
        assert_eq!(
            m.evaluate(&input).unwrap(),
            i16s(&e["output"]),
            "entry {}",
            e["label"]
        );
        let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
        let output_bytes = m.evaluate_bytes(&input_bytes).unwrap();
        assert_eq!(
            blake3::hash(&input_bytes).to_hex().as_str(),
            e["input_hash"].as_str().unwrap()
        );
        assert_eq!(
            blake3::hash(&output_bytes).to_hex().as_str(),
            e["output_hash"].as_str().unwrap()
        );
    }
}

#[test]
fn the_canonical_layer_set_is_the_one_in_the_spec_file() {
    let spec = read_json("assets/canonical_spec.json");
    let declared = spec["architecture"]["layers"].as_array().expect("layers");
    let m = FixedPointMlp::canonical();
    assert_eq!(declared.len(), m.layers().len());
    for (index, (d, layer)) in declared.iter().zip(m.layers()).enumerate() {
        match layer {
            Layer::Relu => assert_eq!(d["kind"], "ReLU", "layer {index}"),
            Layer::Dense(dense) => {
                assert_eq!(d["kind"], "Dense", "layer {index}");
                assert_eq!(d["input_dim"].as_u64().unwrap() as usize, dense.input_dim());
                assert_eq!(
                    d["output_dim"].as_u64().unwrap() as usize,
                    dense.output_dim()
                );
                let w = &spec["weights"][d["weights_ref"].as_str().unwrap()];
                let b = &spec["weights"][d["biases_ref"].as_str().unwrap()];
                assert_eq!(
                    ints(&w["shape"]),
                    vec![dense.input_dim() as i64, dense.output_dim() as i64]
                );
                assert_eq!(ints(&b["shape"]), vec![dense.output_dim() as i64]);
                let flat: Vec<i16> = w["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(i16s)
                    .collect();
                let rebuilt = DenseLayer::new(
                    index,
                    dense.input_dim(),
                    dense.output_dim(),
                    flat,
                    i16s(&b["values"]),
                )
                .unwrap();
                assert_eq!(&rebuilt, dense, "layer {index} differs from the spec file");
            }
        }
    }
    assert_eq!(spec["quantization"]["scale_log2"], 8);
    let ev = &spec["canonical_evaluation"];
    assert_eq!(
        m.evaluate(&i16s(&ev["input"])).unwrap(),
        i16s(&ev["output"])
    );
}
