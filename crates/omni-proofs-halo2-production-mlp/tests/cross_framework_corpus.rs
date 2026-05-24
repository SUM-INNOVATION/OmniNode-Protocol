//! Stage 11d.2 — cross-framework corpus equivalence test.
//!
//! Asserts:
//!   (a) every entry in `tests/fixtures/corpus.json` matches what
//!       the pure-Rust `canonical_evaluate` produces for its input;
//!   (b) every per-framework corpus file (`rumus_corpus.json`,
//!       `pytorch_corpus.json`, `tensorflow_corpus.json`,
//!       `caffe_corpus.json`, `framework_agnostic_corpus.json`)
//!       has the same set of (input, output, input_hash,
//!       output_hash) tuples as the ground-truth corpus, in the
//!       same order;
//!   (c) every corpus file declares the same `spec_hash` /
//!       `spec_name` / `spec_version`.
//!
//! Pure-Rust; no framework runtime invoked. Suitable for PR CI
//! (does NOT require the `verify`/`prove` features).

use omni_proofs_halo2_production_mlp::{
    canonical_evaluate,
    encoding::{encode_canonical_input, encode_canonical_output},
    EXPECTED_PRODUCTION_SPEC_HASH,
};

fn expected_spec_hash_hex() -> String {
    let mut s = String::with_capacity(64);
    for b in EXPECTED_PRODUCTION_SPEC_HASH {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_corpus(filename: &str) -> serde_json::Value {
    let path = fixtures_dir().join(filename);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn entry_input(entry: &serde_json::Value) -> [i16; 16] {
    let arr = entry["input"].as_array().expect("entry.input is array");
    assert_eq!(arr.len(), 16, "entry.input must have 16 elements");
    let mut out = [0i16; 16];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v.as_i64().expect("entry.input[i] is integer") as i16;
    }
    out
}

fn entry_output(entry: &serde_json::Value) -> [i16; 8] {
    let arr = entry["output"].as_array().expect("entry.output is array");
    assert_eq!(arr.len(), 8, "entry.output must have 8 elements");
    let mut out = [0i16; 8];
    for (i, v) in arr.iter().enumerate() {
        out[i] = v.as_i64().expect("entry.output[i] is integer") as i16;
    }
    out
}

#[test]
fn canonical_corpus_matches_pure_rust_evaluator() {
    let corpus = load_corpus("corpus.json");
    assert_eq!(
        corpus["spec_hash"].as_str().unwrap(),
        expected_spec_hash_hex(),
        "corpus.spec_hash drift vs build-time EXPECTED_PRODUCTION_SPEC_HASH"
    );
    let entries = corpus["entries"].as_array().expect("entries is array");
    // Criteria §1.2 Claim C2: committed corpus of `≥ 16 inputs`
    // spanning declared-domain boundaries. The Stage 11d.2 PR
    // commits exactly 16; downstream PRs MAY add more, but never
    // fewer. This assertion catches accidental deletions or
    // truncations during a regen.
    assert!(
        entries.len() >= 16,
        "Stage 11d.2 corpus must have ≥ 16 entries to satisfy criteria §1.2 Claim C2; \
         got {}",
        entries.len()
    );

    for entry in entries {
        let label = entry["label"].as_str().unwrap_or("?");
        let input = entry_input(entry);
        let claimed_output = entry_output(entry);
        let computed = canonical_evaluate(input);
        assert_eq!(
            claimed_output, computed,
            "corpus entry {label}: claimed output {claimed_output:?} differs from \
             canonical_evaluate({input:?}) = {computed:?}"
        );
        let input_hash = hex_blake3(&encode_canonical_input(&input));
        assert_eq!(
            entry["input_hash"].as_str().unwrap(),
            input_hash,
            "corpus entry {label}: input_hash drift"
        );
        let output_hash = hex_blake3(&encode_canonical_output(&computed));
        assert_eq!(
            entry["output_hash"].as_str().unwrap(),
            output_hash,
            "corpus entry {label}: output_hash drift"
        );
    }
}

fn assert_framework_corpus_matches(filename: &str, expected_framework: &str) {
    let truth = load_corpus("corpus.json");
    let fw = load_corpus(filename);

    assert_eq!(fw["spec_hash"], truth["spec_hash"], "{filename}: spec_hash drift");
    assert_eq!(fw["spec_name"], truth["spec_name"], "{filename}: spec_name drift");
    assert_eq!(
        fw["spec_version"], truth["spec_version"],
        "{filename}: spec_version drift"
    );
    assert_eq!(
        fw["framework"].as_str().unwrap_or(""),
        expected_framework,
        "{filename}: framework field mismatch"
    );

    let truth_entries = truth["entries"].as_array().unwrap();
    let fw_entries = fw["entries"].as_array().unwrap();
    assert_eq!(
        truth_entries.len(),
        fw_entries.len(),
        "{filename}: entry count differs from ground truth"
    );

    for (i, (t, f)) in truth_entries.iter().zip(fw_entries.iter()).enumerate() {
        assert_eq!(t["label"], f["label"], "{filename}[{i}]: label drift");
        assert_eq!(t["input"], f["input"], "{filename}[{i}]: input drift");
        assert_eq!(t["output"], f["output"], "{filename}[{i}]: output drift");
        assert_eq!(t["input_hash"], f["input_hash"], "{filename}[{i}]: input_hash drift");
        assert_eq!(t["output_hash"], f["output_hash"], "{filename}[{i}]: output_hash drift");
    }
}

#[test]
fn rumus_corpus_matches_ground_truth() {
    assert_framework_corpus_matches("rumus_corpus.json", "Rumus");
}

#[test]
fn pytorch_corpus_matches_ground_truth() {
    assert_framework_corpus_matches("pytorch_corpus.json", "PyTorch");
}

#[test]
fn tensorflow_corpus_matches_ground_truth() {
    assert_framework_corpus_matches("tensorflow_corpus.json", "TensorFlow");
}

#[test]
fn caffe_corpus_matches_ground_truth() {
    assert_framework_corpus_matches("caffe_corpus.json", "Caffe");
}

#[test]
fn framework_agnostic_corpus_matches_ground_truth() {
    assert_framework_corpus_matches("framework_agnostic_corpus.json", "FrameworkAgnostic");
}
