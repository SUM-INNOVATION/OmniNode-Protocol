//! Stage 11b.1.a — cross-framework equivalence integration test.
//!
//! Loads every framework manifest in `tests/fixtures/*.json` and
//! asserts each one validates against the canonical evaluator and
//! the canonical spec hashes. RUMUS, PyTorch, TensorFlow, and Caffe
//! are equal-status primary compatibility targets; this test treats
//! them all the same way.
//!
//! **Zero framework runtime dependency.** This test runs pure Rust;
//! it does not invoke PyTorch, TensorFlow, Caffe, or RUMUS. The
//! manifests are committed JSON files; the test parses them and
//! asserts arithmetic equivalence with the canonical evaluator plus
//! byte equality of the committed `spec_hash` / `input_hash` /
//! `output_hash` fields against fresh BLAKE3 derivations.

use omni_proofs_halo2_reference::{
    canonical_evaluate, encoding::encode_tensor_4xi16_le, FrameworkManifest, CANONICAL_INPUT,
    CANONICAL_OUTPUT, EXPECTED_SPEC_HASH,
};
use omni_zkml::ModelFramework;

fn expected_spec_hash_hex() -> String {
    let mut s = String::with_capacity(64);
    for b in EXPECTED_SPEC_HASH {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn load_manifest(filename: &str) -> FrameworkManifest {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(filename);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("failed to read fixture at {}: {e}", path.display())
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|e| {
        panic!("failed to parse {} as FrameworkManifest: {e}", path.display())
    })
}

fn all_cases() -> &'static [(&'static str, ModelFramework)] {
    &[
        ("framework_agnostic_manifest.json", ModelFramework::FrameworkAgnostic),
        ("pytorch_manifest.json",            ModelFramework::PyTorch),
        ("tensorflow_manifest.json",         ModelFramework::TensorFlow),
        ("caffe_manifest.json",              ModelFramework::Caffe),
        ("rumus_manifest.json",              ModelFramework::Rumus),
    ]
}

/// Each framework manifest validates against the canonical evaluator
/// (arithmetic equivalence) and against the canonical spec hashes
/// (provenance equivalence).
#[test]
fn every_framework_fixture_matches_canonical_evaluator() {
    let expected_hash_hex = expected_spec_hash_hex();
    for (filename, expected_framework) in all_cases() {
        let manifest = load_manifest(filename);
        manifest
            .validate_against_canonical(*expected_framework, &expected_hash_hex)
            .unwrap_or_else(|e| {
                panic!(
                    "{filename}: framework manifest does not validate against canonical \
                     evaluator: {e}"
                )
            });
    }
}

/// Pin the (input, output) pair across the fixture suite.
#[test]
fn canonical_input_produces_committed_output() {
    let manifest = load_manifest("framework_agnostic_manifest.json");
    let computed = canonical_evaluate(manifest.input);
    assert_eq!(computed, manifest.output);
}

/// The Rust canonical evaluator is the neutral reference
/// implementation. Its output must equal the canonical spec's
/// `canonical_evaluation.output` (the source of truth).
#[test]
fn rust_canonical_evaluator_matches_committed_spec_output() {
    assert_eq!(canonical_evaluate(CANONICAL_INPUT), CANONICAL_OUTPUT);
}

/// All five manifests must carry byte-identical `weights_hash`,
/// `spec_hash`, `input_hash`, and `output_hash` fields. This is
/// the explicit cross-framework hash-equivalence invariant.
#[test]
fn all_manifests_share_identical_hashes() {
    let manifests: Vec<(String, FrameworkManifest)> = all_cases()
        .iter()
        .map(|(f, _)| (f.to_string(), load_manifest(f)))
        .collect();

    let (ref_name, ref_manifest) = &manifests[0];
    let ref_weights = ref_manifest.weights_hash.clone();
    let ref_spec = ref_manifest.spec_hash.clone();
    let ref_input = ref_manifest.input_hash.clone();
    let ref_output = ref_manifest.output_hash.clone();

    for (name, m) in &manifests[1..] {
        assert_eq!(
            m.weights_hash, ref_weights,
            "{name} weights_hash differs from {ref_name}",
        );
        assert_eq!(
            m.spec_hash, ref_spec,
            "{name} spec_hash differs from {ref_name}",
        );
        assert_eq!(
            m.input_hash, ref_input,
            "{name} input_hash differs from {ref_name}",
        );
        assert_eq!(
            m.output_hash, ref_output,
            "{name} output_hash differs from {ref_name}",
        );
    }
}

/// Cross-check the committed hash values against fresh BLAKE3
/// derivations — confirms the JSON files weren't hand-typed wrong.
#[test]
fn committed_hashes_match_fresh_blake3_derivations() {
    let expected_spec = expected_spec_hash_hex();
    let expected_input = hex_blake3(&encode_tensor_4xi16_le(&CANONICAL_INPUT));
    let expected_output = hex_blake3(&encode_tensor_4xi16_le(&CANONICAL_OUTPUT));

    for (filename, _) in all_cases() {
        let m = load_manifest(filename);
        assert_eq!(m.weights_hash, expected_spec, "{filename} weights_hash drift");
        assert_eq!(
            m.spec_hash.as_deref(),
            Some(expected_spec.as_str()),
            "{filename} spec_hash drift",
        );
        assert_eq!(
            m.input_hash.as_deref(),
            Some(expected_input.as_str()),
            "{filename} input_hash drift",
        );
        assert_eq!(
            m.output_hash.as_deref(),
            Some(expected_output.as_str()),
            "{filename} output_hash drift",
        );
    }
}

/// RUMUS-specific test: the manifest is now produced via
/// `tools/rumus_export/` using `rumus = "0.4.0"`'s `fixed::FixedLinear`,
/// so its `generation_mode` is `LiveExport`. This pins the contract
/// added by the four-equal-primaries posture — a future change that
/// regresses RUMUS to `IntendedRepresentation` without surfacing the
/// blocker will fail here.
#[test]
fn rumus_manifest_is_live_export() {
    use omni_proofs_halo2_reference::GenerationMode;
    let m = load_manifest("rumus_manifest.json");
    assert_eq!(
        m.generation_mode,
        GenerationMode::LiveExport,
        "RUMUS manifest must declare LiveExport in Stage 11b.1.a — rumus 0.4.0 ships a \
         deterministic CPU FixedI16 path (fixed::FixedLinear + requantize). A regression to \
         IntendedRepresentation requires surfacing the blocker, not silently downgrading."
    );
    let meta = m
        .generator_metadata
        .as_ref()
        .expect("RUMUS manifest must carry generator_metadata recording its regen path");
    let runtime_mode = meta
        .get("runtime_mode")
        .and_then(|v| v.as_str())
        .expect("generator_metadata.runtime_mode must be set");
    assert_eq!(runtime_mode, "rumus-fixed-linear");
}

/// Caffe-specific test: when the fallback path is used,
/// `generation_mode` must be `PureNumpyCompatibility` and
/// `generator_metadata.runtime_mode` must be
/// `"pure-numpy-emulation"`. When real Caffe is used,
/// `generation_mode` must be `LiveExport` and `runtime_mode` must be
/// `"caffe-runtime"`. The committed fixture exercises whichever the
/// regen host produced.
#[test]
fn caffe_manifest_records_runtime_mode_consistently() {
    use omni_proofs_halo2_reference::GenerationMode;
    let m = load_manifest("caffe_manifest.json");
    let meta = m
        .generator_metadata
        .as_ref()
        .expect("Caffe manifest must carry generator_metadata recording its runtime mode");
    let runtime_mode = meta
        .get("runtime_mode")
        .and_then(|v| v.as_str())
        .expect("generator_metadata.runtime_mode must be set");
    let caffe_runtime_present = meta
        .get("caffe_runtime_present")
        .and_then(|v| v.as_bool())
        .expect("generator_metadata.caffe_runtime_present must be a bool");

    match m.generation_mode {
        GenerationMode::PureNumpyCompatibility => {
            assert_eq!(runtime_mode, "pure-numpy-emulation");
            assert!(
                !caffe_runtime_present,
                "PureNumpyCompatibility requires caffe_runtime_present=false",
            );
        }
        GenerationMode::LiveExport => {
            assert_eq!(runtime_mode, "caffe-runtime");
            assert!(
                caffe_runtime_present,
                "LiveExport requires caffe_runtime_present=true",
            );
        }
        other => panic!(
            "Caffe manifest generation_mode must be PureNumpyCompatibility or LiveExport, \
             got {other:?}"
        ),
    }
}
