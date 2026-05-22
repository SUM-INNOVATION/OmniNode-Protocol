//! Stage 11b.1.a — cross-framework equivalence integration test.
//!
//! Loads every framework manifest in `tests/fixtures/*.json` and
//! asserts each one validates against the canonical evaluator. The
//! canonical evaluator (`omni_proofs_halo2_reference::canonical::
//! canonical_evaluate`) is the single source of truth — if any
//! framework's manifest disagrees, this test fails loudly and
//! identifies which framework drifted.
//!
//! **Zero framework runtime dependency.** This test runs pure Rust;
//! it does not invoke PyTorch, TensorFlow, Caffe, or RUMUS. The
//! manifests are committed JSON files; the test parses them and
//! asserts arithmetic equivalence with the canonical evaluator.

use omni_proofs_halo2_reference::{
    canonical_evaluate, FrameworkManifest, EXPECTED_SPEC_HASH,
};
use omni_zkml::ModelFramework;

fn expected_spec_hash_hex() -> String {
    let mut s = String::with_capacity(64);
    for b in EXPECTED_SPEC_HASH {
        s.push_str(&format!("{:02x}", b));
    }
    s
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

/// The architectural-validation invariant for Stage 11b.1.a.
///
/// Every framework manifest must:
///   - declare the matching `framework` field;
///   - reuse the canonical input bytes;
///   - report an output equal to the canonical evaluator's output
///     for that input (byte-for-byte);
///   - reuse the `weights_hash` equal to `EXPECTED_SPEC_HASH`.
///
/// If any framework drifts, the test surfaces which one.
#[test]
fn every_framework_fixture_matches_canonical_evaluator() {
    let expected_hash_hex = expected_spec_hash_hex();
    let cases: &[(&str, ModelFramework)] = &[
        ("framework_agnostic_manifest.json", ModelFramework::FrameworkAgnostic),
        ("pytorch_manifest.json",            ModelFramework::PyTorch),
        ("tensorflow_manifest.json",         ModelFramework::TensorFlow),
        ("caffe_manifest.json",              ModelFramework::Caffe),
        ("rumus_manifest.json",              ModelFramework::Rumus),
    ];

    for (filename, expected_framework) in cases {
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

/// Pin that the canonical evaluator's output for the canonical input
/// equals what every fixture manifest reports. Same content as the
/// above test but framed positively — a single concrete (input, output)
/// pair pinned across the fixture suite.
#[test]
fn canonical_input_produces_committed_output() {
    let manifest = load_manifest("framework_agnostic_manifest.json");
    let computed = canonical_evaluate(manifest.input);
    assert_eq!(computed, manifest.output,
               "canonical evaluator output drifted from FrameworkAgnostic fixture");
}

/// RUMUS-specific test: assert that the RUMUS manifest carries the
/// `IntendedRepresentation` generation_mode (since it is explicitly
/// not a live export). This pins the Stage 11b.1.a documentation
/// contract: a future PR that mistakenly upgrades RUMUS to
/// `LiveExport` without also wiring a real runtime path fails here.
#[test]
fn rumus_manifest_is_marked_intended_representation_until_runtime_lands() {
    use omni_proofs_halo2_reference::GenerationMode;
    let m = load_manifest("rumus_manifest.json");
    assert_eq!(
        m.generation_mode,
        GenerationMode::IntendedRepresentation,
        "RUMUS manifest must declare IntendedRepresentation in Stage 11b.1.a — \
         RUMUS does not yet expose a deterministic CPU fixed-point integer-dense path. \
         If a future stage upgrades to LiveExport, the regen path must actually run RUMUS."
    );
    assert!(
        m.notes.as_ref().is_some_and(|n| n.contains("RUMUS")),
        "RUMUS manifest must carry an explanatory `notes` field"
    );
}
